#include "pose_estimator.h"

#include <chrono>
#include <cmath>
#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>

#include <nlohmann/json.hpp>

#include "profiling.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <opencv2/imgproc.hpp>

#include "nelder_mead.h"
#include "visualizer.h"

namespace pose_matching {

PoseEstimator::PoseEstimator(const maskgen::CameraParams& camera_params,
                             const std::string& model_path, float model_scale)
    : camera_params_(camera_params), principal_axis_(2) {
  if (!mesh_.LoadFromFile(model_path, model_scale)) {
    throw std::runtime_error("Failed to load mesh: " + model_path);
  }

  const auto& verts = mesh_.vertices();
  const size_t n = verts.size() / 3;

  float min_p[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
  float max_p[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
  mesh_centroid_[0] = mesh_centroid_[1] = mesh_centroid_[2] = 0.0f;

  for (size_t i = 0; i < n; ++i) {
    for (int d = 0; d < 3; ++d) {
      float v = verts[i * 3 + d];
      min_p[d] = std::min(min_p[d], v);
      max_p[d] = std::max(max_p[d], v);
      mesh_centroid_[d] += v;
    }
  }
  for (int d = 0; d < 3; ++d) {
    mesh_centroid_[d] /= static_cast<float>(n);
    mesh_extent_[d] = max_p[d] - min_p[d];
  }

  float max_extent = 0;
  for (int d = 0; d < 3; ++d) {
    if (mesh_extent_[d] > max_extent) {
      max_extent = mesh_extent_[d];
      principal_axis_ = d;
    }
  }

  std::cout << "Mesh centroid: (" << mesh_centroid_[0] << ", " << mesh_centroid_[1]
            << ", " << mesh_centroid_[2] << ")\n";
  std::cout << "Mesh extents: (" << mesh_extent_[0] << ", " << mesh_extent_[1]
            << ", " << mesh_extent_[2] << ")\n";
  std::cout << "Principal axis: " << principal_axis_ << " (extent=" << max_extent << ")\n";

  maskgen::CameraParams render_params = camera_params_;
  render_params.eye_x = 0;
  render_params.eye_y = 0;
  render_params.eye_z = 0;
  render_params.target_x = 0;
  render_params.target_y = 0;
  render_params.target_z = 1;
  render_params.up_x = 0;
  render_params.up_y = -1;
  render_params.up_z = 0;

  camera_params_ = render_params;
  generator_ = std::make_unique<maskgen::MaskGenerator>(camera_params_);
}

void PoseEstimator::SetVisualizer(Visualizer* viz) { viz_ = viz; }

cv::Mat PoseEstimator::RenderPose(const Pose6D& pose) {
  maskgen::MeshPose mp;
  mp.tx = pose.tx;
  mp.ty = pose.ty;
  mp.tz = pose.tz;
  mp.rx = pose.rx;
  mp.ry = pose.ry;
  mp.rz = pose.rz;
  return generator_->Generate(mesh_, mp);
}

double PoseEstimator::ComputeIoU(const cv::Mat& a, const cv::Mat& b) const {
  int a_count = 0, b_count = 0, ab_count = 0;
  for (int r = 0; r < a.rows; ++r) {
    const auto* pa = a.ptr<uint8_t>(r);
    const auto* pb = b.ptr<uint8_t>(r);
    for (int c = 0; c < a.cols; ++c) {
      bool va = pa[c] > 127;
      bool vb = pb[c] > 127;
      a_count += va;
      b_count += vb;
      ab_count += (va && vb);
    }
  }
  if (a_count + b_count - ab_count == 0) return 0.0;
  return static_cast<double>(ab_count) / (a_count + b_count - ab_count);
}

static glm::mat4 RotationFromDirection(const glm::vec3& target_dir, float in_plane_angle,
                                       int principal_axis) {
  glm::vec3 source_axis(0, 0, 0);
  source_axis[principal_axis] = 1.0f;

  float dot = glm::dot(source_axis, target_dir);
  glm::mat4 R_align;

  if (dot > 0.9999f) {
    R_align = glm::mat4(1.0f);
  } else if (dot < -0.9999f) {
    glm::vec3 perp(0, 1, 0);
    if (std::abs(glm::dot(source_axis, perp)) > 0.9f) {
      perp = glm::vec3(1, 0, 0);
    }
    glm::vec3 rot_axis = glm::normalize(glm::cross(source_axis, perp));
    R_align = glm::rotate(glm::mat4(1.0f), glm::pi<float>(), rot_axis);
  } else {
    glm::vec3 axis = glm::normalize(glm::cross(source_axis, target_dir));
    float angle = glm::acos(glm::clamp(dot, -1.0f, 1.0f));
    R_align = glm::rotate(glm::mat4(1.0f), angle, axis);
  }

  glm::mat4 R_inplane = glm::rotate(glm::mat4(1.0f), in_plane_angle, source_axis);

  return R_align * R_inplane;
}

static void ExtractEulerZYX(const glm::mat4& R, double& rx, double& ry, double& rz) {
  glm::mat3 R3(R);
  float ry_f = glm::asin(glm::clamp(-R3[0][2], -1.0f, 1.0f));
  float rx_f = std::atan2(R3[1][2], R3[2][2]);
  float rz_f = std::atan2(R3[0][1], R3[0][0]);
  rx = static_cast<double>(rx_f);
  ry = static_cast<double>(ry_f);
  rz = static_cast<double>(rz_f);
}

static std::vector<glm::vec3> FibonacciSphere(int n) {
  std::vector<glm::vec3> points;
  points.reserve(n);
  const float golden = glm::pi<float>() * (3.0f - glm::sqrt(5.0f));
  for (int i = 0; i < n; ++i) {
    float theta = golden * static_cast<float>(i);
    float phi = glm::acos(1.0f - 2.0f * (static_cast<float>(i) + 0.5f) / static_cast<float>(n));
    points.emplace_back(glm::sin(phi) * glm::cos(theta),
                        glm::sin(phi) * glm::sin(theta),
                        glm::cos(phi));
  }
  return points;
}

std::vector<ScoredCandidate> PoseEstimator::CoarseSearch(
    const cv::Mat& input_mask, const cv::Mat& dt_input, const EstimationParams& params) {
  cv::Moments m = cv::moments(input_mask, true);
  if (m.m00 == 0) return {};

  double u_cx = m.m10 / m.m00;
  double v_cy = m.m01 / m.m00;

  double input_area = m.m00;

  double area_input = cv::countNonZero(input_mask);

  std::vector<double> depths;
  double depth_step = (params.depth_max - params.depth_min) / std::max(params.num_depth - 1, 1);
  for (int i = 0; i < params.num_depth; ++i) {
    depths.push_back(params.depth_min + i * depth_step);
  }

  if (input_area > 0) {
    double fx = camera_params_.fx;
    double fy = camera_params_.fy;
    double mesh_area_estimate = mesh_extent_[principal_axis_] *
                                *std::max_element(mesh_extent_, mesh_extent_ + 3);
    double estimated_depth =
        std::sqrt(mesh_area_estimate * fx * fy / input_area) * 0.5;
    if (estimated_depth > params.depth_min && estimated_depth < params.depth_max) {
      bool too_close = false;
      for (double d : depths) {
        if (std::abs(d - estimated_depth) < depth_step * 0.3) {
          too_close = true;
          break;
        }
      }
      if (!too_close) {
        depths.push_back(estimated_depth);
      }
    }
  }

  auto directions = FibonacciSphere(params.num_directions);
  std::vector<ScoredCandidate> candidates;
  candidates.reserve(directions.size() * params.num_in_plane * depths.size());

  int candidate_index = 0;
  for (const auto& dir : directions) {
    for (int ip = 0; ip < params.num_in_plane; ++ip) {
      float in_plane = 2.0f * glm::pi<float>() * static_cast<float>(ip) /
                       static_cast<float>(params.num_in_plane);

      glm::mat4 R = RotationFromDirection(dir, in_plane, principal_axis_);

      double rx, ry, rz;
      ExtractEulerZYX(R, rx, ry, rz);

      glm::vec4 c_rot = R * glm::vec4(mesh_centroid_[0], mesh_centroid_[1],
                                       mesh_centroid_[2], 1.0f);

      for (double depth : depths) {
        Pose6D pose;
        pose.rx = rx;
        pose.ry = ry;
        pose.rz = rz;

        double tz_total = depth;
        pose.tx = (u_cx - camera_params_.cx) * tz_total / camera_params_.fx - c_rot.x;
        pose.ty = (v_cy - camera_params_.cy) * tz_total / camera_params_.fy - c_rot.y;
        pose.tz = tz_total - c_rot.z;

        if (pose.tz < 0.01) continue;

        cv::Mat rendered = RenderPose(pose);
        double iou = ComputeIoU(rendered, input_mask);

        if (viz_) {
          viz_->LogCandidate(candidate_index, pose, rendered, iou);
        }
        candidate_index++;

        candidates.push_back({pose, iou, 1.0 - iou});
      }
    }
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const ScoredCandidate& a, const ScoredCandidate& b) {
              return a.cost < b.cost;
            });

  // Diversify: keep only one candidate per unique direction
  std::vector<ScoredCandidate> diverse;
  auto dir_key = [](const Pose6D& p) -> int {
    return static_cast<int>(p.rx * 100) + static_cast<int>(p.ry * 10000) +
           static_cast<int>(p.rz * 1000000);
  };
  std::set<int> seen_dirs;
  for (const auto& c : candidates) {
    // Group by similar direction: quantize rotation
    int key = static_cast<int>(c.pose.rx * 10) * 100000 +
              static_cast<int>(c.pose.ry * 10) * 1000 +
              static_cast<int>(c.pose.rz * 10);
    if (seen_dirs.insert(key).second) {
      diverse.push_back(c);
    }
    if (static_cast<int>(diverse.size()) >= params.top_k_coarse) break;
  }
  candidates = diverse;

  std::cout << "Top " << candidates.size() << " coarse candidates:\n";
  for (size_t i = 0; i < candidates.size(); ++i) {
    glm::mat4 Rc = glm::mat4(1.0f);
    Rc = glm::rotate(Rc, static_cast<float>(candidates[i].pose.rz), glm::vec3(0, 0, 1));
    Rc = glm::rotate(Rc, static_cast<float>(candidates[i].pose.ry), glm::vec3(0, 1, 0));
    Rc = glm::rotate(Rc, static_cast<float>(candidates[i].pose.rx), glm::vec3(1, 0, 0));
    glm::vec3 src(0, 0, 0); src[principal_axis_] = 1.0f;
    glm::vec3 d = glm::normalize(glm::vec3(Rc * glm::vec4(src, 0.0f)));
    std::cout << "  [" << i << "] iou=" << candidates[i].iou
              << " cost=" << candidates[i].cost
              << " tz=" << candidates[i].pose.tz
              << " dir=(" << d.x << "," << d.y << "," << d.z << ")\n";
  }

  if (viz_) {
    viz_->LogCoarseComplete(candidates);
  }

  return candidates;
}

SearchResult PoseEstimator::RefinePose(const ScoredCandidate& initial,
                                       const cv::Mat& input_mask,
                                       const cv::Mat& dt_input,
                                       int max_iterations,
                                       const NelderMeadOptions& nm_opts,
                                       int refine_index) {
  std::vector<double> x = {initial.pose.tx, initial.pose.ty, initial.pose.tz,
                           initial.pose.rx, initial.pose.ry, initial.pose.rz};

  double area_input = cv::countNonZero(input_mask);
  double scale_factor = area_input > 0 ? 1.0 / std::sqrt(area_input) : 1.0;

  cv::Mat dt_input_f;
  dt_input.convertTo(dt_input_f, CV_32F);

  int viz_iteration = 0;
  RefineProfiler prof;
  auto cost = [&](const std::vector<double>& params) -> double {
    prof.cost_evals++;
    auto t0 = std::chrono::high_resolution_clock::now();

    Pose6D p;
    p.tx = params[0];
    p.ty = params[1];
    p.tz = params[2];
    p.rx = params[3];
    p.ry = params[4];
    p.rz = params[5];

    if (p.tz < 0.01) {
      prof.total_wall_ms += std::chrono::duration<double, std::milli>(
          std::chrono::high_resolution_clock::now() - t0).count();
      return 1e6;
    }

    cv::Mat rendered;
    {
      ScopedTimer t(prof.generate_ms);
      rendered = RenderPose(p);
    }

    cv::Mat diff;
    {
      ScopedTimer t(prof.absdiff_ms);
      cv::absdiff(rendered, input_mask, diff);
      diff.convertTo(diff, CV_32F, 1.0 / 255.0);
    }

    cv::Mat dt_rendered;
    {
      ScopedTimer t(prof.dt_ms);
      cv::distanceTransform(255 - rendered, dt_rendered, cv::DIST_L2, 5);
      dt_rendered.convertTo(dt_rendered, CV_32F);
    }

    double chamfer_val;
    {
      ScopedTimer t(prof.chamfer_ms);
      cv::Mat dt_combined = cv::max(dt_input_f, dt_rendered);
      chamfer_val = cv::sum(dt_combined.mul(diff))[0] * scale_factor;
    }

    double area_ratio;
    {
      ScopedTimer t(prof.area_ms);
      double area_rendered = cv::countNonZero(rendered);
      area_ratio = area_input > 0
                       ? std::abs(area_rendered - area_input) / area_input
                       : 0.0;
    }

    double c = chamfer_val + 0.5 * area_ratio;

    if (viz_ && viz_iteration % 5 == 0) {
      ScopedTimer t(prof.viz_ms);
      double cur_iou = ComputeIoU(rendered, input_mask);
      viz_->LogRefineStep(refine_index * 10000 + viz_iteration, p, rendered, c, cur_iou);
    }
    viz_iteration++;

    prof.total_wall_ms += std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();

    return c;
  };

  std::vector<double> initial_step = {
      0.02, 0.02, 0.03,
      0.3, 0.3, 0.3,
  };

  NelderMead(cost, x, initial_step, max_iterations, nm_opts);

  prof.Print("RefinePose #" + std::to_string(refine_index));

  Pose6D refined;
  refined.tx = x[0];
  refined.ty = x[1];
  refined.tz = x[2];
  refined.rx = x[3];
  refined.ry = x[4];
  refined.rz = x[5];

  cv::Mat final_rendered = RenderPose(refined);
  double final_iou = ComputeIoU(final_rendered, input_mask);

  return {refined, final_iou};
}

std::vector<ScoredCandidate> PoseEstimator::LocalSearch(
    const std::vector<ScoredCandidate>& coarse_best,
    const cv::Mat& input_mask, const cv::Mat& dt_input,
    const EstimationParams& params) {

  cv::Mat dt_input_f;
  dt_input.convertTo(dt_input_f, CV_32F);
  double area_input = cv::countNonZero(input_mask);
  double scale_factor = area_input > 0 ? 1.0 / std::sqrt(area_input) : 1.0;

  cv::Moments m = cv::moments(input_mask, true);
  double u_cx = m.m10 / m.m00;
  double v_cy = m.m01 / m.m00;

  float half_angle = static_cast<float>(params.local_cone_half_angle_deg * M_PI / 180.0);
  int n_local_dirs = params.local_directions;
  int n_local_ip = params.local_in_plane;

  std::vector<ScoredCandidate> all_local;

  int n_coarse_for_local = std::min(static_cast<int>(coarse_best.size()), 10);
  for (int ci = 0; ci < n_coarse_for_local; ++ci) {
    const auto& cand = coarse_best[ci];
    glm::mat4 R_cand = glm::mat4(1.0f);
    R_cand = glm::rotate(R_cand, static_cast<float>(cand.pose.rz),
                         glm::vec3(0, 0, 1));
    R_cand = glm::rotate(R_cand, static_cast<float>(cand.pose.ry),
                         glm::vec3(0, 1, 0));
    R_cand = glm::rotate(R_cand, static_cast<float>(cand.pose.rx),
                         glm::vec3(1, 0, 0));

    glm::vec3 source_axis(0, 0, 0);
    source_axis[principal_axis_] = 1.0f;
    glm::vec3 cand_dir = glm::normalize(glm::vec3(R_cand * glm::vec4(source_axis, 0.0f)));

    glm::vec3 perp1;
    if (std::abs(glm::dot(cand_dir, glm::vec3(0, 1, 0))) < 0.9f) {
      perp1 = glm::normalize(glm::cross(cand_dir, glm::vec3(0, 1, 0)));
    } else {
      perp1 = glm::normalize(glm::cross(cand_dir, glm::vec3(1, 0, 0)));
    }
    glm::vec3 perp2 = glm::normalize(glm::cross(cand_dir, perp1));

    for (int di = 0; di < n_local_dirs; ++di) {
      float theta1 = 2.0f * glm::pi<float>() * static_cast<float>(di) /
                     static_cast<float>(n_local_dirs);

      float cone_angles[] = {
          half_angle * 0.33f,
          half_angle * 0.67f,
          half_angle * 1.0f,
      };
      float theta2 = cone_angles[di % 3];

      glm::vec3 local_dir = glm::normalize(
          cand_dir * std::cos(theta2) +
          perp1 * std::sin(theta2) * std::cos(theta1) +
          perp2 * std::sin(theta2) * std::sin(theta1));

      for (int flip = 0; flip < 2; ++flip) {
        glm::vec3 final_dir = (flip == 0) ? local_dir : -local_dir;

        for (int ip = 0; ip < n_local_ip; ++ip) {
          float in_plane = 2.0f * glm::pi<float>() * static_cast<float>(ip) /
                           static_cast<float>(n_local_ip);

          glm::mat4 R = RotationFromDirection(final_dir, in_plane, principal_axis_);
          double rx, ry, rz;
          ExtractEulerZYX(R, rx, ry, rz);

          glm::vec4 c_rot = R * glm::vec4(mesh_centroid_[0], mesh_centroid_[1],
                                           mesh_centroid_[2], 1.0f);

          for (int d = 0; d < params.local_depth; ++d) {
            double depth_frac = params.local_depth > 1
                ? static_cast<double>(d) / (params.local_depth - 1) : 0.5;
            double depth_range = 0.06;
            double depth_offset = (depth_frac - 0.5) * depth_range;
            double base_tz = cand.pose.tz + depth_offset;

            if (base_tz < params.depth_min || base_tz > params.depth_max) continue;

            Pose6D pose;
            pose.rx = rx;
            pose.ry = ry;
            pose.rz = rz;

            pose.tx = (u_cx - camera_params_.cx) * base_tz / camera_params_.fx - c_rot.x;
            pose.ty = (v_cy - camera_params_.cy) * base_tz / camera_params_.fy - c_rot.y;
            pose.tz = base_tz - c_rot.z;

            if (pose.tz < 0.01) continue;

            cv::Mat rendered = RenderPose(pose);

            cv::Mat diff;
            cv::absdiff(rendered, input_mask, diff);
            diff.convertTo(diff, CV_32F, 1.0 / 255.0);

            cv::Mat dt_rendered;
            cv::distanceTransform(255 - rendered, dt_rendered, cv::DIST_L2, 5);
            dt_rendered.convertTo(dt_rendered, CV_32F);

            cv::Mat dt_combined = cv::max(dt_input_f, dt_rendered);
            double chamfer = cv::sum(dt_combined.mul(diff))[0] * scale_factor;

            double area_rendered = cv::countNonZero(rendered);
            double area_ratio = area_input > 0
                                    ? std::abs(area_rendered - area_input) / area_input
                                    : 0.0;

            double cost = chamfer + 0.5 * area_ratio;
            double iou = ComputeIoU(rendered, input_mask);

            all_local.push_back({pose, iou, cost});
          }
        }
      }
    }
  }

  std::sort(all_local.begin(), all_local.end(),
            [](const ScoredCandidate& a, const ScoredCandidate& b) {
              return a.cost < b.cost;
            });

  if (static_cast<int>(all_local.size()) > params.top_k_local) {
    all_local.resize(params.top_k_local);
  }

  return all_local;
}

SearchResult PoseEstimator::Estimate(const cv::Mat& input_mask,
                                     const EstimationParams& params) {
  auto t_total_start = std::chrono::high_resolution_clock::now();

  auto t0 = std::chrono::high_resolution_clock::now();
  cv::Mat binary_mask;
  if (input_mask.channels() > 1) {
    cv::cvtColor(input_mask, binary_mask, cv::COLOR_BGR2GRAY);
  } else {
    binary_mask = input_mask.clone();
  }
  cv::threshold(binary_mask, binary_mask, 127, 255, cv::THRESH_BINARY);

  if (viz_) {
    viz_->SetInputMask(binary_mask);
    viz_->SetCamera(camera_params_.fx, camera_params_.fy, camera_params_.cx,
                    camera_params_.cy, camera_params_.width, camera_params_.height);
    viz_->SetMesh(mesh_.vertices().data(), mesh_.vertices().size() / 3,
                  mesh_.indices().data(), mesh_.indices().size());
  }

  cv::Mat dt_input;
  cv::distanceTransform(255 - binary_mask, dt_input, cv::DIST_L2, 5);
  auto t1 = std::chrono::high_resolution_clock::now();
  double preprocess_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "[Timing] Preprocessing: " << preprocess_ms << " ms\n";

  auto t_coarse_start = std::chrono::high_resolution_clock::now();
  auto coarse = CoarseSearch(binary_mask, dt_input, params);
  auto t_coarse_end = std::chrono::high_resolution_clock::now();
  double coarse_ms = std::chrono::duration<double, std::milli>(t_coarse_end - t_coarse_start).count();
  std::cout << "[Timing] CoarseSearch: " << coarse_ms << " ms\n";

  if (coarse.empty()) {
    SearchResult result;
    result.iou = 0;
    return result;
  }

  auto t_local_start = std::chrono::high_resolution_clock::now();
  auto local = LocalSearch(coarse, binary_mask, dt_input, params);
  auto t_local_end = std::chrono::high_resolution_clock::now();
  double local_ms = std::chrono::duration<double, std::milli>(t_local_end - t_local_start).count();
  std::cout << "[Timing] LocalSearch: " << local_ms << " ms\n";

  // Combine coarse and local candidates for refinement
  std::vector<ScoredCandidate> refine_candidates;

  // Add all coarse candidates directly for refinement
  int n_coarse_refine = std::min(static_cast<int>(coarse.size()),
                                  params.top_k_coarse);
  for (int i = 0; i < n_coarse_refine; ++i) {
    refine_candidates.push_back(coarse[i]);
  }

  // Deduplicate by direction before adding local results
  if (!local.empty()) {
    for (const auto& lc : local) {
      bool duplicate = false;
      for (const auto& rc : refine_candidates) {
        double dtx = std::abs(lc.pose.tx - rc.pose.tx);
        double dty = std::abs(lc.pose.ty - rc.pose.ty);
        double dtz = std::abs(lc.pose.tz - rc.pose.tz);
        double drx = std::abs(lc.pose.rx - rc.pose.rx);
        double dry = std::abs(lc.pose.ry - rc.pose.ry);
        double drz = std::abs(lc.pose.rz - rc.pose.rz);
        if (dtx < 0.01 && dty < 0.01 && dtz < 0.01 &&
            drx < 0.1 && dry < 0.1 && drz < 0.1) {
          duplicate = true;
          break;
        }
      }
      if (!duplicate) {
        refine_candidates.push_back(lc);
      }
    }
  }

  // Sort by cost and take top-k for actual refinement
  std::sort(refine_candidates.begin(), refine_candidates.end(),
            [](const ScoredCandidate& a, const ScoredCandidate& b) {
              return a.cost < b.cost;
            });

  SearchResult best_result;
  best_result.iou = -1;

  auto t_correct_start = std::chrono::high_resolution_clock::now();
  // Also evaluate the correct pose if available (for debugging)
  // Check if correct_pose.json exists
  {
    std::ifstream cpf("correct_pose.json");
    if (cpf.is_open()) {
      nlohmann::json cpj;
      cpf >> cpj;
      Pose6D correct_pose;
      correct_pose.tx = cpj.value("tx", 0.0);
      correct_pose.ty = cpj.value("ty", 0.0);
      correct_pose.tz = cpj.value("tz", 0.0);
      correct_pose.rx = cpj.value("rx", 0.0);
      correct_pose.ry = cpj.value("ry", 0.0);
      correct_pose.rz = cpj.value("rz", 0.0);
      if (cpj.contains("rx_deg")) correct_pose.rx = cpj["rx_deg"].get<double>() * M_PI / 180.0;
      if (cpj.contains("ry_deg")) correct_pose.ry = cpj["ry_deg"].get<double>() * M_PI / 180.0;
      if (cpj.contains("rz_deg")) correct_pose.rz = cpj["rz_deg"].get<double>() * M_PI / 180.0;

      cv::Mat correct_rendered = RenderPose(correct_pose);
      double correct_iou = ComputeIoU(correct_rendered, binary_mask);

      double area_input = cv::countNonZero(binary_mask);
      double scale_factor = area_input > 0 ? 1.0 / std::sqrt(area_input) : 1.0;
      cv::Mat dt_input_f;
      dt_input.convertTo(dt_input_f, CV_32F);

      cv::Mat diff;
      cv::absdiff(correct_rendered, binary_mask, diff);
      diff.convertTo(diff, CV_32F, 1.0 / 255.0);
      cv::Mat dt_rendered;
      cv::distanceTransform(255 - correct_rendered, dt_rendered, cv::DIST_L2, 5);
      dt_rendered.convertTo(dt_rendered, CV_32F);
      cv::Mat dt_combined = cv::max(dt_input_f, dt_rendered);
      double chamfer = cv::sum(dt_combined.mul(diff))[0] * scale_factor;
      double area_rendered = cv::countNonZero(correct_rendered);
      double area_ratio = area_input > 0 ? std::abs(area_rendered - area_input) / area_input : 0.0;
      double correct_cost = chamfer + 0.5 * area_ratio;

      std::cout << "\n=== Correct Pose Evaluation ===\n";
      std::cout << "Correct IoU: " << correct_iou << "\n";
      std::cout << "Correct chamfer: " << chamfer << "\n";
      std::cout << "Correct cost: " << correct_cost << "\n";
      std::cout << "Correct area_ratio: " << area_ratio << "\n";
      std::cout << "Correct rendered area: " << area_rendered << " vs input: " << area_input << "\n";

      // Also evaluate the best result's cost for comparison
      // (will be filled after refine)
      correct_pose_for_comparison_ = correct_pose;
      correct_cost_for_comparison_ = correct_cost;
      correct_iou_for_comparison_ = correct_iou;
    }
  }
  auto t_correct_end = std::chrono::high_resolution_clock::now();
  double correct_ms = std::chrono::duration<double, std::milli>(t_correct_end - t_correct_start).count();
  std::cout << "[Timing] CorrectPose eval: " << correct_ms << " ms\n";

  auto t_refine_start = std::chrono::high_resolution_clock::now();
  int refine_count = std::min(static_cast<int>(refine_candidates.size()), params.max_refine_candidates);
  for (int i = 0; i < refine_count; ++i) {
    SearchResult refined =
        RefinePose(refine_candidates[i], binary_mask, dt_input, params.nelder_mead_iterations, params.nm_options, i);
    if (refined.iou > best_result.iou) {
      best_result = refined;
    }
  }
  auto t_refine_end = std::chrono::high_resolution_clock::now();
  double refine_ms = std::chrono::duration<double, std::milli>(t_refine_end - t_refine_start).count();
  std::cout << "[Timing] RefinePose (x" << refine_count << "): " << refine_ms << " ms\n";

  if (viz_) {
    cv::Mat final_rendered = RenderPose(best_result.pose);
    viz_->LogFinalResult(best_result.pose, final_rendered, best_result.iou);
  }

  if (correct_cost_for_comparison_ >= 0) {
    cv::Mat result_rendered = RenderPose(best_result.pose);
    double area_input = cv::countNonZero(binary_mask);
    double scale_factor = area_input > 0 ? 1.0 / std::sqrt(area_input) : 1.0;
    cv::Mat dt_input_f;
    dt_input.convertTo(dt_input_f, CV_32F);
    cv::Mat diff;
    cv::absdiff(result_rendered, binary_mask, diff);
    diff.convertTo(diff, CV_32F, 1.0 / 255.0);
    cv::Mat dt_rendered;
    cv::distanceTransform(255 - result_rendered, dt_rendered, cv::DIST_L2, 5);
    dt_rendered.convertTo(dt_rendered, CV_32F);
    cv::Mat dt_combined = cv::max(dt_input_f, dt_rendered);
    double result_chamfer = cv::sum(dt_combined.mul(diff))[0] * scale_factor;
    double result_area = cv::countNonZero(result_rendered);
    double result_area_ratio = area_input > 0 ? std::abs(result_area - area_input) / area_input : 0.0;
    double result_cost = result_chamfer + 0.5 * result_area_ratio;

    std::cout << "\n=== Comparison ===\n";
    std::cout << "Result cost: " << result_cost << " (IoU=" << best_result.iou << ")\n";
    std::cout << "Correct cost: " << correct_cost_for_comparison_ << " (IoU=" << correct_iou_for_comparison_ << ")\n";
    if (result_cost < correct_cost_for_comparison_) {
      std::cout << ">> Result has LOWER cost (better 2D fit) than correct pose\n";
    } else {
      std::cout << ">> Correct pose has lower cost\n";
    }
  }

  auto t_total_end = std::chrono::high_resolution_clock::now();
  double total_ms = std::chrono::duration<double, std::milli>(t_total_end - t_total_start).count();
  std::cout << "[Timing] Estimate total: " << total_ms << " ms\n";
  std::cout << "[Timing]   Preprocessing: " << preprocess_ms << " ms ("
            << 100.0 * preprocess_ms / total_ms << "%)\n";
  std::cout << "[Timing]   CoarseSearch:  " << coarse_ms << " ms ("
            << 100.0 * coarse_ms / total_ms << "%)\n";
  std::cout << "[Timing]   LocalSearch:   " << local_ms << " ms ("
            << 100.0 * local_ms / total_ms << "%)\n";
  std::cout << "[Timing]   CorrectPose:   " << correct_ms << " ms ("
            << 100.0 * correct_ms / total_ms << "%)\n";
  std::cout << "[Timing]   RefinePose:    " << refine_ms << " ms ("
            << 100.0 * refine_ms / total_ms << "%)\n";

  return best_result;
}

}  // namespace pose_matching
