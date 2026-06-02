#include "pose_estimator.h"

#include <cmath>
#include <algorithm>
#include <cfloat>
#include <numeric>

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
    const cv::Mat& input_mask, const EstimationParams& params) {
  cv::Moments m = cv::moments(input_mask, true);
  if (m.m00 == 0) return {};

  double u_cx = m.m10 / m.m00;
  double v_cy = m.m01 / m.m00;

  double input_area = m.m00;
  double mesh_area_estimate = mesh_extent_[principal_axis_] *
                              *std::max_element(mesh_extent_, mesh_extent_ + 3);

  std::vector<double> depths;
  double depth_step = (params.depth_max - params.depth_min) / std::max(params.num_depth - 1, 1);
  for (int i = 0; i < params.num_depth; ++i) {
    depths.push_back(params.depth_min + i * depth_step);
  }

  if (input_area > 0 && mesh_area_estimate > 0) {
    double fx = camera_params_.fx;
    double fy = camera_params_.fy;
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

        candidates.push_back({pose, iou});
      }
    }
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const ScoredCandidate& a, const ScoredCandidate& b) {
              return a.iou > b.iou;
            });

  if (static_cast<int>(candidates.size()) > params.top_k_coarse) {
    candidates.resize(params.top_k_coarse);
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
                                       int refine_index) {
  std::vector<double> x = {initial.pose.tx, initial.pose.ty, initial.pose.tz,
                           initial.pose.rx, initial.pose.ry, initial.pose.rz};

  double area_input = cv::countNonZero(input_mask);
  double scale_factor = area_input > 0 ? 1.0 / std::sqrt(area_input) : 1.0;

  cv::Mat dt_input_f;
  dt_input.convertTo(dt_input_f, CV_32F);

  int viz_iteration = 0;
  auto cost = [&](const std::vector<double>& params) -> double {
    Pose6D p;
    p.tx = params[0];
    p.ty = params[1];
    p.tz = params[2];
    p.rx = params[3];
    p.ry = params[4];
    p.rz = params[5];

    if (p.tz < 0.01) return 1e6;

    cv::Mat rendered = RenderPose(p);

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

    double c = chamfer + 0.5 * area_ratio;

    if (viz_ && viz_iteration % 5 == 0) {
      double cur_iou = ComputeIoU(rendered, input_mask);
      viz_->LogRefineStep(refine_index * 10000 + viz_iteration, p, rendered, c, cur_iou);
    }
    viz_iteration++;

    return c;
  };

  std::vector<double> initial_step = {
      0.005, 0.005, 0.01,
      0.05, 0.05, 0.05,
  };

  NelderMead(cost, x, initial_step, max_iterations);

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

SearchResult PoseEstimator::Estimate(const cv::Mat& input_mask,
                                     const EstimationParams& params) {
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

  auto coarse = CoarseSearch(binary_mask, params);

  if (coarse.empty()) {
    SearchResult result;
    result.iou = 0;
    return result;
  }

  SearchResult best_result;
  best_result.iou = -1;

  int refine_count = std::min(static_cast<int>(coarse.size()), 3);
  for (int i = 0; i < refine_count; ++i) {
    SearchResult refined =
        RefinePose(coarse[i], binary_mask, dt_input, params.nelder_mead_iterations, i);
    if (refined.iou > best_result.iou) {
      best_result = refined;
    }
  }

  if (viz_) {
    cv::Mat final_rendered = RenderPose(best_result.pose);
    viz_->LogFinalResult(best_result.pose, final_rendered, best_result.iou);
  }

  return best_result;
}

}  // namespace pose_matching
