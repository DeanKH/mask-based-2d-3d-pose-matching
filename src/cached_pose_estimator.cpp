#include "cached_pose_estimator.h"

#include <chrono>
#include <cmath>
#include <algorithm>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>

#include <nlohmann/json.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <opencv2/imgproc.hpp>

#include "nelder_mead.h"
#include "visualizer.h"

namespace pose_matching {

CachedPoseEstimator::CachedPoseEstimator(
    const maskgen::CameraParams& camera_params,
    const std::string& model_path, float model_scale,
    const std::string& cache_path)
    : camera_params_(camera_params), principal_axis_(2) {
  if (!mesh_.LoadFromFile(model_path, model_scale)) {
    throw std::runtime_error("Failed to load mesh: " + model_path);
  }

  const auto& verts = mesh_.vertices();
  const size_t n = verts.size() / 3;
  mesh_centroid_[0] = mesh_centroid_[1] = mesh_centroid_[2] = 0.0f;
  float min_p[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
  float max_p[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
  for (size_t i = 0; i < n; ++i) {
    for (int d = 0; d < 3; ++d) {
      float v = verts[i * 3 + d];
      min_p[d] = std::min(min_p[d], v);
      max_p[d] = std::max(max_p[d], v);
      mesh_centroid_[d] += v;
    }
  }
  float max_extent = 0;
  for (int d = 0; d < 3; ++d) {
    mesh_centroid_[d] /= static_cast<float>(n);
    mesh_extent_[d] = max_p[d] - min_p[d];
    if (mesh_extent_[d] > max_extent) {
      max_extent = mesh_extent_[d];
      principal_axis_ = d;
    }
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  std::vector<uint8_t> file_buffer;
  if (!ReadCache(cache_path, cache_)) {
    throw std::runtime_error("Failed to read cache file: " + cache_path);
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  double cache_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "[Timing] Cache loading: " << cache_ms << " ms\n";
  std::cout << "  Coarse entries: " << cache_.coarse_entries.size() << "\n";
  std::cout << "  Local entries: " << cache_.local_entries.size() << "\n";
  std::cout << "  Mask data: " << cache_.mask_data.size() / (1024.0 * 1024.0) << " MB\n";

  camera_params_ = camera_params;
  camera_params_.eye_x = 0;
  camera_params_.eye_y = 0;
  camera_params_.eye_z = 0;
  camera_params_.target_x = 0;
  camera_params_.target_y = 0;
  camera_params_.target_z = 1;
  camera_params_.up_x = 0;
  camera_params_.up_y = -1;
  camera_params_.up_z = 0;

  generator_ = std::make_unique<maskgen::MaskGenerator>(camera_params_);
}

void CachedPoseEstimator::SetVisualizer(Visualizer* viz) { viz_ = viz; }

cv::Mat CachedPoseEstimator::DecodeMask(size_t mask_offset, uint32_t mask_size) const {
  const uint8_t* compressed = cache_.mask_data.data() + mask_offset;
  int packed_cols = (cache_.header.image_width + 7) / 8;
  size_t expected = static_cast<size_t>(packed_cols) * cache_.header.image_height;
  std::vector<uint8_t> packed = ZlibDecompress(compressed, mask_size, expected);
  return UnpackMask1Bit(packed.data(), packed.size(),
                        cache_.header.image_width, cache_.header.image_height);
}

cv::Mat CachedPoseEstimator::ShiftMask(const cv::Mat& mask, double dx, double dy) const {
  if (std::abs(dx) < 0.5 && std::abs(dy) < 0.5) return mask;

  cv::Mat M = (cv::Mat_<float>(2, 3) << 1, 0, dx, 0, 1, dy);
  cv::Mat shifted;
  cv::warpAffine(mask, shifted, M, mask.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT,
                 cv::Scalar(0));
  return shifted;
}

double CachedPoseEstimator::ComputeIoU(const cv::Mat& a, const cv::Mat& b) const {
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

std::vector<ScoredCandidate> CachedPoseEstimator::CachedCoarseSearch(
    const cv::Mat& input_mask, const cv::Mat& dt_input,
    const EstimationParams& params) {
  cv::Moments m = cv::moments(input_mask, true);
  if (m.m00 == 0) return {};

  double u_cx = m.m10 / m.m00;
  double v_cy = m.m01 / m.m00;

  double shift_u = u_cx - cache_.header.cx;
  double shift_v = v_cy - cache_.header.cy;

  std::vector<ScoredCandidate> candidates;
  candidates.reserve(cache_.coarse_entries.size());

  int candidate_index = 0;
  for (const auto& entry : cache_.coarse_entries) {
    if (entry.mask_size == 0) {
      candidate_index++;
      continue;
    }

    cv::Mat base_mask = DecodeMask(entry.mask_offset, entry.mask_size);
    cv::Mat shifted = ShiftMask(base_mask, shift_u, shift_v);

    double iou = ComputeIoU(shifted, input_mask);

    double tz_total = entry.depth;
    double tz = tz_total - entry.crot_z;

    Pose6D pose;
    pose.rx = entry.rx;
    pose.ry = entry.ry;
    pose.rz = entry.rz;
    pose.tx = (u_cx - cache_.header.cx) * tz_total / cache_.header.fx - entry.crot_x;
    pose.ty = (v_cy - cache_.header.cy) * tz_total / cache_.header.fy - entry.crot_y;
    pose.tz = tz;

    if (viz_) {
      viz_->LogCandidate(candidate_index, pose, shifted, iou);
    }
    candidate_index++;

    candidates.push_back({pose, iou, 1.0 - iou});
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const ScoredCandidate& a, const ScoredCandidate& b) {
              return a.cost < b.cost;
            });

  std::vector<ScoredCandidate> diverse;
  std::set<int> seen_dirs;
  for (const auto& c : candidates) {
    int key = static_cast<int>(c.pose.rx * 10) * 100000 +
              static_cast<int>(c.pose.ry * 10) * 1000 +
              static_cast<int>(c.pose.rz * 10);
    if (seen_dirs.insert(key).second) {
      diverse.push_back(c);
    }
    if (static_cast<int>(diverse.size()) >= params.top_k_coarse) break;
  }

  std::cout << "Top " << diverse.size() << " coarse candidates (cached):\n";
  for (size_t i = 0; i < diverse.size(); ++i) {
    std::cout << "  [" << i << "] iou=" << diverse[i].iou
              << " cost=" << diverse[i].cost
              << " tz=" << diverse[i].pose.tz << "\n";
  }

  if (viz_) {
    viz_->LogCoarseComplete(diverse);
  }

  return diverse;
}

std::vector<ScoredCandidate> CachedPoseEstimator::CachedLocalSearch(
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

  double shift_u = u_cx - cache_.header.cx;
  double shift_v = v_cy - cache_.header.cy;

  std::set<uint32_t> needed_coarse_idx;
  int n_coarse = std::min(static_cast<int>(coarse_best.size()), 10);

  for (size_t ei = 0; ei < cache_.coarse_entries.size(); ++ei) {
    const auto& entry = cache_.coarse_entries[ei];
    for (int ci = 0; ci < n_coarse; ++ci) {
      double drx = std::abs(entry.rx - coarse_best[ci].pose.rx);
      double dry = std::abs(entry.ry - coarse_best[ci].pose.ry);
      double drz = std::abs(entry.rz - coarse_best[ci].pose.rz);
      if (drx < 0.01 && dry < 0.01 && drz < 0.01) {
        uint32_t dir_idx = static_cast<uint32_t>(ei / cache_.header.num_depth);
        needed_coarse_idx.insert(dir_idx);
        break;
      }
    }
  }

  std::vector<ScoredCandidate> all_local;

  for (const auto& local_entry : cache_.local_entries) {
    if (needed_coarse_idx.find(local_entry.coarse_idx) == needed_coarse_idx.end()) {
      continue;
    }

    if (local_entry.mask_size == 0) continue;

    cv::Mat base_mask = DecodeMask(local_entry.mask_offset, local_entry.mask_size);
    cv::Mat shifted = ShiftMask(base_mask, shift_u, shift_v);

    cv::Mat diff;
    cv::absdiff(shifted, input_mask, diff);
    diff.convertTo(diff, CV_32F, 1.0 / 255.0);

    cv::Mat dt_rendered;
    cv::distanceTransform(255 - shifted, dt_rendered, cv::DIST_L2, 5);
    dt_rendered.convertTo(dt_rendered, CV_32F);

    cv::Mat dt_combined = cv::max(dt_input_f, dt_rendered);
    double chamfer = cv::sum(dt_combined.mul(diff))[0] * scale_factor;

    double area_rendered = cv::countNonZero(shifted);
    double area_ratio = area_input > 0
                            ? std::abs(area_rendered - area_input) / area_input
                            : 0.0;

    double cost = chamfer + 0.5 * area_ratio;
    double iou = ComputeIoU(shifted, input_mask);

    double tz_total = local_entry.depth;
    double tz = tz_total - local_entry.crot_z;

    Pose6D pose;
    pose.rx = local_entry.rx;
    pose.ry = local_entry.ry;
    pose.rz = local_entry.rz;
    pose.tx = (u_cx - cache_.header.cx) * tz_total / cache_.header.fx - local_entry.crot_x;
    pose.ty = (v_cy - cache_.header.cy) * tz_total / cache_.header.fy - local_entry.crot_y;
    pose.tz = tz;

    all_local.push_back({pose, iou, cost});
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

SearchResult CachedPoseEstimator::RefinePose(const ScoredCandidate& initial,
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

    maskgen::MeshPose mp;
    mp.tx = p.tx;
    mp.ty = p.ty;
    mp.tz = p.tz;
    mp.rx = p.rx;
    mp.ry = p.ry;
    mp.rz = p.rz;
    cv::Mat rendered = generator_->Generate(mesh_, mp);

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
      0.02, 0.02, 0.03,
      0.3, 0.3, 0.3,
  };

  NelderMead(cost, x, initial_step, max_iterations);

  Pose6D refined;
  refined.tx = x[0];
  refined.ty = x[1];
  refined.tz = x[2];
  refined.rx = x[3];
  refined.ry = x[4];
  refined.rz = x[5];

  maskgen::MeshPose mp;
  mp.tx = refined.tx;
  mp.ty = refined.ty;
  mp.tz = refined.tz;
  mp.rx = refined.rx;
  mp.ry = refined.ry;
  mp.rz = refined.rz;
  cv::Mat final_rendered = generator_->Generate(mesh_, mp);
  double final_iou = ComputeIoU(final_rendered, input_mask);

  return {refined, final_iou};
}

SearchResult CachedPoseEstimator::Estimate(const cv::Mat& input_mask,
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
  auto coarse = CachedCoarseSearch(binary_mask, dt_input, params);
  auto t_coarse_end = std::chrono::high_resolution_clock::now();
  double coarse_ms = std::chrono::duration<double, std::milli>(t_coarse_end - t_coarse_start).count();
  std::cout << "[Timing] CachedCoarseSearch: " << coarse_ms << " ms\n";

  if (coarse.empty()) {
    SearchResult result;
    result.iou = 0;
    return result;
  }

  auto t_local_start = std::chrono::high_resolution_clock::now();
  auto local = CachedLocalSearch(coarse, binary_mask, dt_input, params);
  auto t_local_end = std::chrono::high_resolution_clock::now();
  double local_ms = std::chrono::duration<double, std::milli>(t_local_end - t_local_start).count();
  std::cout << "[Timing] CachedLocalSearch: " << local_ms << " ms\n";

  std::vector<ScoredCandidate> refine_candidates;

  int n_coarse_refine = std::min(static_cast<int>(coarse.size()), params.top_k_coarse);
  for (int i = 0; i < n_coarse_refine; ++i) {
    refine_candidates.push_back(coarse[i]);
  }

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

  std::sort(refine_candidates.begin(), refine_candidates.end(),
            [](const ScoredCandidate& a, const ScoredCandidate& b) {
              return a.cost < b.cost;
            });

  SearchResult best_result;
  best_result.iou = -1;

  auto t_correct_start = std::chrono::high_resolution_clock::now();
  double correct_cost_for_comparison = -1;
  double correct_iou_for_comparison = -1;
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

      maskgen::MeshPose mp;
      mp.tx = correct_pose.tx;
      mp.ty = correct_pose.ty;
      mp.tz = correct_pose.tz;
      mp.rx = correct_pose.rx;
      mp.ry = correct_pose.ry;
      mp.rz = correct_pose.rz;
      cv::Mat correct_rendered = generator_->Generate(mesh_, mp);
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
      correct_cost_for_comparison = chamfer + 0.5 * area_ratio;
      correct_iou_for_comparison = correct_iou;

      std::cout << "\n=== Correct Pose Evaluation ===\n";
      std::cout << "Correct IoU: " << correct_iou << "\n";
      std::cout << "Correct cost: " << correct_cost_for_comparison << "\n";
    }
  }
  auto t_correct_end = std::chrono::high_resolution_clock::now();
  double correct_ms = std::chrono::duration<double, std::milli>(t_correct_end - t_correct_start).count();
  std::cout << "[Timing] CorrectPose eval: " << correct_ms << " ms\n";

  auto t_refine_start = std::chrono::high_resolution_clock::now();
  int refine_count = std::min(static_cast<int>(refine_candidates.size()), 10);
  for (int i = 0; i < refine_count; ++i) {
    SearchResult refined =
        RefinePose(refine_candidates[i], binary_mask, dt_input, params.nelder_mead_iterations, i);
    if (refined.iou > best_result.iou) {
      best_result = refined;
    }
  }
  auto t_refine_end = std::chrono::high_resolution_clock::now();
  double refine_ms = std::chrono::duration<double, std::milli>(t_refine_end - t_refine_start).count();
  std::cout << "[Timing] RefinePose (x" << refine_count << "): " << refine_ms << " ms\n";

  if (correct_cost_for_comparison >= 0) {
    maskgen::MeshPose mp;
    mp.tx = best_result.pose.tx;
    mp.ty = best_result.pose.ty;
    mp.tz = best_result.pose.tz;
    mp.rx = best_result.pose.rx;
    mp.ry = best_result.pose.ry;
    mp.rz = best_result.pose.rz;
    cv::Mat result_rendered = generator_->Generate(mesh_, mp);
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
    std::cout << "Correct cost: " << correct_cost_for_comparison << " (IoU=" << correct_iou_for_comparison << ")\n";
  }

  if (viz_) {
    maskgen::MeshPose mp;
    mp.tx = best_result.pose.tx;
    mp.ty = best_result.pose.ty;
    mp.tz = best_result.pose.tz;
    mp.rx = best_result.pose.rx;
    mp.ry = best_result.pose.ry;
    mp.rz = best_result.pose.rz;
    cv::Mat final_rendered = generator_->Generate(mesh_, mp);
    viz_->LogFinalResult(best_result.pose, final_rendered, best_result.iou);
  }

  auto t_total_end = std::chrono::high_resolution_clock::now();
  double total_ms = std::chrono::duration<double, std::milli>(t_total_end - t_total_start).count();
  std::cout << "[Timing] Estimate total: " << total_ms << " ms\n";

  return best_result;
}

}  // namespace pose_matching
