#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <maskgen/camera.h>
#include <maskgen/mask_generator.h>
#include <maskgen/mesh.h>

#include <opencv2/core.hpp>

#include "nelder_mead.h"

namespace pose_matching {

enum class RefineMethod {
  NelderMead,
  LevenbergMarquardt,
  GaussNewton,
};

class Visualizer;

struct Pose6D {
  double tx = 0;
  double ty = 0;
  double tz = 0;
  double rx = 0;
  double ry = 0;
  double rz = 0;
};

struct SearchResult {
  Pose6D pose;
  double iou = 0;
};

struct ScoredCandidate {
  Pose6D pose;
  double iou;
  double cost;
};

struct EstimationParams {
  int num_directions = 48;
  int num_in_plane = 4;
  int num_depth = 5;
  double depth_min = 0.05;
  double depth_max = 0.50;
  int top_k_coarse = 10;
  int nelder_mead_iterations = 200;
  int local_directions = 6;
  int local_in_plane = 6;
  int local_depth = 3;
  double local_cone_half_angle_deg = 30.0;
  int top_k_local = 5;
  int max_refine_candidates = 5;
  NelderMeadOptions nm_options;
  RefineMethod refine_method = RefineMethod::NelderMead;
  int contour_points = 1000;
  int lm_max_iterations = 100;
  double lm_relative_tol = 1e-6;
  double lm_absolute_tol = 1e-6;
  double early_termination_iou = 1.0;
  std::string int_mask_dir;
};

class PoseEstimator {
 public:
  PoseEstimator(const maskgen::CameraParams& camera_params,
                const std::string& model_path, float model_scale);

  void SetVisualizer(Visualizer* viz);

  SearchResult Estimate(const cv::Mat& input_mask,
                        const EstimationParams& params = EstimationParams());

 private:
  cv::Mat RenderPose(const Pose6D& pose);
  double ComputeIoU(const cv::Mat& a, const cv::Mat& b) const;

  std::vector<ScoredCandidate> CoarseSearch(const cv::Mat& input_mask,
                                            const cv::Mat& dt_input,
                                            const EstimationParams& params);

  std::vector<ScoredCandidate> LocalSearch(
      const std::vector<ScoredCandidate>& coarse_best,
      const cv::Mat& input_mask, const cv::Mat& dt_input,
      const EstimationParams& params);

  SearchResult RefinePose(const ScoredCandidate& initial, const cv::Mat& input_mask,
                          const cv::Mat& dt_input, int max_iterations,
                          const NelderMeadOptions& nm_opts, int refine_index,
                          maskgen::MaskGenerator* generator);

  maskgen::CameraParams camera_params_;
  maskgen::Mesh mesh_;
  std::unique_ptr<maskgen::MaskGenerator> generator_;
  Visualizer* viz_ = nullptr;

  int principal_axis_;
  float mesh_centroid_[3];
  float mesh_extent_[3];
  Pose6D correct_pose_for_comparison_{};
  double correct_cost_for_comparison_ = -1;
  double correct_iou_for_comparison_ = -1;
};

}  // namespace pose_matching
