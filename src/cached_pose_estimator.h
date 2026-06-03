#pragma once

#include <memory>
#include <string>
#include <vector>

#include <maskgen/camera.h>
#include <maskgen/mask_generator.h>
#include <maskgen/mesh.h>

#include <opencv2/core.hpp>

#include "cache_format.h"
#include "pose_estimator.h"

namespace pose_matching {

class CachedPoseEstimator {
 public:
  CachedPoseEstimator(const maskgen::CameraParams& camera_params,
                      const std::string& model_path, float model_scale,
                      const std::string& cache_path);

  void SetVisualizer(Visualizer* viz);

  SearchResult Estimate(const cv::Mat& input_mask,
                        const EstimationParams& params = EstimationParams());

 private:
  cv::Mat DecodeMask(size_t mask_offset, uint32_t mask_size) const;

  cv::Mat ShiftMask(const cv::Mat& mask, double dx, double dy) const;

  double ComputeIoU(const cv::Mat& a, const cv::Mat& b) const;

  std::vector<ScoredCandidate> CachedCoarseSearch(const cv::Mat& input_mask,
                                                   const cv::Mat& dt_input,
                                                   const EstimationParams& params);

  std::vector<ScoredCandidate> CachedLocalSearch(
      const std::vector<ScoredCandidate>& coarse_best,
      const cv::Mat& input_mask, const cv::Mat& dt_input,
      const EstimationParams& params);

  SearchResult RefinePose(const ScoredCandidate& initial, const cv::Mat& input_mask,
                          const cv::Mat& dt_input, int max_iterations, int refine_index);

  CacheData cache_;
  maskgen::CameraParams camera_params_;
  maskgen::Mesh mesh_;
  std::unique_ptr<maskgen::MaskGenerator> generator_;
  Visualizer* viz_ = nullptr;

  int principal_axis_;
  float mesh_centroid_[3];
  float mesh_extent_[3];
};

}  // namespace pose_matching
