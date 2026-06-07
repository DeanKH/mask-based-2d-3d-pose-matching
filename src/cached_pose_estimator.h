#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include <maskgen/camera.h>
#include <maskgen/mask_generator.h>
#include <maskgen/mesh.h>

#include <opencv2/core.hpp>

#include "cache_format.h"
#include "contour_sampler.h"
#include "nelder_mead.h"
#include "pose_estimator.h"
#include "refine_lm.h"

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

  double ShiftedIoU(const cv::Mat& base, const cv::Mat& input,
                    int du, int dv, int base_count, int input_count) const;

  std::vector<ScoredCandidate> CachedCoarseSearch(const cv::Mat& input_mask,
                                                   const cv::Mat& dt_input,
                                                   const EstimationParams& params);

  std::vector<ScoredCandidate> CachedLocalSearch(
      const std::vector<ScoredCandidate>& coarse_best,
      const cv::Mat& input_mask, const cv::Mat& dt_input,
      const EstimationParams& params);

  SearchResult RefinePose(const ScoredCandidate& initial, const cv::Mat& input_mask,
                          const cv::Mat& dt_input, int max_iterations,
                          const NelderMeadOptions& nm_opts, int refine_index,
                          maskgen::MaskGenerator* generator,
                          const std::atomic<bool>* abort_flag = nullptr);

  CacheData cache_;
  std::vector<cv::Mat> decoded_coarse_masks_;
  std::vector<int> decoded_coarse_areas_;
  std::vector<cv::Mat> decoded_coarse_masks_small_;
  std::vector<int> decoded_coarse_areas_small_;
  std::vector<cv::Mat> decoded_local_masks_small_;
  std::vector<int> decoded_local_areas_small_;
  int small_w_ = 0;
  int small_h_ = 0;
  maskgen::CameraParams camera_params_;
  maskgen::Mesh mesh_;
  std::unique_ptr<maskgen::MaskGenerator> generator_;
  Visualizer* viz_ = nullptr;

  int principal_axis_;
  float mesh_centroid_[3];
  float mesh_extent_[3];
};

}  // namespace pose_matching
