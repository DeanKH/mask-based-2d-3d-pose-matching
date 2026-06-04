#pragma once

#include "pose_estimator.h"

#include <glm/glm.hpp>
#include <maskgen/camera.h>
#include <maskgen/mask_generator.h>
#include <maskgen/mesh.h>
#include <opencv2/core.hpp>
#include <vector>

namespace pose_matching {

enum class OptimizerType {
  LevenbergMarquardt,
  GaussNewton,
};

struct LMOptions {
  OptimizerType optimizer = OptimizerType::LevenbergMarquardt;
  int max_iterations = 100;
  double relative_tol = 1e-6;
  double absolute_tol = 1e-6;
};

SearchResult RefinePoseLM(
    const ScoredCandidate& initial,
    const cv::Mat& input_mask,
    const std::vector<glm::vec3>& contour_points,
    const maskgen::CameraParams& camera_params,
    const maskgen::Mesh& mesh,
    maskgen::MaskGenerator& generator,
    const LMOptions& opts,
    int refine_index);

}  // namespace pose_matching
