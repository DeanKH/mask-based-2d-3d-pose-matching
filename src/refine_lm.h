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

// LM/GN による6DoF精密化。
// edge_dt が空でなく edge_weight > 0 の場合、シルエットDT因子に加えて
// 入力画像エッジDT因子 (輪郭点が画像エッジに近いほど誤差小) をグラフに追加する。
SearchResult RefinePoseLM(
    const ScoredCandidate& initial,
    const cv::Mat& input_mask,
    const std::vector<glm::vec3>& contour_points,
    const maskgen::CameraParams& camera_params,
    const maskgen::Mesh& mesh,
    maskgen::MaskGenerator& generator,
    const LMOptions& opts,
    int refine_index,
    const cv::Mat& edge_dt = cv::Mat(),
    double edge_weight = 0.0);

}  // namespace pose_matching
