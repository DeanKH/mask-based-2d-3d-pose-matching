#pragma once

#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <maskgen/camera.h>
#include <maskgen/mask_generator.h>
#include <maskgen/mesh.h>

#include <opencv2/core.hpp>

#include "bobyqa.h"
#include "nelder_mead.h"

namespace pose_matching {

enum class RefineMethod {
  NelderMead,
  LevenbergMarquardt,
  GaussNewton,
  BOBYQA,
};

enum class CandidateSortMetric {
  IoU,
  HuMoments,
  ZernikeMoments,
  IoUZernikeMoments,
  CentroidIoU,
  CentroidDT_L1,
  CentroidDT_L2,
  DT_IoU,
  CentralMoments,
  ShapeContext,
  ContourChamfer,
  FourierDescriptor,
};

// 候補姿勢の軸方向制約。
// モデル座標系の axis (+x/+y/+z) が、カメラ座標系の参照方向 ref_dir から
// max_angle_deg 以内に向いている姿勢のみを候補として許可する。
struct AxisConstraint {
  int axis = 2;                            // モデル軸: 0=x, 1=y, 2=z
  float ref_dir[3] = {0.0f, 0.0f, -1.0f};  // カメラ座標系の参照方向 (正規化済み)
  float max_angle_deg = 30.0f;             // 許容する最大角度 [deg]
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
  BobyqaOptions bobyqa_options;
  RefineMethod refine_method = RefineMethod::NelderMead;
  int contour_points = 1000;
  int lm_max_iterations = 100;
  double lm_relative_tol = 1e-6;
  double lm_absolute_tol = 1e-6;
  double early_termination_iou = 1.0;
  bool use_cpu = false;
  CandidateSortMetric sort_metric = CandidateSortMetric::IoU;
  std::string int_mask_dir;

  // ---- 候補姿勢の制約 (CLI: --axis-constraint / --tx-range / --ty-range) ----
  // 軸方向制約。モデルの指定軸がカメラ座標系の参照方向から何度以内か。
  // 指定がない場合 (=empty) は全姿勢を探索する。
  std::vector<AxisConstraint> axis_constraints;
  // モデル原点位置 (tx, ty) の許容範囲 [m] (カメラ座標系)。
  bool has_tx_range = false;
  double tx_min = 0.0;
  double tx_max = 0.0;
  bool has_ty_range = false;
  double ty_min = 0.0;
  double ty_max = 0.0;

  // ---- ノミナル位置サンプリングモード (CLI: --nominal-position) ----
  // 有効時、CoarseSearch はマスク重心投影ではなく
  // モデル原点のノミナル位置 ± nominal_tol のグリッドから tx/ty をサンプルする。
  // tz (原点Z) は num_depth 段階で nominal_tz ± nominal_tol_tz をサンプル。
  bool use_nominal_position = false;
  double nominal_tx = 0.0;
  double nominal_ty = 0.0;
  double nominal_tz = 0.0;
  double nominal_tol_tx = 0.02;
  double nominal_tol_ty = 0.02;
  double nominal_tol_tz = 0.03;
  int nominal_grid = 3;  // tx/ty それぞれのサンプル数 (±tol を含む等間隔)

  // ---- フォールバック ----
  // 制約付き推定の最終IoUが fallback_iou 未満の場合、制約なしで再推定する。
  bool fallback_unconstrained = true;
  double fallback_iou = 0.70;
};

// ============================================================
// 候補姿勢の制約ヘルパー (ヘッダオンリー: cache_builder からも利用)
// ============================================================

// 回転 (オイラー角 ZYX: R = Rz * Ry * Rx) がすべての軸制約を満たすか判定する。
inline bool RotationSatisfiesAxisConstraints(
    double rx, double ry, double rz,
    const std::vector<AxisConstraint>& constraints) {
  if (constraints.empty()) return true;

  constexpr double kPi = 3.14159265358979323846;
  const double cx = std::cos(rx), sx = std::sin(rx);
  const double cy = std::cos(ry), sy = std::sin(ry);
  const double cz = std::cos(rz), sz = std::sin(rz);
  const double Rxm[3][3] = {{1, 0, 0}, {0, cx, -sx}, {0, sx, cx}};
  const double Rym[3][3] = {{cy, 0, sy}, {0, 1, 0}, {-sy, 0, cy}};
  const double Rzm[3][3] = {{cz, -sz, 0}, {sz, cz, 0}, {0, 0, 1}};

  double Rzy[3][3], R[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      Rzy[i][j] = 0;
      for (int k = 0; k < 3; ++k) Rzy[i][j] += Rzm[i][k] * Rym[k][j];
    }
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      R[i][j] = 0;
      for (int k = 0; k < 3; ++k) R[i][j] += Rzy[i][k] * Rxm[k][j];
    }

  for (const auto& c : constraints) {
    double e[3] = {0, 0, 0};
    e[c.axis] = 1.0;
    double dir[3];
    for (int i = 0; i < 3; ++i)
      dir[i] = R[i][0] * e[0] + R[i][1] * e[1] + R[i][2] * e[2];

    double ref_len = std::sqrt(c.ref_dir[0] * c.ref_dir[0] +
                               c.ref_dir[1] * c.ref_dir[1] +
                               c.ref_dir[2] * c.ref_dir[2]);
    if (ref_len < 1e-9f) return false;
    double dot = (dir[0] * c.ref_dir[0] + dir[1] * c.ref_dir[1] +
                  dir[2] * c.ref_dir[2]) / ref_len;
    if (dot < std::cos(c.max_angle_deg * kPi / 180.0)) return false;
  }
  return true;
}

// 姿勢が軸制約・tx/ty範囲をすべて満たすか判定する。
// (tz の範囲は既存の depth_min/depth_max が担うためここでは判定しない)
inline bool PoseSatisfiesConstraints(const Pose6D& pose,
                                     const EstimationParams& params) {
  if (!RotationSatisfiesAxisConstraints(pose.rx, pose.ry, pose.rz,
                                        params.axis_constraints)) {
    return false;
  }
  if (params.has_tx_range &&
      (pose.tx < params.tx_min || pose.tx > params.tx_max)) {
    return false;
  }
  if (params.has_ty_range &&
      (pose.ty < params.ty_min || pose.ty > params.ty_max)) {
    return false;
  }
  return true;
}

inline bool EstimationHasConstraints(const EstimationParams& params) {
  return !params.axis_constraints.empty() || params.has_tx_range ||
         params.has_ty_range || params.use_nominal_position;
}

// 制約を除去したパラメータのコピーを返す (フォールバック用)。
inline EstimationParams StripPoseConstraints(EstimationParams params) {
  params.axis_constraints.clear();
  params.has_tx_range = false;
  params.has_ty_range = false;
  params.use_nominal_position = false;
  params.fallback_unconstrained = false;
  return params;
}

// ============================================================
// CLI 指定文字列のパーサ
//   軸制約  : "z:0,0,-1:30"   (モデル軸:カメラ参照方向:最大角deg)
//   範囲    : "-0.05:0.05"    (min:max)
//   ベクトル: "0.02,-0.01,0.15"
// ============================================================

inline bool ParseAxisConstraintSpec(const std::string& spec, AxisConstraint& out) {
  char axis = 0;
  double dx, dy, dz, angle;
  if (std::sscanf(spec.c_str(), " %c : %lf , %lf , %lf : %lf",
                  &axis, &dx, &dy, &dz, &angle) != 5) {
    return false;
  }
  int idx = (axis == 'x') ? 0 : (axis == 'y') ? 1 : (axis == 'z') ? 2 : -1;
  if (idx < 0) return false;
  double norm = std::sqrt(dx * dx + dy * dy + dz * dz);
  if (norm < 1e-9 || angle < 0.0 || angle >= 180.0) return false;
  out.axis = idx;
  out.ref_dir[0] = static_cast<float>(dx / norm);
  out.ref_dir[1] = static_cast<float>(dy / norm);
  out.ref_dir[2] = static_cast<float>(dz / norm);
  out.max_angle_deg = static_cast<float>(angle);
  return true;
}

inline bool ParseRangeSpec(const std::string& spec, double& lo, double& hi) {
  if (std::sscanf(spec.c_str(), "%lf:%lf", &lo, &hi) != 2) return false;
  return lo <= hi;
}

inline bool ParseVec3Spec(const std::string& spec, double out[3]) {
  return std::sscanf(spec.c_str(), "%lf,%lf,%lf", &out[0], &out[1], &out[2]) == 3;
}

inline std::string AxisConstraintToString(const AxisConstraint& c) {
  const char axis_name = (c.axis == 0) ? 'x' : (c.axis == 1) ? 'y' : 'z';
  char buf[128];
  std::snprintf(buf, sizeof(buf), "%c:(%.3f,%.3f,%.3f):%.1fdeg", axis_name,
                static_cast<double>(c.ref_dir[0]),
                static_cast<double>(c.ref_dir[1]),
                static_cast<double>(c.ref_dir[2]),
                static_cast<double>(c.max_angle_deg));
  return std::string(buf);
}

class PoseEstimator {
 public:
  PoseEstimator(const maskgen::CameraParams& camera_params,
                const std::string& model_path, float model_scale);

  void SetVisualizer(Visualizer* viz);

  SearchResult Estimate(const cv::Mat& input_mask,
                        const EstimationParams& params = EstimationParams());

 private:
  // 制約適用後の本体。Estimate はフォールバック (制約なし再試行) を制御する。
  SearchResult EstimateImpl(const cv::Mat& input_mask,
                            const EstimationParams& params);

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
                          const NelderMeadOptions& nm_opts,
                          const BobyqaOptions& bobyqa_opts,
                          RefineMethod refine_method, int refine_index,
                          const EstimationParams* constraint_params,
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
