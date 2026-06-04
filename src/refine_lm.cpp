#include "refine_lm.h"

#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/nonlinear/Symbol.h>
#include <gtsam/nonlinear/Values.h>

#include <glm/gtc/matrix_transform.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <iostream>

namespace pose_matching {

class ContourDTFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
 public:
  using Base = gtsam::NoiseModelFactor1<gtsam::Pose3>;

  ContourDTFactor(
      gtsam::Key key,
      const std::vector<gtsam::Point3>& contour_points,
      const cv::Mat& dt_map,
      const cv::Mat& dt_grad_x,
      const cv::Mat& dt_grad_y,
      double fx, double fy, double cx, double cy,
      int width, int height,
      const gtsam::SharedNoiseModel& model)
      : Base(model, key),
        contour_points_(contour_points),
        dt_map_(dt_map),
        dt_grad_x_(dt_grad_x),
        dt_grad_y_(dt_grad_y),
        fx_(fx), fy_(fy), cx_(cx), cy_(cy),
        width_(width), height_(height) {}

  gtsam::Vector evaluateError(
      const gtsam::Pose3& pose,
      gtsam::OptionalMatrixType H = nullptr) const override {
    const int N = static_cast<int>(contour_points_.size());
    gtsam::Vector errors(N);
    if (H) {
      H->resize(N, 6);
      H->setZero();
    }

    const double max_error = 50.0;

    for (int i = 0; i < N; ++i) {
      gtsam::Matrix36 J_transform;
      gtsam::Point3 p_world = pose.transformFrom(contour_points_[i], J_transform);

      double xv = p_world.x();
      double yv = p_world.y();
      double zv = p_world.z() - 1.0;

      if (zv >= -0.001) {
        errors(i) = max_error;
        continue;
      }

      double u = -fx_ * xv / zv + cx_;
      double v = fy_ * yv / zv + cy_;

      int ui = static_cast<int>(std::round(u));
      int vi = static_cast<int>(std::round(v));

      if (ui < 1 || ui >= width_ - 1 || vi < 1 || vi >= height_ - 1) {
        errors(i) = max_error;
        continue;
      }

      float dt_val = dt_map_.at<float>(vi, ui);
      errors(i) = std::min(static_cast<double>(dt_val), max_error);

      if (H) {
        float gx = dt_grad_x_.at<float>(vi, ui);
        float gy = dt_grad_y_.at<float>(vi, ui);

        double inv_zv = 1.0 / zv;
        double inv_zv2 = inv_zv * inv_zv;

        Eigen::Matrix<double, 1, 3> J_dt_view;
        J_dt_view << gx * (-fx_) * inv_zv,
            gy * fy_ * inv_zv,
            gx * fx_ * xv * inv_zv2 - gy * fy_ * yv * inv_zv2;

        H->row(i) = J_dt_view * J_transform;
      }
    }

    return errors;
  }

 private:
  std::vector<gtsam::Point3> contour_points_;
  cv::Mat dt_map_;
  cv::Mat dt_grad_x_;
  cv::Mat dt_grad_y_;
  double fx_, fy_, cx_, cy_;
  int width_, height_;
};

static gtsam::Pose3 Pose6DToGTSAM(const Pose6D& p) {
  gtsam::Rot3 R =
      gtsam::Rot3::Rz(p.rz) * gtsam::Rot3::Ry(p.ry) * gtsam::Rot3::Rx(p.rx);
  return gtsam::Pose3(R, gtsam::Point3(p.tx, p.ty, p.tz));
}

static Pose6D GTSAMToPose6D(const gtsam::Pose3& pose) {
  gtsam::Vector3 rpy = pose.rotation().rpy();
  Pose6D p;
  p.rx = rpy(0);
  p.ry = rpy(1);
  p.rz = rpy(2);
  p.tx = pose.translation().x();
  p.ty = pose.translation().y();
  p.tz = pose.translation().z();
  return p;
}

static double ComputeIoU(const cv::Mat& a, const cv::Mat& b) {
  cv::Mat inter, uni;
  cv::bitwise_and(a, b, inter);
  cv::bitwise_or(a, b, uni);
  double i = cv::countNonZero(inter);
  double u = cv::countNonZero(uni);
  return u > 0 ? i / u : 0;
}

SearchResult RefinePoseLM(
    const ScoredCandidate& initial,
    const cv::Mat& input_mask,
    const std::vector<glm::vec3>& contour_points,
    const maskgen::CameraParams& camera_params,
    const maskgen::Mesh& mesh,
    maskgen::MaskGenerator& generator,
    const LMOptions& opts,
    int refine_index) {
  auto t_start = std::chrono::high_resolution_clock::now();

  cv::Mat edges;
  cv::Canny(input_mask, edges, 50, 150);
  cv::Mat dt_edges;
  cv::distanceTransform(255 - edges, dt_edges, cv::DIST_L2, cv::DIST_MASK_5);
  dt_edges.convertTo(dt_edges, CV_32F);

  cv::Mat dt_grad_x, dt_grad_y;
  cv::Sobel(dt_edges, dt_grad_x, CV_32F, 1, 0, 3);
  cv::Sobel(dt_edges, dt_grad_y, CV_32F, 0, 1, 3);

  auto t_dt = std::chrono::high_resolution_clock::now();
  double dt_ms = std::chrono::duration<double, std::milli>(t_dt - t_start).count();

  std::vector<gtsam::Point3> gtsam_points;
  gtsam_points.reserve(contour_points.size());
  for (const auto& p : contour_points) {
    gtsam_points.emplace_back(p.x, p.y, p.z);
  }

  std::cout << "[RefineLM #" << refine_index << "] Initial IoU=" << initial.iou << "\n";

  int N = static_cast<int>(gtsam_points.size());
  auto base_noise = gtsam::noiseModel::Isotropic::Sigma(N, 1.0);
  auto robust_noise = gtsam::noiseModel::Robust::Create(
      gtsam::noiseModel::mEstimator::Huber::Create(5.0), base_noise);

  gtsam::Pose3 initial_gtsam = Pose6DToGTSAM(initial.pose);
  auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector6() << 0.02, 0.02, 0.02, 0.005, 0.005, 0.005).finished());

  gtsam::NonlinearFactorGraph graph;
  graph.emplace_shared<ContourDTFactor>(
      gtsam::Symbol('x', 0), gtsam_points, dt_edges, dt_grad_x, dt_grad_y,
      camera_params.fx, camera_params.fy, camera_params.cx, camera_params.cy,
      camera_params.width, camera_params.height, robust_noise);
  graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
      gtsam::Symbol('x', 0), initial_gtsam, prior_noise);

  gtsam::Values initial_estimate;
  initial_estimate.insert(gtsam::Symbol('x', 0), initial_gtsam);

  double cost_before = graph.error(initial_estimate);

  auto t_opt_start = std::chrono::high_resolution_clock::now();

  gtsam::Values result;
  if (opts.optimizer == OptimizerType::LevenbergMarquardt) {
    gtsam::LevenbergMarquardtParams params;
    params.setMaxIterations(opts.max_iterations);
    params.setRelativeErrorTol(opts.relative_tol);
    params.setAbsoluteErrorTol(opts.absolute_tol);
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial_estimate, params);
    result = optimizer.optimize();
  } else {
    gtsam::GaussNewtonParams params;
    params.setMaxIterations(opts.max_iterations);
    params.setRelativeErrorTol(opts.relative_tol);
    params.setAbsoluteErrorTol(opts.absolute_tol);
    gtsam::GaussNewtonOptimizer optimizer(graph, initial_estimate, params);
    result = optimizer.optimize();
  }

  auto t_opt_end = std::chrono::high_resolution_clock::now();
  double opt_ms = std::chrono::duration<double, std::milli>(t_opt_end - t_opt_start).count();

  gtsam::Pose3 optimized = result.at<gtsam::Pose3>(gtsam::Symbol('x', 0));
  Pose6D refined = GTSAMToPose6D(optimized);
  double cost_after = graph.error(result);

  maskgen::MeshPose mp;
  mp.tx = refined.tx;
  mp.ty = refined.ty;
  mp.tz = refined.tz;
  mp.rx = refined.rx;
  mp.ry = refined.ry;
  mp.rz = refined.rz;
  cv::Mat final_rendered = generator.Generate(mesh, mp);
  double final_iou = ComputeIoU(final_rendered, input_mask);

  maskgen::MeshPose init_mp;
  init_mp.tx = initial.pose.tx;
  init_mp.ty = initial.pose.ty;
  init_mp.tz = initial.pose.tz;
  init_mp.rx = initial.pose.rx;
  init_mp.ry = initial.pose.ry;
  init_mp.rz = initial.pose.rz;
  cv::Mat init_rendered = generator.Generate(mesh, init_mp);
  double init_iou = ComputeIoU(init_rendered, input_mask);

  if (final_iou < init_iou) {
    final_iou = init_iou;
    refined = initial.pose;
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

  std::cout << "[RefineLM #" << refine_index
            << "] DT: " << dt_ms << " ms"
            << ", opt: " << opt_ms << " ms"
            << ", total: " << total_ms << " ms"
            << ", cost: " << cost_before << " -> " << cost_after
            << ", IoU: " << final_iou
            << (refined.tx == initial.pose.tx ? " (reverted)" : "")
            << "\n";

  return {refined, final_iou};
}

}  // namespace pose_matching
