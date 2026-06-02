#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "pose_estimator.h"

namespace pose_matching {

class Visualizer {
 public:
  virtual ~Visualizer() = default;

  virtual void SetInputMask(const cv::Mat& mask) = 0;
  virtual void SetCamera(double fx, double fy, double cx, double cy, int w, int h) = 0;
  virtual void SetMesh(const float* vertices, size_t vertex_count,
                       const uint32_t* indices, size_t index_count) = 0;

  virtual void LogCandidate(int index, const Pose6D& pose, const cv::Mat& rendered_mask,
                            double iou) = 0;
  virtual void LogCoarseComplete(const std::vector<ScoredCandidate>& top_k) = 0;
  virtual void LogRefineStep(int iteration, const Pose6D& pose, const cv::Mat& rendered_mask,
                             double cost, double iou) = 0;
  virtual void LogFinalResult(const Pose6D& pose, const cv::Mat& rendered_mask, double iou) = 0;
};

}  // namespace pose_matching
