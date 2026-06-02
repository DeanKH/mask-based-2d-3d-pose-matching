#pragma once

#include <rerun.hpp>

#include "visualizer.h"

namespace pose_matching {

class RerunVisualizer : public Visualizer {
 public:
  explicit RerunVisualizer(const std::string& recording_path);
  ~RerunVisualizer() override = default;

  void SetInputMask(const cv::Mat& mask) override;
  void SetCamera(double fx, double fy, double cx, double cy, int w, int h) override;
  void SetMesh(const float* vertices, size_t vertex_count, const uint32_t* indices,
               size_t index_count) override;

  void LogCandidate(int index, const Pose6D& pose, const cv::Mat& rendered_mask,
                    double iou) override;
  void LogCoarseComplete(const std::vector<ScoredCandidate>& top_k) override;
  void LogRefineStep(int iteration, const Pose6D& pose, const cv::Mat& rendered_mask,
                     double cost, double iou) override;
  void LogFinalResult(const Pose6D& pose, const cv::Mat& rendered_mask, double iou) override;

 private:
  rerun::Collection<rerun::Position3D> ConvertVertices(const float* vertices, size_t count);
  rerun::Collection<rerun::TriangleIndices> ConvertIndices(const uint32_t* indices,
                                                           size_t count);
  rerun::Image MatToImage(const cv::Mat& mat);
  rerun::Image CreateOverlay(const cv::Mat& input_mask, const cv::Mat& rendered_mask);

  rerun::RecordingStream rec_;
  cv::Mat input_mask_;
  bool mesh_logged_ = false;
};

}  // namespace pose_matching
