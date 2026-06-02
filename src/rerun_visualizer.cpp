#include "rerun_visualizer.h"

#include <cmath>
#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <opencv2/imgproc.hpp>

namespace pose_matching {

RerunVisualizer::RerunVisualizer() : rec_("pose_matching") {}

RerunVisualizer::RerunVisualizer(const std::string& recording_path)
    : rec_("pose_matching") {
  rec_.save(recording_path).exit_on_failure();
  std::cout << "Rerun recording: " << recording_path << "\n";
}

std::unique_ptr<RerunVisualizer> RerunVisualizer::ConnectGrpc(const std::string& url) {
  auto viz = std::unique_ptr<RerunVisualizer>(new RerunVisualizer());
  auto err = viz->rec_.connect_grpc(url);
  if (err.is_err()) {
    std::cerr << "Rerun connect error: " << err.description << "\n";
    return nullptr;
  }
  std::cout << "Rerun streaming to: " << url << "\n";
  return viz;
}

rerun::Collection<rerun::Position3D> RerunVisualizer::ConvertVertices(const float* vertices,
                                                                       size_t count) {
  std::vector<rerun::Position3D> out;
  out.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    out.emplace_back(vertices[i * 3], vertices[i * 3 + 1], vertices[i * 3 + 2]);
  }
  return rerun::Collection<rerun::Position3D>::take_ownership(std::move(out));
}

rerun::Collection<rerun::TriangleIndices> RerunVisualizer::ConvertIndices(const uint32_t* indices,
                                                                          size_t count) {
  std::vector<rerun::TriangleIndices> out;
  out.reserve(count / 3);
  for (size_t i = 0; i < count; i += 3) {
    out.emplace_back(indices[i], indices[i + 1], indices[i + 2]);
  }
  return rerun::Collection<rerun::TriangleIndices>::take_ownership(std::move(out));
}

rerun::Image RerunVisualizer::MatToImage(const cv::Mat& mat) {
  cv::Mat rgb;
  if (mat.channels() == 1) {
    cv::cvtColor(mat, rgb, cv::COLOR_GRAY2RGB);
  } else {
    rgb = mat.clone();
  }

  std::vector<uint8_t> data(rgb.total() * rgb.channels());
  std::memcpy(data.data(), rgb.data, data.size());

  return rerun::Image::from_rgb24(
      rerun::Collection<uint8_t>::take_ownership(std::move(data)),
      {static_cast<uint32_t>(rgb.cols), static_cast<uint32_t>(rgb.rows)});
}

rerun::Image RerunVisualizer::CreateOverlay(const cv::Mat& input_mask,
                                             const cv::Mat& rendered_mask) {
  cv::Mat overlay(input_mask.rows, input_mask.cols, CV_8UC3, cv::Scalar(0, 0, 0));

  for (int r = 0; r < input_mask.rows; ++r) {
    const auto* p_in = input_mask.ptr<uint8_t>(r);
    const auto* p_ren = rendered_mask.ptr<uint8_t>(r);
    auto* p_out = overlay.ptr<cv::Vec3b>(r);
    for (int c = 0; c < input_mask.cols; ++c) {
      bool in_mask = p_in[c] > 127;
      bool ren_mask = p_ren[c] > 127;
      if (in_mask && ren_mask) {
        p_out[c] = cv::Vec3b(0, 200, 0);
      } else if (ren_mask) {
        p_out[c] = cv::Vec3b(0, 0, 200);
      } else if (in_mask) {
        p_out[c] = cv::Vec3b(200, 100, 0);
      }
    }
  }

  std::vector<uint8_t> data(overlay.total() * 3);
  std::memcpy(data.data(), overlay.data, data.size());
  return rerun::Image::from_rgb24(
      rerun::Collection<uint8_t>::take_ownership(std::move(data)),
      {static_cast<uint32_t>(overlay.cols), static_cast<uint32_t>(overlay.rows)});
}

static rerun::archetypes::Transform3D MakeTransform(const Pose6D& pose) {
  glm::mat4 model(1.0f);
  model = glm::translate(model, glm::vec3(pose.tx, pose.ty, pose.tz));
  model = glm::rotate(model, static_cast<float>(pose.rz), glm::vec3(0, 0, 1));
  model = glm::rotate(model, static_cast<float>(pose.ry), glm::vec3(0, 1, 0));
  model = glm::rotate(model, static_cast<float>(pose.rx), glm::vec3(1, 0, 0));

  glm::mat3 R3(model);

  rerun::datatypes::Vec3D columns[3] = {
      {R3[0][0], R3[0][1], R3[0][2]},
      {R3[1][0], R3[1][1], R3[1][2]},
      {R3[2][0], R3[2][1], R3[2][2]},
  };

  return rerun::archetypes::Transform3D(
      rerun::components::Translation3D{static_cast<float>(pose.tx),
                                       static_cast<float>(pose.ty),
                                       static_cast<float>(pose.tz)},
      columns);
}

void RerunVisualizer::SetInputMask(const cv::Mat& mask) {
  input_mask_ = mask.clone();
  rec_.log("masks/input", MatToImage(mask));
}

void RerunVisualizer::SetCamera(double fx, double fy, double cx, double cy, int w, int h) {
  float fx_f = static_cast<float>(fx);
  float fy_f = static_cast<float>(fy);
  float cx_f = static_cast<float>(cx);
  float cy_f = static_cast<float>(cy);

  std::array<float, 9> K_col_major = {
      fx_f, 0.0f, cx_f,  // column 0
      0.0f, fy_f, cy_f,  // column 1
      0.0f, 0.0f, 1.0f,  // column 2
  };

  rec_.log("scene/camera",
           rerun::Pinhole(rerun::components::PinholeProjection(K_col_major))
               .with_resolution(w, h));
  rec_.log_static("scene/camera", rerun::ViewCoordinates::RDF);
}

void RerunVisualizer::SetMesh(const float* vertices, size_t vertex_count,
                                const uint32_t* indices, size_t index_count) {
  mesh_vertices_.assign(vertices, vertices + vertex_count * 3);
  mesh_indices_.assign(indices, indices + index_count);

  auto positions = ConvertVertices(vertices, vertex_count);
  auto triangles = ConvertIndices(indices, index_count);

  rec_.log("scene/model_template",
           rerun::Mesh3D(std::move(positions)).with_triangle_indices(std::move(triangles)));
  mesh_logged_ = true;
}

void RerunVisualizer::LogCandidate(int index, const Pose6D& pose, const cv::Mat& rendered_mask,
                                     double iou) {
  rec_.set_time_sequence("candidate", index);

  rec_.log("masks/rendered", MatToImage(rendered_mask));
  rec_.log("masks/overlay", CreateOverlay(input_mask_, rendered_mask));
  rec_.log("scores/iou", rerun::Scalars(iou));

  rec_.log("scene/object", MakeTransform(pose));
  if (!mesh_vertices_.empty()) {
    auto positions = ConvertVertices(mesh_vertices_.data(), mesh_vertices_.size() / 3);
    auto triangles = ConvertIndices(mesh_indices_.data(), mesh_indices_.size());
    rec_.log("scene/object/mesh",
             rerun::Mesh3D(std::move(positions)).with_triangle_indices(std::move(triangles)));
  }

  std::string label = "candidate " + std::to_string(index) + " | IoU=" +
                      std::to_string(iou).substr(0, 6) +
                      " | t=(" + std::to_string(pose.tx).substr(0, 6) + ", " +
                      std::to_string(pose.ty).substr(0, 6) + ", " +
                      std::to_string(pose.tz).substr(0, 6) + ")" +
                      " | r=(" +
                      std::to_string(pose.rx * 180.0 / M_PI).substr(0, 6) + ", " +
                      std::to_string(pose.ry * 180.0 / M_PI).substr(0, 6) + ", " +
                      std::to_string(pose.rz * 180.0 / M_PI).substr(0, 6) + ") deg";
  rec_.log("info/pose", rerun::TextLog(label));
}

void RerunVisualizer::LogCoarseComplete(const std::vector<ScoredCandidate>& top_k) {
  rec_.log("info/coarse_complete",
           rerun::TextLog("Coarse search complete. Top " + std::to_string(top_k.size()) +
                          " candidates selected for refinement."));
}

void RerunVisualizer::LogRefineStep(int iteration, const Pose6D& pose,
                                      const cv::Mat& rendered_mask, double cost, double iou) {
  rec_.set_time_sequence("refine", iteration);

  rec_.log("masks/refined", MatToImage(rendered_mask));
  rec_.log("masks/overlay_refined", CreateOverlay(input_mask_, rendered_mask));
  rec_.log("scores/cost", rerun::Scalars(cost));
  rec_.log("scores/iou_refined", rerun::Scalars(iou));

  rec_.log("scene/object_refined", MakeTransform(pose));
  if (!mesh_vertices_.empty()) {
    auto positions = ConvertVertices(mesh_vertices_.data(), mesh_vertices_.size() / 3);
    auto triangles = ConvertIndices(mesh_indices_.data(), mesh_indices_.size());
    rec_.log("scene/object_refined/mesh",
             rerun::Mesh3D(std::move(positions)).with_triangle_indices(std::move(triangles)));
  }

  rec_.log("info/refine",
           rerun::TextLog("iter=" + std::to_string(iteration) +
                          " | cost=" + std::to_string(cost).substr(0, 8) +
                          " | IoU=" + std::to_string(iou).substr(0, 6)));
}

void RerunVisualizer::LogFinalResult(const Pose6D& pose, const cv::Mat& rendered_mask,
                                      double iou) {
  rec_.set_time_sequence("result", 0);

  rec_.log("masks/final", MatToImage(rendered_mask));
  rec_.log("masks/overlay_final", CreateOverlay(input_mask_, rendered_mask));
  rec_.log("scores/final_iou", rerun::Scalars(iou));

  rec_.log("scene/final", MakeTransform(pose));

  if (!mesh_vertices_.empty()) {
    auto positions = ConvertVertices(mesh_vertices_.data(), mesh_vertices_.size() / 3);
    auto triangles = ConvertIndices(mesh_indices_.data(), mesh_indices_.size());
    rec_.log("scene/final/mesh",
             rerun::Mesh3D(std::move(positions)).with_triangle_indices(std::move(triangles)));
  }

  rec_.log("info/final",
           rerun::TextLog("FINAL | IoU=" + std::to_string(iou).substr(0, 6) +
                          " | tx=" + std::to_string(pose.tx) + " ty=" + std::to_string(pose.ty) +
                          " tz=" + std::to_string(pose.tz) + " | rx=" +
                          std::to_string(pose.rx * 180.0 / M_PI) + " ry=" +
                          std::to_string(pose.ry * 180.0 / M_PI) + " rz=" +
                          std::to_string(pose.rz * 180.0 / M_PI) + " deg"));
}

}  // namespace pose_matching
