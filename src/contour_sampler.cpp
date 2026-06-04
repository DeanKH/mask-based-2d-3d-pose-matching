#include "contour_sampler.h"

#include <glm/gtc/matrix_transform.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

namespace pose_matching {

namespace {

glm::mat4 ComputeModelMatrix(const maskgen::MeshPose& pose) {
  glm::mat4 m(1.0f);
  m = glm::translate(m, glm::vec3(pose.tx, pose.ty, pose.tz));
  m = glm::rotate(m, static_cast<float>(pose.rz), glm::vec3(0, 0, 1));
  m = glm::rotate(m, static_cast<float>(pose.ry), glm::vec3(0, 1, 0));
  m = glm::rotate(m, static_cast<float>(pose.rx), glm::vec3(1, 0, 0));
  return m;
}

glm::mat4 ComputeViewMatrix(const maskgen::CameraParams& cam) {
  return glm::lookAt(
      glm::vec3(cam.eye_x, cam.eye_y, cam.eye_z),
      glm::vec3(cam.target_x, cam.target_y, cam.target_z),
      glm::vec3(cam.up_x, cam.up_y, cam.up_z));
}

glm::mat4 ComputeProjectionMatrix(const maskgen::CameraParams& cam) {
  float w = static_cast<float>(cam.width);
  float h = static_cast<float>(cam.height);
  float n = static_cast<float>(cam.near_plane);
  float f = static_cast<float>(cam.far_plane);
  glm::mat4 p(0.0f);
  p[0][0] = 2.0f * static_cast<float>(cam.fx) / w;
  p[2][0] = 1.0f - 2.0f * static_cast<float>(cam.cx) / w;
  p[1][1] = -2.0f * static_cast<float>(cam.fy) / h;
  p[2][1] = 1.0f - 2.0f * static_cast<float>(cam.cy) / h;
  p[2][2] = -(f + n) / (f - n);
  p[2][3] = -1.0f;
  p[3][2] = -2.0f * f * n / (f - n);
  return p;
}

bool RayTriangleIntersect(
    const glm::vec3& origin, const glm::vec3& dir,
    float v0x, float v0y, float v0z,
    float v1x, float v1y, float v1z,
    float v2x, float v2y, float v2z,
    float& t) {
  const float EPS = 1e-6f;
  float e1x = v1x - v0x, e1y = v1y - v0y, e1z = v1z - v0z;
  float e2x = v2x - v0x, e2y = v2y - v0y, e2z = v2z - v0z;

  float hx = dir.y * e2z - dir.z * e2y;
  float hy = dir.z * e2x - dir.x * e2z;
  float hz = dir.x * e2y - dir.y * e2x;

  float a = e1x * hx + e1y * hy + e1z * hz;
  if (a > -EPS && a < EPS) return false;

  float f = 1.0f / a;
  float sx = origin.x - v0x, sy = origin.y - v0y, sz = origin.z - v0z;

  float u = f * (sx * hx + sy * hy + sz * hz);
  if (u < 0.0f || u > 1.0f) return false;

  float qx = sy * e1z - sz * e1y;
  float qy = sz * e1x - sx * e1z;
  float qz = sx * e1y - sy * e1x;

  float v = f * (dir.x * qx + dir.y * qy + dir.z * qz);
  if (v < 0.0f || u + v > 1.0f) return false;

  t = f * (e2x * qx + e2y * qy + e2z * qz);
  return t > EPS;
}

}  // namespace

std::vector<glm::vec3> SampleContour3D(
    const maskgen::Mesh& mesh,
    const maskgen::MeshPose& initial_pose,
    const maskgen::CameraParams& camera_params,
    maskgen::MaskGenerator& generator,
    int num_points) {
  cv::Mat rendered = generator.Generate(mesh, initial_pose);

  cv::Mat binary;
  cv::threshold(rendered, binary, 127, 255, cv::THRESH_BINARY);

  cv::Mat eroded;
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::erode(binary, eroded, kernel);
  cv::Mat contour_pixels = binary - eroded;

  std::vector<cv::Point> contour_pts;
  for (int y = 0; y < contour_pixels.rows; ++y) {
    for (int x = 0; x < contour_pixels.cols; ++x) {
      if (contour_pixels.at<uchar>(y, x) > 0) {
        contour_pts.emplace_back(x, y);
      }
    }
  }

  if (contour_pts.empty()) {
    std::cerr << "[ContourSampler] No contour pixels found\n";
    return {};
  }

  glm::mat4 model = ComputeModelMatrix(initial_pose);
  glm::mat4 view = ComputeViewMatrix(camera_params);
  glm::mat4 proj = ComputeProjectionMatrix(camera_params);
  glm::mat4 mvp = proj * view * model;
  glm::mat4 inv_mvp = glm::inverse(mvp);

  int width = camera_params.width;
  int height = camera_params.height;

  const float* verts = mesh.vertices().data();
  const uint32_t* indices = mesh.indices().data();
  size_t num_tris = mesh.indices().size() / 3;

  std::vector<glm::vec3> points_3d;
  points_3d.reserve(contour_pts.size());

  for (const auto& px : contour_pts) {
    float nx = 2.0f * static_cast<float>(px.x) / width - 1.0f;
    float ny = 2.0f * static_cast<float>(px.y) / height - 1.0f;

    glm::vec4 near_pt = inv_mvp * glm::vec4(nx, ny, 0.0f, 1.0f);
    glm::vec4 far_pt = inv_mvp * glm::vec4(nx, ny, 1.0f, 1.0f);
    near_pt /= near_pt.w;
    far_pt /= far_pt.w;

    glm::vec3 origin(near_pt);
    glm::vec3 dir = glm::normalize(glm::vec3(far_pt) - origin);

    float best_t = 1e18f;
    for (size_t ti = 0; ti < num_tris; ++ti) {
      uint32_t i0 = indices[ti * 3];
      uint32_t i1 = indices[ti * 3 + 1];
      uint32_t i2 = indices[ti * 3 + 2];

      float hit_t;
      if (RayTriangleIntersect(
              origin, dir,
              verts[i0 * 3], verts[i0 * 3 + 1], verts[i0 * 3 + 2],
              verts[i1 * 3], verts[i1 * 3 + 1], verts[i1 * 3 + 2],
              verts[i2 * 3], verts[i2 * 3 + 1], verts[i2 * 3 + 2],
              hit_t)) {
        if (hit_t < best_t) best_t = hit_t;
      }
    }

    if (best_t < 1e17f) {
      points_3d.push_back(origin + dir * best_t);
    }
  }

  if (points_3d.empty()) {
    std::cerr << "[ContourSampler] No 3D contour points could be extracted\n";
    return {};
  }

  if (static_cast<int>(points_3d.size()) > num_points) {
    std::mt19937 rng(42);
    std::shuffle(points_3d.begin(), points_3d.end(), rng);
    points_3d.resize(num_points);
  }

  std::cout << "[ContourSampler] Sampled " << points_3d.size()
            << " 3D contour points (from "
            << contour_pts.size() << " contour pixels)\n";

  return points_3d;
}

}  // namespace pose_matching
