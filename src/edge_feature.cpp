#include "edge_feature.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <opencv2/imgproc.hpp>

namespace pose_matching {

cv::Mat ExtractEdgeMap(const cv::Mat& rgb, double low_threshold,
                       double high_threshold) {
  cv::Mat gray;
  if (rgb.channels() == 1) {
    gray = rgb;
  } else {
    cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);
  }

  cv::Mat blurred;
  cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 0);

  cv::Mat edges;
  cv::Canny(blurred, edges, low_threshold, high_threshold);
  return edges;
}

cv::Mat ComputeEdgeDistanceTransform(const cv::Mat& edge_map) {
  cv::Mat dt;
  cv::distanceTransform(255 - edge_map, dt, cv::DIST_L2, 5);
  cv::Mat dt_f;
  dt.convertTo(dt_f, CV_32F);
  return dt_f;
}

double ContourEdgeCost(const cv::Mat& rendered_mask, const cv::Mat& edge_dt,
                       double max_distance) {
  if (edge_dt.empty() || rendered_mask.empty()) return 1.0;
  if (max_distance <= 0.0) return 1.0;

  cv::Mat binary;
  if (rendered_mask.type() == CV_8U) {
    cv::threshold(rendered_mask, binary, 127, 255, cv::THRESH_BINARY);
  } else {
    rendered_mask.convertTo(binary, CV_8U);
    cv::threshold(binary, binary, 127, 255, cv::THRESH_BINARY);
  }

  // 輪郭画素 = binary - erode(binary)
  cv::Mat eroded;
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::erode(binary, eroded, kernel);
  cv::Mat contour = binary - eroded;

  double sum = 0;
  int count = 0;
  for (int y = 0; y < contour.rows; ++y) {
    const uint8_t* row = contour.ptr<uint8_t>(y);
    const float* dt_row = edge_dt.ptr<float>(y);
    for (int x = 0; x < contour.cols; ++x) {
      if (row[x]) {
        sum += std::min(static_cast<double>(dt_row[x]), max_distance);
        ++count;
      }
    }
  }
  if (count == 0) return 1.0;
  return (sum / static_cast<double>(count)) / max_distance;
}

}  // namespace pose_matching
