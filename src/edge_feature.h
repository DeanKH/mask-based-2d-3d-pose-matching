#pragma once

#include <opencv2/core.hpp>

namespace pose_matching {

// RGB画像からCannyエッジマップを抽出する。
// グレースケール化 → ガウスぼかし(3x3) → Canny。
cv::Mat ExtractEdgeMap(const cv::Mat& rgb, double low_threshold,
                      double high_threshold);

// エッジマップの距離変換 (非エッジ画素から最近傍エッジまでのL2距離, CV_32F)。
cv::Mat ComputeEdgeDistanceTransform(const cv::Mat& edge_map);

// レンダリングマスクの輪郭画素をエッジDT上で評価した正規化コスト [0,1]。
// 輪郭が画像エッジに近いほど0に近づく。輪郭が空の場合は1.0。
//   cost = mean(min(dt, max_distance)) / max_distance
double ContourEdgeCost(const cv::Mat& rendered_mask, const cv::Mat& edge_dt,
                       double max_distance);

}  // namespace pose_matching
