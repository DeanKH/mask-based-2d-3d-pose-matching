#include "cached_pose_estimator.h"

#include <chrono>
#include <cmath>
#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <thread>

#include "profiling.h"

#include "contour_sampler.h"
#include "refine_lm.h"

#include <nlohmann/json.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "nelder_mead.h"
#include "visualizer.h"

namespace {

double ComputeHuDistance(const cv::Mat& a, const cv::Mat& b) {
  cv::Mat hu_a, hu_b;
  cv::HuMoments(cv::moments(a), hu_a);
  cv::HuMoments(cv::moments(b), hu_b);
  double dist = 0;
  for (int i = 0; i < 7; ++i) {
    double ha = hu_a.at<double>(i);
    double hb = hu_b.at<double>(i);
    double la = -std::copysign(std::log10(std::abs(ha) + 1e-30), ha);
    double lb = -std::copysign(std::log10(std::abs(hb) + 1e-30), hb);
    double d = la - lb;
    dist += d * d;
  }
  return std::sqrt(dist);
}

double ZernikeRadial(int n, int m, double r) {
  double sum = 0;
  int s_max = (n - m) / 2;
  for (int s = 0; s <= s_max; ++s) {
    double coeff = (s % 2 == 0 ? 1.0 : -1.0);
    coeff *= std::tgamma(n - s + 1);
    coeff /= std::tgamma(s + 1);
    coeff /= std::tgamma((n + m) / 2 - s + 1);
    coeff /= std::tgamma((n - m) / 2 - s + 1);
    sum += coeff * std::pow(r, n - 2 * s);
  }
  return sum;
}

std::vector<double> ComputeZernikeFeatures(const cv::Mat& mask, int max_order) {
  cv::Moments mom = cv::moments(mask, true);
  if (mom.m00 == 0) return {};
  double cx = mom.m10 / mom.m00;
  double cy = mom.m01 / mom.m00;

  double max_r = 0;
  for (int y = 0; y < mask.rows; ++y) {
    const uint8_t* row = mask.ptr<uint8_t>(y);
    for (int x = 0; x < mask.cols; ++x) {
      if (row[x] == 0) continue;
      double dx = x - cx, dy = y - cy;
      max_r = std::max(max_r, std::sqrt(dx * dx + dy * dy));
    }
  }
  if (max_r < 1.0) max_r = 1.0;

  struct Px { double r, theta; };
  std::vector<Px> pixels;
  for (int y = 0; y < mask.rows; ++y) {
    const uint8_t* row = mask.ptr<uint8_t>(y);
    for (int x = 0; x < mask.cols; ++x) {
      if (row[x] == 0) continue;
      double dx = (x - cx) / max_r, dy = (y - cy) / max_r;
      double r = std::sqrt(dx * dx + dy * dy);
      if (r > 1.0) continue;
      pixels.push_back({r, std::atan2(dy, dx)});
    }
  }

  std::vector<double> features;
  for (int n = 0; n <= max_order; ++n) {
    for (int m = 0; m <= n; ++m) {
      if ((n - m) % 2 != 0) continue;
      double re = 0, im = 0;
      for (const auto& px : pixels) {
        double R = ZernikeRadial(n, m, px.r);
        re += R * std::cos(m * px.theta);
        im += R * std::sin(m * px.theta);
      }
      double norm = (n + 1) / M_PI;
      features.push_back(std::sqrt(re * re + im * im) * norm);
    }
  }
  return features;
}

double ComputeZernikeDistance(const cv::Mat& a, const cv::Mat& b, int max_order = 8) {
  auto fa = ComputeZernikeFeatures(a, max_order);
  auto fb = ComputeZernikeFeatures(b, max_order);
  if (fa.size() != fb.size() || fa.empty()) return 1e6;
  double na = 0, nb = 0;
  for (size_t i = 0; i < fa.size(); ++i) {
    na += fa[i] * fa[i];
    nb += fb[i] * fb[i];
  }
  na = std::sqrt(na);
  nb = std::sqrt(nb);
  if (na > 0) for (auto& v : fa) v /= na;
  if (nb > 0) for (auto& v : fb) v /= nb;
  double dist = 0;
  for (size_t i = 0; i < fa.size(); ++i) {
    double d = fa[i] - fb[i];
    dist += d * d;
  }
  return std::sqrt(dist);
}

cv::Mat ShiftMaskByCentroid(const cv::Mat& reference, const cv::Mat& mask) {
  cv::Moments mi = cv::moments(reference, true);
  cv::Moments mr = cv::moments(mask, true);
  if (mi.m00 == 0 || mr.m00 == 0) return cv::Mat::zeros(mask.rows, mask.cols, mask.type());
  int du = static_cast<int>(std::round(mi.m10 / mi.m00 - mr.m10 / mr.m00));
  int dv = static_cast<int>(std::round(mi.m01 / mi.m00 - mr.m01 / mr.m00));
  cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, du, 0, 1, dv);
  cv::Mat shifted;
  cv::warpAffine(mask, shifted, M, mask.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, 0);
  return shifted;
}

double ComputeMaskIoU(const cv::Mat& a, const cv::Mat& b) {
  cv::Mat ab;
  cv::bitwise_and(a, b, ab);
  int ab_count = cv::countNonZero(ab);
  int a_count = cv::countNonZero(a);
  int b_count = cv::countNonZero(b);
  int denom = a_count + b_count - ab_count;
  return denom > 0 ? static_cast<double>(ab_count) / denom : 0.0;
}

double ComputeCentroidAlignedIoU(const cv::Mat& input, const cv::Mat& rendered) {
  cv::Mat shifted = ShiftMaskByCentroid(input, rendered);
  return ComputeMaskIoU(input, shifted);
}

double ComputeAvgDT(const cv::Mat& mask, const cv::Mat& dt, bool use_l2) {
  double sum = 0;
  int count = 0;
  for (int y = 0; y < mask.rows; ++y) {
    const uint8_t* row = mask.ptr<uint8_t>(y);
    const float* dt_row = dt.ptr<float>(y);
    for (int x = 0; x < mask.cols; ++x) {
      if (row[x]) {
        sum += use_l2 ? dt_row[x] * dt_row[x] : static_cast<double>(dt_row[x]);
        ++count;
      }
    }
  }
  if (count == 0) return 1e6;
  return use_l2 ? std::sqrt(sum / count) : sum / count;
}

double ComputeCentroidDT(const cv::Mat& input, const cv::Mat& rendered,
                          const cv::Mat& dt, bool use_l2) {
  cv::Mat shifted = ShiftMaskByCentroid(input, rendered);
  return ComputeAvgDT(shifted, dt, use_l2);
}

double ComputeDT_IoU(const cv::Mat& input, const cv::Mat& rendered,
                      const cv::Mat& dt_l2) {
  cv::Mat shifted = ShiftMaskByCentroid(input, rendered);
  double avg_dt = ComputeAvgDT(shifted, dt_l2, true);
  double iou = ComputeMaskIoU(input, shifted);
  return avg_dt * (1.0 - iou);
}

double ComputeCentralMomentsDist(const cv::Mat& input, const cv::Mat& rendered) {
  cv::Mat shifted = ShiftMaskByCentroid(input, rendered);
  cv::Moments ma = cv::moments(input, true);
  cv::Moments mb = cv::moments(shifted, true);
  if (ma.m00 == 0 || mb.m00 == 0) return 1e6;
  double a_m[] = {ma.nu20, ma.nu11, ma.nu02, ma.nu30, ma.nu21, ma.nu12, ma.nu03};
  double b_m[] = {mb.nu20, mb.nu11, mb.nu02, mb.nu30, mb.nu21, mb.nu12, mb.nu03};
  double dist = 0;
  for (int i = 0; i < 7; ++i) {
    double d = a_m[i] - b_m[i];
    dist += d * d;
  }
  return std::sqrt(dist);
}

std::vector<cv::Point2f> ExtractLargestContour(const cv::Mat& mask) {
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
  if (contours.empty()) return {};
  size_t best = 0;
  for (size_t i = 1; i < contours.size(); ++i)
    if (contours[i].size() > contours[best].size()) best = i;
  std::vector<cv::Point2f> pts;
  for (auto& p : contours[best]) pts.emplace_back(p.x, p.y);
  return pts;
}

std::vector<cv::Point2f> ResampleContour(const std::vector<cv::Point2f>& pts, int n) {
  if (static_cast<int>(pts.size()) < 2 || n < 2) return pts;
  std::vector<float> cum_len(pts.size());
  cum_len[0] = 0;
  for (size_t i = 1; i < pts.size(); ++i) {
    float dx = pts[i].x - pts[i - 1].x;
    float dy = pts[i].y - pts[i - 1].y;
    cum_len[i] = cum_len[i - 1] + std::sqrt(dx * dx + dy * dy);
  }
  float total = cum_len.back();
  if (total < 1e-6f) return pts;
  std::vector<cv::Point2f> resampled(n);
  int seg = 0;
  for (int i = 0; i < n; ++i) {
    float target = total * i / n;
    while (seg < static_cast<int>(cum_len.size()) - 2 && cum_len[seg + 1] < target) ++seg;
    float denom = cum_len[seg + 1] - cum_len[seg];
    float t = denom > 0 ? (target - cum_len[seg]) / denom : 0;
    resampled[i].x = pts[seg].x + t * (pts[seg + 1].x - pts[seg].x);
    resampled[i].y = pts[seg].y + t * (pts[seg + 1].y - pts[seg].y);
  }
  return resampled;
}

double ComputeShapeContextDist(const cv::Mat& input, const cv::Mat& rendered) {
  cv::Mat shifted = ShiftMaskByCentroid(input, rendered);
  auto pa = ExtractLargestContour(input);
  auto pb = ExtractLargestContour(shifted);
  if (pa.empty() || pb.empty()) return 1e6;
  int n_pts = 100;
  pa = ResampleContour(pa, n_pts);
  pb = ResampleContour(pb, n_pts);
  int n_bins_r = 5, n_bins_t = 12;
  int n_bins = n_bins_r * n_bins_t;
  auto computeDesc = [&](const std::vector<cv::Point2f>& pts) {
    std::vector<std::vector<float>> desc(n_pts, std::vector<float>(n_bins, 0));
    for (int i = 0; i < n_pts; ++i) {
      for (int j = 0; j < n_pts; ++j) {
        if (i == j) continue;
        float dx = pts[j].x - pts[i].x;
        float dy = pts[j].y - pts[i].y;
        float r = std::sqrt(dx * dx + dy * dy);
        float theta = std::atan2(dy, dx);
        int r_bin = std::min(n_bins_r - 1, static_cast<int>(std::log(r + 1) / std::log(2)));
        int t_bin = static_cast<int>((theta + static_cast<float>(M_PI)) /
                                      (2 * static_cast<float>(M_PI)) * n_bins_t) % n_bins_t;
        desc[i][r_bin * n_bins_t + t_bin]++;
      }
    }
    for (auto& d : desc) {
      float s = 0;
      for (auto v : d) s += v;
      if (s > 0) for (auto& v : d) v /= s;
    }
    return desc;
  };
  auto da = computeDesc(pa);
  auto db = computeDesc(pb);
  double total = 0;
  for (int i = 0; i < n_pts; ++i) {
    double chi2 = 0;
    for (int k = 0; k < n_bins; ++k) {
      double s = da[i][k] + db[i][k];
      if (s > 0) { double d = da[i][k] - db[i][k]; chi2 += d * d / s; }
    }
    total += chi2;
  }
  return total / n_pts;
}

double ComputeContourChamferDist(const cv::Mat& input, const cv::Mat& rendered) {
  cv::Mat shifted = ShiftMaskByCentroid(input, rendered);
  auto pa = ExtractLargestContour(input);
  auto pb = ExtractLargestContour(shifted);
  if (pa.empty() || pb.empty()) return 1e6;
  int n_pts = 200;
  pa = ResampleContour(pa, n_pts);
  pb = ResampleContour(pb, n_pts);
  double forward = 0;
  for (auto& p : pa) {
    double min_d = 1e9;
    for (auto& q : pb) { double d = (p.x-q.x)*(p.x-q.x)+(p.y-q.y)*(p.y-q.y); if (d < min_d) min_d = d; }
    forward += std::sqrt(min_d);
  }
  forward /= n_pts;
  double backward = 0;
  for (auto& p : pb) {
    double min_d = 1e9;
    for (auto& q : pa) { double d = (p.x-q.x)*(p.x-q.x)+(p.y-q.y)*(p.y-q.y); if (d < min_d) min_d = d; }
    backward += std::sqrt(min_d);
  }
  backward /= n_pts;
  return (forward + backward) / 2;
}

double ComputeFourierDescriptorDist(const cv::Mat& input, const cv::Mat& rendered) {
  cv::Mat shifted = ShiftMaskByCentroid(input, rendered);
  auto pa = ExtractLargestContour(input);
  auto pb = ExtractLargestContour(shifted);
  if (pa.empty() || pb.empty()) return 1e6;
  int n_pts = 128;
  pa = ResampleContour(pa, n_pts);
  pb = ResampleContour(pb, n_pts);
  auto computeDesc = [&](const std::vector<cv::Point2f>& pts) -> std::vector<double> {
    cv::Mat cmplx(1, n_pts, CV_64FC2);
    cv::Vec2d ctr(0, 0);
    for (int i = 0; i < n_pts; ++i) { cmplx.at<cv::Vec2d>(0, i) = cv::Vec2d(pts[i].x, pts[i].y); ctr += cmplx.at<cv::Vec2d>(0, i); }
    ctr /= n_pts;
    for (int i = 0; i < n_pts; ++i) cmplx.at<cv::Vec2d>(0, i) -= ctr;
    cv::Mat dft_r;
    cv::dft(cmplx, dft_r);
    int K = 32;
    std::vector<double> desc(K);
    for (int i = 0; i < K; ++i) { auto v = dft_r.at<cv::Vec2d>(0, i + 1); desc[i] = std::sqrt(v[0]*v[0]+v[1]*v[1]); }
    double norm = desc[0] > 0 ? desc[0] : 1.0;
    for (auto& d : desc) d /= norm;
    return desc;
  };
  auto da = computeDesc(pa);
  auto db = computeDesc(pb);
  double dist = 0;
  for (size_t i = 0; i < da.size(); ++i) { double d = da[i] - db[i]; dist += d * d; }
  return std::sqrt(dist);
}

}  // namespace

namespace pose_matching {

CachedPoseEstimator::CachedPoseEstimator(
    const maskgen::CameraParams& camera_params,
    const std::string& model_path, float model_scale,
    const std::string& cache_path)
    : camera_params_(camera_params), principal_axis_(2) {
  if (!mesh_.LoadFromFile(model_path, model_scale)) {
    throw std::runtime_error("Failed to load mesh: " + model_path);
  }

  const auto& verts = mesh_.vertices();
  const size_t n = verts.size() / 3;
  mesh_centroid_[0] = mesh_centroid_[1] = mesh_centroid_[2] = 0.0f;
  float min_p[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
  float max_p[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
  for (size_t i = 0; i < n; ++i) {
    for (int d = 0; d < 3; ++d) {
      float v = verts[i * 3 + d];
      min_p[d] = std::min(min_p[d], v);
      max_p[d] = std::max(max_p[d], v);
      mesh_centroid_[d] += v;
    }
  }
  float max_extent = 0;
  for (int d = 0; d < 3; ++d) {
    mesh_centroid_[d] /= static_cast<float>(n);
    mesh_extent_[d] = max_p[d] - min_p[d];
    if (mesh_extent_[d] > max_extent) {
      max_extent = mesh_extent_[d];
      principal_axis_ = d;
    }
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  std::vector<uint8_t> file_buffer;
  if (!ReadCache(cache_path, cache_)) {
    throw std::runtime_error("Failed to read cache file: " + cache_path);
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  double cache_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "[Timing] Cache loading: " << cache_ms << " ms\n";
  std::cout << "  Coarse entries: " << cache_.coarse_entries.size() << "\n";
  std::cout << "  Local entries: " << cache_.local_entries.size() << "\n";
  std::cout << "  Mask data: " << cache_.mask_data.size() / (1024.0 * 1024.0) << " MB\n";

  camera_params_ = camera_params;
  camera_params_.eye_x = 0;
  camera_params_.eye_y = 0;
  camera_params_.eye_z = 0;
  camera_params_.target_x = 0;
  camera_params_.target_y = 0;
  camera_params_.target_z = 1;
  camera_params_.up_x = 0;
  camera_params_.up_y = -1;
  camera_params_.up_z = 0;

  generator_ = std::make_unique<maskgen::MaskGenerator>(camera_params_);
  generator_->SetMesh(mesh_);

  auto t_decode_start = std::chrono::high_resolution_clock::now();
  int n_coarse = static_cast<int>(cache_.coarse_entries.size());
  decoded_coarse_masks_.resize(n_coarse);
  decoded_coarse_areas_.resize(n_coarse, 0);
  #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < n_coarse; ++i) {
    if (cache_.coarse_entries[i].mask_size > 0) {
      decoded_coarse_masks_[i] = DecodeMask(cache_.coarse_entries[i].mask_offset,
                                            cache_.coarse_entries[i].mask_size);
      decoded_coarse_areas_[i] = cv::countNonZero(decoded_coarse_masks_[i]);
    }
  }
  auto t_decode_end = std::chrono::high_resolution_clock::now();
  double decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
  std::cout << "[Timing] Pre-decode coarse masks: " << decode_ms << " ms\n";

  small_w_ = cache_.header.image_width / 4;
  small_h_ = cache_.header.image_height / 4;
  decoded_coarse_masks_small_.resize(n_coarse);
  decoded_coarse_areas_small_.resize(n_coarse, 0);
  auto t_small_start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < n_coarse; ++i) {
    if (decoded_coarse_areas_[i] > 0) {
      cv::resize(decoded_coarse_masks_[i], decoded_coarse_masks_small_[i],
                 cv::Size(small_w_, small_h_), 0, 0, cv::INTER_NEAREST);
      decoded_coarse_areas_small_[i] = cv::countNonZero(decoded_coarse_masks_small_[i]);
    }
  }
  auto t_small_end = std::chrono::high_resolution_clock::now();
  double small_ms = std::chrono::duration<double, std::milli>(t_small_end - t_small_start).count();
  std::cout << "[Timing] Downsample coarse masks (1/4): " << small_ms << " ms\n";

  int n_local = static_cast<int>(cache_.local_entries.size());
  decoded_local_masks_small_.resize(n_local);
  decoded_local_areas_small_.resize(n_local, 0);
  auto t_local_start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < n_local; ++i) {
    if (cache_.local_entries[i].mask_size > 0) {
      cv::Mat full = DecodeMask(cache_.local_entries[i].mask_offset,
                                cache_.local_entries[i].mask_size);
      cv::resize(full, decoded_local_masks_small_[i],
                 cv::Size(small_w_, small_h_), 0, 0, cv::INTER_NEAREST);
      decoded_local_areas_small_[i] = cv::countNonZero(decoded_local_masks_small_[i]);
    }
  }
  auto t_local_end = std::chrono::high_resolution_clock::now();
  double local_ms = std::chrono::duration<double, std::milli>(t_local_end - t_local_start).count();
  std::cout << "[Timing] Decode+downsample local masks (1/4): " << local_ms << " ms\n";
}

void CachedPoseEstimator::SetVisualizer(Visualizer* viz) { viz_ = viz; }

cv::Mat CachedPoseEstimator::DecodeMask(size_t mask_offset, uint32_t mask_size) const {
  const uint8_t* compressed = cache_.mask_data.data() + mask_offset;
  int packed_cols = (cache_.header.image_width + 7) / 8;
  size_t expected = static_cast<size_t>(packed_cols) * cache_.header.image_height;
  std::vector<uint8_t> packed = ZlibDecompress(compressed, mask_size, expected);
  return UnpackMask1Bit(packed.data(), packed.size(),
                        cache_.header.image_width, cache_.header.image_height);
}

cv::Mat CachedPoseEstimator::ShiftMask(const cv::Mat& mask, double dx, double dy) const {
  if (std::abs(dx) < 0.5 && std::abs(dy) < 0.5) return mask;

  cv::Mat M = (cv::Mat_<float>(2, 3) << 1, 0, dx, 0, 1, dy);
  cv::Mat shifted;
  cv::warpAffine(mask, shifted, M, mask.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT,
                 cv::Scalar(0));
  return shifted;
}

double CachedPoseEstimator::ComputeIoU(const cv::Mat& a, const cv::Mat& b) const {
  cv::Mat ab;
  cv::bitwise_and(a, b, ab);
  int ab_count = cv::countNonZero(ab);
  int a_count = cv::countNonZero(a);
  int b_count = cv::countNonZero(b);
  int denom = a_count + b_count - ab_count;
  if (denom == 0) return 0.0;
  return static_cast<double>(ab_count) / denom;
}

double CachedPoseEstimator::ShiftedIoU(const cv::Mat& base, const cv::Mat& input,
                                       int du, int dv,
                                       int base_count, int input_count) const {
  int h = input.rows;
  int w = input.cols;

  int b_r0 = std::max(0, -dv);
  int b_r1 = std::min(h, h - dv);
  int b_c0 = std::max(0, -du);
  int b_c1 = std::min(w, w - du);

  int overlap_h = b_r1 - b_r0;
  int overlap_w = b_c1 - b_c0;
  if (overlap_h <= 0 || overlap_w <= 0) return 0.0;

  int i_r0 = b_r0 + dv;
  int i_c0 = b_c0 + du;

  cv::Rect base_rect(b_c0, b_r0, overlap_w, overlap_h);
  cv::Rect input_rect(i_c0, i_r0, overlap_w, overlap_h);

  cv::Mat ab;
  cv::bitwise_and(base(base_rect), input(input_rect), ab);
  int intersection = cv::countNonZero(ab);

  int denom = base_count + input_count - intersection;
  if (denom == 0) return 0.0;
  return static_cast<double>(intersection) / denom;
}

std::vector<ScoredCandidate> CachedPoseEstimator::CachedCoarseSearch(
    const cv::Mat& input_mask, const cv::Mat& dt_input,
    const EstimationParams& params) {
  cv::Moments m = cv::moments(input_mask, true);
  if (m.m00 == 0) return {};

  double u_cx = m.m10 / m.m00;
  double v_cy = m.m01 / m.m00;

  double shift_u = u_cx - cache_.header.cx;
  double shift_v = v_cy - cache_.header.cy;

  int n_entries = static_cast<int>(cache_.coarse_entries.size());
  int input_count = cv::countNonZero(input_mask);
  int du = static_cast<int>(std::round(shift_u));
  int dv = static_cast<int>(std::round(shift_v));

  // Phase 1: Low-res IoU for all entries
  cv::Mat input_mask_small;
  cv::resize(input_mask, input_mask_small, cv::Size(small_w_, small_h_), 0, 0, cv::INTER_NEAREST);
  int input_count_small = cv::countNonZero(input_mask_small);
  int du_small = static_cast<int>(std::round(static_cast<double>(du) / 4.0));
  int dv_small = static_cast<int>(std::round(static_cast<double>(dv) / 4.0));

  std::vector<int> lowres_idx(n_entries);
  std::vector<double> lowres_iou(n_entries, -1.0);

  #pragma omp parallel for schedule(dynamic)
  for (int ei = 0; ei < n_entries; ++ei) {
    lowres_idx[ei] = ei;
    if (decoded_coarse_areas_small_[ei] == 0) continue;
    lowres_iou[ei] = ShiftedIoU(decoded_coarse_masks_small_[ei], input_mask_small,
                                 du_small, dv_small,
                                 decoded_coarse_areas_small_[ei], input_count_small);
  }

  std::sort(lowres_idx.begin(), lowres_idx.end(),
            [&](int a, int b) { return lowres_iou[a] > lowres_iou[b]; });

  // Phase 2: Full-res IoU for top N only
  int n_fullres = std::min(std::max(params.top_k_coarse * 8, 64), n_entries);
  std::vector<ScoredCandidate> fullres_results(n_fullres);

  #pragma omp parallel for schedule(dynamic)
  for (int fi = 0; fi < n_fullres; ++fi) {
    int ei = lowres_idx[fi];
    if (decoded_coarse_areas_[ei] == 0) {
      fullres_results[fi] = {{}, -1.0, 1e6};
      continue;
    }
    double iou = ShiftedIoU(decoded_coarse_masks_[ei], input_mask,
                            du, dv, decoded_coarse_areas_[ei], input_count);

    const auto& entry = cache_.coarse_entries[ei];
    double tz_total = entry.depth;
    double tz = tz_total - entry.crot_z;

    Pose6D pose;
    pose.rx = entry.rx;
    pose.ry = entry.ry;
    pose.rz = entry.rz;
    pose.tx = (u_cx - cache_.header.cx) * tz_total / cache_.header.fx - entry.crot_x;
    pose.ty = (v_cy - cache_.header.cy) * tz_total / cache_.header.fy - entry.crot_y;
    pose.tz = tz;

    fullres_results[fi] = {pose, iou, 1.0 - iou};
  }

  std::vector<ScoredCandidate> candidates;
  candidates.reserve(n_fullres);
  for (int fi = 0; fi < n_fullres; ++fi) {
    if (fullres_results[fi].iou < 0) continue;
    if (viz_) {
      int ei = lowres_idx[fi];
      cv::Mat shifted = ShiftMask(decoded_coarse_masks_[ei], shift_u, shift_v);
      viz_->LogCandidate(fi, fullres_results[fi].pose, shifted, fullres_results[fi].iou);
    }
    candidates.push_back(fullres_results[fi]);
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const ScoredCandidate& a, const ScoredCandidate& b) {
              return a.cost < b.cost;
            });

  std::vector<ScoredCandidate> diverse;
  std::set<int> seen_dirs;
  for (const auto& c : candidates) {
    int key = static_cast<int>(c.pose.rx * 10) * 100000 +
              static_cast<int>(c.pose.ry * 10) * 1000 +
              static_cast<int>(c.pose.rz * 10);
    if (seen_dirs.insert(key).second) {
      diverse.push_back(c);
    }
    if (static_cast<int>(diverse.size()) >= params.top_k_coarse) break;
  }

  std::cout << "Top " << diverse.size() << " coarse candidates (cached):\n";
  for (size_t i = 0; i < diverse.size(); ++i) {
    std::cout << "  [" << i << "] iou=" << diverse[i].iou
              << " cost=" << diverse[i].cost
              << " tz=" << diverse[i].pose.tz << "\n";
  }

  if (viz_) {
    viz_->LogCoarseComplete(diverse);
  }

  return diverse;
}

std::vector<ScoredCandidate> CachedPoseEstimator::CachedLocalSearch(
    const std::vector<ScoredCandidate>& coarse_best,
    const cv::Mat& input_mask, const cv::Mat& dt_input,
    const EstimationParams& params) {

  cv::Mat dt_input_f;
  dt_input.convertTo(dt_input_f, CV_32F);
  double area_input = cv::countNonZero(input_mask);
  double scale_factor = area_input > 0 ? 1.0 / std::sqrt(area_input) : 1.0;

  cv::Moments m = cv::moments(input_mask, true);
  double u_cx = m.m10 / m.m00;
  double v_cy = m.m01 / m.m00;

  double shift_u = u_cx - cache_.header.cx;
  double shift_v = v_cy - cache_.header.cy;

  std::set<uint32_t> needed_coarse_idx;
  int n_coarse = std::min(static_cast<int>(coarse_best.size()), 10);

  for (size_t ei = 0; ei < cache_.coarse_entries.size(); ++ei) {
    const auto& entry = cache_.coarse_entries[ei];
    for (int ci = 0; ci < n_coarse; ++ci) {
      double drx = std::abs(entry.rx - coarse_best[ci].pose.rx);
      double dry = std::abs(entry.ry - coarse_best[ci].pose.ry);
      double drz = std::abs(entry.rz - coarse_best[ci].pose.rz);
      if (drx < 0.01 && dry < 0.01 && drz < 0.01) {
        uint32_t dir_idx = static_cast<uint32_t>(ei / cache_.header.num_depth);
        needed_coarse_idx.insert(dir_idx);
        break;
      }
    }
  }

  std::vector<ScoredCandidate> all_local;

  int n_local = static_cast<int>(cache_.local_entries.size());
  int input_count = static_cast<int>(area_input);
  int du = static_cast<int>(std::round(shift_u));
  int dv = static_cast<int>(std::round(shift_v));

  // Phase 1: Low-res IoU for all entries
  cv::Mat input_mask_small;
  cv::resize(input_mask, input_mask_small, cv::Size(small_w_, small_h_), 0, 0, cv::INTER_NEAREST);
  int input_count_small = cv::countNonZero(input_mask_small);
  int du_small = static_cast<int>(std::round(static_cast<double>(du) / 4.0));
  int dv_small = static_cast<int>(std::round(static_cast<double>(dv) / 4.0));

  std::vector<double> lowres_iou(n_local, -1.0);

  #pragma omp parallel for schedule(dynamic)
  for (int li = 0; li < n_local; ++li) {
    if (decoded_local_areas_small_[li] == 0) continue;
    if (needed_coarse_idx.find(cache_.local_entries[li].coarse_idx) == needed_coarse_idx.end())
      continue;
    lowres_iou[li] = ShiftedIoU(decoded_local_masks_small_[li], input_mask_small,
                                 du_small, dv_small,
                                 decoded_local_areas_small_[li], input_count_small);
  }

  std::vector<int> valid_idx;
  valid_idx.reserve(n_local);
  for (int li = 0; li < n_local; ++li) {
    if (lowres_iou[li] >= 0) valid_idx.push_back(li);
  }
  std::sort(valid_idx.begin(), valid_idx.end(),
            [&](int a, int b) { return lowres_iou[a] > lowres_iou[b]; });

  std::cout << "[LocalSearch] Phase 1: " << valid_idx.size()
            << " candidates filtered by low-res IoU\n";

  // Phase 2: Full Chamfer+IoU for top N only
  int n_full = std::min(std::max(params.top_k_local * 8, 32),
                        static_cast<int>(valid_idx.size()));
  std::vector<ScoredCandidate> fullres_results(n_full);
  std::vector<bool> fullres_valid(n_full, false);

  #pragma omp parallel for schedule(dynamic)
  for (int fi = 0; fi < n_full; ++fi) {
    int li = valid_idx[fi];
    const auto& local_entry = cache_.local_entries[li];

    cv::Mat base_mask = DecodeMask(local_entry.mask_offset, local_entry.mask_size);
    cv::Mat shifted = ShiftMask(base_mask, shift_u, shift_v);

    cv::Mat diff;
    cv::absdiff(shifted, input_mask, diff);
    diff.convertTo(diff, CV_32F, 1.0 / 255.0);

    cv::Mat dt_rendered;
    cv::distanceTransform(255 - shifted, dt_rendered, cv::DIST_L2, 5);
    dt_rendered.convertTo(dt_rendered, CV_32F);

    cv::Mat dt_combined = cv::max(dt_input_f, dt_rendered);
    double chamfer = cv::sum(dt_combined.mul(diff))[0] * scale_factor;

    double area_rendered = cv::countNonZero(shifted);
    double area_ratio = area_input > 0
                            ? std::abs(area_rendered - area_input) / area_input
                            : 0.0;

    double cost = chamfer + 0.5 * area_ratio;
    double iou = ShiftedIoU(base_mask, input_mask, du, dv,
                            static_cast<int>(area_rendered), input_count);

    double tz_total = local_entry.depth;
    double tz = tz_total - local_entry.crot_z;

    Pose6D pose;
    pose.rx = local_entry.rx;
    pose.ry = local_entry.ry;
    pose.rz = local_entry.rz;
    pose.tx = (u_cx - cache_.header.cx) * tz_total / cache_.header.fx - local_entry.crot_x;
    pose.ty = (v_cy - cache_.header.cy) * tz_total / cache_.header.fy - local_entry.crot_y;
    pose.tz = tz;

    fullres_results[fi] = {pose, iou, cost};
    fullres_valid[fi] = true;
  }

  for (int fi = 0; fi < n_full; ++fi) {
    if (fullres_valid[fi]) all_local.push_back(fullres_results[fi]);
  }

  std::sort(all_local.begin(), all_local.end(),
            [](const ScoredCandidate& a, const ScoredCandidate& b) {
              return a.cost < b.cost;
            });

  if (static_cast<int>(all_local.size()) > params.top_k_local) {
    all_local.resize(params.top_k_local);
  }

  return all_local;
}

SearchResult CachedPoseEstimator::RefinePose(const ScoredCandidate& initial,
                                              const cv::Mat& input_mask,
                                              const cv::Mat& dt_input,
                                              int max_iterations,
                                              const NelderMeadOptions& nm_opts,
                                              int refine_index,
                                              maskgen::MaskGenerator* generator,
                                              const std::atomic<bool>* abort_flag) {
  std::vector<double> x = {initial.pose.tx, initial.pose.ty, initial.pose.tz,
                           initial.pose.rx, initial.pose.ry, initial.pose.rz};

  double area_input = cv::countNonZero(input_mask);
  double scale_factor = area_input > 0 ? 1.0 / std::sqrt(area_input) : 1.0;

  cv::Mat dt_input_f;
  dt_input.convertTo(dt_input_f, CV_32F);

  int viz_iteration = 0;
  RefineProfiler prof;

  std::vector<double> best_x = x;
  double best_cost_val = std::numeric_limits<double>::max();

  auto cost = [&](const std::vector<double>& params) -> double {
    if (abort_flag && abort_flag->load(std::memory_order_relaxed)) {
      throw std::runtime_error("aborted");
    }

    prof.cost_evals++;
    auto t0 = std::chrono::high_resolution_clock::now();

    Pose6D p;
    p.tx = params[0];
    p.ty = params[1];
    p.tz = params[2];
    p.rx = params[3];
    p.ry = params[4];
    p.rz = params[5];

    if (p.tz < 0.01) {
      prof.total_wall_ms += std::chrono::duration<double, std::milli>(
          std::chrono::high_resolution_clock::now() - t0).count();
      return 1e6;
    }

    maskgen::MeshPose mp;
    mp.tx = p.tx;
    mp.ty = p.ty;
    mp.tz = p.tz;
    mp.rx = p.rx;
    mp.ry = p.ry;
    mp.rz = p.rz;

    cv::Mat rendered;
    {
      ScopedTimer t(prof.generate_ms);
      rendered = generator->GeneratePose(mp);
    }

    cv::Mat diff;
    {
      ScopedTimer t(prof.absdiff_ms);
      cv::absdiff(rendered, input_mask, diff);
      diff.convertTo(diff, CV_32F, 1.0 / 255.0);
    }

    cv::Mat dt_rendered;
    {
      ScopedTimer t(prof.dt_ms);
      cv::distanceTransform(255 - rendered, dt_rendered, cv::DIST_L2, 5);
      dt_rendered.convertTo(dt_rendered, CV_32F);
    }

    double chamfer_val;
    {
      ScopedTimer t(prof.chamfer_ms);
      cv::Mat dt_combined = cv::max(dt_input_f, dt_rendered);
      chamfer_val = cv::sum(dt_combined.mul(diff))[0] * scale_factor;
    }

    double area_ratio;
    {
      ScopedTimer t(prof.area_ms);
      double area_rendered = cv::countNonZero(rendered);
      area_ratio = area_input > 0
                       ? std::abs(area_rendered - area_input) / area_input
                       : 0.0;
    }

    double c = chamfer_val + 0.5 * area_ratio;

    if (c < best_cost_val) {
      best_cost_val = c;
      best_x = params;
    }

    if (viz_ && viz_iteration % 5 == 0) {
      ScopedTimer t(prof.viz_ms);
      double cur_iou = ComputeIoU(rendered, input_mask);
      viz_->LogRefineStep(refine_index * 10000 + viz_iteration, p, rendered, c, cur_iou);
    }
    viz_iteration++;

    prof.total_wall_ms += std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();

    return c;
  };

  std::vector<double> initial_step = {
      0.02, 0.02, 0.03,
      0.3, 0.3, 0.3,
  };

  bool aborted = false;
  try {
    NelderMead(cost, x, initial_step, max_iterations, nm_opts);
  } catch (const std::runtime_error&) {
    aborted = true;
  }

  if (aborted) {
    x = best_x;
    std::cout << "[EarlyTermination] RefinePose #" << refine_index << " aborted mid-optimization\n";
  }

  prof.Print("RefinePose #" + std::to_string(refine_index));

  Pose6D refined;
  refined.tx = x[0];
  refined.ty = x[1];
  refined.tz = x[2];
  refined.rx = x[3];
  refined.ry = x[4];
  refined.rz = x[5];

  maskgen::MeshPose mp;
  mp.tx = refined.tx;
  mp.ty = refined.ty;
  mp.tz = refined.tz;
  mp.rx = refined.rx;
  mp.ry = refined.ry;
  mp.rz = refined.rz;
  cv::Mat final_rendered = generator->GeneratePose(mp);
  double final_iou = ComputeIoU(final_rendered, input_mask);

  return {refined, final_iou};
}

SearchResult CachedPoseEstimator::Estimate(const cv::Mat& input_mask,
                                            const EstimationParams& params) {
  auto t_total_start = std::chrono::high_resolution_clock::now();

  auto t0 = std::chrono::high_resolution_clock::now();
  cv::Mat binary_mask;
  if (input_mask.channels() > 1) {
    cv::cvtColor(input_mask, binary_mask, cv::COLOR_BGR2GRAY);
  } else {
    binary_mask = input_mask.clone();
  }
  cv::threshold(binary_mask, binary_mask, 127, 255, cv::THRESH_BINARY);

  if (viz_) {
    viz_->SetInputMask(binary_mask);
    viz_->SetCamera(camera_params_.fx, camera_params_.fy, camera_params_.cx,
                    camera_params_.cy, camera_params_.width, camera_params_.height);
    viz_->SetMesh(mesh_.vertices().data(), mesh_.vertices().size() / 3,
                  mesh_.indices().data(), mesh_.indices().size());
  }

  cv::Mat dt_input;
  cv::distanceTransform(255 - binary_mask, dt_input, cv::DIST_L2, 5);
  auto t1 = std::chrono::high_resolution_clock::now();
  double preprocess_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "[Timing] Preprocessing: " << preprocess_ms << " ms\n";

  auto t_coarse_start = std::chrono::high_resolution_clock::now();
  auto coarse = CachedCoarseSearch(binary_mask, dt_input, params);
  auto t_coarse_end = std::chrono::high_resolution_clock::now();
  double coarse_ms = std::chrono::duration<double, std::milli>(t_coarse_end - t_coarse_start).count();
  std::cout << "[Timing] CachedCoarseSearch: " << coarse_ms << " ms\n";

  if (coarse.empty()) {
    SearchResult result;
    result.iou = 0;
    return result;
  }

  auto t_local_start = std::chrono::high_resolution_clock::now();
  auto local = CachedLocalSearch(coarse, binary_mask, dt_input, params);
  auto t_local_end = std::chrono::high_resolution_clock::now();
  double local_ms = std::chrono::duration<double, std::milli>(t_local_end - t_local_start).count();
  std::cout << "[Timing] CachedLocalSearch: " << local_ms << " ms\n";

  std::vector<ScoredCandidate> refine_candidates;

  int n_coarse_refine = std::min(static_cast<int>(coarse.size()), params.top_k_coarse);
  for (int i = 0; i < n_coarse_refine; ++i) {
    refine_candidates.push_back(coarse[i]);
  }

  if (!local.empty()) {
    for (const auto& lc : local) {
      bool duplicate = false;
      for (const auto& rc : refine_candidates) {
        double dtx = std::abs(lc.pose.tx - rc.pose.tx);
        double dty = std::abs(lc.pose.ty - rc.pose.ty);
        double dtz = std::abs(lc.pose.tz - rc.pose.tz);
        double drx = std::abs(lc.pose.rx - rc.pose.rx);
        double dry = std::abs(lc.pose.ry - rc.pose.ry);
        double drz = std::abs(lc.pose.rz - rc.pose.rz);
        if (dtx < 0.01 && dty < 0.01 && dtz < 0.01 &&
            drx < 0.1 && dry < 0.1 && drz < 0.1) {
          duplicate = true;
          break;
        }
      }
      if (!duplicate) {
        refine_candidates.push_back(lc);
      }
    }
  }

  auto t_sort_start = std::chrono::high_resolution_clock::now();

  if (params.sort_metric == CandidateSortMetric::IoU) {
    std::sort(refine_candidates.begin(), refine_candidates.end(),
              [](const ScoredCandidate& a, const ScoredCandidate& b) {
                return a.cost < b.cost;
              });
  } else {
    cv::Mat input_dt_l1, input_dt_l2;
    bool needs_dt = (params.sort_metric == CandidateSortMetric::CentroidDT_L1 ||
                     params.sort_metric == CandidateSortMetric::CentroidDT_L2 ||
                     params.sort_metric == CandidateSortMetric::DT_IoU);
    if (needs_dt) {
      cv::distanceTransform(binary_mask, input_dt_l2, cv::DIST_L2, 5);
      cv::distanceTransform(binary_mask, input_dt_l1, cv::DIST_L1, 3);
    }

    std::vector<std::pair<double, size_t>> scored_idx(refine_candidates.size());
    for (size_t i = 0; i < refine_candidates.size(); ++i) {
      const auto& cand = refine_candidates[i];
      maskgen::MeshPose mp;
      mp.tx = cand.pose.tx;
      mp.ty = cand.pose.ty;
      mp.tz = cand.pose.tz;
      mp.rx = cand.pose.rx;
      mp.ry = cand.pose.ry;
      mp.rz = cand.pose.rz;
      cv::Mat rendered = generator_->GeneratePose(mp);
      double score;
      switch (params.sort_metric) {
        case CandidateSortMetric::HuMoments:
          score = ComputeHuDistance(rendered, binary_mask); break;
        case CandidateSortMetric::ZernikeMoments:
          score = ComputeZernikeDistance(rendered, binary_mask); break;
        case CandidateSortMetric::IoUZernikeMoments:
          score = (1.0 - cand.iou) * ComputeZernikeDistance(rendered, binary_mask); break;
        case CandidateSortMetric::CentroidIoU:
          score = 1.0 - ComputeCentroidAlignedIoU(binary_mask, rendered); break;
        case CandidateSortMetric::CentroidDT_L1:
          score = ComputeCentroidDT(binary_mask, rendered, input_dt_l1, false); break;
        case CandidateSortMetric::CentroidDT_L2:
          score = ComputeCentroidDT(binary_mask, rendered, input_dt_l2, true); break;
        case CandidateSortMetric::DT_IoU:
          score = ComputeDT_IoU(binary_mask, rendered, input_dt_l2); break;
        case CandidateSortMetric::CentralMoments:
          score = ComputeCentralMomentsDist(binary_mask, rendered); break;
        case CandidateSortMetric::ShapeContext:
          score = ComputeShapeContextDist(binary_mask, rendered); break;
        case CandidateSortMetric::ContourChamfer:
          score = ComputeContourChamferDist(binary_mask, rendered); break;
        case CandidateSortMetric::FourierDescriptor:
          score = ComputeFourierDescriptorDist(binary_mask, rendered); break;
        default:
          score = cand.iou; break;
      }
      scored_idx[i] = {score, i};
    }
    std::sort(scored_idx.begin(), scored_idx.end());
    std::vector<ScoredCandidate> sorted;
    sorted.reserve(refine_candidates.size());
    for (const auto& [sc, idx] : scored_idx) {
      sorted.push_back(refine_candidates[idx]);
    }
    refine_candidates = std::move(sorted);

    const char* mname = "Unknown";
    switch (params.sort_metric) {
      case CandidateSortMetric::HuMoments:        mname = "Hu"; break;
      case CandidateSortMetric::ZernikeMoments:   mname = "Zernike"; break;
      case CandidateSortMetric::IoUZernikeMoments: mname = "IoU*Zernike"; break;
      case CandidateSortMetric::CentroidIoU:      mname = "CentroidIoU"; break;
      case CandidateSortMetric::CentroidDT_L1:    mname = "CentroidDT_L1"; break;
      case CandidateSortMetric::CentroidDT_L2:    mname = "CentroidDT_L2"; break;
      case CandidateSortMetric::DT_IoU:           mname = "DT_IoU"; break;
      case CandidateSortMetric::CentralMoments:   mname = "CentralMoments"; break;
      case CandidateSortMetric::ShapeContext:     mname = "ShapeContext"; break;
      case CandidateSortMetric::ContourChamfer:  mname = "ContourChamfer"; break;
      case CandidateSortMetric::FourierDescriptor: mname = "FourierDescriptor"; break;
      default: break;
    }
    std::cout << "[SortMetric] " << mname << " sort scores:\n";
    for (size_t i = 0; i < refine_candidates.size(); ++i) {
      std::cout << "  [" << i << "] score=" << scored_idx[i].first
                << " iou=" << refine_candidates[i].iou << "\n";
    }
  }

  auto t_sort_end = std::chrono::high_resolution_clock::now();
  double sort_ms = std::chrono::duration<double, std::milli>(t_sort_end - t_sort_start).count();
  {
    const char* mname = "IoU";
    switch (params.sort_metric) {
      case CandidateSortMetric::HuMoments:        mname = "Hu"; break;
      case CandidateSortMetric::ZernikeMoments:   mname = "Zernike"; break;
      case CandidateSortMetric::IoUZernikeMoments: mname = "IoU*Zernike"; break;
      case CandidateSortMetric::CentroidIoU:      mname = "CentroidIoU"; break;
      case CandidateSortMetric::CentroidDT_L1:    mname = "CentroidDT_L1"; break;
      case CandidateSortMetric::CentroidDT_L2:    mname = "CentroidDT_L2"; break;
      case CandidateSortMetric::DT_IoU:           mname = "DT_IoU"; break;
      case CandidateSortMetric::CentralMoments:   mname = "CentralMoments"; break;
      case CandidateSortMetric::ShapeContext:     mname = "ShapeContext"; break;
      case CandidateSortMetric::ContourChamfer:  mname = "ContourChamfer"; break;
      case CandidateSortMetric::FourierDescriptor: mname = "FourierDescriptor"; break;
      default: break;
    }
    std::cout << "[Timing] Sort candidates (" << mname << "): " << sort_ms << " ms\n";
  }

  SearchResult best_result;
  best_result.iou = -1;

  auto t_correct_start = std::chrono::high_resolution_clock::now();
  double correct_cost_for_comparison = -1;
  double correct_iou_for_comparison = -1;
  {
    std::ifstream cpf("correct_pose.json");
    if (cpf.is_open()) {
      nlohmann::json cpj;
      cpf >> cpj;
      Pose6D correct_pose;
      correct_pose.tx = cpj.value("tx", 0.0);
      correct_pose.ty = cpj.value("ty", 0.0);
      correct_pose.tz = cpj.value("tz", 0.0);
      correct_pose.rx = cpj.value("rx", 0.0);
      correct_pose.ry = cpj.value("ry", 0.0);
      correct_pose.rz = cpj.value("rz", 0.0);
      if (cpj.contains("rx_deg")) correct_pose.rx = cpj["rx_deg"].get<double>() * M_PI / 180.0;
      if (cpj.contains("ry_deg")) correct_pose.ry = cpj["ry_deg"].get<double>() * M_PI / 180.0;
      if (cpj.contains("rz_deg")) correct_pose.rz = cpj["rz_deg"].get<double>() * M_PI / 180.0;

      maskgen::MeshPose mp;
      mp.tx = correct_pose.tx;
      mp.ty = correct_pose.ty;
      mp.tz = correct_pose.tz;
      mp.rx = correct_pose.rx;
      mp.ry = correct_pose.ry;
      mp.rz = correct_pose.rz;
      cv::Mat correct_rendered = generator_->GeneratePose(mp);
      double correct_iou = ComputeIoU(correct_rendered, binary_mask);

      double area_input = cv::countNonZero(binary_mask);
      double scale_factor = area_input > 0 ? 1.0 / std::sqrt(area_input) : 1.0;
      cv::Mat dt_input_f;
      dt_input.convertTo(dt_input_f, CV_32F);
      cv::Mat diff;
      cv::absdiff(correct_rendered, binary_mask, diff);
      diff.convertTo(diff, CV_32F, 1.0 / 255.0);
      cv::Mat dt_rendered;
      cv::distanceTransform(255 - correct_rendered, dt_rendered, cv::DIST_L2, 5);
      dt_rendered.convertTo(dt_rendered, CV_32F);
      cv::Mat dt_combined = cv::max(dt_input_f, dt_rendered);
      double chamfer = cv::sum(dt_combined.mul(diff))[0] * scale_factor;
      double area_rendered = cv::countNonZero(correct_rendered);
      double area_ratio = area_input > 0 ? std::abs(area_rendered - area_input) / area_input : 0.0;
      correct_cost_for_comparison = chamfer + 0.5 * area_ratio;
      correct_iou_for_comparison = correct_iou;

      std::cout << "\n=== Correct Pose Evaluation ===\n";
      std::cout << "Correct IoU: " << correct_iou << "\n";
      std::cout << "Correct cost: " << correct_cost_for_comparison << "\n";
    }
  }
  auto t_correct_end = std::chrono::high_resolution_clock::now();
  double correct_ms = std::chrono::duration<double, std::milli>(t_correct_end - t_correct_start).count();
  std::cout << "[Timing] CorrectPose eval: " << correct_ms << " ms\n";

  auto t_refine_start = std::chrono::high_resolution_clock::now();
  int refine_count = std::min(static_cast<int>(refine_candidates.size()), params.max_refine_candidates);

  if (!params.int_mask_dir.empty()) {
    std::filesystem::create_directories(params.int_mask_dir);
    for (int i = 0; i < refine_count; ++i) {
      const auto& cand = refine_candidates[i];
      maskgen::MeshPose mp;
      mp.tx = cand.pose.tx;
      mp.ty = cand.pose.ty;
      mp.tz = cand.pose.tz;
      mp.rx = cand.pose.rx;
      mp.ry = cand.pose.ry;
      mp.rz = cand.pose.rz;
      cv::Mat rendered = generator_->GeneratePose(mp);

      cv::Mat overlay(binary_mask.rows, binary_mask.cols, CV_8UC3, cv::Scalar(0, 0, 0));
      cv::Mat input_color;
      cv::cvtColor(binary_mask, input_color, cv::COLOR_GRAY2BGR);
      input_color.setTo(cv::Scalar(0, 0, 255), binary_mask);
      cv::Mat render_color;
      cv::cvtColor(rendered, render_color, cv::COLOR_GRAY2BGR);
      render_color.setTo(cv::Scalar(0, 255, 0), rendered);
      cv::add(input_color, render_color, overlay);

      std::ostringstream path;
      path << params.int_mask_dir << "/candidate_" << std::setw(2) << std::setfill('0') << i
           << "_iou" << std::fixed << std::setprecision(3) << cand.iou
           << ".png";
      cv::imwrite(path.str(), overlay);
      std::cout << "[IntMask] Saved: " << path.str() << "\n";
    }
  }

  if (params.refine_method == RefineMethod::NelderMead) {
    const unsigned int hw_threads = std::thread::hardware_concurrency();
    const int num_threads = std::max(1, static_cast<int>(hw_threads));
    const int actual_threads = std::min(num_threads, refine_count);

    std::vector<std::unique_ptr<maskgen::MaskGenerator>> thread_generators(actual_threads);
    for (int t = 0; t < actual_threads; ++t) {
      thread_generators[t] = std::make_unique<maskgen::MaskGenerator>(camera_params_);
      thread_generators[t]->SetMesh(mesh_);
    }

    std::vector<SearchResult> refine_results(refine_count);
    std::atomic<int> next_idx{0};
    std::atomic<bool> early_stop{false};

    auto worker = [&](int thread_id) {
      while (true) {
        if (early_stop.load(std::memory_order_relaxed)) break;
        int i = next_idx.fetch_add(1);
        if (i >= refine_count) break;
        refine_results[i] = RefinePose(refine_candidates[i], binary_mask, dt_input,
                                       params.nelder_mead_iterations, params.nm_options, i,
                                       thread_generators[thread_id].get(), &early_stop);
        std::cout << "[Timing] RefinePose NM candidate #" << i << " iou " << refine_results[i].iou << "\n";
        if (refine_results[i].iou >= params.early_termination_iou) {
          std::cout << "[EarlyTermination] IoU " << refine_results[i].iou
                    << " >= " << params.early_termination_iou
                    << ", skipping remaining candidates\n";
          early_stop.store(true, std::memory_order_relaxed);
        }
      }
    };

    std::vector<std::thread> threads;
    threads.reserve(actual_threads);
    for (int t = 0; t < actual_threads; ++t) {
      threads.emplace_back(worker, t);
    }
    for (auto& th : threads) {
      th.join();
    }

    size_t best_idx = 0;
    for (int i = 0; i < refine_count; ++i) {
      if (refine_results[i].iou > best_result.iou) {
        best_idx = i;
        best_result = refine_results[i];
      }
    }
    std::cout << "[Timing] RefinePose NM (x" << refine_count << ", " << actual_threads << " threads), best_idx=" << best_idx << "\n";
  } else {
    LMOptions lm_opts;
    lm_opts.optimizer = (params.refine_method == RefineMethod::GaussNewton)
                            ? OptimizerType::GaussNewton
                            : OptimizerType::LevenbergMarquardt;
    lm_opts.max_iterations = params.lm_max_iterations;
    lm_opts.relative_tol = params.lm_relative_tol;
    lm_opts.absolute_tol = params.lm_absolute_tol;

    for (int i = 0; i < refine_count; ++i) {
      auto t_sample_start = std::chrono::high_resolution_clock::now();
      maskgen::MeshPose init_mp;
      init_mp.tx = refine_candidates[i].pose.tx;
      init_mp.ty = refine_candidates[i].pose.ty;
      init_mp.tz = refine_candidates[i].pose.tz;
      init_mp.rx = refine_candidates[i].pose.rx;
      init_mp.ry = refine_candidates[i].pose.ry;
      init_mp.rz = refine_candidates[i].pose.rz;

      auto contour_3d = SampleContour3D(
          mesh_, init_mp, camera_params_, *generator_, params.contour_points);
      auto t_sample_end = std::chrono::high_resolution_clock::now();
      double sample_ms = std::chrono::duration<double, std::milli>(t_sample_end - t_sample_start).count();
      std::cout << "[Timing] Contour sampling #" << i << ": " << sample_ms << " ms\n";

      if (contour_3d.empty()) continue;

      SearchResult refined = RefinePoseLM(
          refine_candidates[i], binary_mask, contour_3d,
          camera_params_, mesh_, *generator_, lm_opts, i);
      if (refined.iou > best_result.iou) {
        best_result = refined;
      }
      if (refined.iou >= params.early_termination_iou) {
        std::cout << "[EarlyTermination] IoU " << refined.iou
                  << " >= " << params.early_termination_iou
                  << ", skipping remaining candidates\n";
        break;
      }
    }
    std::cout << "[Timing] RefinePose LM (x" << refine_count << ")\n";
  }
  auto t_refine_end = std::chrono::high_resolution_clock::now();
  double refine_ms = std::chrono::duration<double, std::milli>(t_refine_end - t_refine_start).count();
  std::cout << "[Timing] RefinePose (x" << refine_count << "): " << refine_ms << " ms\n";

  if (correct_cost_for_comparison >= 0) {
    maskgen::MeshPose mp;
    mp.tx = best_result.pose.tx;
    mp.ty = best_result.pose.ty;
    mp.tz = best_result.pose.tz;
    mp.rx = best_result.pose.rx;
    mp.ry = best_result.pose.ry;
    mp.rz = best_result.pose.rz;
    cv::Mat result_rendered = generator_->GeneratePose(mp);
    double area_input = cv::countNonZero(binary_mask);
    double scale_factor = area_input > 0 ? 1.0 / std::sqrt(area_input) : 1.0;
    cv::Mat dt_input_f;
    dt_input.convertTo(dt_input_f, CV_32F);
    cv::Mat diff;
    cv::absdiff(result_rendered, binary_mask, diff);
    diff.convertTo(diff, CV_32F, 1.0 / 255.0);
    cv::Mat dt_rendered;
    cv::distanceTransform(255 - result_rendered, dt_rendered, cv::DIST_L2, 5);
    dt_rendered.convertTo(dt_rendered, CV_32F);
    cv::Mat dt_combined = cv::max(dt_input_f, dt_rendered);
    double result_chamfer = cv::sum(dt_combined.mul(diff))[0] * scale_factor;
    double result_area = cv::countNonZero(result_rendered);
    double result_area_ratio = area_input > 0 ? std::abs(result_area - area_input) / area_input : 0.0;
    double result_cost = result_chamfer + 0.5 * result_area_ratio;

    std::cout << "\n=== Comparison ===\n";
    std::cout << "Result cost: " << result_cost << " (IoU=" << best_result.iou << ")\n";
    std::cout << "Correct cost: " << correct_cost_for_comparison << " (IoU=" << correct_iou_for_comparison << ")\n";
  }

  if (viz_) {
    maskgen::MeshPose mp;
    mp.tx = best_result.pose.tx;
    mp.ty = best_result.pose.ty;
    mp.tz = best_result.pose.tz;
    mp.rx = best_result.pose.rx;
    mp.ry = best_result.pose.ry;
    mp.rz = best_result.pose.rz;
    cv::Mat final_rendered = generator_->GeneratePose(mp);
    viz_->LogFinalResult(best_result.pose, final_rendered, best_result.iou);
  }

  auto t_total_end = std::chrono::high_resolution_clock::now();
  double total_ms = std::chrono::duration<double, std::milli>(t_total_end - t_total_start).count();
  std::cout << "[Timing] Estimate total: " << total_ms << " ms\n";

  return best_result;
}

}  // namespace pose_matching
