#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <maskgen/camera.h>
#include <maskgen/mask_generator.h>
#include <maskgen/mesh.h>

#include "cached_pose_estimator.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

static void PrintUsage(const char* program) {
  std::cerr
      << "Usage: " << program << " [options]\n\n"
      << "Options:\n"
      << "  --dataset PATH       Dataset root containing mask.png files (default: ./test/data)\n"
      << "  --camera PATH        Camera intrinsics JSON (default: ./camera_info.json)\n"
      << "  --model-dir PATH     Directory containing <object>.step files (default: ./mask-generation/model)\n"
      << "  --cache-dir PATH     Directory containing <object>.bin cache files (default: ./cache)\n"
      << "  --output-dir PATH    Output base directory (default: ./benchmark)\n"
      << "  --scale FLOAT        Model scale factor (default: 0.001)\n"
      << "  --max-refine INT     Max refinement iterations (default: 200)\n"
      << "  --max-candidates INT Max refine candidates (default: 5)\n"
      << "  --fatol FLOAT        NM/BOBYQA function tolerance (default: 1e-4)\n"
      << "  --xatol FLOAT        NM/BOBYQA parameter tolerance (default: 1e-3)\n"
      << "  --patience INT       NM stagnation patience (default: 10)\n"
      << "  --refine-method STR  Refine method: lm, gn, nelder-mead, bobyqa (default: nelder-mead)\n"
      << "  --use-cpu            Force CPU cost evaluation instead of GPU compute shader\n"
      << "  --contour-points INT Number of contour sample points (default: 1000)\n"
      << "  --lm-iterations INT  LM/GN max iterations (default: 100)\n"
      << "  --lm-tolerance FLOAT LM/GN convergence tolerance (default: 1e-6)\n"
      << "  --early-termination-iou FLOAT  Stop refining when IoU exceeds this (default: disabled)\n"
      << "  --sort-metric STR    Sort metric: iou, centroid_iou, etc. (default: iou)\n"
      << "  --bobyqa-population INT  BOBYQA interpolation points: 0=auto(2n+1), 28=full quadratic (default: 0)\n"
      << "  --bobyqa-step-scale FLOAT  BOBYQA initial step scale factor (default: 1.0)\n"
      << "  -h, --help           Show this help\n";
}

struct BenchmarkConfig {
  fs::path dataset = "./test/data";
  fs::path camera = "./camera_info.json";
  fs::path model_dir = "./mask-generation/model";
  fs::path cache_dir = "./cache";
  fs::path output_dir = "./benchmark";
  float model_scale = 0.001f;
  pose_matching::EstimationParams est_params;
};

static double Deg2Rad(double d) { return d * M_PI / 180.0; }

static void EulerToMatrix(double rx, double ry, double rz,
                          double R[3][3]) {
  double cx = std::cos(rx), sx = std::sin(rx);
  double cy = std::cos(ry), sy = std::sin(ry);
  double cz = std::cos(rz), sz = std::sin(rz);
  double Rx[3][3] = {
      {1, 0, 0},
      {0, cx, -sx},
      {0, sx, cx},
  };
  double Ry[3][3] = {
      {cy, 0, sy},
      {0, 1, 0},
      {-sy, 0, cy},
  };
  double Rz[3][3] = {
      {cz, -sz, 0},
      {sz, cz, 0},
      {0, 0, 1},
  };
  double tmp[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      tmp[i][j] = 0;
      for (int k = 0; k < 3; ++k) tmp[i][j] += Rz[i][k] * Ry[k][j];
    }
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      R[i][j] = 0;
      for (int k = 0; k < 3; ++k) R[i][j] += tmp[i][k] * Rx[k][j];
    }
}

static double RotationErrorDeg(double rx1, double ry1, double rz1,
                               double rx2, double ry2, double rz2) {
  double R1[3][3], R2[3][3];
  EulerToMatrix(rx1, ry1, rz1, R1);
  EulerToMatrix(rx2, ry2, rz2, R2);
  double Rrel[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      Rrel[i][j] = 0;
      for (int k = 0; k < 3; ++k) Rrel[i][j] += R1[k][i] * R2[k][j];
    }
  double trace = Rrel[0][0] + Rrel[1][1] + Rrel[2][2];
  double cos_angle = (trace - 1.0) / 2.0;
  cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
  return std::acos(cos_angle) * 180.0 / M_PI;
}

static double TranslationError(double tx1, double ty1, double tz1,
                               double tx2, double ty2, double tz2) {
  double dx = tx1 - tx2, dy = ty1 - ty2, dz = tz1 - tz2;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

struct SampleResult {
  std::string sample_id;
  std::string object;
  double iou = 0;
  double time_ms = 0;
  double tx, ty, tz, rx, ry, rz;
  double gt_tx, gt_ty, gt_tz, gt_rx, gt_ry, gt_rz;
  double trans_error_m;
  double rot_error_deg;
  bool success = false;
  std::string error_msg;
};

struct Stats {
  double min, max, mean, median, p5, p95, std_dev;
};

static Stats ComputeStats(std::vector<double> values) {
  Stats s{};
  if (values.empty()) return s;
  std::sort(values.begin(), values.end());
  s.min = values.front();
  s.max = values.back();
  double sum = 0;
  for (double v : values) sum += v;
  s.mean = sum / values.size();
  size_t n = values.size();
  s.median = (n % 2 == 0) ? (values[n / 2 - 1] + values[n / 2]) / 2.0 : values[n / 2];
  s.p5 = values[std::max(0, static_cast<int>(n * 0.05) - 1)];
  s.p95 = values[std::min(static_cast<int>(n) - 1, static_cast<int>(n * 0.95))];
  double sq_sum = 0;
  for (double v : values) sq_sum += (v - s.mean) * (v - s.mean);
  s.std_dev = std::sqrt(sq_sum / n);
  return s;
}

static json StatsToJson(const Stats& s) {
  return json{
      {"min", s.min}, {"max", s.max},   {"mean", s.mean},
      {"median", s.median}, {"p5", s.p5}, {"p95", s.p95},
      {"std", s.std_dev},
  };
}

static std::vector<fs::path> FindMaskFiles(const fs::path& root) {
  std::vector<fs::path> masks;
  for (auto& entry : fs::recursive_directory_iterator(root)) {
    if (entry.is_regular_file() && entry.path().filename() == "mask.png") {
      masks.push_back(entry.path());
    }
  }
  std::sort(masks.begin(), masks.end());
  return masks;
}

struct ObjectGroup {
  std::string object_name;
  std::vector<fs::path> mask_paths;
};

static std::vector<ObjectGroup> GroupByObject(
    const std::vector<fs::path>& masks, const fs::path& dataset_root) {
  std::string root_name = dataset_root.filename().string();
  std::map<std::string, std::vector<fs::path>> grouped;
  for (const auto& mask : masks) {
    auto rel = fs::relative(mask.parent_path(), dataset_root);
    auto it = rel.begin();
    std::string object_name;
    if (rel.has_parent_path()) {
      object_name = it->string();
    } else {
      object_name = root_name;
    }
    grouped[object_name].push_back(mask);
  }
  std::vector<ObjectGroup> result;
  for (auto& [name, paths] : grouped) {
    result.push_back({name, std::move(paths)});
  }
  return result;
}

static cv::Mat RenderOverlay(const cv::Mat& rgb, const cv::Mat& result_mask) {
  cv::Mat overlay = rgb.clone();
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(result_mask, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);
  cv::drawContours(overlay, contours, -1, cv::Scalar(0, 255, 0), 2);
  cv::Mat mask_colored;
  cv::cvtColor(result_mask, mask_colored, cv::COLOR_GRAY2BGR);
  mask_colored.setTo(cv::Scalar(0, 255, 0), result_mask);
  cv::addWeighted(overlay, 1.0, mask_colored, 0.3, 0, overlay);
  return overlay;
}

static json PoseToJson(const pose_matching::Pose6D& p) {
  return json{
      {"tx", p.tx}, {"ty", p.ty}, {"tz", p.tz},
      {"rx", p.rx}, {"ry", p.ry}, {"rz", p.rz},
      {"rx_deg", p.rx * 180.0 / M_PI},
      {"ry_deg", p.ry * 180.0 / M_PI},
      {"rz_deg", p.rz * 180.0 / M_PI},
  };
}

int main(int argc, char* argv[]) {
  std::vector<std::string> args(argv, argv + argc);
  BenchmarkConfig cfg;

  for (size_t i = 1; i < args.size(); ++i) {
    const auto& arg = args[i];
    if (arg == "-h" || arg == "--help") {
      PrintUsage(args[0].c_str());
      return 0;
    } else if (arg == "--dataset") {
      cfg.dataset = args[++i];
    } else if (arg == "--camera") {
      cfg.camera = args[++i];
    } else if (arg == "--model-dir") {
      cfg.model_dir = args[++i];
    } else if (arg == "--cache-dir") {
      cfg.cache_dir = args[++i];
    } else if (arg == "--output-dir") {
      cfg.output_dir = args[++i];
    } else if (arg == "--scale") {
      cfg.model_scale = std::stof(args[++i]);
    } else if (arg == "--max-refine") {
      cfg.est_params.nelder_mead_iterations = std::stoi(args[++i]);
    } else if (arg == "--max-candidates") {
      ++i;
      cfg.est_params.top_k_coarse = std::stoi(args[i]);
      cfg.est_params.max_refine_candidates = std::stoi(args[i]);
    } else if (arg == "--fatol") {
      cfg.est_params.nm_options.fatol = std::stod(args[++i]);
      cfg.est_params.bobyqa_options.ftol_rel = std::stod(args[i]);
    } else if (arg == "--xatol") {
      cfg.est_params.nm_options.xatol = std::stod(args[++i]);
      cfg.est_params.bobyqa_options.xtol_rel = std::stod(args[i]);
    } else if (arg == "--patience") {
      cfg.est_params.nm_options.patience = std::stoi(args[++i]);
    } else if (arg == "--refine-method") {
      const std::string& m = args[++i];
      if (m == "lm") cfg.est_params.refine_method = pose_matching::RefineMethod::LevenbergMarquardt;
      else if (m == "gn") cfg.est_params.refine_method = pose_matching::RefineMethod::GaussNewton;
      else if (m == "nelder-mead" || m == "nm") cfg.est_params.refine_method = pose_matching::RefineMethod::NelderMead;
      else if (m == "bobyqa") cfg.est_params.refine_method = pose_matching::RefineMethod::BOBYQA;
      else { std::cerr << "Error: unknown refine method: " << m << "\n"; return 1; }
    } else if (arg == "--use-cpu") {
      cfg.est_params.use_cpu = true;
    } else if (arg == "--contour-points") {
      cfg.est_params.contour_points = std::stoi(args[++i]);
    } else if (arg == "--lm-iterations") {
      cfg.est_params.lm_max_iterations = std::stoi(args[++i]);
    } else if (arg == "--lm-tolerance") {
      cfg.est_params.lm_relative_tol = std::stod(args[++i]);
      cfg.est_params.lm_absolute_tol = cfg.est_params.lm_relative_tol;
    } else if (arg == "--early-termination-iou") {
      cfg.est_params.early_termination_iou = std::stod(args[++i]);
    } else if (arg == "--sort-metric") {
      const std::string& sm = args[++i];
      if (sm == "iou") cfg.est_params.sort_metric = pose_matching::CandidateSortMetric::IoU;
      else if (sm == "hu") cfg.est_params.sort_metric = pose_matching::CandidateSortMetric::HuMoments;
      else if (sm == "zernike") cfg.est_params.sort_metric = pose_matching::CandidateSortMetric::ZernikeMoments;
      else if (sm == "iou_zernike") cfg.est_params.sort_metric = pose_matching::CandidateSortMetric::IoUZernikeMoments;
      else if (sm == "centroid_iou") cfg.est_params.sort_metric = pose_matching::CandidateSortMetric::CentroidIoU;
      else if (sm == "centroid_dt_l1") cfg.est_params.sort_metric = pose_matching::CandidateSortMetric::CentroidDT_L1;
      else if (sm == "centroid_dt_l2") cfg.est_params.sort_metric = pose_matching::CandidateSortMetric::CentroidDT_L2;
      else if (sm == "dt_iou") cfg.est_params.sort_metric = pose_matching::CandidateSortMetric::DT_IoU;
      else if (sm == "central_moments") cfg.est_params.sort_metric = pose_matching::CandidateSortMetric::CentralMoments;
      else if (sm == "shape_context") cfg.est_params.sort_metric = pose_matching::CandidateSortMetric::ShapeContext;
      else if (sm == "contour_chamfer") cfg.est_params.sort_metric = pose_matching::CandidateSortMetric::ContourChamfer;
      else if (sm == "fourier") cfg.est_params.sort_metric = pose_matching::CandidateSortMetric::FourierDescriptor;
      else { std::cerr << "Error: unknown sort metric: " << sm << "\n"; return 1; }
    } else if (arg == "--bobyqa-population") {
      cfg.est_params.bobyqa_options.population = std::stoi(args[++i]);
    } else if (arg == "--bobyqa-step-scale") {
      cfg.est_params.bobyqa_options.initial_step_scale = std::stod(args[++i]);
    } else {
      std::cerr << "Error: unknown option: " << arg << "\n";
      PrintUsage(args[0].c_str());
      return 1;
    }
  }

  if (!fs::exists(cfg.dataset)) {
    std::cerr << "Error: dataset path does not exist: " << cfg.dataset << "\n";
    return 1;
  }
  if (!fs::exists(cfg.camera)) {
    std::cerr << "Error: camera JSON does not exist: " << cfg.camera << "\n";
    return 1;
  }

  json cam_json;
  {
    std::ifstream f(cfg.camera);
    f >> cam_json;
  }
  maskgen::CameraParams camera_params;
  camera_params.width = cam_json.value("width", 640);
  camera_params.height = cam_json.value("height", 480);
  camera_params.fx = cam_json.value("fx", 500.0);
  camera_params.fy = cam_json.value("fy", 500.0);
  camera_params.cx = cam_json.value("cx", static_cast<double>(camera_params.width) / 2.0);
  camera_params.cy = cam_json.value("cy", static_cast<double>(camera_params.height) / 2.0);

  auto mask_files = FindMaskFiles(cfg.dataset);
  if (mask_files.empty()) {
    std::cerr << "Error: no mask.png files found under " << cfg.dataset << "\n";
    return 1;
  }
  std::cout << "Found " << mask_files.size() << " mask files" << std::endl;

  auto object_groups = GroupByObject(mask_files, cfg.dataset);
  std::cout << "Objects: ";
  for (const auto& og : object_groups) std::cout << og.object_name << " ";
  std::cout << "\n";

  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  std::ostringstream ts_ss;
  ts_ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
  std::string timestamp = ts_ss.str();

  std::string refine_tag = "nelder-mead";
  switch (cfg.est_params.refine_method) {
    case pose_matching::RefineMethod::LevenbergMarquardt: refine_tag = "lm"; break;
    case pose_matching::RefineMethod::GaussNewton: refine_tag = "gn"; break;
    case pose_matching::RefineMethod::BOBYQA: refine_tag = "bobyqa"; break;
    default: break;
  }

  std::string sort_metric_tag = "iou";
  switch (cfg.est_params.sort_metric) {
    case pose_matching::CandidateSortMetric::IoU: sort_metric_tag = "iou"; break;
    case pose_matching::CandidateSortMetric::HuMoments: sort_metric_tag = "hu"; break;
    case pose_matching::CandidateSortMetric::ZernikeMoments: sort_metric_tag = "zernike"; break;
    case pose_matching::CandidateSortMetric::IoUZernikeMoments: sort_metric_tag = "iou_zernike"; break;
    case pose_matching::CandidateSortMetric::CentroidIoU: sort_metric_tag = "centroid_iou"; break;
    case pose_matching::CandidateSortMetric::CentroidDT_L1: sort_metric_tag = "centroid_dt_l1"; break;
    case pose_matching::CandidateSortMetric::CentroidDT_L2: sort_metric_tag = "centroid_dt_l2"; break;
    case pose_matching::CandidateSortMetric::DT_IoU: sort_metric_tag = "dt_iou"; break;
    case pose_matching::CandidateSortMetric::CentralMoments: sort_metric_tag = "central_moments"; break;
    case pose_matching::CandidateSortMetric::ShapeContext: sort_metric_tag = "shape_context"; break;
    case pose_matching::CandidateSortMetric::ContourChamfer: sort_metric_tag = "contour_chamfer"; break;
    case pose_matching::CandidateSortMetric::FourierDescriptor: sort_metric_tag = "fourier"; break;
  }

  fs::path out_folder = cfg.output_dir / (timestamp + "_" + refine_tag);
  fs::create_directories(out_folder);
  std::cout << "Output: " << out_folder << "\n";

  std::vector<SampleResult> all_results;

  maskgen::CameraParams render_params = camera_params;
  render_params.eye_x = 0;
  render_params.eye_y = 0;
  render_params.eye_z = 0;
  render_params.target_x = 0;
  render_params.target_y = 0;
  render_params.target_z = 1;
  render_params.up_x = 0;
  render_params.up_y = -1;
  render_params.up_z = 0;

  for (const auto& og : object_groups) {
    fs::path model_path = cfg.model_dir / (og.object_name + ".step");
    fs::path cache_path = cfg.cache_dir / (og.object_name + ".bin");

    if (!fs::exists(model_path)) {
      std::cerr << "[WARN] Model not found for object '" << og.object_name
                << "': " << model_path << " — skipping\n";
      continue;
    }
    if (!fs::exists(cache_path)) {
      std::cerr << "[WARN] Cache not found for object '" << og.object_name
                << "': " << cache_path << " — skipping\n";
      continue;
    }

    std::cout << "\n=== Object: " << og.object_name << " ("
              << og.mask_paths.size() << " samples) ===\n";

    pose_matching::CachedPoseEstimator estimator(camera_params,
                                                  model_path.string(),
                                                  cfg.model_scale,
                                                  cache_path.string());

    maskgen::MaskGenerator overlay_gen(render_params);
    maskgen::Mesh mesh;
    mesh.LoadFromFile(model_path.string(), cfg.model_scale);

    for (const auto& mask_path : og.mask_paths) {
      fs::path sample_dir = mask_path.parent_path();
      auto rel = fs::relative(sample_dir, cfg.dataset);
      fs::path rel_full =
          rel.has_parent_path() ? rel : fs::path(og.object_name) / rel;
      std::string sample_id = rel_full.string();

      fs::path rgb_path = sample_dir / "rgb.png";
      fs::path pose_gt_path = sample_dir / "pose.json";

      SampleResult sr;
      sr.sample_id = sample_id;
      sr.object = og.object_name;

      cv::Mat input_mask = cv::imread(mask_path.string(), cv::IMREAD_GRAYSCALE);
      if (input_mask.empty()) {
        sr.error_msg = "Failed to read mask";
        std::cerr << "[ERROR] " << sample_id << ": cannot read mask\n";
        all_results.push_back(sr);
        continue;
      }

      json gt_pose;
      if (fs::exists(pose_gt_path)) {
        std::ifstream f(pose_gt_path);
        f >> gt_pose;
        sr.gt_tx = gt_pose.value("tx", 0.0);
        sr.gt_ty = gt_pose.value("ty", 0.0);
        sr.gt_tz = gt_pose.value("tz", 0.0);
        sr.gt_rx = gt_pose.value("rx", 0.0);
        sr.gt_ry = gt_pose.value("ry", 0.0);
        sr.gt_rz = gt_pose.value("rz", 0.0);
      }

      std::cout << "  [" << sample_id << "] ";

      try {
        auto t_start = std::chrono::high_resolution_clock::now();
        auto result = estimator.Estimate(input_mask, cfg.est_params);
        auto t_end = std::chrono::high_resolution_clock::now();
        sr.time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        sr.iou = result.iou;
        sr.tx = result.pose.tx;
        sr.ty = result.pose.ty;
        sr.tz = result.pose.tz;
        sr.rx = result.pose.rx;
        sr.ry = result.pose.ry;
        sr.rz = result.pose.rz;
        sr.trans_error_m = TranslationError(sr.tx, sr.ty, sr.tz,
                                            sr.gt_tx, sr.gt_ty, sr.gt_tz);
        sr.rot_error_deg = RotationErrorDeg(sr.rx, sr.ry, sr.rz,
                                            sr.gt_rx, sr.gt_ry, sr.gt_rz);
        sr.success = true;

        std::cout << "IoU=" << std::fixed << std::setprecision(4) << sr.iou
                  << "  time=" << std::setprecision(1) << sr.time_ms << "ms"
                  << "  t_err=" << std::setprecision(4) << sr.trans_error_m << "m"
                  << "  r_err=" << sr.rot_error_deg << "deg\n";

        if (fs::exists(rgb_path)) {
          cv::Mat rgb = cv::imread(rgb_path.string());
          if (!rgb.empty()) {
            maskgen::MeshPose mp;
            mp.tx = sr.tx; mp.ty = sr.ty; mp.tz = sr.tz;
            mp.rx = sr.rx; mp.ry = sr.ry; mp.rz = sr.rz;
            cv::Mat result_mask = overlay_gen.Generate(mesh, mp);
            cv::Mat overlay = RenderOverlay(rgb, result_mask);

            fs::path overlay_path = out_folder / rel_full / "final_overlay.png";
            fs::create_directories(overlay_path.parent_path());
            cv::imwrite(overlay_path.string(), overlay);
          }
        }
      } catch (const std::exception& e) {
        sr.error_msg = e.what();
        sr.success = false;
        std::cerr << "ERROR: " << e.what() << "\n";
      }

      all_results.push_back(sr);
    }
  }

  std::vector<double> ious, times, trans_errors, rot_errors;
  for (const auto& r : all_results) {
    if (r.success) {
      ious.push_back(r.iou);
      times.push_back(r.time_ms);
      trans_errors.push_back(r.trans_error_m);
      rot_errors.push_back(r.rot_error_deg);
    }
  }

  json per_object_summary;
  std::map<std::string, std::vector<double>> obj_ious, obj_times;
  for (const auto& r : all_results) {
    if (r.success) {
      obj_ious[r.object].push_back(r.iou);
      obj_times[r.object].push_back(r.time_ms);
    }
  }
  for (const auto& [obj, vals] : obj_ious) {
    per_object_summary[obj] = {
        {"iou", StatsToJson(ComputeStats(obj_ious[obj]))},
        {"time_ms", StatsToJson(ComputeStats(obj_times[obj]))},
        {"count", vals.size()},
    };
  }

  json results_json;
  results_json["timestamp"] = timestamp;
  results_json["refine_method"] = refine_tag;
  results_json["parameters"] = {
      {"dataset", cfg.dataset.string()},
      {"max_candidates", cfg.est_params.max_refine_candidates},
      {"max_refine", cfg.est_params.nelder_mead_iterations},
      {"fatol", cfg.est_params.nm_options.fatol},
      {"xatol", cfg.est_params.nm_options.xatol},
      {"contour_points", cfg.est_params.contour_points},
      {"use_cpu", cfg.est_params.use_cpu},
      {"early_termination_iou", cfg.est_params.early_termination_iou},
      {"sort_metric", sort_metric_tag},
      {"bobyqa_population", cfg.est_params.bobyqa_options.population},
      {"bobyqa_step_scale", cfg.est_params.bobyqa_options.initial_step_scale},
  };
  results_json["summary"] = {
      {"total_samples", all_results.size()},
      {"successful", ious.size()},
      {"failed", all_results.size() - ious.size()},
      {"iou", StatsToJson(ComputeStats(ious))},
      {"time_ms", StatsToJson(ComputeStats(times))},
      {"translation_error_m", StatsToJson(ComputeStats(trans_errors))},
      {"rotation_error_deg", StatsToJson(ComputeStats(rot_errors))},
      {"per_object", per_object_summary},
  };

  json results_map = json::object();
  for (const auto& r : all_results) {
    json rj;
    rj["object"] = r.object;
    rj["success"] = r.success;
    if (r.success) {
      rj["pose"] = {
          {"tx", r.tx}, {"ty", r.ty}, {"tz", r.tz},
          {"rx", r.rx}, {"ry", r.ry}, {"rz", r.rz},
      };
      rj["iou"] = r.iou;
      rj["time_ms"] = r.time_ms;
      rj["ground_truth"] = {
          {"tx", r.gt_tx}, {"ty", r.gt_ty}, {"tz", r.gt_tz},
          {"rx", r.gt_rx}, {"ry", r.gt_ry}, {"rz", r.gt_rz},
      };
      rj["translation_error_m"] = r.trans_error_m;
      rj["rotation_error_deg"] = r.rot_error_deg;
    } else {
      rj["error"] = r.error_msg;
    }
    results_map[r.sample_id] = rj;
  }
  results_json["results"] = results_map;

  fs::path result_path = out_folder / "result.json";
  std::ofstream ofs(result_path);
  ofs << results_json.dump(2) << "\n";
  ofs.close();

  std::cout << "\n=== Benchmark Summary ===\n";
  if (!ious.empty()) {
    auto iou_stats = ComputeStats(ious);
    auto time_stats = ComputeStats(times);
    std::cout << "IoU: mean=" << std::fixed << std::setprecision(4) << iou_stats.mean
              << " median=" << iou_stats.median
              << " min=" << iou_stats.min
              << " max=" << iou_stats.max << "\n";
    std::cout << "Time: mean=" << std::setprecision(1) << time_stats.mean << "ms"
              << " median=" << time_stats.median << "ms"
              << " total=" << std::accumulate(times.begin(), times.end(), 0.0) << "ms\n";
  }
  std::cout << "Results saved to: " << result_path << "\n";

  return 0;
}
