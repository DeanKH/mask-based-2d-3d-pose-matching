#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "cached_pose_estimator.h"
#include "rerun_visualizer.h"
#include "visualizer.h"

#ifdef ENABLE_PROFILER
#include <gperftools/profiler.h>
#endif

static void PrintUsage(const char* program) {
  std::cerr
      << "Usage: " << program
      << " --mask <mask.png> --camera <camera_info.json> --model <model.step> "
         "--cache <cache.bin> [options]\n\n"
      << "Required:\n"
      << "  --mask PATH          Segmentation mask image\n"
      << "  --camera PATH        Camera intrinsics JSON\n"
      << "  --model PATH         3D model file (STEP, PLY, OBJ, STL)\n"
      << "  --cache PATH         Pre-built cache file\n\n"
      << "Options:\n"
      << "  --scale FLOAT        Model scale factor (default: 1.0, 0.001 for mm->m)\n"
      << "  --image PATH         RGB image for visualization\n"
      << "  --output PATH        Output visualization path (default: result.png)\n"
      << "  --output-pose PATH   Output pose JSON path (default: pose_result.json)\n"
       << "  --max-refine INT     Max refinement iterations (default: 200)\n"
       << "  --max-candidates INT Max refine candidates (default: 5)\n"
       << "  --fatol FLOAT        NM function tolerance (default: 1e-4)\n"
       << "  --xatol FLOAT        NM parameter tolerance (default: 1e-3)\n"
       << "  --patience INT       NM stagnation patience (default: 10)\n"
       << "  --refine-method STR  Refine method: lm, gn, nelder-mead (default: lm)\n"
       << "  --contour-points INT Number of contour sample points (default: 1000)\n"
       << "  --lm-iterations INT  LM/GN max iterations (default: 100)\n"
        << "  --lm-tolerance FLOAT LM/GN convergence tolerance (default: 1e-6)\n"
       << "  --early-termination-iou FLOAT  Stop refining remaining candidates when IoU exceeds this (default: disabled)\n"
       << "  --rerun [DIR]        Enable rerun visualization, save to DIR (default: .)\n"
       << "  --profile PATH       Write gperftools CPU profile to PATH\n"
       << "  -h, --help           Show this help\n";
}

int main(int argc, char* argv[]) {
  std::vector<std::string> args(argv, argv + argc);

  std::string mask_path;
  std::string camera_path;
  std::string model_path;
  std::string cache_path;
  std::string image_path;
  std::string output_path = "result.png";
  std::string output_pose_path = "pose_result.json";
  std::string rerun_dir;
  bool rerun_enabled = false;
  std::string profile_path;
  float model_scale = 1.0f;
  pose_matching::EstimationParams est_params;

  for (size_t i = 1; i < args.size(); ++i) {
    const auto& arg = args[i];
    if (arg == "-h" || arg == "--help") {
      PrintUsage(args[0].c_str());
      return 0;
    } else if (arg == "--mask") {
      if (++i >= args.size()) return 1;
      mask_path = args[i];
    } else if (arg == "--camera") {
      if (++i >= args.size()) return 1;
      camera_path = args[i];
    } else if (arg == "--model") {
      if (++i >= args.size()) return 1;
      model_path = args[i];
    } else if (arg == "--cache") {
      if (++i >= args.size()) return 1;
      cache_path = args[i];
    } else if (arg == "--scale") {
      if (++i >= args.size()) return 1;
      model_scale = std::stof(args[i]);
    } else if (arg == "--image") {
      if (++i >= args.size()) return 1;
      image_path = args[i];
    } else if (arg == "--output") {
      if (++i >= args.size()) return 1;
      output_path = args[i];
    } else if (arg == "--output-pose") {
      if (++i >= args.size()) return 1;
      output_pose_path = args[i];
    } else if (arg == "--max-refine") {
      if (++i >= args.size()) return 1;
      est_params.nelder_mead_iterations = std::stoi(args[i]);
    } else if (arg == "--max-candidates") {
      if (++i >= args.size()) return 1;
      est_params.top_k_coarse = std::stoi(args[i]);
      est_params.max_refine_candidates = std::stoi(args[i]);
    } else if (arg == "--fatol") {
      if (++i >= args.size()) return 1;
      est_params.nm_options.fatol = std::stod(args[i]);
    } else if (arg == "--xatol") {
      if (++i >= args.size()) return 1;
      est_params.nm_options.xatol = std::stod(args[i]);
    } else if (arg == "--patience") {
      if (++i >= args.size()) return 1;
      est_params.nm_options.patience = std::stoi(args[i]);
    } else if (arg == "--refine-method") {
      if (++i >= args.size()) return 1;
      const std::string& m = args[i];
      if (m == "lm") est_params.refine_method = pose_matching::RefineMethod::LevenbergMarquardt;
      else if (m == "gn") est_params.refine_method = pose_matching::RefineMethod::GaussNewton;
      else if (m == "nelder-mead" || m == "nm") est_params.refine_method = pose_matching::RefineMethod::NelderMead;
      else { std::cerr << "Error: unknown refine method: " << m << "\n"; return 1; }
    } else if (arg == "--contour-points") {
      if (++i >= args.size()) return 1;
      est_params.contour_points = std::stoi(args[i]);
    } else if (arg == "--lm-iterations") {
      if (++i >= args.size()) return 1;
      est_params.lm_max_iterations = std::stoi(args[i]);
    } else if (arg == "--lm-tolerance") {
      if (++i >= args.size()) return 1;
      est_params.lm_relative_tol = std::stod(args[i]);
      est_params.lm_absolute_tol = est_params.lm_relative_tol;
    } else if (arg == "--early-termination-iou") {
      if (++i >= args.size()) return 1;
      est_params.early_termination_iou = std::stod(args[i]);
    } else if (arg == "--profile") {
      if (++i >= args.size()) return 1;
      profile_path = args[i];
    } else if (arg == "--rerun") {
      if (i + 1 < args.size() && args[i + 1][0] != '-') {
        rerun_dir = args[++i];
      } else {
        rerun_dir = ".";
      }
      rerun_enabled = true;
    } else if (arg[0] == '-') {
      std::cerr << "Error: unknown option: " << arg << "\n";
      return 1;
    }
  }

  if (mask_path.empty() || camera_path.empty() || model_path.empty() || cache_path.empty()) {
    std::cerr << "Error: --mask, --camera, --model, and --cache are required\n";
    PrintUsage(args[0].c_str());
    return 1;
  }

  nlohmann::json cam_json;
  {
    std::ifstream f(camera_path);
    if (!f.is_open()) {
      std::cerr << "Error: cannot open camera JSON: " << camera_path << "\n";
      return 1;
    }
    f >> cam_json;
  }

  maskgen::CameraParams camera_params;
  camera_params.width = cam_json.value("width", 640);
  camera_params.height = cam_json.value("height", 480);
  camera_params.fx = cam_json.value("fx", 500.0);
  camera_params.fy = cam_json.value("fy", 500.0);
  camera_params.cx = cam_json.value("cx", static_cast<double>(camera_params.width) / 2.0);
  camera_params.cy = cam_json.value("cy", static_cast<double>(camera_params.height) / 2.0);

  cv::Mat input_mask = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
  if (input_mask.empty()) {
    std::cerr << "Error: cannot read mask image: " << mask_path << "\n";
    return 1;
  }

  std::cout << "Mask: " << mask_path << " (" << input_mask.cols << "x" << input_mask.rows << ")\n";
  std::cout << "Camera: fx=" << camera_params.fx << " fy=" << camera_params.fy
            << " cx=" << camera_params.cx << " cy=" << camera_params.cy << "\n";
  std::cout << "Model: " << model_path << " (scale=" << model_scale << ")\n";
  std::cout << "Cache: " << cache_path << "\n";

  try {
    auto t_init_start = std::chrono::high_resolution_clock::now();
    pose_matching::CachedPoseEstimator estimator(camera_params, model_path, model_scale,
                                                  cache_path);
    auto t_init_end = std::chrono::high_resolution_clock::now();
    double init_ms = std::chrono::duration<double, std::milli>(t_init_end - t_init_start).count();
    std::cout << "[Timing] Initialization (mesh + cache): " << init_ms << " ms\n";

    std::unique_ptr<pose_matching::Visualizer> viz;
    if (rerun_enabled) {
      auto now = std::chrono::system_clock::now();
      auto time_t = std::chrono::system_clock::to_time_t(now);
      std::ostringstream oss;
      oss << rerun_dir << "/"
          << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << ".rrd";
      viz = std::make_unique<pose_matching::RerunVisualizer>(oss.str());
      estimator.SetVisualizer(viz.get());
    }

#ifdef ENABLE_PROFILER
    if (!profile_path.empty()) {
      ProfilerStart(profile_path.c_str());
    }
#endif

    auto t_est_start = std::chrono::high_resolution_clock::now();
    auto result = estimator.Estimate(input_mask, est_params);
    auto t_est_end = std::chrono::high_resolution_clock::now();
    double est_ms = std::chrono::duration<double, std::milli>(t_est_end - t_est_start).count();
    std::cout << "[Timing] Estimate (from main): " << est_ms << " ms\n";

#ifdef ENABLE_PROFILER
    if (!profile_path.empty()) {
      ProfilerStop();
    }
#endif

    std::cout << "\n=== Result ===\n";
    std::cout << "Translation: tx=" << result.pose.tx << " ty=" << result.pose.ty
              << " tz=" << result.pose.tz << "\n";
    std::cout << "Rotation (rad): rx=" << result.pose.rx << " ry=" << result.pose.ry
              << " rz=" << result.pose.rz << "\n";
    std::cout << "Rotation (deg): rx=" << result.pose.rx * 180.0 / M_PI
              << " ry=" << result.pose.ry * 180.0 / M_PI
              << " rz=" << result.pose.rz * 180.0 / M_PI << "\n";
    std::cout << "IoU: " << result.iou << "\n";

    nlohmann::json pose_json;
    pose_json["tx"] = result.pose.tx;
    pose_json["ty"] = result.pose.ty;
    pose_json["tz"] = result.pose.tz;
    pose_json["rx"] = result.pose.rx;
    pose_json["ry"] = result.pose.ry;
    pose_json["rz"] = result.pose.rz;
    pose_json["rx_deg"] = result.pose.rx * 180.0 / M_PI;
    pose_json["ry_deg"] = result.pose.ry * 180.0 / M_PI;
    pose_json["rz_deg"] = result.pose.rz * 180.0 / M_PI;
    pose_json["iou"] = result.iou;

    {
      std::ofstream f(output_pose_path);
      if (f.is_open()) {
        f << pose_json.dump(2) << "\n";
        std::cout << "Pose saved to: " << output_pose_path << "\n";
      }
    }

    if (!image_path.empty()) {
      cv::Mat rgb = cv::imread(image_path);
      if (!rgb.empty()) {
        maskgen::MeshPose mp;
        mp.tx = result.pose.tx;
        mp.ty = result.pose.ty;
        mp.tz = result.pose.tz;
        mp.rx = result.pose.rx;
        mp.ry = result.pose.ry;
        mp.rz = result.pose.rz;

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

        maskgen::MaskGenerator gen(render_params);
        maskgen::Mesh mesh;
        mesh.LoadFromFile(model_path, model_scale);
        cv::Mat result_mask = gen.Generate(mesh, mp);

        cv::Mat overlay = rgb.clone();
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(result_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(overlay, contours, -1, cv::Scalar(0, 255, 0), 2);

        cv::Mat mask_colored;
        cv::cvtColor(result_mask, mask_colored, cv::COLOR_GRAY2BGR);
        mask_colored.setTo(cv::Scalar(0, 255, 0), result_mask);
        cv::addWeighted(overlay, 1.0, mask_colored, 0.3, 0, overlay);

        if (!cv::imwrite(output_path, overlay)) {
          std::cerr << "Error: failed to write output image: " << output_path << "\n";
        } else {
          std::cout << "Visualization saved to: " << output_path << "\n";
        }
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
