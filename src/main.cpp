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

#include "pose_estimator.h"
#include "rerun_visualizer.h"
#include "visualizer.h"

static void PrintUsage(const char* program) {
  std::cerr
      << "Usage: " << program
      << " --mask <mask.png> --camera <camera_info.json> --model <model.step> "
         "[options]\n\n"
      << "Required:\n"
      << "  --mask PATH          Segmentation mask image\n"
      << "  --camera PATH        Camera intrinsics JSON\n"
      << "  --model PATH         3D model file (STEP, PLY, OBJ, STL)\n\n"
      << "Options:\n"
      << "  --scale FLOAT        Model scale factor (default: 1.0, 0.001 for mm->m)\n"
      << "  --image PATH         RGB image for visualization\n"
      << "  --output PATH        Output visualization path (default: result.png)\n"
      << "  --output-pose PATH   Output pose JSON path (default: pose_result.json)\n"
      << "  --depth-min FLOAT    Min search depth in meters (default: 0.05)\n"
      << "  --depth-max FLOAT    Max search depth in meters (default: 0.50)\n"
      << "  --directions INT     Number of orientation samples (default: 24)\n"
      << "  --in-plane INT       Number of in-plane rotation samples (default: 4)\n"
      << "  --depth-steps INT    Number of depth samples (default: 5)\n"
      << "  --top-k INT          Top K candidates for refinement (default: 10)\n"
      << "  --max-refine INT     Max refinement iterations (default: 200)\n"
       << "  --rerun [DIR]        Enable rerun visualization, save to DIR (default: .)\n"
      << "\nCandidate pose constraints (speed up search by restricting poses):\n"
      << "  --axis-constraint SPEC  Axis direction constraint, repeatable (e.g. \"z:0,0,-1:30\"\n"
      << "                          = model +Z within 30 deg of camera dir (0,0,-1))\n"
      << "  --tx-range MIN:MAX   Model origin tx allowed range in meters\n"
      << "  --ty-range MIN:MAX   Model origin ty allowed range in meters\n"
      << "  --nominal-position X,Y,Z  Sample tx/ty on a grid around this model origin\n"
      << "                          position (meters, camera frame) instead of deriving\n"
      << "                          them from the mask centroid\n"
      << "  --nominal-tol DX,DY,DZ   Nominal grid half-extents (default: 0.02,0.02,0.03)\n"
      << "  --nominal-grid INT   tx/ty grid samples per axis (default: 3 = -tol,0,+tol)\n"
      << "  --no-fallback        Disable unconstrained retry on low IoU\n"
      << "  --fallback-iou FLOAT Retry without constraints when final IoU < this (default: 0.70)\n"
      << "  -h, --help           Show this help\n";
}

int main(int argc, char* argv[]) {
  std::vector<std::string> args(argv, argv + argc);

  std::string mask_path;
  std::string camera_path;
  std::string model_path;
  std::string image_path;
  std::string output_path = "result.png";
  std::string output_pose_path = "pose_result.json";
  std::string rerun_dir;
  bool rerun_enabled = false;
  float model_scale = 1.0f;
  pose_matching::EstimationParams est_params;

  for (size_t i = 1; i < args.size(); ++i) {
    const auto& arg = args[i];
    if (arg == "-h" || arg == "--help") {
      PrintUsage(args[0].c_str());
      return 0;
    } else if (arg == "--mask") {
      if (++i >= args.size()) {
        std::cerr << "Error: --mask requires a path\n";
        return 1;
      }
      mask_path = args[i];
    } else if (arg == "--camera") {
      if (++i >= args.size()) {
        std::cerr << "Error: --camera requires a path\n";
        return 1;
      }
      camera_path = args[i];
    } else if (arg == "--model") {
      if (++i >= args.size()) {
        std::cerr << "Error: --model requires a path\n";
        return 1;
      }
      model_path = args[i];
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
    } else if (arg == "--depth-min") {
      if (++i >= args.size()) return 1;
      est_params.depth_min = std::stod(args[i]);
    } else if (arg == "--depth-max") {
      if (++i >= args.size()) return 1;
      est_params.depth_max = std::stod(args[i]);
    } else if (arg == "--directions") {
      if (++i >= args.size()) return 1;
      est_params.num_directions = std::stoi(args[i]);
    } else if (arg == "--in-plane") {
      if (++i >= args.size()) return 1;
      est_params.num_in_plane = std::stoi(args[i]);
    } else if (arg == "--depth-steps") {
      if (++i >= args.size()) return 1;
      est_params.num_depth = std::stoi(args[i]);
    } else if (arg == "--top-k") {
      if (++i >= args.size()) return 1;
      est_params.top_k_coarse = std::stoi(args[i]);
    } else if (arg == "--max-refine") {
      if (++i >= args.size()) return 1;
      est_params.nelder_mead_iterations = std::stoi(args[i]);
    } else if (arg == "--axis-constraint") {
      if (++i >= args.size()) {
        std::cerr << "Error: --axis-constraint requires a spec like \"z:0,0,-1:30\"\n";
        return 1;
      }
      pose_matching::AxisConstraint ac;
      if (!pose_matching::ParseAxisConstraintSpec(args[i], ac)) {
        std::cerr << "Error: invalid axis constraint spec: " << args[i]
                  << " (expected \"x|y|z:dx,dy,dz:max_deg\")\n";
        return 1;
      }
      if (est_params.axis_constraints.size() >= 4) {
        std::cerr << "Error: too many --axis-constraint entries (max 4)\n";
        return 1;
      }
      est_params.axis_constraints.push_back(ac);
      std::cout << "Axis constraint: " << pose_matching::AxisConstraintToString(ac) << "\n";
    } else if (arg == "--tx-range") {
      if (++i >= args.size()) return 1;
      double lo, hi;
      if (!pose_matching::ParseRangeSpec(args[i], lo, hi)) {
        std::cerr << "Error: invalid --tx-range: " << args[i] << " (expected MIN:MAX)\n";
        return 1;
      }
      est_params.has_tx_range = true;
      est_params.tx_min = lo;
      est_params.tx_max = hi;
    } else if (arg == "--ty-range") {
      if (++i >= args.size()) return 1;
      double lo, hi;
      if (!pose_matching::ParseRangeSpec(args[i], lo, hi)) {
        std::cerr << "Error: invalid --ty-range: " << args[i] << " (expected MIN:MAX)\n";
        return 1;
      }
      est_params.has_ty_range = true;
      est_params.ty_min = lo;
      est_params.ty_max = hi;
    } else if (arg == "--nominal-position") {
      if (++i >= args.size()) return 1;
      double v[3];
      if (!pose_matching::ParseVec3Spec(args[i], v)) {
        std::cerr << "Error: invalid --nominal-position: " << args[i]
                  << " (expected X,Y,Z in meters)\n";
        return 1;
      }
      est_params.use_nominal_position = true;
      est_params.nominal_tx = v[0];
      est_params.nominal_ty = v[1];
      est_params.nominal_tz = v[2];
    } else if (arg == "--nominal-tol") {
      if (++i >= args.size()) return 1;
      double v[3];
      if (!pose_matching::ParseVec3Spec(args[i], v) || v[0] < 0 || v[1] < 0 || v[2] < 0) {
        std::cerr << "Error: invalid --nominal-tol: " << args[i]
                  << " (expected DX,DY,DZ >= 0 in meters)\n";
        return 1;
      }
      est_params.nominal_tol_tx = v[0];
      est_params.nominal_tol_ty = v[1];
      est_params.nominal_tol_tz = v[2];
    } else if (arg == "--nominal-grid") {
      if (++i >= args.size()) return 1;
      est_params.nominal_grid = std::stoi(args[i]);
      if (est_params.nominal_grid < 1) {
        std::cerr << "Error: --nominal-grid must be >= 1\n";
        return 1;
      }
    } else if (arg == "--no-fallback") {
      est_params.fallback_unconstrained = false;
    } else if (arg == "--fallback-iou") {
      if (++i >= args.size()) return 1;
      est_params.fallback_iou = std::stod(args[i]);
      if (est_params.fallback_iou <= 0.0 || est_params.fallback_iou > 1.0) {
        std::cerr << "Error: --fallback-iou must be in (0, 1]\n";
        return 1;
      }
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

  if (mask_path.empty() || camera_path.empty() || model_path.empty()) {
    std::cerr << "Error: --mask, --camera, and --model are required\n";
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

  std::cout << "Mask: " << mask_path << " (" << input_mask.cols << "x" << input_mask.rows
            << ")\n";
  std::cout << "Camera: fx=" << camera_params.fx << " fy=" << camera_params.fy
            << " cx=" << camera_params.cx << " cy=" << camera_params.cy << "\n";
  std::cout << "Model: " << model_path << " (scale=" << model_scale << ")\n";
  std::cout << "Search: " << est_params.num_directions << " directions x "
            << est_params.num_in_plane << " in-plane x " << est_params.num_depth
            << " depths = "
            << est_params.num_directions * est_params.num_in_plane * est_params.num_depth
            << " candidates\n";
  if (!est_params.axis_constraints.empty()) {
    std::cout << "Axis constraints: " << est_params.axis_constraints.size() << "\n";
    for (const auto& ac : est_params.axis_constraints) {
      std::cout << "  - " << pose_matching::AxisConstraintToString(ac) << "\n";
    }
  }
  if (est_params.has_tx_range || est_params.has_ty_range) {
    std::cout << "Origin range: "
              << (est_params.has_tx_range
                      ? "tx[" + std::to_string(est_params.tx_min) + "," +
                            std::to_string(est_params.tx_max) + "] "
                      : "")
              << (est_params.has_ty_range
                      ? "ty[" + std::to_string(est_params.ty_min) + "," +
                            std::to_string(est_params.ty_max) + "]"
                      : "")
              << "\n";
  }
  if (est_params.use_nominal_position) {
    std::cout << "Nominal position: (" << est_params.nominal_tx << ","
              << est_params.nominal_ty << "," << est_params.nominal_tz << ") tol=("
              << est_params.nominal_tol_tx << "," << est_params.nominal_tol_ty << ","
              << est_params.nominal_tol_tz << ") grid=" << est_params.nominal_grid << "x"
              << est_params.nominal_grid << "\n";
  }
  std::cout << "Fallback: "
            << (est_params.fallback_unconstrained
                    ? "enabled (retry unconstrained if IoU < " +
                          std::to_string(est_params.fallback_iou) + ")"
                    : "disabled")
            << "\n";

  try {
    auto t_load_start = std::chrono::high_resolution_clock::now();
    pose_matching::PoseEstimator estimator(camera_params, model_path, model_scale);
    auto t_load_end = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();
    std::cout << "[Timing] Model loading: " << load_ms << " ms\n";

    std::unique_ptr<pose_matching::Visualizer> viz;
    if (rerun_enabled) {
      auto now = std::chrono::system_clock::now();
      auto time_t = std::chrono::system_clock::to_time_t(now);
      std::ostringstream oss;
      oss << rerun_dir << "/"
          << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S")
          << ".rrd";
      std::string rerun_path = oss.str();
      viz = std::make_unique<pose_matching::RerunVisualizer>(rerun_path);
      estimator.SetVisualizer(viz.get());
    }

    auto t_est_start = std::chrono::high_resolution_clock::now();
    auto result = estimator.Estimate(input_mask, est_params);
    auto t_est_end = std::chrono::high_resolution_clock::now();
    double est_ms = std::chrono::duration<double, std::milli>(t_est_end - t_est_start).count();
    std::cout << "[Timing] Estimate (from main): " << est_ms << " ms\n";

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
      auto t_viz_start = std::chrono::high_resolution_clock::now();
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
      auto t_viz_end = std::chrono::high_resolution_clock::now();
      double viz_ms = std::chrono::duration<double, std::milli>(t_viz_end - t_viz_start).count();
      std::cout << "[Timing] Visualization output: " << viz_ms << " ms\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
