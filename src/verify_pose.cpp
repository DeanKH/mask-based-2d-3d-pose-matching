#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <nlohmann/json.hpp>
#include <rerun.hpp>

#include <maskgen/camera.h>
#include <maskgen/mask_generator.h>
#include <maskgen/mesh.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

static void PrintUsage(const char* program) {
  std::cerr << "Usage: " << program
            << " --model <model.step> --camera <camera_info.json>"
            << " --image <color.png> --pose <correct_pose.json> [options]\n\n"
            << "Required:\n"
            << "  --model PATH    3D model file (STEP, PLY, OBJ, STL)\n"
            << "  --camera PATH   Camera intrinsics JSON\n"
            << "  --image PATH    Color image\n"
            << "  --pose PATH     Correct pose JSON (deg format)\n\n"
            << "Options:\n"
            << "  --scale FLOAT   Model scale factor (default: 0.001)\n"
            << "  --rerun URL     Rerun gRPC URL (default: rerun+http://127.0.0.1:9876/proxy)\n"
            << "  -h, --help      Show this help\n";
}

int main(int argc, char* argv[]) {
  std::vector<std::string> args(argv, argv + argc);

  std::string model_path;
  std::string camera_path;
  std::string image_path;
  std::string pose_path;
  float model_scale = 0.001f;
  std::string rerun_url = "rerun+http://127.0.0.1:9876/proxy";

  for (size_t i = 1; i < args.size(); ++i) {
    const auto& arg = args[i];
    if (arg == "-h" || arg == "--help") {
      PrintUsage(args[0].c_str());
      return 0;
    } else if (arg == "--model") {
      if (++i >= args.size()) return 1;
      model_path = args[i];
    } else if (arg == "--camera") {
      if (++i >= args.size()) return 1;
      camera_path = args[i];
    } else if (arg == "--image") {
      if (++i >= args.size()) return 1;
      image_path = args[i];
    } else if (arg == "--pose") {
      if (++i >= args.size()) return 1;
      pose_path = args[i];
    } else if (arg == "--scale") {
      if (++i >= args.size()) return 1;
      model_scale = std::stof(args[i]);
    } else if (arg == "--rerun") {
      if (++i >= args.size()) return 1;
      rerun_url = args[i];
    } else if (arg[0] == '-') {
      std::cerr << "Error: unknown option: " << arg << "\n";
      return 1;
    }
  }

  if (model_path.empty() || camera_path.empty() || image_path.empty() || pose_path.empty()) {
    std::cerr << "Error: --model, --camera, --image, and --pose are required\n";
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

  nlohmann::json pose_json;
  {
    std::ifstream f(pose_path);
    if (!f.is_open()) {
      std::cerr << "Error: cannot open pose JSON: " << pose_path << "\n";
      return 1;
    }
    f >> pose_json;
  }

  maskgen::CameraParams camera_params;
  camera_params.width = cam_json.value("width", 640);
  camera_params.height = cam_json.value("height", 480);
  camera_params.fx = cam_json.value("fx", 500.0);
  camera_params.fy = cam_json.value("fy", 500.0);
  camera_params.cx = cam_json.value("cx", static_cast<double>(camera_params.width) / 2.0);
  camera_params.cy = cam_json.value("cy", static_cast<double>(camera_params.height) / 2.0);

  double tx = pose_json.value("tx", 0.0);
  double ty = pose_json.value("ty", 0.0);
  double tz = pose_json.value("tz", 0.0);
  double rx_deg = pose_json.value("rx_deg", 0.0);
  double ry_deg = pose_json.value("ry_deg", 0.0);
  double rz_deg = pose_json.value("rz_deg", 0.0);
  double rx = rx_deg * M_PI / 180.0;
  double ry = ry_deg * M_PI / 180.0;
  double rz = rz_deg * M_PI / 180.0;

  std::cout << "Camera: fx=" << camera_params.fx << " fy=" << camera_params.fy
            << " cx=" << camera_params.cx << " cy=" << camera_params.cy << "\n";
  std::cout << "Pose (deg): rx=" << rx_deg << " ry=" << ry_deg << " rz=" << rz_deg << "\n";
  std::cout << "Translation: tx=" << tx << " ty=" << ty << " tz=" << tz << "\n";

  cv::Mat rgb = cv::imread(image_path);
  if (rgb.empty()) {
    std::cerr << "Error: cannot read image: " << image_path << "\n";
    return 1;
  }

  maskgen::Mesh mesh;
  if (!mesh.LoadFromFile(model_path, model_scale)) {
    std::cerr << "Error: failed to load mesh: " << model_path << "\n";
    return 1;
  }

  maskgen::CameraParams render_params;
  render_params.width = camera_params.width;
  render_params.height = camera_params.height;
  render_params.fx = camera_params.fx;
  render_params.fy = camera_params.fy;
  render_params.cx = camera_params.cx;
  render_params.cy = camera_params.cy;
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
  maskgen::MeshPose mp;
  mp.tx = tx;
  mp.ty = ty;
  mp.tz = tz;
  mp.rx = rx;
  mp.ry = ry;
  mp.rz = rz;
  cv::Mat rendered_mask = gen.Generate(mesh, mp);

  cv::Mat overlay = rgb.clone();
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(rendered_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  cv::drawContours(overlay, contours, -1, cv::Scalar(0, 255, 0), 2);
  cv::Mat mask_colored;
  cv::cvtColor(rendered_mask, mask_colored, cv::COLOR_GRAY2BGR);
  mask_colored.setTo(cv::Scalar(0, 255, 0), rendered_mask);
  cv::addWeighted(overlay, 1.0, mask_colored, 0.3, 0, overlay);

  std::cout << "Mask pixels: " << cv::countNonZero(rendered_mask) << "\n";

  rerun::RecordingStream rec("verify_pose");
  auto err = rec.connect_grpc(rerun_url);
  if (err.is_err()) {
    std::cerr << "Rerun connect error: " << err.description << "\n";
    return 1;
  }
  std::cout << "Connected to rerun: " << rerun_url << "\n";

  {
    float fx_f = static_cast<float>(camera_params.fx);
    float fy_f = static_cast<float>(camera_params.fy);
    float cx_f = static_cast<float>(camera_params.cx);
    float cy_f = static_cast<float>(camera_params.cy);

    std::array<float, 9> K_col_major = {
        fx_f, 0.0f, cx_f,
        0.0f, fy_f, cy_f,
        0.0f, 0.0f, 1.0f,
    };

    rec.log("scene/camera",
            rerun::Pinhole(rerun::components::PinholeProjection(K_col_major))
                .with_resolution(camera_params.width, camera_params.height));
    rec.log_static("scene/camera", rerun::ViewCoordinates::RDF);
  }

  {
    const auto& verts = mesh.vertices();
    const auto& idxs = mesh.indices();
    size_t vcount = verts.size() / 3;
    size_t icount = idxs.size();

    std::vector<rerun::Position3D> positions;
    positions.reserve(vcount);
    for (size_t i = 0; i < vcount; ++i) {
      positions.emplace_back(verts[i * 3], verts[i * 3 + 1], verts[i * 3 + 2]);
    }

    std::vector<rerun::TriangleIndices> triangles;
    triangles.reserve(icount / 3);
    for (size_t i = 0; i < icount; i += 3) {
      triangles.emplace_back(idxs[i], idxs[i + 1], idxs[i + 2]);
    }

    rec.log("scene/pose/mesh",
            rerun::Mesh3D(rerun::Collection<rerun::Position3D>::take_ownership(std::move(positions)))
                .with_triangle_indices(
                    rerun::Collection<rerun::TriangleIndices>::take_ownership(std::move(triangles))));
  }

  {
    glm::mat4 model(1.0f);
    model = glm::translate(model, glm::vec3(tx, ty, tz));
    model = glm::rotate(model, static_cast<float>(rz), glm::vec3(0, 0, 1));
    model = glm::rotate(model, static_cast<float>(ry), glm::vec3(0, 1, 0));
    model = glm::rotate(model, static_cast<float>(rx), glm::vec3(1, 0, 0));

    glm::mat3 R3(model);

    rerun::datatypes::Vec3D columns[3] = {
        {R3[0][0], R3[0][1], R3[0][2]},
        {R3[1][0], R3[1][1], R3[1][2]},
        {R3[2][0], R3[2][1], R3[2][2]},
    };

    rec.log("scene/pose",
            rerun::archetypes::Transform3D(
                rerun::components::Translation3D{static_cast<float>(tx),
                                                 static_cast<float>(ty),
                                                 static_cast<float>(tz)},
                columns));
  }

  {
    std::vector<uint8_t> img_data(rgb.total() * rgb.channels());
    std::memcpy(img_data.data(), rgb.data, img_data.size());
    rec.log("images/color",
            rerun::Image::from_rgb24(
                rerun::Collection<uint8_t>::take_ownership(std::move(img_data)),
                {static_cast<uint32_t>(rgb.cols), static_cast<uint32_t>(rgb.rows)}));
  }

  {
    cv::Mat rgb_mask;
    cv::cvtColor(rendered_mask, rgb_mask, cv::COLOR_GRAY2RGB);
    std::vector<uint8_t> mask_data(rgb_mask.total() * rgb_mask.channels());
    std::memcpy(mask_data.data(), rgb_mask.data, mask_data.size());
    rec.log("images/mask",
            rerun::Image::from_rgb24(
                rerun::Collection<uint8_t>::take_ownership(std::move(mask_data)),
                {static_cast<uint32_t>(rgb_mask.cols), static_cast<uint32_t>(rgb_mask.rows)}));
  }

  {
    std::vector<uint8_t> overlay_data(overlay.total() * overlay.channels());
    std::memcpy(overlay_data.data(), overlay.data, overlay_data.size());
    rec.log("images/overlay",
            rerun::Image::from_rgb24(
                rerun::Collection<uint8_t>::take_ownership(std::move(overlay_data)),
                {static_cast<uint32_t>(overlay.cols), static_cast<uint32_t>(overlay.rows)}));
  }

  rec.log("info/pose",
          rerun::TextLog("tx=" + std::to_string(tx) + " ty=" + std::to_string(ty) +
                         " tz=" + std::to_string(tz) +
                         " | rx=" + std::to_string(rx_deg) +
                         " ry=" + std::to_string(ry_deg) +
                         " rz=" + std::to_string(rz_deg) + " deg"));

  std::cout << "Done. Data sent to rerun.\n";
  return 0;
}
