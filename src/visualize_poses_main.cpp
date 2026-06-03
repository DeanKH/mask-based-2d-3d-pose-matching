#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <nlohmann/json.hpp>
#include <rerun.hpp>

#include <maskgen/camera.h>
#include <maskgen/mesh.h>

static void PrintUsage(const char* program) {
  std::cerr
      << "Usage: " << program
      << " --camera <camera_info.json>"
      << " --model <model.step --scale 0.001>"
      << " --pose <pose_result.json>"
      << " [--pose <pose_result.json> ...]"
      << " [options]\n\n"
      << "  Multiple --pose entries are supported. Each pose is rendered in\n"
      << "  a different color on the same Rerun 3D scene.\n\n"
      << "  To overlay multiple models, repeat --model --scale --pose groups.\n"
      << "  When --model appears again, a new model group starts.\n\n"
      << "Required:\n"
      << "  --camera PATH        Camera intrinsics JSON\n"
      << "  --model PATH         3D model file (STEP, PLY, OBJ, STL)\n"
      << "  --pose PATH          Pose result JSON (tx,ty,tz,rx,ry,rz)\n\n"
      << "Options:\n"
      << "  --scale FLOAT        Model scale factor (default: 1.0)\n"
      << "  --rerun PATH         Save .rrd file to PATH (instead of streaming)\n"
      << "  --rerun-grpc URL     Rerun gRPC URL (default: rerun+http://127.0.0.1:9876/proxy)\n"
      << "  -h, --help           Show this help\n";
}

struct PoseEntry {
  double tx, ty, tz;
  double rx, ry, rz;
  double iou;
  std::string source_file;
};

struct ModelGroup {
  std::string model_path;
  float scale = 1.0f;
  std::vector<PoseEntry> poses;
};

static rerun::Collection<rerun::Position3D> ConvertVertices(const float* vertices,
                                                             size_t count) {
  std::vector<rerun::Position3D> out;
  out.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    out.emplace_back(vertices[i * 3], vertices[i * 3 + 1], vertices[i * 3 + 2]);
  }
  return rerun::Collection<rerun::Position3D>::take_ownership(std::move(out));
}

static rerun::Collection<rerun::TriangleIndices> ConvertIndices(const uint32_t* indices,
                                                                 size_t count) {
  std::vector<rerun::TriangleIndices> out;
  out.reserve(count / 3);
  for (size_t i = 0; i < count; i += 3) {
    out.emplace_back(indices[i], indices[i + 1], indices[i + 2]);
  }
  return rerun::Collection<rerun::TriangleIndices>::take_ownership(std::move(out));
}

static rerun::components::Color ModelColor(size_t index) {
  static const rerun::components::Color kColors[] = {
      {200, 50, 50, 255},   {50, 200, 50, 255},   {50, 80, 220, 255},
      {220, 200, 30, 255},  {200, 50, 200, 255},  {50, 200, 200, 255},
      {255, 140, 30, 255},  {140, 50, 200, 255},  {50, 160, 80, 255},
      {200, 100, 50, 255},
  };
  constexpr size_t kNumColors = sizeof(kColors) / sizeof(kColors[0]);
  return kColors[index % kNumColors];
}

static rerun::archetypes::Transform3D MakeTransform(double tx, double ty, double tz,
                                                      double rx, double ry, double rz) {
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

  return rerun::archetypes::Transform3D(
      rerun::components::Translation3D{static_cast<float>(tx), static_cast<float>(ty),
                                       static_cast<float>(tz)},
      columns);
}

int main(int argc, char* argv[]) {
  std::vector<std::string> args(argv, argv + argc);

  std::string camera_path;
  std::vector<ModelGroup> groups;
  std::string rerun_path;
  std::string rerun_url = "rerun+http://127.0.0.1:9876/proxy";
  bool use_file = false;

  groups.emplace_back();

  for (size_t i = 1; i < args.size(); ++i) {
    const auto& arg = args[i];
    if (arg == "-h" || arg == "--help") {
      PrintUsage(args[0].c_str());
      return 0;
    } else if (arg == "--camera") {
      if (++i >= args.size()) return 1;
      camera_path = args[i];
    } else if (arg == "--model") {
      if (++i >= args.size()) return 1;
      if (!groups.back().model_path.empty() && groups.back().poses.empty()) {
      } else if (!groups.back().model_path.empty()) {
        groups.emplace_back();
      }
      groups.back().model_path = args[i];
    } else if (arg == "--scale") {
      if (++i >= args.size()) return 1;
      groups.back().scale = std::stof(args[i]);
    } else if (arg == "--pose") {
      if (++i >= args.size()) return 1;
      PoseEntry entry;
      entry.source_file = args[i];

      std::ifstream f(args[i]);
      if (!f.is_open()) {
        std::cerr << "Error: cannot open pose JSON: " << args[i] << "\n";
        return 1;
      }
      nlohmann::json j;
      f >> j;

      entry.tx = j.value("tx", 0.0);
      entry.ty = j.value("ty", 0.0);
      entry.tz = j.value("tz", 0.0);
      entry.iou = j.value("iou", 0.0);

      if (j.contains("rx_deg")) {
        double rx_deg = j.value("rx_deg", 0.0);
        double ry_deg = j.value("ry_deg", 0.0);
        double rz_deg = j.value("rz_deg", 0.0);
        entry.rx = rx_deg * M_PI / 180.0;
        entry.ry = ry_deg * M_PI / 180.0;
        entry.rz = rz_deg * M_PI / 180.0;
      } else {
        entry.rx = j.value("rx", 0.0);
        entry.ry = j.value("ry", 0.0);
        entry.rz = j.value("rz", 0.0);
      }

      groups.back().poses.push_back(std::move(entry));
    } else if (arg == "--rerun") {
      if (++i >= args.size()) return 1;
      rerun_path = args[i];
      use_file = true;
    } else if (arg == "--rerun-grpc") {
      if (++i >= args.size()) return 1;
      rerun_url = args[i];
    } else if (arg[0] == '-') {
      std::cerr << "Error: unknown option: " << arg << "\n";
      return 1;
    }
  }

  while (!groups.empty() && groups.back().poses.empty()) {
    groups.pop_back();
  }

  if (camera_path.empty() || groups.empty()) {
    std::cerr << "Error: --camera and at least one --model/--pose pair are required\n";
    PrintUsage(args[0].c_str());
    return 1;
  }

  maskgen::CameraParams camera_params;
  {
    std::ifstream f(camera_path);
    if (!f.is_open()) {
      std::cerr << "Error: cannot open camera JSON: " << camera_path << "\n";
      return 1;
    }
    nlohmann::json j;
    f >> j;
    camera_params.width = j.value("width", 640);
    camera_params.height = j.value("height", 480);
    camera_params.fx = j.value("fx", 500.0);
    camera_params.fy = j.value("fy", 500.0);
    camera_params.cx = j.value("cx", static_cast<double>(camera_params.width) / 2.0);
    camera_params.cy = j.value("cy", static_cast<double>(camera_params.height) / 2.0);
  }

  std::cout << "Camera: fx=" << camera_params.fx << " fy=" << camera_params.fy
            << " cx=" << camera_params.cx << " cy=" << camera_params.cy
            << " (" << camera_params.width << "x" << camera_params.height << ")\n";
  std::cout << "Model groups: " << groups.size() << "\n";
  size_t total_poses = 0;
  for (const auto& g : groups) total_poses += g.poses.size();
  std::cout << "Total poses: " << total_poses << "\n";

  rerun::RecordingStream rec("visualize_poses");
  if (use_file) {
    rec.save(rerun_path).exit_on_failure();
    std::cout << "Saving to: " << rerun_path << "\n";
  } else {
    auto err = rec.connect_grpc(rerun_url);
    if (err.is_err()) {
      std::cerr << "Rerun connect error: " << err.description << "\n";
      return 1;
    }
    std::cout << "Streaming to: " << rerun_url << "\n";
  }

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
    std::vector<rerun::Position3D> origin_pt = {rerun::Position3D(0.0f, 0.0f, 0.0f)};
    rec.log("scene/origin",
            rerun::Points3D(rerun::Collection<rerun::Position3D>::take_ownership(std::move(origin_pt)))
                .with_radii(rerun::Collection<float>{0.005f})
                .with_colors(rerun::Collection<rerun::components::Color>{{255, 255, 255, 255}}));
  }

  size_t pose_global_idx = 0;
  for (size_t gi = 0; gi < groups.size(); ++gi) {
    const auto& group = groups[gi];

    std::cout << "\nGroup " << gi << ": model=" << group.model_path
              << " scale=" << group.scale << " poses=" << group.poses.size() << "\n";

    maskgen::Mesh mesh;
    if (!mesh.LoadFromFile(group.model_path, group.scale)) {
      std::cerr << "Error: failed to load mesh: " << group.model_path << "\n";
      return 1;
    }

    const auto& verts = mesh.vertices();
    const auto& idxs = mesh.indices();
    auto positions = ConvertVertices(verts.data(), verts.size() / 3);
    auto triangles = ConvertIndices(idxs.data(), idxs.size());

    std::string group_prefix = "scene/group" + std::to_string(gi);

    rec.log(group_prefix + "/template",
            rerun::Mesh3D(rerun::Collection<rerun::Position3D>(positions))
                .with_triangle_indices(rerun::Collection<rerun::TriangleIndices>(triangles))
                .with_vertex_colors(rerun::Collection<rerun::components::Color>{ModelColor(gi)}));

    for (size_t pi = 0; pi < group.poses.size(); ++pi) {
      const auto& pose = group.poses[pi];
      auto color = ModelColor(pose_global_idx % 10);

      std::string entity_prefix = group_prefix + "/pose" + std::to_string(pi);

      rec.log(entity_prefix,
              MakeTransform(pose.tx, pose.ty, pose.tz, pose.rx, pose.ry, pose.rz));

      rec.log(entity_prefix + "/mesh",
              rerun::Mesh3D(rerun::Collection<rerun::Position3D>(positions))
                  .with_triangle_indices(rerun::Collection<rerun::TriangleIndices>(triangles))
                  .with_vertex_colors(rerun::Collection<rerun::components::Color>{color}));

      std::string label =
          "group" + std::to_string(gi) + "/pose" + std::to_string(pi) +
          " | IoU=" + std::to_string(pose.iou).substr(0, 6) +
          " | t=(" + std::to_string(pose.tx).substr(0, 7) + ", " +
          std::to_string(pose.ty).substr(0, 7) + ", " +
          std::to_string(pose.tz).substr(0, 7) + ")" +
          " | r=(" + std::to_string(pose.rx * 180.0 / M_PI).substr(0, 7) + ", " +
          std::to_string(pose.ry * 180.0 / M_PI).substr(0, 7) + ", " +
          std::to_string(pose.rz * 180.0 / M_PI).substr(0, 7) + ") deg" +
          " | " + pose.source_file;

      rec.log("info/" + entity_prefix, rerun::TextLog(label));

      std::cout << "  pose" << pi << ": IoU=" << pose.iou
                << " tx=" << pose.tx << " ty=" << pose.ty << " tz=" << pose.tz
                << " rx=" << pose.rx * 180.0 / M_PI
                << " ry=" << pose.ry * 180.0 / M_PI
                << " rz=" << pose.rz * 180.0 / M_PI << " deg\n";

      ++pose_global_idx;
    }
  }

  std::cout << "\nDone. " << pose_global_idx << " poses visualized.\n";
  return 0;
}
