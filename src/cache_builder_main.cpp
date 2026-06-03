#include <atomic>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <condition_variable>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <maskgen/camera.h>
#include <maskgen/mask_generator.h>
#include <maskgen/mesh.h>

#include <opencv2/imgproc.hpp>

#include "cache_format.h"
#include "pose_estimator.h"

using namespace pose_matching;

static glm::mat4 RotationFromDirection(const glm::vec3& target_dir, float in_plane_angle,
                                        int principal_axis) {
  glm::vec3 source_axis(0, 0, 0);
  source_axis[principal_axis] = 1.0f;

  float dot = glm::dot(source_axis, target_dir);
  glm::mat4 R_align;

  if (dot > 0.9999f) {
    R_align = glm::mat4(1.0f);
  } else if (dot < -0.9999f) {
    glm::vec3 perp(0, 1, 0);
    if (std::abs(glm::dot(source_axis, perp)) > 0.9f) {
      perp = glm::vec3(1, 0, 0);
    }
    glm::vec3 rot_axis = glm::normalize(glm::cross(source_axis, perp));
    R_align = glm::rotate(glm::mat4(1.0f), glm::pi<float>(), rot_axis);
  } else {
    glm::vec3 axis = glm::normalize(glm::cross(source_axis, target_dir));
    float angle = glm::acos(glm::clamp(dot, -1.0f, 1.0f));
    R_align = glm::rotate(glm::mat4(1.0f), angle, axis);
  }

  glm::mat4 R_inplane = glm::rotate(glm::mat4(1.0f), in_plane_angle, source_axis);
  return R_align * R_inplane;
}

static void ExtractEulerZYX(const glm::mat4& R, double& rx, double& ry, double& rz) {
  glm::mat3 R3(R);
  float ry_f = glm::asin(glm::clamp(-R3[0][2], -1.0f, 1.0f));
  float rx_f = std::atan2(R3[1][2], R3[2][2]);
  float rz_f = std::atan2(R3[0][1], R3[0][0]);
  rx = static_cast<double>(rx_f);
  ry = static_cast<double>(ry_f);
  rz = static_cast<double>(rz_f);
}

static std::vector<glm::vec3> FibonacciSphere(int n) {
  std::vector<glm::vec3> points;
  points.reserve(n);
  const float golden = glm::pi<float>() * (3.0f - glm::sqrt(5.0f));
  for (int i = 0; i < n; ++i) {
    float theta = golden * static_cast<float>(i);
    float phi = glm::acos(1.0f - 2.0f * (static_cast<float>(i) + 0.5f) / static_cast<float>(n));
    points.emplace_back(glm::sin(phi) * glm::cos(theta),
                        glm::sin(phi) * glm::sin(theta),
                        glm::cos(phi));
  }
  return points;
}

struct MeshInfo {
  float centroid[3];
  float extent[3];
  int principal_axis;
};

static MeshInfo ComputeMeshInfo(const maskgen::Mesh& mesh) {
  MeshInfo info{};
  const auto& verts = mesh.vertices();
  const size_t n = verts.size() / 3;

  float min_p[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
  float max_p[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};

  for (size_t i = 0; i < n; ++i) {
    for (int d = 0; d < 3; ++d) {
      float v = verts[i * 3 + d];
      min_p[d] = std::min(min_p[d], v);
      max_p[d] = std::max(max_p[d], v);
      info.centroid[d] += v;
    }
  }
  float max_extent = 0;
  for (int d = 0; d < 3; ++d) {
    info.centroid[d] /= static_cast<float>(n);
    info.extent[d] = max_p[d] - min_p[d];
    if (info.extent[d] > max_extent) {
      max_extent = info.extent[d];
      info.principal_axis = d;
    }
  }
  return info;
}

struct CoarseWorkItem {
  double rx, ry, rz;
  float crot_x, crot_y, crot_z;
  double depth;
};

struct LocalWorkItem {
  uint32_t coarse_idx;
  double rx, ry, rz;
  float crot_x, crot_y, crot_z;
  double depth;
};

struct RenderedMask {
  std::vector<uint8_t> compressed;
  double rx, ry, rz;
  float crot_x, crot_y, crot_z;
  double depth;
  uint32_t coarse_idx;
  bool valid;
};

static void PrintUsage(const char* program) {
  std::cerr
      << "Usage: " << program
      << " --camera <camera_info.json> --model <model.step> --output <cache.bin> "
         "[options]\n\n"
      << "Required:\n"
      << "  --camera PATH   Camera intrinsics JSON\n"
      << "  --model PATH    3D model file (STEP, PLY, OBJ, STL)\n"
      << "  --output PATH   Output cache file path\n\n"
      << "Options:\n"
      << "  --scale FLOAT   Model scale factor (default: 1.0, 0.001 for mm->m)\n"
      << "  --depth-min FLOAT   Min search depth (default: 0.05)\n"
      << "  --depth-max FLOAT   Max search depth (default: 0.50)\n"
      << "  --directions INT    Number of orientation samples (default: 48)\n"
      << "  --in-plane INT      Number of in-plane rotation samples (default: 4)\n"
      << "  --depth-steps INT   Number of depth samples (default: 5)\n"
      << "  --local-directions INT  Local search directions (default: 6)\n"
      << "  --local-in-plane INT    Local in-plane samples (default: 6)\n"
      << "  --local-depth INT       Local depth samples (default: 3)\n"
      << "  --local-cone-angle FLOAT  Local cone half angle deg (default: 30.0)\n"
      << "  --threads INT       Number of render threads (default: 4)\n"
      << "  -h, --help          Show this help\n";
}

int main(int argc, char* argv[]) {
  std::vector<std::string> args(argv, argv + argc);

  std::string camera_path;
  std::string model_path;
  std::string output_path;
  float model_scale = 1.0f;
  int num_threads = 4;

  EstimationParams params;

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
      model_path = args[i];
    } else if (arg == "--output") {
      if (++i >= args.size()) return 1;
      output_path = args[i];
    } else if (arg == "--scale") {
      if (++i >= args.size()) return 1;
      model_scale = std::stof(args[i]);
    } else if (arg == "--depth-min") {
      if (++i >= args.size()) return 1;
      params.depth_min = std::stod(args[i]);
    } else if (arg == "--depth-max") {
      if (++i >= args.size()) return 1;
      params.depth_max = std::stod(args[i]);
    } else if (arg == "--directions") {
      if (++i >= args.size()) return 1;
      params.num_directions = std::stoi(args[i]);
    } else if (arg == "--in-plane") {
      if (++i >= args.size()) return 1;
      params.num_in_plane = std::stoi(args[i]);
    } else if (arg == "--depth-steps") {
      if (++i >= args.size()) return 1;
      params.num_depth = std::stoi(args[i]);
    } else if (arg == "--local-directions") {
      if (++i >= args.size()) return 1;
      params.local_directions = std::stoi(args[i]);
    } else if (arg == "--local-in-plane") {
      if (++i >= args.size()) return 1;
      params.local_in_plane = std::stoi(args[i]);
    } else if (arg == "--local-depth") {
      if (++i >= args.size()) return 1;
      params.local_depth = std::stoi(args[i]);
    } else if (arg == "--local-cone-angle") {
      if (++i >= args.size()) return 1;
      params.local_cone_half_angle_deg = std::stod(args[i]);
    } else if (arg == "--threads") {
      if (++i >= args.size()) return 1;
      num_threads = std::stoi(args[i]);
    } else if (arg[0] == '-') {
      std::cerr << "Error: unknown option: " << arg << "\n";
      return 1;
    }
  }

  if (camera_path.empty() || model_path.empty() || output_path.empty()) {
    std::cerr << "Error: --camera, --model, and --output are required\n";
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

  camera_params.eye_x = 0;
  camera_params.eye_y = 0;
  camera_params.eye_z = 0;
  camera_params.target_x = 0;
  camera_params.target_y = 0;
  camera_params.target_z = 1;
  camera_params.up_x = 0;
  camera_params.up_y = -1;
  camera_params.up_z = 0;

  std::cout << "Camera: fx=" << camera_params.fx << " fy=" << camera_params.fy
            << " cx=" << camera_params.cx << " cy=" << camera_params.cy
            << " (" << camera_params.width << "x" << camera_params.height << ")\n";
  std::cout << "Threads: " << num_threads << "\n";

  auto t_load_start = std::chrono::high_resolution_clock::now();
  maskgen::Mesh mesh;
  if (!mesh.LoadFromFile(model_path, model_scale)) {
    std::cerr << "Error: failed to load mesh: " << model_path << "\n";
    return 1;
  }
  auto t_load_end = std::chrono::high_resolution_clock::now();
  double load_ms = std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();
  std::cout << "[Timing] Mesh loading: " << load_ms << " ms\n";

  MeshInfo mesh_info = ComputeMeshInfo(mesh);
  std::cout << "Mesh centroid: (" << mesh_info.centroid[0] << ", " << mesh_info.centroid[1]
            << ", " << mesh_info.centroid[2] << ")\n";
  std::cout << "Mesh extents: (" << mesh_info.extent[0] << ", " << mesh_info.extent[1]
            << ", " << mesh_info.extent[2] << ")\n";
  std::cout << "Principal axis: " << mesh_info.principal_axis << "\n";

  auto directions = FibonacciSphere(params.num_directions);

  std::vector<double> depths;
  double depth_step = (params.depth_max - params.depth_min) / std::max(params.num_depth - 1, 1);
  for (int i = 0; i < params.num_depth; ++i) {
    depths.push_back(params.depth_min + i * depth_step);
  }

  // === Build coarse work items ===
  std::vector<CoarseWorkItem> coarse_work;
  coarse_work.reserve(static_cast<size_t>(params.num_directions) * params.num_in_plane * depths.size());

  for (const auto& dir : directions) {
    for (int ip = 0; ip < params.num_in_plane; ++ip) {
      float in_plane = 2.0f * glm::pi<float>() * static_cast<float>(ip) /
                       static_cast<float>(params.num_in_plane);

      glm::mat4 R = RotationFromDirection(dir, in_plane, mesh_info.principal_axis);
      double rx, ry, rz;
      ExtractEulerZYX(R, rx, ry, rz);
      glm::vec4 c_rot = R * glm::vec4(mesh_info.centroid[0], mesh_info.centroid[1],
                                       mesh_info.centroid[2], 1.0f);

      for (double depth : depths) {
        CoarseWorkItem item;
        item.rx = rx;
        item.ry = ry;
        item.rz = rz;
        item.crot_x = c_rot.x;
        item.crot_y = c_rot.y;
        item.crot_z = c_rot.z;
        item.depth = depth;
        coarse_work.push_back(item);
      }
    }
  }

  // === Build local work items ===
  struct CoarseDirInfo {
    double rx, ry, rz;
    float crot_x, crot_y, crot_z;
  };

  std::vector<CoarseDirInfo> coarse_dir_infos;
  coarse_dir_infos.reserve(directions.size() * params.num_in_plane);

  for (const auto& dir : directions) {
    for (int ip = 0; ip < params.num_in_plane; ++ip) {
      float in_plane = 2.0f * glm::pi<float>() * static_cast<float>(ip) /
                       static_cast<float>(params.num_in_plane);
      glm::mat4 R = RotationFromDirection(dir, in_plane, mesh_info.principal_axis);
      double rx, ry, rz;
      ExtractEulerZYX(R, rx, ry, rz);
      glm::vec4 c_rot = R * glm::vec4(mesh_info.centroid[0], mesh_info.centroid[1],
                                       mesh_info.centroid[2], 1.0f);
      coarse_dir_infos.push_back({rx, ry, rz, c_rot.x, c_rot.y, c_rot.z});
    }
  }

  std::vector<LocalWorkItem> local_work;

  int n_local_dirs = params.local_directions;
  int n_local_ip = params.local_in_plane;
  float half_angle = static_cast<float>(params.local_cone_half_angle_deg * M_PI / 180.0);
  int num_coarse_dirs = static_cast<int>(coarse_dir_infos.size());
  double base_tz_total = (params.depth_min + params.depth_max) / 2.0;

  for (int ci = 0; ci < num_coarse_dirs; ++ci) {
    const auto& dir_info = coarse_dir_infos[ci];

    glm::mat4 R_cand = glm::mat4(1.0f);
    R_cand = glm::rotate(R_cand, static_cast<float>(dir_info.rz), glm::vec3(0, 0, 1));
    R_cand = glm::rotate(R_cand, static_cast<float>(dir_info.ry), glm::vec3(0, 1, 0));
    R_cand = glm::rotate(R_cand, static_cast<float>(dir_info.rx), glm::vec3(1, 0, 0));

    glm::vec3 source_axis(0, 0, 0);
    source_axis[mesh_info.principal_axis] = 1.0f;
    glm::vec3 cand_dir = glm::normalize(glm::vec3(R_cand * glm::vec4(source_axis, 0.0f)));

    glm::vec3 perp1;
    if (std::abs(glm::dot(cand_dir, glm::vec3(0, 1, 0))) < 0.9f) {
      perp1 = glm::normalize(glm::cross(cand_dir, glm::vec3(0, 1, 0)));
    } else {
      perp1 = glm::normalize(glm::cross(cand_dir, glm::vec3(1, 0, 0)));
    }
    glm::vec3 perp2 = glm::normalize(glm::cross(cand_dir, perp1));

    for (int di = 0; di < n_local_dirs; ++di) {
      float theta1 = 2.0f * glm::pi<float>() * static_cast<float>(di) /
                     static_cast<float>(n_local_dirs);
      float cone_angles[] = {
          half_angle * 0.33f,
          half_angle * 0.67f,
          half_angle * 1.0f,
      };
      float theta2 = cone_angles[di % 3];

      glm::vec3 local_dir = glm::normalize(
          cand_dir * std::cos(theta2) +
          perp1 * std::sin(theta2) * std::cos(theta1) +
          perp2 * std::sin(theta2) * std::sin(theta1));

      for (int flip = 0; flip < 2; ++flip) {
        glm::vec3 final_dir = (flip == 0) ? local_dir : -local_dir;

        for (int ip = 0; ip < n_local_ip; ++ip) {
          float in_plane = 2.0f * glm::pi<float>() * static_cast<float>(ip) /
                           static_cast<float>(n_local_ip);

          glm::mat4 R = RotationFromDirection(final_dir, in_plane, mesh_info.principal_axis);
          double rx, ry, rz;
          ExtractEulerZYX(R, rx, ry, rz);
          glm::vec4 c_rot = R * glm::vec4(mesh_info.centroid[0], mesh_info.centroid[1],
                                           mesh_info.centroid[2], 1.0f);

          for (int d = 0; d < params.local_depth; ++d) {
            double depth_frac = params.local_depth > 1
                ? static_cast<double>(d) / (params.local_depth - 1) : 0.5;
            double depth_range = 0.06;
            double depth_offset = (depth_frac - 0.5) * depth_range;
            double tz_total = base_tz_total + depth_offset;

            if (tz_total < params.depth_min || tz_total > params.depth_max) continue;

            double tz = tz_total - c_rot.z;
            if (tz < 0.01) continue;

            LocalWorkItem item;
            item.coarse_idx = static_cast<uint32_t>(ci);
            item.rx = rx;
            item.ry = ry;
            item.rz = rz;
            item.crot_x = c_rot.x;
            item.crot_y = c_rot.y;
            item.crot_z = c_rot.z;
            item.depth = tz_total;
            local_work.push_back(item);
          }
        }
      }
    }
  }

  std::cout << "Coarse work items: " << coarse_work.size() << "\n";
  std::cout << "Local work items: " << local_work.size() << "\n";

  auto render_one = [&](maskgen::MaskGenerator& gen,
                        double rx, double ry, double rz,
                        float crot_x, float crot_y, float crot_z,
                        double depth) -> std::vector<uint8_t> {
    double tz_total = depth;
    double tz = tz_total - crot_z;
    if (tz < 0.01) return {};

    maskgen::MeshPose mp;
    mp.tx = -crot_x;
    mp.ty = -crot_y;
    mp.tz = tz;
    mp.rx = rx;
    mp.ry = ry;
    mp.rz = rz;

    cv::Mat rendered = gen.Generate(mesh, mp);
    cv::Mat binary;
    cv::threshold(rendered, binary, 127, 255, cv::THRESH_BINARY);
    cv::Mat packed = PackMask1Bit(binary);
    return ZlibCompress(packed.data, packed.total() * packed.elemSize());
  };

  // === Parallel coarse rendering ===
  auto t_coarse_start = std::chrono::high_resolution_clock::now();

  size_t total_coarse = coarse_work.size();
  std::vector<std::pair<CoarseWorkItem, std::vector<uint8_t>>> coarse_results(total_coarse);
  std::atomic<size_t> coarse_done{0};

  auto coarse_worker = [&](int tid) {
    maskgen::MaskGenerator gen(camera_params);
    size_t chunk = (total_coarse + num_threads - 1) / num_threads;
    size_t start = tid * chunk;
    size_t end = std::min(start + chunk, total_coarse);

    for (size_t i = start; i < end; ++i) {
      const auto& item = coarse_work[i];
      auto compressed = render_one(gen, item.rx, item.ry, item.rz,
                                   item.crot_x, item.crot_y, item.crot_z, item.depth);
      coarse_results[i] = {item, std::move(compressed)};

      size_t done = coarse_done.fetch_add(1) + 1;
      if (done % 100 == 0 || done == total_coarse) {
        std::cout << "\r[Coarse] " << done << "/" << total_coarse << std::flush;
      }
    }
  };

  {
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
      threads.emplace_back(coarse_worker, t);
    }
    for (auto& th : threads) th.join();
  }

  auto t_coarse_end = std::chrono::high_resolution_clock::now();
  double coarse_ms = std::chrono::duration<double, std::milli>(t_coarse_end - t_coarse_start).count();
  std::cout << "\r[Coarse] " << total_coarse << "/" << total_coarse
            << " (" << coarse_ms / 1000.0 << " s)\n";

  // === Parallel local rendering ===
  auto t_local_start = std::chrono::high_resolution_clock::now();

  size_t total_local = local_work.size();
  std::vector<std::pair<LocalWorkItem, std::vector<uint8_t>>> local_results(total_local);
  std::atomic<size_t> local_done{0};

  auto local_worker = [&](int tid) {
    maskgen::MaskGenerator gen(camera_params);
    size_t chunk = (total_local + num_threads - 1) / num_threads;
    size_t start = tid * chunk;
    size_t end = std::min(start + chunk, total_local);

    for (size_t i = start; i < end; ++i) {
      const auto& item = local_work[i];
      auto compressed = render_one(gen, item.rx, item.ry, item.rz,
                                   item.crot_x, item.crot_y, item.crot_z, item.depth);
      local_results[i] = {item, std::move(compressed)};

      size_t done = local_done.fetch_add(1) + 1;
      if (done % 1000 == 0 || done == total_local) {
        std::cout << "\r[Local] " << done << "/" << total_local << std::flush;
      }
    }
  };

  {
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
      threads.emplace_back(local_worker, t);
    }
    for (auto& th : threads) th.join();
  }

  auto t_local_end = std::chrono::high_resolution_clock::now();
  double local_ms = std::chrono::duration<double, std::milli>(t_local_end - t_local_start).count();
  std::cout << "\r[Local] " << total_local << "/" << total_local
            << " (" << local_ms / 1000.0 << " s)\n";

  // === Assemble cache ===
  auto t_assemble_start = std::chrono::high_resolution_clock::now();

  CacheData cache;

  cache.header.image_width = static_cast<uint32_t>(camera_params.width);
  cache.header.image_height = static_cast<uint32_t>(camera_params.height);
  cache.header.fx = camera_params.fx;
  cache.header.fy = camera_params.fy;
  cache.header.cx = camera_params.cx;
  cache.header.cy = camera_params.cy;
  cache.header.mesh_centroid[0] = mesh_info.centroid[0];
  cache.header.mesh_centroid[1] = mesh_info.centroid[1];
  cache.header.mesh_centroid[2] = mesh_info.centroid[2];
  cache.header.mesh_extent[0] = mesh_info.extent[0];
  cache.header.mesh_extent[1] = mesh_info.extent[1];
  cache.header.mesh_extent[2] = mesh_info.extent[2];
  cache.header.principal_axis = static_cast<uint32_t>(mesh_info.principal_axis);
  cache.header.num_directions = params.num_directions;
  cache.header.num_in_plane = params.num_in_plane;
  cache.header.num_depth = params.num_depth;
  cache.header.depth_min = params.depth_min;
  cache.header.depth_max = params.depth_max;
  cache.header.top_k_coarse = params.top_k_coarse;
  cache.header.nelder_mead_iterations = params.nelder_mead_iterations;
  cache.header.local_directions = params.local_directions;
  cache.header.local_in_plane = params.local_in_plane;
  cache.header.local_depth = params.local_depth;
  cache.header.local_cone_half_angle_deg = params.local_cone_half_angle_deg;
  cache.header.top_k_local = params.top_k_local;

  std::string model_hash = ComputeSHA256(model_path);
  if (!model_hash.empty()) {
    std::memcpy(cache.header.model_hash, model_hash.data(), 32);
  }

  cache.coarse_entries.reserve(coarse_results.size());
  for (auto& [item, compressed] : coarse_results) {
    CoarseEntry entry;
    entry.rx = item.rx;
    entry.ry = item.ry;
    entry.rz = item.rz;
    entry.crot_x = item.crot_x;
    entry.crot_y = item.crot_y;
    entry.crot_z = item.crot_z;
    entry.depth = item.depth;

    if (compressed.empty()) {
      entry.mask_offset = 0;
      entry.mask_size = 0;
    } else {
      entry.mask_offset = static_cast<uint64_t>(cache.mask_data.size());
      entry.mask_size = static_cast<uint32_t>(compressed.size());
      cache.mask_data.insert(cache.mask_data.end(), compressed.begin(), compressed.end());
    }

    cache.coarse_entries.push_back(entry);
  }

  cache.local_entries.reserve(local_results.size());
  for (auto& [item, compressed] : local_results) {
    LocalEntry entry;
    entry.coarse_idx = item.coarse_idx;
    entry.rx = item.rx;
    entry.ry = item.ry;
    entry.rz = item.rz;
    entry.crot_x = item.crot_x;
    entry.crot_y = item.crot_y;
    entry.crot_z = item.crot_z;
    entry.depth = item.depth;

    if (compressed.empty()) {
      entry.mask_offset = 0;
      entry.mask_size = 0;
    } else {
      entry.mask_offset = static_cast<uint64_t>(cache.mask_data.size());
      entry.mask_size = static_cast<uint32_t>(compressed.size());
      cache.mask_data.insert(cache.mask_data.end(), compressed.begin(), compressed.end());
    }

    cache.local_entries.push_back(entry);
  }

  auto t_assemble_end = std::chrono::high_resolution_clock::now();
  double assemble_ms = std::chrono::duration<double, std::milli>(t_assemble_end - t_assemble_start).count();
  std::cout << "[Timing] Assembly: " << assemble_ms << " ms\n";

  auto t_write_start = std::chrono::high_resolution_clock::now();
  if (!WriteCache(output_path, cache)) {
    std::cerr << "Error: failed to write cache file\n";
    return 1;
  }
  auto t_write_end = std::chrono::high_resolution_clock::now();
  double write_ms = std::chrono::duration<double, std::milli>(t_write_end - t_write_start).count();

  double total_size_mb = (sizeof(CacheFileHeader) +
                          cache.coarse_entries.size() * sizeof(CoarseEntry) +
                          cache.local_entries.size() * sizeof(LocalEntry) +
                          cache.mask_data.size()) / (1024.0 * 1024.0);

  std::cout << "\n=== Cache Build Summary ===\n";
  std::cout << "Threads: " << num_threads << "\n";
  std::cout << "Coarse entries: " << cache.coarse_entries.size() << "\n";
  std::cout << "Local entries: " << cache.local_entries.size() << "\n";
  std::cout << "File size: " << total_size_mb << " MB\n";
  std::cout << "Coarse render time: " << coarse_ms / 1000.0 << " s\n";
  std::cout << "Local render time: " << local_ms / 1000.0 << " s\n";
  std::cout << "Assembly time: " << assemble_ms / 1000.0 << " s\n";
  std::cout << "Write time: " << write_ms / 1000.0 << " s\n";
  std::cout << "Total time: " << (coarse_ms + local_ms + assemble_ms + write_ms) / 1000.0 << " s\n";
  std::cout << "Output: " << output_path << "\n";

  return 0;
}
