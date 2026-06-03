#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace pose_matching {

#pragma pack(push, 1)

struct CacheFileHeader {
  char magic[4] = {'P', 'M', 'C', 'K'};
  uint32_t version = 1;
  uint32_t image_width = 0;
  uint32_t image_height = 0;
  double fx = 0;
  double fy = 0;
  double cx = 0;
  double cy = 0;
  float mesh_centroid[3] = {};
  float mesh_extent[3] = {};
  uint32_t principal_axis = 0;
  int32_t num_directions = 48;
  int32_t num_in_plane = 4;
  int32_t num_depth = 5;
  double depth_min = 0.05;
  double depth_max = 0.50;
  int32_t top_k_coarse = 10;
  int32_t nelder_mead_iterations = 200;
  int32_t local_directions = 6;
  int32_t local_in_plane = 6;
  int32_t local_depth = 3;
  double local_cone_half_angle_deg = 30.0;
  int32_t top_k_local = 5;
  char reserved[64] = {};
  char model_hash[32] = {};
  uint32_t num_coarse = 0;
  uint32_t num_local = 0;
  uint64_t coarse_offset = 0;
  uint64_t local_offset = 0;
  uint64_t data_offset = 0;
};

struct CoarseEntry {
  double rx, ry, rz;
  float crot_x, crot_y, crot_z;
  double depth;
  uint64_t mask_offset;
  uint32_t mask_size;
};

struct LocalEntry {
  uint32_t coarse_idx;
  double rx, ry, rz;
  float crot_x, crot_y, crot_z;
  double depth;
  uint64_t mask_offset;
  uint32_t mask_size;
};

#pragma pack(pop)

struct CacheData {
  CacheFileHeader header;
  std::vector<CoarseEntry> coarse_entries;
  std::vector<LocalEntry> local_entries;
  std::vector<uint8_t> mask_data;
};

cv::Mat PackMask1Bit(const cv::Mat& binary_mask);

cv::Mat UnpackMask1Bit(const uint8_t* packed, size_t packed_size, int width, int height);

std::vector<uint8_t> ZlibCompress(const uint8_t* data, size_t size);

std::vector<uint8_t> ZlibDecompress(const uint8_t* data, size_t size, size_t expected_size);

bool WriteCache(const std::string& path, const CacheData& cache);

bool ReadCache(const std::string& path, CacheData& cache);

bool ReadCacheMMAP(const std::string& path, CacheData& cache,
                   std::vector<uint8_t>& file_buffer);

std::string ComputeSHA256(const std::string& file_path);

}  // namespace pose_matching
