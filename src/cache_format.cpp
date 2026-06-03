#include "cache_format.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <zlib.h>
#include <openssl/sha.h>

namespace pose_matching {

cv::Mat PackMask1Bit(const cv::Mat& binary_mask) {
  int w = binary_mask.cols;
  int h = binary_mask.rows;
  int packed_cols = (w + 7) / 8;
  cv::Mat packed(h, packed_cols, CV_8UC1, cv::Scalar(0));

  for (int r = 0; r < h; ++r) {
    const auto* src = binary_mask.ptr<uint8_t>(r);
    auto* dst = packed.ptr<uint8_t>(r);
    for (int c = 0; c < w; ++c) {
      if (src[c] > 127) {
        dst[c / 8] |= static_cast<uint8_t>(0x80 >> (c % 8));
      }
    }
  }
  return packed;
}

cv::Mat UnpackMask1Bit(const uint8_t* packed, size_t packed_size, int width, int height) {
  int packed_cols = (width + 7) / 8;
  cv::Mat mask(height, width, CV_8UC1, cv::Scalar(0));

  for (int r = 0; r < height; ++r) {
    auto* dst = mask.ptr<uint8_t>(r);
    const uint8_t* src = packed + r * packed_cols;
    for (int c = 0; c < width; ++c) {
      if (src[c / 8] & static_cast<uint8_t>(0x80 >> (c % 8))) {
        dst[c] = 255;
      }
    }
  }
  return mask;
}

std::vector<uint8_t> ZlibCompress(const uint8_t* data, size_t size) {
  uLongf compressed_size = compressBound(static_cast<uLong>(size));
  std::vector<uint8_t> out(compressed_size);
  int ret = compress(out.data(), &compressed_size, data, static_cast<uLong>(size));
  if (ret != Z_OK) {
    throw std::runtime_error("zlib compress failed: " + std::to_string(ret));
  }
  out.resize(compressed_size);
  return out;
}

std::vector<uint8_t> ZlibDecompress(const uint8_t* data, size_t size, size_t expected_size) {
  std::vector<uint8_t> out(expected_size);
  uLongf dest_size = static_cast<uLongf>(expected_size);
  int ret = uncompress(out.data(), &dest_size, data, static_cast<uLong>(size));
  if (ret != Z_OK) {
    throw std::runtime_error("zlib decompress failed: " + std::to_string(ret));
  }
  out.resize(dest_size);
  return out;
}

bool WriteCache(const std::string& path, const CacheData& cache) {
  std::ofstream f(path, std::ios::binary);
  if (!f.is_open()) {
    std::cerr << "Error: cannot create cache file: " << path << "\n";
    return false;
  }

  CacheFileHeader header = cache.header;
  header.num_coarse = static_cast<uint32_t>(cache.coarse_entries.size());
  header.num_local = static_cast<uint32_t>(cache.local_entries.size());

  size_t header_size = sizeof(CacheFileHeader);
  size_t coarse_table_size = cache.coarse_entries.size() * sizeof(CoarseEntry);
  size_t local_table_size = cache.local_entries.size() * sizeof(LocalEntry);

  header.coarse_offset = header_size;
  header.local_offset = header.coarse_offset + coarse_table_size;
  header.data_offset = header.local_offset + local_table_size;

  f.write(reinterpret_cast<const char*>(&header), header_size);

  if (!cache.coarse_entries.empty()) {
    f.write(reinterpret_cast<const char*>(cache.coarse_entries.data()),
            coarse_table_size);
  }

  if (!cache.local_entries.empty()) {
    f.write(reinterpret_cast<const char*>(cache.local_entries.data()),
            local_table_size);
  }

  if (!cache.mask_data.empty()) {
    f.write(reinterpret_cast<const char*>(cache.mask_data.data()),
            cache.mask_data.size());
  }

  return f.good();
}

bool ReadCache(const std::string& path, CacheData& cache) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f.is_open()) {
    std::cerr << "Error: cannot open cache file: " << path << "\n";
    return false;
  }

  size_t file_size = static_cast<size_t>(f.tellg());
  f.seekg(0);

  std::vector<uint8_t> buffer(file_size);
  f.read(reinterpret_cast<char*>(buffer.data()), file_size);
  if (!f.good()) {
    std::cerr << "Error: failed to read cache file\n";
    return false;
  }

  if (file_size < sizeof(CacheFileHeader)) {
    std::cerr << "Error: cache file too small\n";
    return false;
  }

  std::memcpy(&cache.header, buffer.data(), sizeof(CacheFileHeader));

  if (std::memcmp(cache.header.magic, "PMCK", 4) != 0) {
    std::cerr << "Error: invalid cache file magic\n";
    return false;
  }

  size_t coarse_count = cache.header.num_coarse;
  size_t local_count = cache.header.num_local;

  cache.coarse_entries.resize(coarse_count);
  if (coarse_count > 0) {
    std::memcpy(cache.coarse_entries.data(),
                buffer.data() + cache.header.coarse_offset,
                coarse_count * sizeof(CoarseEntry));
  }

  cache.local_entries.resize(local_count);
  if (local_count > 0) {
    std::memcpy(cache.local_entries.data(),
                buffer.data() + cache.header.local_offset,
                local_count * sizeof(LocalEntry));
  }

  size_t mask_data_size = file_size - cache.header.data_offset;
  cache.mask_data.resize(mask_data_size);
  if (mask_data_size > 0) {
    std::memcpy(cache.mask_data.data(),
                buffer.data() + cache.header.data_offset,
                mask_data_size);
  }

  return true;
}

bool ReadCacheMMAP(const std::string& path, CacheData& cache,
                   std::vector<uint8_t>& file_buffer) {
  return ReadCache(path, cache);
}

std::string ComputeSHA256(const std::string& file_path) {
  std::ifstream f(file_path, std::ios::binary);
  if (!f.is_open()) {
    return {};
  }

  SHA256_CTX ctx;
  SHA256_Init(&ctx);

  char buf[8192];
  while (f.read(buf, sizeof(buf))) {
    SHA256_Update(&ctx, buf, static_cast<size_t>(f.gcount()));
  }
  if (f.gcount() > 0) {
    SHA256_Update(&ctx, buf, static_cast<size_t>(f.gcount()));
  }

  unsigned char hash[SHA256_DIGEST_LENGTH];
  SHA256_Final(hash, &ctx);

  std::string result(SHA256_DIGEST_LENGTH, '\0');
  std::memcpy(&result[0], hash, SHA256_DIGEST_LENGTH);
  return result;
}

}  // namespace pose_matching
