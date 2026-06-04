#pragma once

#include <chrono>
#include <cstdio>
#include <string>

namespace pose_matching {

struct RefineProfiler {
  double generate_ms = 0;
  double absdiff_ms = 0;
  double dt_ms = 0;
  double chamfer_ms = 0;
  double area_ms = 0;
  double viz_ms = 0;
  double total_wall_ms = 0;
  int cost_evals = 0;

  void Print(const std::string& label) const {
    double categorized = generate_ms + absdiff_ms + dt_ms + chamfer_ms + area_ms + viz_ms;
    double overhead = total_wall_ms - categorized;
    double total = total_wall_ms;
    auto pct = [&](double v) { return total > 0 ? 100.0 * v / total : 0.0; };
    auto avg = [&](double v) { return cost_evals > 0 ? v / cost_evals : 0.0; };

    std::printf("[RefineProfile] %s: %d cost evaluations\n", label.c_str(), cost_evals);
    std::printf("  %-20s %10.1f ms  (%5.1f%%)  avg=%8.2f ms\n",
                "generate", generate_ms, pct(generate_ms), avg(generate_ms));
    std::printf("  %-20s %10.1f ms  (%5.1f%%)  avg=%8.2f ms\n",
                "distance_transform", dt_ms, pct(dt_ms), avg(dt_ms));
    std::printf("  %-20s %10.1f ms  (%5.1f%%)  avg=%8.2f ms\n",
                "absdiff", absdiff_ms, pct(absdiff_ms), avg(absdiff_ms));
    std::printf("  %-20s %10.1f ms  (%5.1f%%)  avg=%8.2f ms\n",
                "chamfer", chamfer_ms, pct(chamfer_ms), avg(chamfer_ms));
    std::printf("  %-20s %10.1f ms  (%5.1f%%)  avg=%8.2f ms\n",
                "area", area_ms, pct(area_ms), avg(area_ms));
    std::printf("  %-20s %10.1f ms  (%5.1f%%)  avg=%8.2f ms\n",
                "viz", viz_ms, pct(viz_ms), avg(viz_ms));
    std::printf("  %-20s %10.1f ms  (%5.1f%%)\n",
                "overhead", overhead, pct(overhead));
    std::printf("  %-20s %10.1f ms\n", "TOTAL", total);
  }
};

class ScopedTimer {
 public:
  explicit ScopedTimer(double& dest) : dest_(dest), start_(std::chrono::high_resolution_clock::now()) {}
  ~ScopedTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    dest_ += std::chrono::duration<double, std::milli>(end - start_).count();
  }

 private:
  double& dest_;
  std::chrono::high_resolution_clock::time_point start_;
};

}  // namespace pose_matching
