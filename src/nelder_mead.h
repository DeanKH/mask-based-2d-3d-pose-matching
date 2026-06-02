#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

namespace pose_matching {

struct NelderMeadOptions {
  double alpha = 1.0;
  double gamma = 2.0;
  double rho = 0.5;
  double sigma = 0.5;
  double xatol = 1e-6;
  double fatol = 1e-8;
};

inline double NelderMead(std::function<double(const std::vector<double>&)> cost,
                         std::vector<double>& x,
                         const std::vector<double>& initial_step,
                         int max_iterations,
                         const NelderMeadOptions& opts = NelderMeadOptions()) {
  const int n = static_cast<int>(x.size());
  const int num_vertices = n + 1;

  std::vector<std::vector<double>> simplex(num_vertices, x);
  for (int i = 0; i < n; ++i) {
    simplex[i + 1][i] += initial_step[i];
  }

  std::vector<double> f_values(num_vertices);
  for (int i = 0; i < num_vertices; ++i) {
    f_values[i] = cost(simplex[i]);
  }

  std::vector<int> indices(num_vertices);
  std::iota(indices.begin(), indices.end(), 0);

  for (int iter = 0; iter < max_iterations; ++iter) {
    std::sort(indices.begin(), indices.end(),
              [&](int a, int b) { return f_values[a] < f_values[b]; });

    double best_f = f_values[indices[0]];
    double worst_f = f_values[indices[num_vertices - 1]];
    double second_worst_f = f_values[indices[num_vertices - 2]];

    if (worst_f - best_f < opts.fatol) {
      bool all_close = true;
      for (int i = 0; i < n; ++i) {
        double min_v = simplex[indices[0]][i];
        double max_v = simplex[indices[0]][i];
        for (int j = 1; j < num_vertices; ++j) {
          min_v = std::min(min_v, simplex[indices[j]][i]);
          max_v = std::max(max_v, simplex[indices[j]][i]);
        }
        if (max_v - min_v > opts.xatol) {
          all_close = false;
          break;
        }
      }
      if (all_close) break;
    }

    std::vector<double> centroid(n, 0.0);
    for (int i = 0; i < num_vertices - 1; ++i) {
      for (int j = 0; j < n; ++j) {
        centroid[j] += simplex[indices[i]][j];
      }
    }
    for (int j = 0; j < n; ++j) {
      centroid[j] /= (num_vertices - 1);
    }

    std::vector<double> reflected(n);
    for (int j = 0; j < n; ++j) {
      reflected[j] = centroid[j] + opts.alpha * (centroid[j] - simplex[indices[num_vertices - 1]][j]);
    }
    double f_reflected = cost(reflected);

    if (f_reflected < second_worst_f && f_reflected >= best_f) {
      simplex[indices[num_vertices - 1]] = reflected;
      f_values[indices[num_vertices - 1]] = f_reflected;
      continue;
    }

    if (f_reflected < best_f) {
      std::vector<double> expanded(n);
      for (int j = 0; j < n; ++j) {
        expanded[j] = centroid[j] + opts.gamma * (reflected[j] - centroid[j]);
      }
      double f_expanded = cost(expanded);
      if (f_expanded < f_reflected) {
        simplex[indices[num_vertices - 1]] = expanded;
        f_values[indices[num_vertices - 1]] = f_expanded;
      } else {
        simplex[indices[num_vertices - 1]] = reflected;
        f_values[indices[num_vertices - 1]] = f_reflected;
      }
      continue;
    }

    std::vector<double> contracted(n);
    for (int j = 0; j < n; ++j) {
      contracted[j] = centroid[j] + opts.rho * (simplex[indices[num_vertices - 1]][j] - centroid[j]);
    }
    double f_contracted = cost(contracted);

    if (f_contracted < worst_f) {
      simplex[indices[num_vertices - 1]] = contracted;
      f_values[indices[num_vertices - 1]] = f_contracted;
      continue;
    }

    for (int i = 1; i < num_vertices; ++i) {
      for (int j = 0; j < n; ++j) {
        simplex[indices[i]][j] =
            simplex[indices[0]][j] + opts.sigma * (simplex[indices[i]][j] - simplex[indices[0]][j]);
      }
      f_values[indices[i]] = cost(simplex[indices[i]]);
    }
  }

  std::sort(indices.begin(), indices.end(),
            [&](int a, int b) { return f_values[a] < f_values[b]; });
  x = simplex[indices[0]];
  return f_values[indices[0]];
}

}  // namespace pose_matching
