#pragma once

#include <atomic>
#include <functional>
#include <limits>
#include <vector>

#include <nlopt.h>

namespace pose_matching {

struct BobyqaOptions {
  double xtol_rel = 1e-4;
  double ftol_rel = 0.0;
};

namespace detail {

struct BobyqaContext {
  std::function<double(const std::vector<double>&)>* cost_fn;
  const std::atomic<bool>* abort_flag;
  nlopt_opt opt;
  bool aborted = false;
};

inline double bobyqa_callback(unsigned n, const double* x,
                              double* grad, void* data) {
  (void)grad;
  auto* ctx = static_cast<BobyqaContext*>(data);

  if (ctx->abort_flag && ctx->abort_flag->load(std::memory_order_relaxed)) {
    ctx->aborted = true;
    nlopt_force_stop(ctx->opt);
    return std::numeric_limits<double>::max();
  }

  std::vector<double> params(x, x + n);
  try {
    return (*ctx->cost_fn)(params);
  } catch (const std::runtime_error&) {
    ctx->aborted = true;
    nlopt_force_stop(ctx->opt);
    return std::numeric_limits<double>::max();
  }
}

}  // namespace detail

inline double Bobyqa(std::function<double(const std::vector<double>&)> cost,
                     std::vector<double>& x,
                     const std::vector<double>& initial_step,
                     int max_iterations,
                     const BobyqaOptions& opts = BobyqaOptions(),
                     const std::atomic<bool>* abort_flag = nullptr,
                     bool* aborted = nullptr) {
  const unsigned n = static_cast<unsigned>(x.size());

  nlopt_opt opt = nlopt_create(NLOPT_LN_BOBYQA, n);

  detail::BobyqaContext ctx;
  ctx.cost_fn = &cost;
  ctx.abort_flag = abort_flag;
  ctx.opt = opt;
  ctx.aborted = false;

  nlopt_set_min_objective(opt, detail::bobyqa_callback, &ctx);

  if (!initial_step.empty()) {
    std::vector<double> step(n);
    for (unsigned i = 0; i < n; ++i) {
      step[i] = i < initial_step.size() ? initial_step[i] : 0.1;
    }
    nlopt_set_initial_step(opt, step.data());
  }

  nlopt_set_xtol_rel(opt, opts.xtol_rel);
  if (opts.ftol_rel > 0.0) {
    nlopt_set_ftol_rel(opt, opts.ftol_rel);
  }
  nlopt_set_maxeval(opt, max_iterations);

  double minf;
  nlopt_optimize(opt, x.data(), &minf);

  if (aborted) {
    *aborted = ctx.aborted ||
               (abort_flag && abort_flag->load(std::memory_order_relaxed));
  }

  nlopt_destroy(opt);

  return minf;
}

}  // namespace pose_matching
