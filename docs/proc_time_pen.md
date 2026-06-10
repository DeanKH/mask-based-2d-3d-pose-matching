# Sort Metric Benchmark Results — Pen

## Test Conditions
- Object: pen (cache/pen.bin)
- Candidates: 30
- RefinePose: Nelder-Mead, 16 threads
- Early termination IoU: 0.95
- GPU: Vulkan

## Summary Table

| Metric | Sort Time | best_idx | Final IoU | Top 10? | Estimate Total |
|--------|-----------|----------|-----------|---------|----------------|
| iou | 0.001ms | 5 | 0.977 | YES | 14231ms |
| centroid_dt_l1 | 1376ms | 6 | 0.977 | YES | 15359ms |
| centroid_dt_l2 | 1368ms | 6 | 0.977 | YES | 15209ms |
| dt_iou | 1376ms | 17 | 0.977 | NO | 17057ms |
| central_moments | 1378ms | 5 | 0.977 | YES | 15669ms |
| shape_context | 1375ms | 20 | 0.977 | NO | 19333ms |
| contour_chamfer | 1407ms | 25 | 0.977 | NO | 23846ms |
| fourier | 1373ms | 3 | 0.977 | YES | 15603ms |

## Analysis

### All metrics converge to the same result
All 8 metrics find the same final pose (IoU=0.9767) through Nelder-Mead refinement.
The difference is only in which sorted candidate leads to the best result (best_idx)
and the total wall-clock time (affected by sort overhead + early termination timing).

### Top 10 Criterion (best_idx <= 10)
- **fourier**: best_idx=3 (best ranking among all metrics)
- **iou**: best_idx=5 (free — 0ms sort overhead)
- **central_moments**: best_idx=5
- **centroid_dt_l1**: best_idx=6
- **centroid_dt_l2**: best_idx=6

### Total Time Criterion (< 20000ms)
All metrics except `contour_chamfer` (23846ms) finish under 20000ms.

### Best Metric for Pen: iou
- The coarse IoU is already a strong indicator for the pen object
- Sort time is negligible (0.001ms vs 1370ms for others)
- best_idx=5 (well within top 10)
- Total Estimate time: 14231ms (fastest overall)

### Comparison with ad object (see docs/proc_time.md)
| Metric | Pen best_idx | Ad best_idx | Pen Total | Ad Total |
|--------|-------------|-------------|-----------|----------|
| iou | 5 | 28 | 14231ms | 20555ms |
| contour_chamfer | 25 | 5 | 23846ms | 12142ms |
| fourier | 3 | 8 | 15603ms | 20631ms |
| shape_context | 20 | 1 | 19333ms | 20651ms |

The ranking is inverted between pen and ad:
- For **ad**: contour_chamfer > shape_context > fourier >> iou
- For **pen**: iou > fourier >> contour_chamfer > shape_context

This suggests that **no single metric is universally optimal**.
The pen is a thin elongated object where coarse IoU already provides
good discrimination, while the ad (more complex shape) benefits from
shape-aware metrics like contour chamfer distance.
