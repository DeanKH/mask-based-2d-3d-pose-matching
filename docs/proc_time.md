# Sort Metric Benchmark Results

## Test Conditions
- Object: ad (cache/ad.bin)
- Candidates: 30-33
- RefinePose: Nelder-Mead, 16 threads
- Early termination IoU: 0.95
- GPU: Vulkan

## Summary Table

| Metric | Sort Time | best_idx | Final IoU | Top 10? | Estimate Total |
|--------|-----------|----------|-----------|---------|----------------|
| iou | 0.001ms | 28 | 0.991 | NO | 20555ms |
| centroid_dt_l1 | 1428ms | 20 | 0.991 | NO | 15804ms |
| centroid_dt_l2 | 1428ms | 17 | 0.991 | NO | 14208ms |
| dt_iou | 1428ms | 17 | 0.991 | NO | 14077ms |
| central_moments | 1459ms | 11 | 0.991 | NO | 12126ms |
| shape_context | 1442ms | 1 | 0.998 | YES | 20651ms |
| contour_chamfer | 1425ms | 5 | 0.991 | YES | 12142ms |
| fourier | 1426ms | 8 | 0.998 | YES | 20631ms |

## Analysis

### Top 10 Criterion (best_idx <= 10)
- **shape_context**: best_idx=1, IoU=0.998 (best ranking but slow)
- **contour_chamfer**: best_idx=5, IoU=0.991 (best balance)
- **fourier**: best_idx=8, IoU=0.998 (good ranking but slow)

### Total Time Criterion (< 20000ms)
- contour_chamfer: 12142ms
- central_moments: 12126ms
- dt_iou: 14077ms
- centroid_dt_l2: 14208ms
- centroid_dt_l1: 15804ms
- iou: 20555ms (exceeds)
- shape_context: 20651ms (exceeds)
- fourier: 20631ms (exceeds)

### Best Metric: contour_chamfer
- Places best candidate at sorted position 5 (well within top 10)
- Total Estimate time: 12142ms (fastest among top-10 metrics)
- Final IoU: 0.991 (> 0.99 threshold)

### Why ShapeContext and Fourier are slow despite early best_idx
All candidates are launched in parallel (16 threads x 2 batches).
Even though the best candidate finishes early, the batch must complete.
The high total time comes from the parallel RefinePose wall time, not the sort.

## Method Details

### 1. Centroid-aligned Distance Transform L1/L2
- Pre-compute DT of input mask (DIST_L1 or DIST_L2)
- Shift rendered mask centroid to align with input centroid
- Score = average DT value at shifted rendered foreground pixels
- L2 is slightly better than L1 (best_idx 17 vs 20)

### 2. DT + IoU
- Centroid-aligned DT_L2 average * (1 - centroid-aligned IoU)
- Combines shape distance with overlap quality
- Similar to CentroidDT_L2 alone (best_idx 17)

### 3. Central Moments
- Shift rendered centroid to align with input
- L2 distance of 7 normalized central moments (nu20..nu03)
- Scale-invariant but NOT rotation-invariant
- best_idx=11, just outside top 10

### 4. Shape Context
- Centroid-aligned contour extraction
- Log-polar histogram descriptors at 100 sampled points (5r x 12theta bins)
- Chi-squared distance between corresponding point descriptors
- **best_idx=1** (excellent ranking), but total time exceeds 20000ms

### 5. Contour Chamfer
- Centroid-aligned contour extraction
- Resample both contours to 200 points
- Bidirectional Chamfer distance (average nearest-point distance)
- **best_idx=5**, total time 12142ms — best overall

### 6. Fourier Descriptor
- Centroid-aligned contour extraction
- Resample to 128 points, center at origin
- DFT, take first 32 magnitude coefficients (normalized by DC)
- L2 distance of normalized descriptors
- best_idx=8, but total time exceeds 20000ms
