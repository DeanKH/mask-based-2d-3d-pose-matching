#!/bin/bash
set -euo pipefail

REFINE_METHODS=("nelder-mead" "bobyqa")
SORT_METRICS=("iou" "hu" "zernike" "iou_zernike" "centroid_iou" "centroid_dt_l1" "centroid_dt_l2" "dt_iou" "central_moments" "shape_context" "contour_chamfer" "fourier")
MAX_CANDIDATES=(10 20 30)

TOTAL=$(( ${#REFINE_METHODS[@]} * ${#SORT_METRICS[@]} * ${#MAX_CANDIDATES[@]} ))
COUNT=0

for refine in "${REFINE_METHODS[@]}"; do
  for metric in "${SORT_METRICS[@]}"; do
    for cand in "${MAX_CANDIDATES[@]}"; do
      COUNT=$((COUNT + 1))
      echo "[$COUNT/$TOTAL] refine=$refine  sort=$metric  candidates=$cand"

      docker compose run --rm \
        --entrypoint /opt/build/benchmark \
        benchmark \
        --dataset /workspace/test/data \
        --camera /workspace/camera_info.json \
        --model-dir /workspace/mask-generation/model \
        --cache-dir /workspace/cache \
        --output-dir /workspace/benchmark \
        --scale 0.001 \
        --max-candidates "$cand" \
        --max-refine 300 \
        --fatol 1e-4 \
        --xatol 1e-4 \
        --contour-points 4000 \
        --refine-method "$refine" \
        --sort-metric "$metric"
    done
  done
done

echo "Done: $COUNT benchmarks completed."
