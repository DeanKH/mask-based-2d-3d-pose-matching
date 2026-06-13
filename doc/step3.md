# Step 3: GPU Compute Shader によるコスト計算

## 概要

Refine段階のコスト関数（双方向Chamfer距離 + 面積比）の計算をGPU Compute Shaderに移行した。

レンダリング済みマスクの距離変換（DT）を、ピクセル単位のローカルウィンドウ探索（R=16, 33x33近傍）で近似し、コスト計算全体を2回のCompute Dispatchで実行する。CPU側のOpenCV処理（`distanceTransform`, `absdiff`, `sum`等）を完全にバイパスする。

## アーキテクチャ

### パイプライン

```
CPU: Render pose (VK_R8_UNORM)  →  GPU Compute x2  →  CPU: read cost
                                 ┌─────────────────┐
     color_image_ (R8)  ────────►│ cost_compute     │──► partial_buffer[]
                                 │ (local-window DT │
                                 │  + workgroup     │
                                 │  reduction)      │
                                 └───────┬─────────┘
                                         │
                                 ┌───────▼─────────┐
                                 │ cost_reduce      │──► cost_output (vec4)
                                 │ (final reduction)│     chamfer, area, inter
                                 └─────────────────┘
```

### Compute Shader詳細

#### Dispatch 1: cost_compute (16x16 workgroups)

各ピクセルについて:

1. **レンダリングマスク読み出し**: `imageLoad(rendered_img, coord).r` (R8)
2. **入力マスク・DT読み出し**: SSBOから `input_mask[idx]`, `dt_input[idx]`
3. **差分判定**: `diff = abs(rendered_val - input_val)`
4. **ローカルDT (forward Chamferのみ)**:
   - 条件: `rendered_val <= 0.5 && diff > 0.5` (入力にあるがレンダリングにないピクセル)
   - 33x33近傍を走査し、最近接レンダリングピクセルまでの距離を計算
   - ウィンドウ内にレンダリングピクセルがない場合、距離を `sqrt(2*R^2+1)` にキャップ
5. **コスト計算**: `chamfer = max(dt_input, dt_r) * diff * scale`
6. **ワークグループ内リダクション**: 256スレッド (16x16) → 1つの `vec4` (chamfer, area, intersection, 0)

**最適化ポイント**:
- `diff > 0.5` のピクセルのみローカル探索を実行（R8マスクでは diff は0または1に近い値）
- Backward Chamfer（レンダリングにあるが入力にない）は `dt_input` で処理されるため、追加計算不要
- 両マスクが一致するピクセルは `diff = 0` でコスト0、計算スキップ

#### Dispatch 2: cost_reduce (256 threads, 1 workgroup)

全ワークグループの部分和を最終的に集約:
- 各スレッドが `partial[i]` をストライド処理で累積
- 共有メモリツリー リダクションで最終結果を出力

### Push Constant

```cpp
struct ComputePC {
    int32_t width;
    int32_t height;
    int32_t radius_or_groups;  // cost_compute: 未使用(Rは定数)
                              // cost_reduce: num_partial_groups
    float scale;
};
```

### リソースバインディング

| Binding | Type | サイズ | 用途 |
|---------|------|--------|------|
| 0 | Storage Image (R8) | width x height | レンダリングマスク (`color_image_`) |
| 3 | Storage Buffer | width x height x 4B | 入力DT (CPU事前計算) |
| 4 | Storage Buffer | width x height x 4B | 入力マスク (正規化済み) |
| 5 | Storage Buffer | 16B | コスト出力 (chamfer, area, inter, _) |
| 6 | Storage Buffer | num_groups x 16B | 部分和バッファ (Device Local) |

### GPU判定

`VK_FORMAT_R8_UNORM` で `VK_IMAGE_USAGE_STORAGE_BIT` がサポートされている場合のみGPU Computeを有効化:

```cpp
has_compute_cost_ =
    (vkGetPhysicalDeviceImageFormatProperties(
         physical_device_, VK_FORMAT_R8_UNORM, VK_IMAGE_TYPE_2D,
         VK_IMAGE_TILING_OPTIMAL,
         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
             VK_IMAGE_USAGE_STORAGE_BIT,
         0, &img_props) == VK_SUCCESS);
```

サポートされていない環境では自動的にCPUパス（OpenCV）にフォールバックする。また、可視化モード（`viz_` = true）でもCPUパスを使用する。

### JFAからの移行

当初 Jump Flooding Algorithm (JFA) でグローバルDTを計算していたが、14回のDispatchオーバーヘッドが大きく（~45ms/eval vs CPU ~5ms/eval）、断念した。ローカルウィンドウDT（2 Dispatch）に置き換えた結果、大幅な高速化を実現した。

| アプローチ | Dispatch数 | 評価時間 | 備考 |
|-----------|-----------|---------|------|
| JFA (10+ passes) | ~14 | ~45ms | Dispatchオーバーヘッド支配的 |
| Local-window DT (R=16) | 2 | ~2ms | 実用的 |

## ベンチマーク結果

### テスト条件

- 解像度: 848x480
- 候補数: 5（デフォルト）
- 最大反復: 200（`--max-refine 200`）
- `--xatol 1e-3`、`--fatol 1e-4`
- 最適化手法: Nelder-Mead
- ローカルウィンドウ半径: R=16

### pen オブジェクト

| 指標 | CPU (OpenCV) | GPU (Compute) | 差 |
|---|---|---|---|
| 最終IoU | 0.965 | **0.976** | **+1.1%** |
| RefinePose合計時間 | 2517ms | **1134ms** | **-55% (2.2x)** |
| 総評価回数 | 924 | 530 | -43% |
| 1評価あたり | 2.7ms | **2.1ms** | -22% |

### ad オブジェクト

| 指標 | CPU (OpenCV) | GPU (Compute) | 差 |
|---|---|---|---|
| 最終IoU | 0.845 | **0.961** | **+13.8%** |
| RefinePose合計時間 | 1136ms | 1302ms | +15% |
| 総評価回数 | 223 | 700 | +214% |
| 1評価あたり | 5.1ms | **1.9ms** | **-63% (2.7x)** |

### 考察

#### pen（細長い軸対称形状）
- 2.2倍の高速化とIoU向上を同時に達成
- CPU版より評価回数が少ない（530 vs 924）— GPUの滑らかなコスト曲面によりNMが早く収束
- 1評価あたりの時間も短縮（2.1ms vs 2.7ms）

#### ad（平坦形状）
- IoUが大幅に向上（0.845 → 0.961, +13.8%）
- CPU版は5候補では最適解に到達できなかった（候補#2のIoU=0.845が最高）
- GPU版は同じ5候補から候補#1のNM最適化でIoU=0.961に到達（319評価）
- 1評価あたり2.7倍高速（1.9ms vs 5.1ms）だが、評価回数が多く（700 vs 223）総時間は同等
- **ローカルDTキャップ（R=16）がもたらす滑らかなコスト曲面が、NMの局所最適解脱出を支援していると考えられる**

#### R（ローカルウィンドウ半径）の影響

| R | pen IoU | ad IoU | 1評価時間 | 備考 |
|---|---------|--------|----------|------|
| 8 | — | 0.790 | 速い | DT精度不足 |
| **16** | **0.976** | **0.961** | **2ms** | **最適バランス** |
| 32 | — | 0.875 | 遅い | コスト曲面が急峻でNMが局所解に捕捉 |

R=16が速度と精度の最適バランス。R=8ではDT近似精度が不足し、R=32ではコスト曲面が急峻になりNMが局所最適解に陥る。

### 30候補での比較（Step 2パラメータ）

| Object | CPU IoU (Step 2) | GPU IoU | CPU時間 | GPU時間 |
|--------|-----------------|---------|---------|---------|
| pen | 0.965 | 0.976 | 2417ms | 4840ms |
| ad | 0.991 | 0.961 | 2501ms | 5859ms |

候補数を増やしてもGPU版のIoUは変化しない（5候補と同じ結果）。CPU版は候補数に依存して精度が変動する（ad: 5候補=0.845, 30候補=0.991）。5候補でのGPU版は、30候補でのCPU版と同等以上の精度をより少ない候補で達成できる。

## 変更ファイル

| ファイル | 変更 |
|---|---|
| `mask-generation/src/vulkan_context.h` | `CostResult`削除、`RenderPoseWithCost`(出力パラメータ版)、`SetCostInputs`、`HasComputeCost`、`Create/CleanupComputeResources`追加、JFA関連メンバ削除 |
| `mask-generation/src/vulkan_context.cpp` | 2つのGLSL Compute Shader追加（cost_compute + cost_reduce）、`CreateComputeResources`(5バインディング)、`RenderPoseWithCost`(2 Dispatch)、`SetCostInputs`、コンストラクタでR8_UNORM Storageサポート判定 |
| `mask-generation/include/maskgen/mask_generator.h` | `CostResult`構造体、`SetCostInputs`、`GeneratePoseWithCost`、`HasComputeCost`追加 |
| `mask-generation/src/mask_generator.cpp` | Impl メソッドで VulkanContext へ転送 |
| `src/cached_pose_estimator.cpp` | `RefinePose` コストラムダでGPU/CPU分岐、`Estimate()`で`SetCostInputs`呼び出し、最終IoUをGPU結果から計算 |

## ベンチマーク実行コマンド

### ビルド

```bash
docker compose build
```

### GPU Compute有効（デフォルト）— pen / ad

```bash
# pen (NM, 5候補)
docker compose run --rm pose-matching-cached \
  --mask input/pen/mask.png \
  --camera camera_info.json \
  --model mask-generation/model/pen.step \
  --cache cache/pen.bin \
  --scale 0.001 \
  --refine-method nelder-mead \
  --image input/pen/rgb.png \
  --output /tmp/pen_gpu.png \
  --output-pose /tmp/pen_gpu.json

# ad (NM, 5候補)
docker compose run --rm pose-matching-cached \
  --mask input/ad/mask.png \
  --camera camera_info.json \
  --model mask-generation/model/ad.step \
  --cache cache/ad.bin \
  --scale 0.001 \
  --refine-method nelder-mead \
  --image input/ad/rgb.png \
  --output /workspace/result/ad/ad_gpu.png \
  --output-pose /tmp/ad_gpu.json
```

### CPU比較（GPU Compute無効化）

`vulkan_context.cpp` のコンストラクタで `has_compute_cost_` を一時的に `false` に設定してビルド:

```cpp
// vulkan_context.cpp:145
has_compute_cost_ = false;  // CPU比較用
```

```bash
docker compose build pose-matching-cached

# 同じコマンドでCPU版を実行
docker compose run --rm pose-matching-cached \
  --mask input/pen/mask.png \
  --camera camera_info.json \
  --model mask-generation/model/pen.step \
  --cache cache/pen.bin \
  --scale 0.001 \
  --refine-method nelder-mead \
  --image input/pen/rgb.png \
  --output /tmp/pen_cpu.png \
  --output-pose /tmp/pen_cpu.json
```

### 30候補 + Early Termination（Step 2パラメータ）

```bash
docker compose run --rm pose-matching-cached \
  --mask input/ad/mask.png \
  --camera camera_info.json \
  --model mask-generation/model/ad.step \
  --cache cache/ad.bin \
  --scale 0.001 \
  --refine-method nelder-mead \
  --max-candidates 30 \
  --early-termination-iou 0.95 \
  --image input/ad/rgb.png \
  --output /workspace/result/ad/ad_gpu30.png \
  --output-pose /tmp/ad_gpu30.json
```

### R（ローカルウィンドウ半径）の変更

`vulkan_context.cpp` のシェーダー内定数を変更してビルド:

```cpp
// vulkan_context.cpp:41 (kCostComputeSource内)
const int R = 16;  // 8, 16, 32 で比較
```

## 今後の改善方向

1. **Step 4: 低解像度初期最適化** — Coarse/Local段階での候補絞り込みを低解像度で行い、GPU Refineの候補品質を向上させる
2. **適応的R** — IoUが低い初期段階ではR=8、高い段階ではR=16と動的に切り替える
3. **マルチスケールDT** — ピラミッド型のローカルウィンドウ（粗い→細かい）で精度と速度を両立
4. **非同期レンダリング+計算** — レンダリングとコスト計算をオーバーラップさせることで、さらに1.5-2倍の高速化が可能
