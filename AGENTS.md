# AGENTS.md

## プロジェクト概要

Mask-based 2D-3D Pose Matching — セグメンテーションマスクと3Dモデルのレンダリングを比較して、既知物体の6-DOF姿勢を推定するC++パイプライン。

## ディレクトリ構造

```
.
├── CMakeLists.txt          # ルートCMake (6つの実行ファイル + cache_lib ライブラリ)
├── Dockerfile              # Ubuntu 24.04 + Vulkan + OCCT + GTSAM (ソースビルド、ENABLE_PROFILER=ON)
├── docker-compose.yml      # 6サービス定義
├── main.py, pyproject.toml, uv.lock  # ルートPythonプロジェクト (uv管理)
├── run_benchmark_sweep.sh  # ベンチマークパラメータスイープ実行スクリプト
├── src/                    # メインのC++ソース
│   ├── main.cpp                       # pose_matching (Coarse+Local+Refine)
│   ├── cached_main.cpp                # pose_matching_cached (キャッシュ利用版、ENABLE_PROFILER対応)
│   ├── benchmark_main.cpp             # benchmark (データセット一括評価)
│   ├── cached_pose_estimator.{h,cpp}  # キャッシュ利用推定器 (CachedCoarse/LocalSearch)
│   ├── cache_builder_main.cpp         # pose_cache_builder (マルチスレッド生成)
│   ├── cache_format.{h,cpp}           # キャッシュのシリアライズ (zlib圧縮 + SHA256ハッシュ、mmap読込対応)
│   ├── contour_sampler.{h,cpp}        # マスク輪郭の3Dサンプリング
│   ├── refine_lm.{h,cpp}              # GTSAMベース LM/GaussNewton 精密化 (RefinePoseLM)
│   ├── edge_feature.{h,cpp}           # エッジベース評価 (Canny + DT, --use-edges)
│   ├── nelder_mead.h                  # ヘッダオンリー Nelder-Mead
│   ├── bobyqa.h                       # ヘッダオンリー NLopt BOBYQA ラッパ (abort対応)
│   ├── pose_estimator.{h,cpp}         # 基本推定器 (RefineMethod/CandidateSortMetric enum定義)
│   ├── verify_pose.cpp                # 姿勢検証 (+ rerun streaming)
│   ├── visualize_poses_main.cpp       # 複数姿勢の3D可視化
│   ├── rerun_visualizer.{h,cpp}       # Rerun SDK による Visualizer 実装
│   ├── profiling.h                    # RefineProfiler / ScopedTimer (cost評価ごとの内訳)
│   └── visualizer.h                   # Visualizer 抽象基底クラス
├── mask-generation/        # サブプロジェクト (maskgenライブラリ)
│   ├── include/maskgen/    # camera.h, mask_generator.h, mesh.h
│   ├── src/                # Vulkanベースのメッシュレンダラ
│   ├── app/main.cpp        # 単体マスク生成アプリ
│   ├── tests/              # Catch2 v3.8.0 単体テスト (FetchContent)
│   ├── cmake/              # maskgenConfig.cmake.in パッケージ設定
│   ├── model/              # 3Dモデル (pen.step など)
│   ├── shaders/            # Vulkanシェーダ
│   └── CMakeLists.txt      # OpenCASCADE/Vulkan/glm 依存
├── benchmark-viewer/       # ベンチマーク結果ビューア (Python, Flask)
├── interactive-3d-annotation/  # Pythonアノテーションツール (uv管理)
├── sam3-scripts/           # SAM3セグメンテーションスクリプト群
├── docs/                   # cache/fast1/fast2/method/speed/param/proc_time/memo/next/oss.md
├── doc/                    # step2.md, step3.md
├── test/data/              # ベンチマーク用データセット
├── record/                 # 過去のRerun録画 (.rrd)
└── 各種データファイル (mask.png, input.png, camera_info.json, pose_result*.json, *.rrd, cpu.prof)
```

## ビルド・実行環境

- **コンテナ**: Docker (Ubuntu 24.04)
- **GPU**: Vulkan (`/dev/dri`必要)、NVIDIA GPU サービス (`pose-matching-cached`/`benchmark` で `deploy.resources` 指定)
- **言語規格**: C++17, CMake 3.20+
- **ビルドタイプ**: Release
- **主要依存ライブラリ**:
  - OpenCV
  - OpenCASCADE / OCCT (STEPファイル読み込み)
  - Vulkan / glm / shaderc (GPUレンダリング)
  - GTSAM (Dockerfile内でソースビルド、Levenberg-Marquardt用、`GTSAM_USE_SYSTEM_EIGEN=ON`)
  - Eigen3 (直接依存、GTSAMと共有)
  - **NLopt** (`libnlopt-dev`、BOBYQA精密化用、`bobyqa.h` で使用)
  - TBB / Boost (GTSAM依存としてシステムインストール)
  - OpenMP (`pose_matching_cached`/`benchmark` で並列化)
  - nlohmann_json
  - Rerun SDK 0.33.0 (FetchContentで自動取得)
  - gperftools (CPUプロファイラ、`ENABLE_PROFILER=ON` で有効化。DockerfileではデフォルトON)
  - zlib / OpenSSL (キャッシュの圧縮・SHA256ハッシュ)

## ビルドコマンド

```bash
docker compose build
```

Dockerfile内で `/opt/build` にビルドされ、以下の6つの実行ファイルが生成される。`cache_lib` は `pose_cache_builder`/`pose_matching_cached`/`benchmark` から静的リンクされる。

| 実行ファイル | エントリポイント | 備考 |
|---|---|---|
| `pose_matching` | `main.cpp` | デフォルトENTRYPOINT。NLoptリンク |
| `verify_pose` | `verify_pose.cpp` | `verify-pose`サービス |
| `pose_cache_builder` | `cache_builder_main.cpp` | maskgen + cache_lib |
| `pose_matching_cached` | `cached_main.cpp` | GTSAM/OpenMP/NLopt/profiler対応 |
| `benchmark` | `benchmark_main.cpp` | GTSAM/OpenMP/NLopt/profiler対応 |
| `visualize_poses` | `visualize_poses_main.cpp` | Rerun 3D描画 |

## 実行コマンド

```bash
# 姿勢推定 (Coarse → Local → Refine)
docker compose run --rm pose-matching

# キャッシュ事前生成 (マルチスレッド、--threads指定)
docker compose run --rm cache-builder

# キャッシュ使った高速姿勢推定 (Nelder-Mead/LM/GaussNewton/BOBYQA 選択可)
# 環境変数 OBJECT_NAME / SORT_METRIC / REFINE_METHOD で切り替え
docker compose run --rm pose-matching-cached

# データセット一括ベンチマーク (結果は benchmark/ に出力)
docker compose run --rm benchmark

# 姿勢検証 (要: rerun --grpc 0.0.0.0:9876/proxy)
docker compose run --rm verify-pose

# 複数姿勢の3D可視化
docker compose run --rm visualize-poses
```

ワークスペース全体が `/workspace` にマウントされ、入出力(マスク、JSON、`.rrd`)はホスト側と共有される。

## Lint / Typecheck

C++プロジェクトのため、Docker内でビルドが成功することで検証とする。`mask-generation/.clang-format` および `mask-generation/.clang-tidy` が設定済み。`mask-generation/tests/` に Catch2 単体テストがあり、`mask-generation` ビルド時に `enable_testing()` される。

## パイプライン構成

3段階のパイプライン。`pose_estimator.h` の `EstimationParams` で各種パラメータ調整可能。精密化手法は `RefineMethod` enum で切り替え（`NelderMead` / `LevenbergMarquardt` / `GaussNewton` / `BOBYQA`）、コマンドライン `--refine-method` で指定。

```
Stage 1: Coarse Search
  Fibonacci球面上の方向サンプル × 面内回転 × 深度
  IoUベースの軽量評価で候補を絞る
  候補のソート指標は CandidateSortMetric で切替可
  (IoU/HuMoments/ZernikeMoments/CentroidIoU/CentroidDT/DT_IoU/
   CentralMoments/ShapeContext/ContourChamfer/FourierDescriptor 等)
  --axis-constraint / --tx-range / --ty-range / --nominal-position で
  候補姿勢の範囲を制限可能 (docs/constraint.md参照)

Stage 2: Local Search
  各coarse候補の周辺をcone sampling (±30°) + 反転
  双方向Chamfer距離ベースの評価

Stage 3: Refinement
  coarse + local の候補から上位 max_refine_candidates 個を6-DOF精密化
  デフォルト手法: Nelder-Mead (双方向Chamfer距離 + 面積比をコスト)
  選択可能: Levenberg-Marquardt / GaussNewton (GTSAM, refine_lm.cpp)
           BOBYQA (NLopt, bobyqa.h)

オプション: エッジベース評価 (--use-edges, src/edge_feature.{h,cpp})
  入力RGB画像のCannyエッジとレンダリング輪郭の一致をコストに加算
  (Coarse/Local/Refineの各段階に適用、LMはGTSAMのエッジDT因子で追加)
  要 --image、GPU compute cost とは併用不可 (自動でCPU評価に切替)
  docs/method.md の「エッジベース評価」参照
```
