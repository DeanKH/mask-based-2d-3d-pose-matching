# AGENTS.md

## プロジェクト概要

Mask-based 2D-3D Pose Matching — セグメンテーションマスクと3Dモデルのレンダリングを比較して、既知物体の6-DOF姿勢を推定するC++パイプライン。

## ディレクトリ構造

```
.
├── CMakeLists.txt          # ルートCMake (5つの実行ファイルを定義)
├── Dockerfile              # Ubuntu 24.04 + Vulkan + OCCT + GTSAM
├── docker-compose.yml      # 5サービス定義
├── src/                    # メインのC++ソース
│   ├── main.cpp                       # pose_matching (Coarse+Local+Nelder-Mead)
│   ├── cached_main.cpp                # pose_matching_cached (キャッシュ利用版)
│   ├── cached_pose_estimator.{h,cpp}
│   ├── cache_builder_main.cpp         # pose_cache_builder
│   ├── cache_format.{h,cpp}           # キャッシュのシリアライズ (zlib+暗号化)
│   ├── contour_sampler.{h,cpp}        # マスク輪郭サンプリング
│   ├── refine_lm.{h,cpp}              # Levenberg-Marquardt精密化
│   ├── nelder_mead.h                  # ヘッダオンリーNelder-Mead
│   ├── pose_estimator.{h,cpp}
│   ├── verify_pose.cpp                # 姿勢検証 (+ rerun streaming)
│   ├── visualize_poses_main.cpp       # 複数姿勢の3D可視化
│   ├── rerun_visualizer.{h,cpp}
│   ├── profiling.h, visualizer.h
├── mask-generation/        # サブプロジェクト (maskgenライブラリ)
│   ├── include/maskgen/    # camera.h, mask_generator.h, mesh.h
│   ├── src/                # Vulkanベースのメッシュレンダラ
│   ├── app/main.cpp        # 単体マスク生成アプリ
│   ├── model/pen.step      # 3Dモデル
│   ├── shaders/            # Vulkanシェーダ
│   └── CMakeLists.txt      # OpenCASCADE/Vulkan/glm 依存
├── interactive-3d-annotation/  # Pythonアノテーションツール (uv管理)
├── sam3-scripts/           # SAM3セグメンテーションスクリプト群
├── docs/                   # cache.md, method.md, speed.md
├── record/                 # 過去のRerun録画 (.rrd)
└── 各種データファイル (mask.png, input.png, camera_info.json, pose_result*.json, *.rrd)
```

## ビルド・実行環境

- **コンテナ**: Docker (Ubuntu 24.04)
- **GPU**: Vulkan (`/dev/dri`必要)
- **言語規格**: C++17, CMake 3.20+
- **ビルドタイプ**: Release
- **主要依存ライブラリ**:
  - OpenCV
  - OpenCASCADE (STEPファイル読み込み)
  - Vulkan / glm / shaderc (GPUレンダリング)
  - GTSAM (ソースビルド、Levenberg-Marquardt用)
  - Eigen3, TBB, Boost
  - nlohmann_json
  - Rerun SDK 0.33.0 (FetchContentで自動取得)
  - gperftools (CPUプロファイラ、ENABLE_PROFILER=ONで有効化)
  - zlib / OpenSSL (キャッシュの圧縮・暗号化)

## ビルドコマンド

```bash
docker compose build
```

Dockerfile内で `/opt/build` にビルドされ、以下の5つの実行ファイルが生成される:

| 実行ファイル | エントリポイント |
|---|---|
| `pose_matching` | デフォルト |
| `verify_pose` | `/opt/build/verify_pose` |
| `pose_cache_builder` | `/opt/build/pose_cache_builder` |
| `pose_matching_cached` | `/opt/build/pose_matching_cached` |
| `visualize_poses` | `/opt/build/visualize_poses` |

## 実行コマンド

```bash
# 姿勢推定 (Coarse → Local → Nelder-Mead)
docker compose run --rm pose-matching

# キャッシュ事前生成
docker compose run --rm cache-builder

# キャッシュ使った高速姿勢推定 (LM/Nelder-Mead選択可)
docker compose run --rm pose-matching-cached

# 姿勢検証 (要: rerun --grpc 0.0.0.0:9876/proxy)
docker compose run --rm verify-pose

# 複数姿勢の3D可視化
docker compose run --rm visualize-poses
```

ワークスペース全体が `/workspace` にマウントされ、入出力(マスク、JSON、`.rrd`)はホスト側と共有される。

## Lint / Typecheck

C++プロジェクトのため、Docker内でビルドが成功することで検証とする。`mask-generation/.clang-format` および `mask-generation/.clang-tidy` が設定済み。

## パイプライン構成

```
Stage 1: Coarse Search
  Fibonacci球面上の方向サンプル × 面内回転 × 深度
  IoUベースの軽量評価で候補を絞る

Stage 2: Local Search
  各coarse候補の周辺をcone sampling (±30°) + 反転
  双方向Chamfer距離ベースの評価

Stage 3: Nelder-Mead Refinement
  coarse + localの候補から最大10個を6-DOF Nelder-Meadで精密化
  双方向Chamfer距離 + 面積比をコスト関数として使用
```
