# 処理時間計測レポート

測定条件: `mask.png` (848x480), `pen.step` (scale=0.001), 48 directions × 4 in-plane × 5 depths = 960 candidates

## 全体サマリ

総実行時間: **285,191 ms (約4分45秒)**

| 処理ステップ | 時間 (ms) | 割合 |
|---|---|---|
| Model loading (STEPファイル読み込み) | 156 | 0.05% |
| Preprocessing (マスク二値化・距離変換) | 6 | 0.00% |
| CoarseSearch (粗い全方向探索) | 38,286 | 13.4% |
| LocalSearch (局所探索) | 97,682 | 34.3% |
| CorrectPose eval (正解ポーズ評価) | 45 | 0.02% |
| RefinePose (Nelder-Mead最適化 ×10) | 149,084 | 52.3% |
| Visualization output | 148 | 0.05% |

## ボトルネック分析

**RefinePose (52%) > LocalSearch (34%) > CoarseSearch (13%)** の3つで全体の **99%** を占めている。

各処理は多数回のポーズレンダリング（Vulkan）を含んでおり、レンダリングが主な時間消費源と考えられる。

## 各ステップの詳細

### Model loading (156 ms)
- OpenCASCADEによるSTEPファイルの読み込みとメッシュ変換
- メッシュ重心・バウンディングボックスの計算

### Preprocessing (6 ms)
- 入力マスクの二値化
- 距離変換 (distance transform) の計算

### CoarseSearch (38,286 ms)
- Fibonacci球面上での方向サンプリング (48方向)
- 面内回転サンプリング (4パターン)
- 深度サンプリング (5段階)
- 計960候補のレンダリングとIoU評価

### LocalSearch (97,682 ms)
- 上位粗探索結果周辺の局所探索
- 円錐内方向サンプリング + 面内回転 + 深度
- Chamfer距離によるコスト評価

### CorrectPose eval (45 ms)
- correct_pose.jsonが存在する場合の正解ポーズ評価 (デバッグ用)

### RefinePose (149,084 ms)
- Nelder-Mead法による最大10候補 × 200イテレーションの最適化
- 各イテレーションでポーズレンダリング + Chamfer距離計算

### Visualization output (148 ms)
- 結果ポーズのレンダリングとオーバーレイ画像の保存
