# BOBYQA パラメータチューニング

## 背景

Nelder-Mead と BOBYQA のベンチマーク比較において、IoU min (ワーストケース精度) で Nelder-Mead が BOBYQA を上回っていた。

| メトリック | Nelder-Mead best | BOBYQA 従来 |
|---|---|---|
| IoU min | 0.8932 | 0.8382 |
| IoU mean | 0.9701 | 0.9649 |
| 時間 | 6195ms | 5038ms |

BOBYQA は速度面で優位 (約19%高速) だが、ワーストケース精度に課題があった。速度を維持したまま IoU min を Nelder-Mead 水準まで引き上げるため、パラメータチューニングを実施した。

## パラメータ概要

BOBYQA (`NLOPT_LN_BOBYQA`) は二次補間モデルを構築して最適化を行う勾配不要法。6パラメータ (tx, ty, tz, rx, ry, rz) に対して以下のオプションが影響する。

| パラメータ | 内容 | デフォルト |
|---|---|---|
| `xtol_rel` | パラメータ変化による収束判定 | 1e-4 |
| `ftol_rel` | コスト値変化による収束判定 (0=無効) | 0.0 |
| `population` | 補間点数 (0=自動 2n+1=13, 28=完全二次) | 0 |
| `initial_step_scale` | 初期ステップのスケール係数 | **1.3** |

`initial_step_scale` は BOBYQA の初期信頼領域半径を決定し、二次モデルの初期探索範囲に直結する。

## チューニング結果

条件: `centroid_iou`, `max_candidates=30`, `max_refine=300`, `contour_points=4000`, GPU cost evaluation

| 設定 | step_scale | xtol | ftol | iou_min | iou_mean | 時間 |
|---|---|---|---|---|---|---|
| BOBYQA 従来 | 1.0 | 1e-4 | 1e-4 | 0.8382 | 0.9649 | 5038ms |
| 厳格tol | 1.0 | 1e-6 | 0 | 0.8388 | 0.9650 | 5505ms |
| 完全二次 | 1.0 (pop=28) | 1e-6 | 0 | 0.8388 | 0.9650 | 5513ms |
| 小ステップ | 0.5 | 1e-6 | 0 | 0.8156 | 0.9590 | 5537ms |
| 大ステップ | 1.5 | 1e-6 | 0 | 0.8800 | 0.9647 | 5605ms |
| **推奨** | **1.3** | **1e-4** | **1e-4** | **0.8811** | **0.9634** | **5003ms** |
| 厳格+中s | 1.3 | 1e-6 | 0 | 0.8811 | 0.9642 | 5569ms |
| 過大ステップ | 2.0 | 1e-6 | 0 | 0.8685 | 0.9628 | 5568ms |

### Nelder-Mead との比較 (推奨設定)

| メトリック | Nelder-Mead | BOBYQA (推奨) | 差 |
|---|---|---|---|
| IoU min | 0.8932 | 0.8811 | -0.012 |
| IoU mean | 0.9701 | 0.9634 | -0.007 |
| 時間 | 6195ms | 5003ms | **-19%** |

## 知見

### initial_step_scale が最大の影響 (step_scale=1.3 が最適)

BOBYQA の初期ステップを30%拡大することで、二次補間モデルの初期探索範囲が広がり、ローカルミニマに陥りにくくなる。IoU min が 0.8382 → 0.8811 に大幅改善。Nelder-Mead との差距を 0.055 から 0.012 に縮小 (78%改善)。

1.3 より大きくすると (1.5, 2.0) 逆に悪化する。初期ステップが大きすぎると二次モデルの近似精度が落ち、精密な収束が困難になる。

### population (補間点数) は影響なし

完全二次モデル (population=28) とデフォルト (2n+1=13) で結果に差がない。6パラメータでは 2n+1=13 点で十分なモデル精度が得られている。

### tolerances は二次的効果

`xtol_rel` を 1e-4 → 1e-6 に厳格化しても、IoU min は不变。iou_mean がわずかに上昇 (+0.0008) するが、時間が +10% 増加する。コストに見合わないため、従来通り `xtol_rel=1e-4`, `ftol_rel=1e-4` を推奨。

### ステップ縮小は逆効果

step_scale=0.5 では IoU min が 0.8156 に悪化。探索範囲が狭くなり、悪い初期解から脱出できなくなる。

## 変更ファイル

| ファイル | 変更内容 |
|---|---|
| `src/bobyqa.h` | `population`, `initial_step_scale` を `BobyqaOptions` に追加、デフォルト `initial_step_scale=1.3` |
| `src/benchmark_main.cpp` | `--bobyqa-population`, `--bobyqa-step-scale` CLIフラグ追加 |
| `run_benchmark_sweep.sh` | BOBYQA実行時に `--bobyqa-step-scale 1.3` を自動付与 |
