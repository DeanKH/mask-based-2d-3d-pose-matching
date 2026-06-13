# Step 2: BOBYQAへの最適化手法変更

## 概要

Refine段階の最適化アルゴリズムを Nelder-Mead から BOBYQA (Bound Optimization BY Quadratic Approximation) に変更した。

[NLopt](https://github.com/stevengj/nlopt) ライブラリの `LN_BOBYQA` を使用し、`src/bobyqa.h` にヘッダオンリーラッパーを実装した。

## 変更内容

### 新規ファイル

- `src/bobyqa.h` — NLoptの `LN_BOBYQA` をラップするヘッダオンリー実装
  - `NelderMead()` と同じインターフェース（`std::function` コスト関数、初期ステップ、最大反復）
  - abort フラグ対応（`nlopt_force_stop` + コールバック内例外捕捉）
  - C API（`nlopt.h`）を使用し、C/C++境界越えの例外送出を防止

### 変更ファイル

| ファイル | 変更 |
|---|---|
| `Dockerfile` | `libnlopt-dev` 追加 |
| `CMakeLists.txt` | `find_library(NLOPT_LIBRARY nlopt)` を追加、`pose_matching`/`pose_matching_cached` にリンク |
| `src/pose_estimator.h` | `RefineMethod::BOBYQA` 追加、`BobyqaOptions` 構造体、`EstimationParams` に `bobyqa_options` フィールド、`RefinePose` シグネチャ更新 |
| `src/cached_pose_estimator.h` | `bobyqa.h` インクルード、`RefinePose` シグネチャ更新 |
| `src/cached_pose_estimator.cpp` | `RefinePose` 内で NM/BOBYQA を分岐、`Estimate` で BOBYQA を NM と同じ並列ブロックに統合 |
| `src/cached_main.cpp` | `--refine-method bobyqa` 追加、`--xatol`/`--fatol` を BOBYQA にもマップ |
| `src/pose_estimator.cpp` | 非キャッシュ版 `RefinePose` にも BOBYQA 対応 |

### 使い方

```bash
docker compose run --rm pose-matching-cached \
  --refine-method bobyqa \
  [その他オプション]
```

`--xatol` / `--fatol` は Nelder-Mead と BOBYQA の両方に適用される。

## アーキテクチャ

### BOBYQA vs Nelder-Mead

| 特徴 | Nelder-Mead | BOBYQA |
|---|---|---|
| 戦略 | シンプレックス（n+1頂点）の反射・膨張・収縮 | 二次モデル補間による信頼領域法 |
| 初期評価回数 | n+1 (=7 for 6D) | 2n+1 (=13 for 6D) |
| 収束特性 | 線形〜超線形 | 超線形（二次モデル近似） |
| 評価回数のばらつき | 大きい（20〜170回） | 小さい（90〜170回） |
| 微分 | 不要 | 不要（コスト関数そのまま流用） |

### コールバック設計

NLoptのコールバックはC境界を越えるため、C++例外をそのまま伝播できない。`bobyqa_callback` 内で例外を捕捉し、`nlopt_force_stop()` で安全に最適化を中断する:

```cpp
inline double bobyqa_callback(unsigned n, const double* x,
                              double* grad, void* data) {
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
```

## ベンチマーク結果

### テスト条件

- 解像度: 848x480
- 候補数: 30（`--max-candidates 30`）
- 最大反復: 200（`--max-refine 200`）
- `--xatol 1e-3`、`--fatol 1e-4`
- Early termination IoU: 0.95
- スレッド数: 14
- ソートメトリック: centroid_iou

### pen オブジェクト

| 指標 | Nelder-Mead | BOBYQA | 差 |
|---|---|---|---|
| 最終IoU | 0.965226 | 0.958777 | -0.7% |
| RefinePose合計時間 | 2417ms | 1918ms | **-21%** |
| 総評価回数 | ~2470 | ~1380 | **-44%** |
| 最良候補(#4)の評価回数 | 165 | 98 | -41% |
| 最良候補(#4)の時間 | 1338ms | 791ms | -41% |

### ad オブジェクト

| 指標 | Nelder-Mead | BOBYQA | 差 |
|---|---|---|---|
| 最終IoU | 0.991322 | 0.993642 | **+0.2%** |
| RefinePose合計時間 | 2501ms | 2721ms | +9% |
| 総評価回数 | ~2610 | ~2751 | +5% |
| 最良候補(#14)の評価回数 | 162 | 167 | +3% |

### 考察

- **pen（細長い軸対称形状）**: BOBYQAが評価回数を大幅に削減し高速化。IoUは0.7%低下したが0.95閾値を超える。軸対称物体は回転方向の自由度が大きく、BOBYQAの二次モデルが早めに収束判定を出した可能性がある。
- **ad（平坦形状）**: BOBYQAがIoUをわずかに向上。評価回数・時間はNMと同等。平坦な物体は最適化空間が滑らかで、BOBYQAの二次近似が有効に働いたと考えられる。
- **early termination時の中断コスト**: BOBYQAは初期モデル構築（13評価）のみで中断可能。NMは最悪170評価後に中断する場合があり、並列実行時の無駄が大きい。

## 精度向上に寄与しそうなパラメータ

### `--xatol`（パラメータ相対許容誤差）

BOBYQAの `xtol_rel` に対応。パラメータベクトルの相対変化がこの値未満になると収束と判定する。

- デフォルト: `1e-3`（NMの `xatol` と共有）
- **精度向上**: `1e-4` 以下に設定すると収束が遅くなるが、最終IoUが向上する可能性がある。特にpenオブジェクトでの0.7%低下を回復できる可能性が高い。
- 注意: BOBYQAは相対許容誤差のため、パラメータのスケールに依存する。回転（rad）と並進（m）でスケールが異なるため、`1e-3`でも回転方向では約0.06°の精度を意味する。

### `--fatol`（関数値相対許容誤差）

BOBYQAの `ftol_rel` に対応。コスト関数値の相対変化がこの値未満になると収束と判定する。

- デフォルト: `1e-4`（NMの `fatol` と共有）
- **精度向上**: `1e-5` 以下に設定すると、より小さな改善も追跡するようになる。ただし、コスト関数（Chamfer距離 + 面積比）はノイズを含むため、小さくしすぎると無駄な評価が増える。
- 注意: 現在のデフォルトでは `ftol_rel = 1e-4` だが、`BobyqaOptions` の本来のデフォルトは `0.0`（無効）。`--fatol` で上書きしない場合、関数値判定は行われない。

### `--max-refine`（最大評価回数）

BOBYQAの `maxeval` に対応。

- デフォルト: 200
- **精度向上**: 300〜500に増やすことで、より多くの信頼領域ステップを許可し、最終IoUが向上する。BOBYQAは1ステップあたり1評価のみ消費するため、200評価 = 初期モデル13 + 最適化ステップ187回。
- ペイオフ: pen候補#4では98評価で収束したため、200でも十分余裕がある。ad候補#14では167評価を使用しており、200の上限に近い。

### 初期ステップサイズ（`initial_step`、コード内固定）

現在のコードでは `RefinePose` 内で固定値を使用:

```cpp
std::vector<double> initial_step = {
    0.02, 0.02, 0.03,   // tx, ty, tz (m)
    0.3, 0.3, 0.3,      // rx, ry, rz (rad)
};
```

BOBYQAでは `nlopt_set_initial_step()` で設定される。この値が探索範囲を決定する。

- **精度向上**: ステップを小さくすると局所的な精密化が進むが、大域最適解を見逃すリスクがある。逆に大きくすると探索範囲が広がるが収束が遅くなる。
- 物体ごとに最適なステップサイズが異なる可能性がある。pen（細長い）は回転方向のステップを小さく、ad（平坦）は並進方向のステップを小さくすると効果的かもしれない。

### 推奨されるチューニング方針

| パラメータ | 現在値 | 精度重視 | 速度重視 |
|---|---|---|---|
| `--xatol` | 1e-3 | 1e-4 | 1e-3 |
| `--fatol` | 1e-4 | 1e-6 | 1e-3 |
| `--max-refine` | 200 | 300 | 150 |

penオブジェクトの精度をNM水準に引き上げるには `--xatol 1e-4` を試すことを推奨する。
