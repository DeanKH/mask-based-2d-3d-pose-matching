# 候補姿勢の制約 (Pose Constraints)

推定時間を短縮するため、候補姿勢の範囲を事前知識で制限する機能。

対応する注意点・設計は以下の通り:

| 要件 | 実装 |
|---|---|
| Z正方向の制約 | `--axis-constraint` (軸方向円錐制約) |
| モデル原点相対の制約 | `--tx-range` / `--ty-range` (原点位置の箱制約) + `--nominal-position` (ノミナル位置グリッドサンプリング) |
| 推定深度の自動追加 | 従来どおりマスク面積から推定した深度をノミナル箱内に自動追加 |
| フォールバック | 最終IoU < 閾値なら制約なしで再試行、`--no-fallback` で無効化可 |

## CLI オプション (案1: 個別フラグ)

すべての実行ファイル (`pose_matching` / `pose_matching_cached` / `benchmark`) で指定可能。
`pose_cache_builder` は `--axis-constraint` のみ (キャッシュに焼き込むため)。

```bash
# 軸方向制約: モデルの+Z軸がカメラ座標 (0,0,-1) 方向 (カメラに向く) から30°以内
--axis-constraint "z:0,0,-1:30"

# 複数指定可 (例: +Zは手前向き、+Yは下向き)
--axis-constraint "z:0,0,-1:30" --axis-constraint "y:0,-1,0:45"

# モデル原点位置 (tx, ty) の許容範囲 [m] (カメラ座標系)
--tx-range "-0.05:0.05"
--ty-range "-0.10:0.10"

# ノミナル位置モード: 原点位置をマスク重心からでなくノミナル位置±グリッドからサンプル
--nominal-position "0.02,-0.01,0.15"   # 原点のノミナル位置 X,Y,Z [m]
--nominal-tol "0.02,0.02,0.03"          # グリッド半径 (デフォルト 0.02,0.02,0.03)
--nominal-grid 3                        # tx/ty それぞれのサンプル数 (デフォルト 3 = -tol,0,+tol)

# フォールバック
--fallback-iou 0.70   # 最終IoUがこの値未満なら制約なしで再試行 (デフォルト 0.70)
--no-fallback         # フォールバック無効化
```

tz (原点Z) の範囲は既存の `--depth-min` / `--depth-max` (重心深度の絶対範囲) が担う。

## 指定形式

| オプション | 形式 | 例 |
|---|---|---|
| `--axis-constraint` | `<モデル軸>:<参照方向x,y,z>:<最大角deg>` (最大4個) | `z:0,0,-1:30` |
| `--tx-range` / `--ty-range` | `<min>:<max>` | `-0.05:0.05` |
| `--nominal-position` / `--nominal-tol` | `<X,Y,Z>` (m) | `0.02,-0.01,0.15` |
| `--nominal-grid` | 整数 (>=1) | `3` |

## 動作

### CoarseSearch

1. 各 (Fibonacci方向 × 面内回転) について軸制約を判定 (深度・並進サンプリング前に弾く)
2. **非ノミナルモード**: tx/ty は従来どおりマスク重心投影から決定し、tx/ty範囲外の候補を除去
3. **ノミナルモード**: tx/ty をノミナル位置 ± tol のグリッド (`nominal_grid`² 点) から直接サンプル。
   原点Zは `nominal_tz ± nominal_tol_tz` を `num_depth` 段階でサンプル (重心深度に変換後
   `depth_min`/`depth_max` でクリップ)
4. マスク面積から推定した重心深度 (既存の自動追加) は、ノミナルモードでは原点Zが箱内に
   収まる場合のみ方向ごとに追加される

### LocalSearch

- 軸制約外の回転 (反転方向を含む) はレンダリング/評価前に除去
- ノミナルモードでは coarse 候補の原点位置を引き継ぎ、回転・深度のみ探索
  (従来から LocalSearch は tx/ty を独立には探索していない)

### Refinement (Nelder-Mead / BOBYQA)

- 軸制約違反の姿勢はコスト `1e6` のペナルティで排除
- BOBYQA では tx/ty 範囲を NLopt のネイティボックス制約 (`nlopt_set_lower/upper_bounds`) として渡す
- **LM/GN (`refine_lm`) は制約を強制しない** (初期候補は既に制約内で、LMは局所収束するため)

### キャッシュとの関係

キャッシュに焼き込まれるのは **軸制約のみ**:

- `pose_cache_builder --axis-constraint "z:0,0,-1:30"` で制約円錐外の回転を
  レンダリングしない (エントリ自体は `mask_size=0` で残すためインデックス構造は不変)。
  これによりキャッシュ容量と生成時間が大幅に減る
  (30°円錅 ≈ 球面の6.7% → coarseマスク数は約1/15)
- tx/ty範囲・ノミナル位置はマスクのピクセルシフトで実行時に吸収されるため
  キャッシュには記録しない。**無制約キャッシュ (=v1または軸制約なしv2) があれば
  制約付き実行時に実行時フィルタで対応できる**
- キャッシュヘッダは version 2 になり軸制約を記録。v1キャッシュも引き続き読める
  (無制約扱い)
- 実行時にキャッシュのカバー範囲を検証:
  - 要求する軸制約円錐がキャッシュの円錐に包含されない → エラー
  - 要求する深度範囲 (`--depth-min`/`--depth-max`、`pose_matching_cached`/`benchmark` も対応)
    がキャッシュ生成時の範囲を超える → エラー。範囲が狭い場合はキャッシュエントリを
    実行時にフィルタして評価対象を減らす

### フォールバック

`Estimate()` の最後に制約付き結果のIoUを判定:

```
最終IoU < fallback_iou (デフォルト 0.70)
  → 制約を除去して再推定し、良い方を採用
```

- coarse候補が制約で全滅した場合 (IoU=0) もこのパスで回収される
- `pose-matching` (非キャッシュ) は常に再試行可能
- `pose_matching_cached` / `benchmark` は **キャッシュが無制約で生成されている場合のみ**
  再試行可能。軸制約付きキャッシュでは再試行せず警告を出す
  (無制約キャッシュを作り直せば有効化できる)

## 速度への効果見積り

| 制約 | coarse候補数の削減 |
|---|---|
| 軸円錅30° | 約1/15 (立体角比 6.7%) |
| 軸円錅45° | 約1/7 (14.6%) |
| Local Search の反転 (flip) | 軸制約でほぼ全滅 → 約1/2 |
| キャッシュ | coarse 960→約130枚、local 41,472→約3,000枚 (maskデータ ~130MB→~15MB) |

## 使用例

```bash
# ペンがカメラに先端を向けていることが分かっている場合
docker compose run --rm pose-matching-cached \
  --axis-constraint "z:0,0,-1:40" \
  --depth-min 0.10 --depth-max 0.30

# 置き場所がほぼ分かっている場合 (ノミナル位置モード)
docker compose run --rm pose-matching-cached \
  --nominal-position "0.02,-0.01,0.15" \
  --nominal-tol "0.02,0.02,0.03" --nominal-grid 3 \
  --no-fallback

# 制約付きキャッシュの生成 (マスクが球面の一部だけでよいので高速・小容量)
docker compose run --rm cache-builder \
  --axis-constraint "z:0,0,-1:30" \
  --output /workspace/cache/pen_constrained.bin
```

## 既知の制限・今後の拡張

- 制約角が狭い場合、全域Fibonacci→フィルタ方式では残る方向サンプルが疧になる。
  効果確認後、LocalSearch の円錅サンプリングコードを再利用して円錅内を直接
  均等サンプリングする方式 (案4の第2段階) に置換すると同候補数で密度が上がる
- LM/GN 精密化では制約を強制しない
- `--nominal-grid` を大きくすると coarse の評価数は grid² 倍に増える
  (非キャッシュ版はレンダリングが増えるため注意。キャッシュ版はシフト評価のみで安価)
