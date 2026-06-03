# ポーズ推定パイプライン

入力: 2Dセグメンテーションマスク + 3Dモデル + カメラ内部パラメータ
出力: 6DoFポーズ (tx, ty, tz, rx, ry, rz)

パイプライン全体の流れ:

```
入力マスク → PreProcessing → CoarseSearch → LocalSearch → RefinePose → 最適ポーズ
```

---

## PreProcessing

入力マスクを探索に使える形式に変換する。

1. **二値化**: 多チャンネルの場合はグレースケール変換後、閾値127で二値化
2. **距離変換 (Distance Transform)**: マスク背景部の距離変換画像 `dt_input` を計算。Chamfer距離の計算で使用する。`cv::distanceTransform(255 - mask, dt, DIST_L2, 5)`

---

## CoarseSearch

3Dモデルの全方向的な粗い探索を行い、IoUスコアで上位候補を絞り込む。

### 方向サンプリング

- **Fibonacci球サンプリング** で単位球面上に `num_directions`(デフォルト48)個の方向ベクトルを均等に配置
- 各方向について **面内回転** を `num_in_plane`(デフォルト4)パターン生成
- 方向ベクトル → 回転行列への変換は `RotationFromDirection` で行う:
  - メッシュの主軸(principal axis)を方向ベクトルにアラインする回転 `R_align` を計算
  - 主軸周りの面内回転 `R_inplane` を適用
  - 最終回転 = `R_align * R_inplane`

### 深度サンプリング

- `depth_min`〜`depth_max` を `num_depth`(デフォルト5)段階で等間隔サンプリング
- マスク面積とメッシュ寸法から **推定深度** を計算し、既存サンプルと近すぎない場合に追加:
  ```
  estimated_depth = sqrt(mesh_area × fx × fy / input_area) × 0.5
  ```

### 並進の決定

各 (方向, 面内回転, 深度) の組合わせについて:
- 回転されたメッシュ重心 `c_rot` を計算
- 入力マスクの重心 `(u_cx, v_cy)` をカメラ座標に投影するよう並進を決定:
  ```
  tx = (u_cx - cx) × tz / fx - c_rot.x
  ty = (v_cy - cy) × tz / fy - c_rot.y
  tz = depth - c_rot.z
  ```
- `tz < 0.01` の候補は除外（カメラより後方）

### 評価

- 各候補ポーズでマスクをレンダリングし、入力マスクとの **IoU** を計算
- コスト = `1.0 - IoU`

### 多様化 (Diversification)

- IoUでソート後、回転を量子化して同じ方向の候補を重複排除
- 上位 `top_k_coarse`(デフォルト10)個を返す

### 候補数

```
num_directions × num_in_plane × num_depth = 48 × 4 × 5 = 960 (デフォルト)
```

---

## LocalSearch

CoarseSearchの上位候補周辺をより密に探索し、Chamfer距離ベースのコストで評価する。

### 探索範囲

- CoarseSearchの上位10候補それぞれについて:
  - 候補の方向ベクトル `cand_dir` を取得
  - `cand_dir` に直交する2つの基底ベクトル `perp1`, `perp2` を計算
  - 円錐内に `local_directions`(デフォルト6)方向をサンプリング:
    - 円錐角度は `local_cone_half_angle_deg`(デフォルト30°)の 0.33倍, 0.67倍, 1.0倍の3段階
    - 各角度で方位角を等間隔にサンプリング
  - 各方向について正負両方向 (`flip = 0, 1`) を生成
  - 各方向について面内回転 `local_in_plane`(デフォルト6)パターン
  - 各組合わせについて深度オフセット `local_depth`(デフォルト3)段階 (範囲 ±0.03m)

### 並進の決定

CoarseSearchと同様に、入力マスク重心を投影するよう並進を計算。

### コスト関数

```
cost = chamfer + 0.5 × area_ratio
```

- **chamfer**: 入力マスクとレンダリングマスクの差分ピクセルに距離変換値を乗算した和を、入力面積の平方根で正規化
  ```
  diff = |rendered - input|
  dt_combined = max(dt_input, dt_rendered)
  chamfer = sum(dt_combined × diff) / sqrt(input_area)
  ```
- **area_ratio**: レンダリング面積と入力面積の相対差
  ```
  area_ratio = |area_rendered - area_input| / area_input
  ```

### 結果

- コストでソートし、上位 `top_k_local`(デフォルト5)個を返す

### 候補数 (1つの粗探索候補あたり)

```
local_directions × 2(flip) × local_in_plane × local_depth = 6 × 2 × 6 × 3 = 216
```

---

## RefinePose

粗探索・局所探索の上位候補をNelder-Mead法で6DoF最適化する。

### 最適化パラメータ

6次元ベクトル: `[tx, ty, tz, rx, ry, rz]`

### 初期ステップサイズ

```
[0.02, 0.02, 0.03, 0.3, 0.3, 0.3]
```

- 並進: 2〜3cm
- 回転: 0.3 rad (約17°)

### コスト関数

LocalSearchと同じ:
```
cost = chamfer + 0.5 × area_ratio
```
- `tz < 0.01` の場合はペナルティ `1e6` を返す

### Nelder-Mead法

- シンプレックス法による勾配不要最適化
- 最大 `nelder_mead_iterations`(デフォルト200)イテレーション
- 収束条件: 関数値の差 < `fatol(1e-8)` かつ全次元のスプレッド < `xatol(1e-6)`
- 各イテレーションの操作: 反射(α=1.0), 拡張(γ=2.0), 収縮(ρ=0.5), 縮小(σ=0.5)

### 候補の選定

- CoarseSearchの結果 + LocalSearchの結果（重複排除済み）をコストでソート
- 上位最大10候補をそれぞれ独立にRefinePose
- 最終的にIoUが最も高い結果を採用

---

## パラメータ一覧

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `num_directions` | 48 | Fibonacci球上の方向サンプル数 |
| `num_in_plane` | 4 | 面内回転のサンプル数 |
| `num_depth` | 5 | 深度サンプル数 |
| `depth_min` | 0.05 m | 探索深度の最小値 |
| `depth_max` | 0.50 m | 探索深度の最大値 |
| `top_k_coarse` | 10 | 粗探索の上位取得数 |
| `nelder_mead_iterations` | 200 | Nelder-Meadの最大イテレーション |
| `local_directions` | 6 | 局所探索の方向サンプル数 |
| `local_in_plane` | 6 | 局所探索の面内回転サンプル数 |
| `local_depth` | 3 | 局所探索の深度サンプル数 |
| `local_cone_half_angle_deg` | 30.0 | 局所探索の円錐半角 |
| `top_k_local` | 5 | 局所探索の上位取得数 |
