# Mask-based 2D-3D Pose Matching

セグメンテーションマスクと3Dモデルのレンダリングを比較して、既知物体の6-DOF姿勢を推定するパイプライン。

## 前提

- Docker & Docker Compose
- GPU (Vulkan対応、`/dev/dri`が利用可能であること)

## 入力ファイル

| ファイル | 説明 | フォーマット |
|---|---|---|
| `mask.png` | セグメンテーションマスク (白=物体) | 8-bit grayscale PNG |
| `input.png` | カラー画像 (可視化用、任意) | RGB PNG |
| `camera_info.json` | カメラ内部パラメータ | JSON |
| `model.step` | 3Dモデル | STEP/PLY/OBJ/STL |

### camera_info.json の形式

```json
{
  "width": 848,
  "height": 480,
  "fx": 434.225,
  "fy": 433.209,
  "cx": 423.769,
  "cy": 238.928
}
```

### correct_pose.json の形式 (verify-pose用)

```json
{
  "tx": 0.023,
  "ty": -0.01,
  "tz": 0.157,
  "rx_deg": 162.68,
  "ry_deg": -15.75,
  "rz_deg": 274.85
}
```

座標系: OpenCV (X=右, Y=下, Z=前方), 回転: ZYX (R = Rz \* Ry \* Rx)

## ビルド

```bash
docker compose build
```

## 実行

### 姿勢推定 (pose-matching)

```bash
docker compose run --rm pose-matching
```

デフォルト引数は `docker-compose.yml` で定義されています。

#### オプション

```
--mask PATH          セグメンテーションマスク画像 (必須)
--camera PATH        カメラ内部パラメータJSON (必須)
--model PATH         3Dモデルファイル (必須)
--scale FLOAT        モデルスケール (mm→mなら0.001)
--image PATH         オーバーレイ表示用カラー画像
--output PATH        出力可視化画像 (デフォルト: result.png)
--output-pose PATH   出力姿勢JSON (デフォルト: pose_result.json)
--depth-min FLOAT    探索深度最小 [m] (デフォルト: 0.05)
--depth-max FLOAT    探索深度最大 [m] (デフォルト: 0.50)
--directions INT     方向サンプル数 (デフォルト: 48)
--in-plane INT       面内回転サンプル数 (デフォルト: 4)
--depth-steps INT    深度サンプル数 (デフォルト: 5)
--top-k INT          coarse候補数 (デフォルト: 10)
--max-refine INT     Nelder-Mead最大反復 (デフォルト: 200)
--rerun [DIR]        Rerun可視化を有効化、.rrdファイルをDIRに保存
```

#### 引数を変更する例

```bash
# 方向サンプル数を増やして精度向上
docker compose run --rm pose-matching \
  --directions 96 --in-plane 8 --max-refine 400

# Rerun可視化なし
docker compose run --rm pose-matching \
  --mask /workspace/mask.png \
  --camera /workspace/camera_info.json \
  --model /workspace/mask-generation/model/pen.step \
  --scale 0.001
```

### 姿勢検証 (verify-pose)

指定した姿勢でマスクをレンダリングし、rerun viewerにストリーミング表示します。

```bash
# 事前にrerun viewerを起動しておく
rerun --grpc 0.0.0.0:9876/proxy

# 別ターミナルで実行
docker compose run --rm verify-pose
```

verify-poseは以下を出力します:
- `verify_overlay.png`: レンダリングマスク(緑)と入力マスク(赤)のオーバーレイ画像
- コンソールにIoU、マスクピクセル数
- rerun viewerに3Dメッシュ + カメラ + 画像

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

## 出力

### pose_result.json

```json
{
  "iou": 0.966,
  "tx": 0.029,
  "ty": -0.013,
  "tz": 0.198,
  "rx": -3.09,
  "ry": -0.719,
  "rz": -1.632,
  "rx_deg": -177.05,
  "ry_deg": -41.20,
  "rz_deg": -93.51
}
```

### result.png

入力カラー画像に推定姿勢のレンダリングマスク(緑)をオーバーレイした画像。

### Rerun (.rrd)

`--rerun`指定時にタイムスタンプ付きファイル名 (`YYYYMMDD_HHMMSS.rrd`) で保存。
coarse候補、refine履歴、最終結果の3Dメッシュ + マスク画像を含む。
