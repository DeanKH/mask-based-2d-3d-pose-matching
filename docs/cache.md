# キャッシュ方針

## 概要

CADモデルとカメラパラメータが固定の環境で、CoarseSearch・LocalSearchのレンダリング結果を事前キャッシュし、ピクセルシフト近似により実行時のレンダリングを省略して高速化する。

## アーキテクチャ

2つの実行ファイルによる2段構成:

```
pose_cache_builder                    pose_matching_cached
  モデル + カメラ + パラメータ           キャッシュ + マスク画像
        ↓                                    ↓
  [キャッシュファイル生成]    →    [シフト近似で高速ポーズ推定]
       1回のみ (~4分)                毎回 (~数秒+RefinePose)
```

| プログラム | 役割 | 実行タイミング |
|---|---|---|
| `pose_cache_builder` | モデル+カメラパラメータからキャッシュファイル生成 | モデル/カメラ変更時のみ（1回） |
| `pose_matching_cached` | キャッシュファイル+入力マスクからポーズ推定 | 毎回 |

## ピクセルシフト近似の原理

現在のCoarseSearch/LocalSearchでは、入力マスクの重心 (u_cx, v_cy) に投影 centroid が一致するよう tx/ty を計算する:

```
tx = (u_cx - cx) × tz / fx - c_rot.x
ty = (v_cy - cy) × tz / fy - c_rot.y
```

この tx/ty は入力に依存するためキャッシュ化できない。

**解決策**: 全マスクを画像中心 (cx, cy) に投影 centroid が一致する位置でレンダリングしておく。実行時は入力マスク重心との差分だけマスクをピクセルシフトする。

```
シフト量: Δu = u_cx - cx,  Δv = v_cy - cy
→ cv::warpAffine(cached_mask, shifted, translation_matrix)
```

### 誤差の見積り

- 透視投影におけるピクセルシフトは centroid depth では正確
- 頂点ごとの深度差による視差誤差は Δz / tz に比例
- ペンの場合: 深度幅 15mm / tz = 200mm → 誤差 ~7.5%
- CoarseSearch (IoUベース) と LocalSearch (Chamferベース) の評価に十分な精度
- RefinePose (Nelder-Mead) で最終的に精密調整されるため近似誤差は最終結果に影響しない

## キャッシュ対象

| 段階 | キャッシュするマスク数 | 内訳 |
|---|---|---|
| CoarseSearch | 960 | 48方向 × 4面内回転 × 5深度 |
| LocalSearch | 41,472 | 192粗方向 × (6方向 × 2反転 × 6面内 × 3深度) |
| RefinePose | なし | Nelder-Meadの反復でポーズが逐次変動するため毎回レンダリング |

## ファイルフォーマット

### 仕様

- 形式: カスタムバイナリ + zlib 圧縮
- マスク解像度: 原解像度 (848×480)
- マスク符号化: 1-bit pack → zlib deflate
- アクセス方式: mmap でオンデマンド読み込み

### バイナリレイアウト

```
┌──────────────────────────────────────┐
│ File Header                          │
│  magic:         "PMCK" (4 bytes)     │
│  version:       uint32               │
│  image width:    uint32              │
│  image height:   uint32              │
│  camera params:  fx, fy, cx, cy      │  4 × float64
│  mesh centroid:  x, y, z             │  3 × float32
│  mesh extent:    x, y, z             │  3 × float32
│  principal_axis:  uint32             │
│  search params:  (all EstimationParams)│
│  model_hash:     SHA-256 (32 bytes)  │ ← キャッシュ整合性確認用
│  num_coarse:     uint32              │
│  num_local:      uint32              │
│  coarse_offset:  uint64              │ ← coarse index table 位置
│  local_offset:   uint64              │ ← local index table 位置
│  data_offset:    uint64              │ ← マスクデータ開始位置
├──────────────────────────────────────┤
│ Coarse Index Table                   │
│  [0]: {                             │ × num_coarse (960)
│    rx, ry, rz:    3 × float64       │   回転 (オイラー角)
│    crot_x, crot_y, crot_z: 3×float32│   回転済み重心
│    depth:         float64            │   深度
│    mask_offset:   uint64             │   マスクデータへのオフセット
│    mask_size:     uint32             │   圧縮済みマスクサイズ
│  }                                   │
│  [1]: ...                            │
├──────────────────────────────────────┤
│ Local Index Table                    │
│  [0]: {                             │ × num_local (41472)
│    coarse_idx:    uint32             │   所属する粗方向のインデックス
│    rx, ry, rz:    3 × float64       │
│    crot_x, crot_y, crot_z: 3×float32│
│    depth:         float64            │
│    mask_offset:   uint64             │
│    mask_size:     uint32             │
│  }                                   │
│  [1]: ...                            │
├──────────────────────────────────────┤
│ Mask Data (compressed)               │
│  [coarse mask 0][coarse mask 1]...   │   各: zlib(1-bit packed)
│  [local mask 0][local mask 1]...     │
└──────────────────────────────────────┘
```

### 1-bit pack 方式

binary mask (0/255) を 8 ピクセル = 1 バイトに pack:

```
元: [255, 0, 0, 255, 255, 0, 0, 0] (8 bytes)
後: [0b10011000]                      (1 byte)
```

zlib でさらに圧縮。オブジェクト面積が小さいほど黒画素が多く高圧縮率となる。

### 整合性確認

キャッシュ読み込み時に model_hash (SHA-256) を元のモデルファイルと照合し、モデル変更後に古いキャッシュが使われるのを防ぐ。

## ストレージ見積り

| 項目 | 計算 | サイズ |
|---|---|---|
| 1 マスク (848×480, 1-bit pack) | 848×480/8 | ~50 KB |
| 960 coarse masks (zlib 圧縮) | ~5 KB/枚 | ~5 MB |
| 41,472 local masks (zlib 圧縮) | ~3 KB/枚 | ~120 MB |
| メタデータ + インデックス | | ~2 MB |
| **合計** | | **~130 MB** |

## 実行時の読み込み戦略

mmap でファイル全体を仮想メモリにマップし、OS がページ単位で必要分だけロードする。

```
[mmap でファイル全体をマップ]
  ↓

CoarseSearch:
  → 960 個の coarse mask を全デコード (~5 MB, 瞬時)
  → 各マスクを (u_cx - cx, v_cy - cy) だけシフト (cv::warpAffine)
  → IoU 評価
  → 上位 10 候補を選出

LocalSearch:
  → top-10 粗方向の coarse_idx を特定
  → その coarse_idx に属する local mask だけデコード
    (全 41,472 のうち ~216 × 10 = 2,160 個、~6 MB)
  → シフト + Chamfer 距離評価

RefinePose:
  → キャッシュ不使用（毎回レンダリング）
  → シフト近似候補のポーズを初期値として Nelder-Mead 最適化
```

## 見積り効果

| ステージ | 現在 | キャッシュ後 | 削減 |
|---|---|---|---|
| CoarseSearch | 38秒 | ~0.5秒 | 99% |
| LocalSearch | 98秒 | ~3秒 | 97% |
| RefinePose | 149秒 | 149秒（変更なし） | 0% |
| **合計** | **285秒** | **~153秒** | **46%** |

### 残りのボトルネック

RefinePose (149秒) は Nelder-Mead の反復でポーズが逐次変動するためキャッシュ不可。さらなる高速化には以下のアプローチが必要:

- GPU バッファ (vertex/index/readback) のキャッシュ再利用
- Nelder-Mead の収束判定の早期化
- 並列コマンドバッファ投入

## 依存ライブラリ

| ライブラリ | 用途 | 状態 |
|---|---|---|
| zlib | マスク圧縮/展開 | Ubuntu runtime 搭載済み、`zlib1g-dev` を Dockerfile に追加 |
| OpenCV | `cv::warpAffine` (シフト), `cv::distanceTransform` | 既存 |

## レンダリング時の正位置での tx/ty 計算

キャッシュ生成時、各候補を画像中心に投影するための tx/ty:

```
tx = (cx - cx) × tz / fx - c_rot.x = -c_rot.x
ty = (cy - cy) × tz / fy - c_rot.y = -c_rot.y
tz = depth - c_rot.z
```

実行時にシフト量から本来の tx/ty を復元:

```
tx = (u_cx - cx) × tz / fx - c_rot.x
ty = (v_cy - cy) × tz / fy - c_rot.y
```

これにより RefinePose の初期値として正確なポーズを渡すことができる。
