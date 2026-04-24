# NavVLA

## Overview

NavVLA は、`OmniVLA-edge` を用いた移動ロボット向けナビゲーション実装です。  
本リポジトリには、以下の 2 つの機能が含まれます。

- `OmniVLA-edge` のファインチューニング
- ROS 2 ノードによるナビゲーション推論

`OmniVLA` 本体はサブモジュールとして管理され、本リポジトリ側では訓練スクリプト、データセット読み込み、ROS 2 ノード、設定ファイルを提供します。

データセット構成は `dataset.yaml` の `datasets` ネストに記述し、複数データセットをまとめて学習できます。

## Training

### 1. 事前準備

- `OmniVLA` サブモジュールを取得する
- `OmniVLA-edge` の重みを `training/config/train.yaml` の `weights_path` に配置する
- 学習用データセットを用意し、`training/config/dataset.yaml` を更新する
- TensorBoard を使う場合は `tensorboard` をインストールする

### 2. 設定ファイル

- 学習設定: [training/config/train.yaml](https://github.com/kyo0221/NavVLA/blob/main/training/config/train.yaml)
- モデル設定: [training/config/network.yaml](https://github.com/kyo0221/NavVLA/blob/main/training/config/network.yaml)
- データセット設定: [training/config/dataset.yaml](https://github.com/kyo0221/NavVLA/blob/main/training/config/dataset.yaml)

### 3. 実行コマンド

```bash
python3 train.py \
  --config training/config/train.yaml \
  --network-config training/config/network.yaml \
  --dataset-config training/config/dataset.yaml
```

### 4. TensorBoard

学習時には TensorBoard ログを出力します。  
デフォルトの出力先は `training/runs/tensorboard` です。

```bash
tensorboard --logdir training/runs/tensorboard
```

記録される主な loss:

- `loss/train_total`
- `loss/train/<dataset_name>_train`
- `loss/train_datasets_total`
- `loss/eval/<dataset_name>_test`
- `loss/eval_total`


## Dataset

### 要求ディレクトリ構成

各データセットは以下の形式を想定しています。

```text
<DATA_FOLDER>/
  <traj_name_1>/
    traj_data.pkl
    0.jpg
    1.jpg
    2.jpg
    ...
  <traj_name_2>/
    traj_data.pkl
    0.jpg
    1.jpg
    2.jpg
    ...
```

train / test split 側は `traj_names.txt` を持つディレクトリを指定します。

```text
<SPLIT_FOLDER>/
  traj_names.txt
```

### データセット設定項目

| 項目 | 内容 |
|---|---|
| `data_folder` | 画像と `traj_data.pkl` を含むデータセット本体ディレクトリ |
| `train` | `traj_names.txt` を含む train split ディレクトリ |
| `test` | `traj_names.txt` を含む test split ディレクトリ |
| `end_slack` | 軌道末尾の余裕フレーム数 |
| `goals_per_obs` | 1 観測あたりに何回 goal をサンプリングするか |
| `waypoint_spacing` | waypoint の時間間隔 |
| `modality_id` | 使用モダリティ ID |


## Navigation

### 概要

ROS 2 ノード `navigation` は、画像入力を購読し、`OmniVLA-edge` で推定した軌跡を `nav_msgs/Path` と `geometry_msgs/Twist` に変換して出力します。

起動ファイル:

- [deployment/launch/navigation.launch.py](https://github.com/kyo0221/NavVLA/blob/main/deployment/launch/navigation.launch.py)

設定ファイル:

- [deployment/config/nav.yaml](https://github.com/kyo0221/NavVLA/blob/main/deployment/config/nav.yaml)
- [deployment/config/preprocess.yaml](https://github.com/kyo0221/NavVLA/blob/main/deployment/config/preprocess.yaml)

### 起動方法

```bash
ros2 launch navvla navigation.launch.py
```

### Topic 一覧

| Topic | 型 | 方向 | 内容 |
|---|---|---|---|
| `/image_raw` | `sensor_msgs/msg/Image` | Subscribe | 現在観測画像 |
| `/autonomous` | `std_msgs/msg/Bool` | Subscribe | 自律動作の有効 / 無効 |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | Publish | 速度指令 |
| `/path` | `nav_msgs/msg/Path` | Publish | 推定 waypoint 列 |

### 主な `nav.yaml` 項目

| 項目 | 内容 |
|---|---|
| `weights_path` | 推論に使う `OmniVLA-edge` 重み |
| `context_size` | 観測履歴長 |
| `len_traj_pred` | 予測 waypoint 長 |
| `interval_ms` | 推論周期 |
| `modality_id` | 使用モダリティ |
| `metric_waypoint_spacing` | waypoint のメートル換算係数 |
| `goal_pose` | 目標姿勢 |
| `goal_image_path` | ゴール画像パス |
| `lan_prompt` | 言語指示 |
| `linear_max_vel` | 最大並進速度 |
| `angular_max_vel` | 最大角速度 |

### `modality_id` の意味

| ID | 内容 |
|---:|---|
| 0 | satellite only |
| 1 | pose and satellite |
| 2 | satellite and image |
| 3 | all |
| 4 | pose only |
| 5 | pose and image |
| 6 | image only |
| 7 | language only |
| 8 | language and pose |

## ライセンス

本リポジトリの独自実装部分は MIT License を想定しています。  
一方で `OmniVLA/` はサブモジュールとして管理される別プロジェクトであり、`OmniVLA` 側のライセンスに従います。

- 本リポジトリ独自コード: MIT
- `OmniVLA/`: [OmniVLA/LICENSE](https://github.com/open-rdc/OmniVLA/blob/main/LICENSE) に従う
