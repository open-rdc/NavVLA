# つくば 魚眼 実機実験 (2026-10-17)

言語指示のみの OmniVLA-edge (fisheye_v1) を、トポロジカルマップ上のベイズフィルタ(feat/toponav の `toponav.py`, PlaceNet 特徴)で
推定したノードの指示文で走らせる。トポマップは full1 (つくばのコース全体) から作った 947 ノード(10 フレームおき)。
横断歩道の手前のノード(full1 で止まった 8 か所, 75 ノード)は「Stop at the edge of the road」で止まる。渡るときは人が `/cross` を送る。
pfoe(CLIP) は魚眼のつくばでは位置が追えなかったので使わない(launch の `use_pfoe:=true` で起動はできる)。

## 1. 取得

```bash
cd ~/ros2_ws/src
git clone -b exp/tsukuba-1017 --recurse-submodules git@github.com:kyo0221/NavVLA.git
# 既にある場合
cd ~/ros2_ws/src/NavVLA && git fetch && git switch exp/tsukuba-1017 && git pull && git submodule update --init
```

OmniVLA はサブモジュール。`git submodule update --init` を忘れると `No module named 'OmniVLA.inference'` で落ちる。

## 2. git に無いファイルを外付けSSDから置く

| SSD (`D:\weights\navvla\`) | 置き場所 (リポジトリ内) |
|---|---|
| `fisheye_v1_omnivla/model_best.pth` | `deployment/weights/omnivla-edge_fisheye_v1_best.pth` (名前を変える) |
| `placenet/placenet.pt` | `deployment/weights/placenet.pt` |
| `topomap/tsukuba_toponav/` (topomap.yaml と images/) | `data/tsukuba_toponav/` |
| `clip/ViT-B-32.pt` | `~/.cache/clip/ViT-B-32.pt` (無い場合だけ。現地にネットが無いと初回起動で落ちる) |

`setup.py` は**ビルド時に** `deployment/weights/` と `data/` をインストール先へコピーする。置いてからビルドする。

## 3. ビルド・起動

```bash
cd ~/ros2_ws && colcon build --packages-select navvla && source install/setup.bash
ros2 launch navvla navigation.launch.py
ros2 topic pub --once /autonomous std_msgs/msg/Bool "{data: true}"   # 自律開始
```

- カメラは `/image_raw` (rgb8)。
- 位置推定は自律走行していない間も回る。起動後スタート地点で 20 秒ほど置くと推定が定まる
  (ログの `Toponav node=... score=...`。score 0.4 未満の間は指示を変えない)。
- 起動直後は過去画像が溜まるまで 3.2 秒 (16 周期) 動かない。
- GPU 必須 (OmniVLA のモデルは CPU だと `Device index must not be negative` で落ちる)。
- `No module named 'OmniVLA'` / `'model_omnivla_edge'` が出たら
  `export PYTHONPATH=~/ros2_ws/src/NavVLA:~/ros2_ws/src/NavVLA/OmniVLA/inference:$PYTHONPATH`。

## 4. 横断歩道

止まったら、安全を確認して

```bash
ros2 topic pub --once /cross std_msgs/msg/Empty "{}"
```

停止の指示を「指示なし(=進む)」に置き換える。停止以外の指示が 3 秒続いたら置き換えを解除する
(`nav.yaml` の `cross_release_sec`)。`/debug_image` には実際にモデルへ渡している指示が出る。

## 5. この実験用の設定 (nav.yaml)

| 項目 | 値 | 理由 |
|---|---|---|
| `context_spacing` | 3 | 学習の `waypoint_spacing: 3`。過去画像を 0.6 秒おきに取る |
| `metric_waypoint_spacing` | 0.51 | 学習の正規化 0.17 x 3 |
| `stop_distance` | 0.6 | 3 秒先の予測点が 0.6m 未満なら速度 0。検証走行で通常区間の誤停止 0%、停止区間の 86% で停止 |
| `toponav_window_lower/upper` | 0 / 2 | 1 周期で「その場 or 1 ノード前進」。feat/toponav の [-1,1) は追従しなかった |
| `modality_id` | 7 | 言語のみで学習した重み |

トポマップを作り直すとき: `python3 deployment/scripts/create_topomap.py <データセット> --output-dir data/tsukuba_toponav --saved-step 10 --crop-size 0`
(データセットは `traj_names.txt` と `<軌跡>/{*.jpg,traj_prompt.txt}`。ノードの指示は traj_prompt.txt の同じフレームの行)
