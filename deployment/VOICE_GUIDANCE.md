# Voice Guidance ノード セットアップ

`pfoe_node` が publish する言語指示トピック `/prompt`（`std_msgs/String`）を購読し、
TTS（text-to-speech）で読み上げるノード。どの地点でどの言語指示が出ているかを
耳で確認できるようにするためのもの。

- ノード本体: `deployment/navvla/voice_guidance.py`
- 実行名: `voice_node`（`setup.py` の `console_scripts` に登録）
- 同じ指示の連呼を防ぐため、テキストが**変化したときだけ**発話する

```
/image_feature → pfoe_node → /prompt (String) → voice_node → 🔊
```

---

## 必要なもの

### Python ライブラリ

```bash
pip install pyttsx3
```

- `pyttsx3` … オフライン TTS。バックエンドに espeak-ng を使う。

### システムパッケージ（apt）

```bash
sudo apt install -y espeak-ng alsa-utils libasound2-plugins pulseaudio-utils
```

| パッケージ | 役割 |
|---|---|
| `espeak-ng` | TTS エンジン本体（音声合成） |
| `alsa-utils` | `aplay`。pyttsx3 が wav 再生に内部で使う |
| `libasound2-plugins` | ALSA → PulseAudio ブリッジ（WSL/WSLg で必要） |
| `pulseaudio-utils` | `paplay`。音声経路の動作確認用 |

> **なぜ aplay と pulse の両方が要るか**
> pyttsx3 は Linux では `aplay`（ALSA）で再生する実装。一方 WSLg のスピーカー出力は
> PulseAudio 経由。そのため ALSA の出力を pulse に橋渡しする設定が必要（下記）。
> 実機（ALSA で直接スピーカーが鳴る環境）では `libasound2-plugins` と `~/.asoundrc` は不要なことが多い。

### WSL / WSLg の音声設定

`aplay`（ALSA）の既定出力を PulseAudio に向けるため、`~/.asoundrc` を作成する。

```
pcm.!default pulse
ctl.!default pulse
```

---

## ビルド

```bash
cd ~/ros2_ws
colcon build --packages-select navvla
source install/setup.bash
```

> `voice_node` を `ros2 run` で呼ぶには `setup.py` の `console_scripts` への登録が必要:
> ```python
> "console_scripts": [
>     "navigation_node = navvla.navigation:main",
>     "voice_node = navvla.voice_guidance:main",
> ],
> ```
> ソースを編集するたびにビルドし直すのが面倒なら `colcon build --symlink-install` を使う。

---

## 実行

### 単体で起動

```bash
ros2 run navvla voice_node
```

`voice_guidance ready` が出れば待機中。

### navigation 一式と一緒に起動

`navigation.launch.py` に voice_node が登録済みなので、launch すれば自動で立ち上がる。

```bash
ros2 launch navvla navigation.launch.py
```

---

## 動作確認

別端末で `/prompt` に手動で publish して読み上げを確認する。

```bash
source install/setup.bash

# 読み上げる
ros2 topic pub -1 /prompt std_msgs/msg/String "data: 'Stop at the white line'"

# 同じ文 → 連呼防止で黙る
ros2 topic pub -1 /prompt std_msgs/msg/String "data: 'Stop at the white line'"

# 別の文 → また読み上げる
ros2 topic pub -1 /prompt std_msgs/msg/String "data: 'Turn left at the corner'"

# ダミー文 → スキップして黙る
ros2 topic pub -1 /prompt std_msgs/msg/String "data: 'No language instruction'"
```

---

## 調整

- **話速**: `voice_guidance.py` の `engine.setProperty("rate", 150)` の数値を変更。
  小さいほどゆっくり（例: 120〜100）。
- **音量**: `engine.setProperty("volume", 1.0)`（0.0〜1.0）。

---

## トラブルシュート（WSL / WSLg）

| 症状 | 原因 | 対処 |
|---|---|---|
| `sh: 1: aplay: not found` | alsa-utils 未インストール | `sudo apt install alsa-utils` |
| `aplay: cannot find card '0'` / `Unknown PCM default` | `~/.asoundrc` 未設定 | 上記 `~/.asoundrc` を作成（要 `libasound2-plugins`） |
| ノードは動くが音が聞こえない | WSLg→Windows 音声転送が切れている（`data_send: send failed`） | Windows の PowerShell で `wsl --shutdown` → WSL 開き直し |

### 音声経路だけを切り分けて確認するコマンド

```bash
# 合成だけ確認（wav が出来れば espeak-ng は正常）
espeak-ng "test" -w /tmp/test.wav

# PulseAudio 経路で鳴るか
paplay /tmp/test.wav

# ALSA(aplay) 経路で鳴るか（pyttsx3 と同じ経路）
aplay /tmp/test.wav

# espeak-ng → pulse 直結
espeak-ng --stdout "test" | paplay
```
