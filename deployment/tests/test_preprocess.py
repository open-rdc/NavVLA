"""実機の画像前処理(deployment/navvla/preprocess.py)が学習時の読み込み(training/data/dataset.py)と揃っているかのテスト。

学習データは 640x480 の魚眼画像を切り出さずに 96x96 / 224x224 へ縮小している。
実機側も同じ見え方になっていること(左右を切り落としていないこと)を確かめる。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "OmniVLA" / "inference"))  # utils_policy が model_omnivla_edge を直接 import するため

from deployment.navvla.preprocess import image_to_cv2  # noqa: E402
from training.data.dataset import EdgeNavigationDataset  # noqa: E402

W, H = 640, 480
OBS_SIZE = (96, 96)
CLIP_SIZE = (224, 224)


def _image_msg(rgb: np.ndarray) -> SimpleNamespace:
    """sensor_msgs/Image(rgb8) の代わり。image_to_cv2 が使う属性だけ持たせる。"""
    return SimpleNamespace(data=rgb.tobytes(), height=rgb.shape[0], width=rgb.shape[1])


def _deploy_images(rgb: np.ndarray) -> tuple[Image.Image, Image.Image]:
    """navigation.py の image_callback と同じ手順で (96x96 の観測画像, CLIP 用画像) を作る。"""
    cv_image = image_to_cv2(_image_msg(rgb), CLIP_SIZE)
    obs_image = Image.fromarray(cv_image[:, :, ::-1])
    return obs_image.resize(OBS_SIZE), obs_image.resize(CLIP_SIZE)


def _train_images(jpg: Path) -> tuple[Image.Image, Image.Image]:
    """EdgeNavigationDataset.load_image と同じ読み込み(normalize なし)。"""
    ds = SimpleNamespace(data_folder=jpg.parent.parent, normalize=False)
    to_pil = lambda t: Image.fromarray((t.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
    obs = EdgeNavigationDataset.load_image(ds, jpg.parent.name, int(jpg.stem), OBS_SIZE)
    clip = EdgeNavigationDataset.load_image(ds, jpg.parent.name, int(jpg.stem), CLIP_SIZE)
    return to_pil(obs), to_pil(clip)


def _scene() -> np.ndarray:
    """左右 80px(中央正方形の切り出しで捨てられる幅)に色を付けた 640x480 の画像。"""
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[:, :, 1] = np.linspace(0, 255, W, dtype=np.uint8)[None, :]  # 横方向のグラデーション
    rgb[:, :80] = (255, 0, 0)
    rgb[:, -80:] = (0, 0, 255)
    return rgb


def test_edges_are_not_cropped():
    obs, clip = _deploy_images(_scene())
    for img in (obs, clip):
        a = np.asarray(img)
        assert a[:, 0].mean(axis=0) == pytest.approx([255, 0, 0], abs=2)  # 左端は赤のまま
        assert a[:, -1].mean(axis=0) == pytest.approx([0, 0, 255], abs=2)  # 右端は青のまま


def test_matches_training_loader(tmp_path):
    rgb = _scene()
    jpg = tmp_path / "traj" / "0.jpg"
    jpg.parent.mkdir()
    Image.fromarray(rgb).save(jpg, quality=95)  # 学習データと同じく JPEG を経由する
    for deploy, train in zip(_deploy_images(rgb), _train_images(jpg)):
        assert deploy.size == train.size
        diff = np.abs(np.asarray(deploy, dtype=np.int16) - np.asarray(train, dtype=np.int16))
        # 縮小の手順(実機は 224 経由の 2 段)と JPEG の分だけずれるが、見え方は同じ
        assert diff.mean() < 3.0, diff.mean()


REAL_JPG = Path.home() / "tsukuba_data/laps/rosbag2_2026_10_03-15_39_55/episode01/100.jpg"


@pytest.mark.skipif(not REAL_JPG.exists(), reason="実データ(つくばの周回)が無い環境では飛ばす")
def test_matches_training_loader_on_real_frame(tmp_path):
    """実際の魚眼画像で、学習時と実機の入力のずれを確かめる。"""
    rgb = np.asarray(Image.open(REAL_JPG).convert("RGB"))
    assert rgb.shape == (H, W, 3)
    for deploy, train in zip(_deploy_images(rgb), _train_images(REAL_JPG)):
        diff = np.abs(np.asarray(deploy, dtype=np.int16) - np.asarray(train, dtype=np.int16))
        assert diff.mean() < 3.0, diff.mean()
