"""Tests for ViNT/NoMaD-style train/test trajectory splitting.

The splitting policy mirrors the upstream visualnav-transformer ``data_split.py``:
- the split is performed at the *trajectory* level (an entire trajectory folder
  goes to either train or test, never both -> no frame-level leakage),
- trajectories are randomly shuffled (optionally seeded for reproducibility) and
  partitioned by a ratio (default 0.8),
- the resulting names are persisted to ``<dir>/<dataset>/{train,test}/traj_names.txt``
  so the same split is reused across runs.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from training.data.split import (
    ensure_split,
    list_trajectories,
    read_traj_names,
    split_trajectories,
    write_split,
)


def _make_traj_dir(root: Path, name: str, with_pkl: bool = True) -> None:
    d = root / name
    d.mkdir(parents=True)
    if with_pkl:
        with (d / "traj_data.pkl").open("wb") as f:
            pickle.dump({"position": [[0.0, 0.0]], "yaw": [0.0]}, f)


def test_list_trajectories_returns_only_traj_dirs_sorted(tmp_path: Path) -> None:
    _make_traj_dir(tmp_path, "episode03")
    _make_traj_dir(tmp_path, "episode01")
    _make_traj_dir(tmp_path, "episode02")
    _make_traj_dir(tmp_path, "not_a_traj", with_pkl=False)  # missing traj_data.pkl
    (tmp_path / "loose_file.txt").write_text("x")  # not a directory

    names = list_trajectories(tmp_path)

    assert names == ["episode01", "episode02", "episode03"]


def test_split_is_disjoint_and_covers_all() -> None:
    names = [f"episode{i:02d}" for i in range(1, 37)]
    train, test = split_trajectories(names, split=0.8, seed=0)

    assert set(train).isdisjoint(set(test))
    assert sorted(train + test) == sorted(names)


def test_split_ratio_matches_upstream_floor_rule() -> None:
    names = [f"episode{i:02d}" for i in range(1, 37)]  # 36 trajectories
    train, test = split_trajectories(names, split=0.8, seed=0)

    # upstream: split_index = int(split * len); train = [:idx], test = [idx:]
    assert len(train) == int(0.8 * 36)  # 28
    assert len(test) == 36 - int(0.8 * 36)  # 8


def test_split_is_reproducible_with_same_seed() -> None:
    names = [f"episode{i:02d}" for i in range(1, 37)]
    a = split_trajectories(names, split=0.8, seed=42)
    b = split_trajectories(names, split=0.8, seed=42)

    assert a == b


def test_split_does_not_mutate_input() -> None:
    names = [f"episode{i:02d}" for i in range(1, 11)]
    original = list(names)
    split_trajectories(names, split=0.8, seed=0)

    assert names == original


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    train_names = ["episode01", "episode05", "episode09"]
    test_names = ["episode02", "episode07"]

    train_dir, test_dir = write_split(tmp_path, train_names, test_names)

    assert (train_dir / "traj_names.txt").exists()
    assert (test_dir / "traj_names.txt").exists()
    # split files live inside the dataset folder
    assert train_dir == tmp_path / "train"
    assert test_dir == tmp_path / "test"

    assert read_traj_names(train_dir) == train_names
    assert read_traj_names(test_dir) == test_names


def test_read_traj_names_strips_blank_lines(tmp_path: Path) -> None:
    folder = tmp_path / "split"
    folder.mkdir()
    (folder / "traj_names.txt").write_text("episode01\nepisode02\n\n")

    assert read_traj_names(folder) == ["episode01", "episode02"]


def test_write_split_clears_stale_entries(tmp_path: Path) -> None:
    write_split(tmp_path, ["episode01", "episode02"], ["episode03"])
    # regenerate with a different split -> stale traj_names must not survive
    train_dir, test_dir = write_split(tmp_path, ["episode09"], ["episode08"])

    assert read_traj_names(train_dir) == ["episode09"]
    assert read_traj_names(test_dir) == ["episode08"]


def test_read_traj_names_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_traj_names(tmp_path / "does_not_exist")


# --- training-time auto-split (ensure_split) ---------------------------------
# The split is generated automatically the first time training runs on a dataset
# and reused verbatim afterwards; there is no standalone split step.

def test_ensure_split_generates_and_persists_when_missing(tmp_path: Path) -> None:
    for i in range(1, 11):
        _make_traj_dir(tmp_path, f"episode{i:02d}")

    train, test = ensure_split(tmp_path, split=0.8, seed=0)

    assert (tmp_path / "train" / "traj_names.txt").exists()
    assert (tmp_path / "test" / "traj_names.txt").exists()
    assert set(train).isdisjoint(set(test))
    assert sorted(train + test) == [f"episode{i:02d}" for i in range(1, 11)]
    assert len(train) == int(0.8 * 10)


def test_ensure_split_reuses_existing_split_verbatim(tmp_path: Path) -> None:
    for i in range(1, 11):
        _make_traj_dir(tmp_path, f"episode{i:02d}")
    # A hand-written split that a fresh shuffle would never reproduce.
    write_split(tmp_path, ["episode03"], ["episode07"])

    train, test = ensure_split(tmp_path, split=0.8, seed=0)

    assert train == ["episode03"]
    assert test == ["episode07"]


def test_ensure_split_is_stable_across_calls(tmp_path: Path) -> None:
    for i in range(1, 11):
        _make_traj_dir(tmp_path, f"episode{i:02d}")

    first = ensure_split(tmp_path, split=0.8, seed=0)
    second = ensure_split(tmp_path, split=0.8, seed=0)

    assert first == second


def test_ensure_split_raises_when_no_trajectories(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ensure_split(tmp_path, split=0.8, seed=0)
