"""ViNT/NoMaD-style train/test trajectory splitting.

This mirrors the trajectory-level split policy used by the visualnav-transformer
family (ViNT, NoMaD, OmniVLA). An entire trajectory folder is assigned to either
train or test -- never both -- so individual (observation, action) frames never
leak across the split. Trajectories are shuffled (optionally seeded) and split by
a ratio, then the resulting names are persisted to
``<data_dir>/{train,test}/traj_names.txt`` so the same split is reused across
training runs. The split is generated automatically the first time training runs
on a dataset (see :func:`ensure_split`); there is no standalone split step.

Note on residual overlap: a trajectory-level split removes frame-level leakage but
does *not* guarantee spatial independence when every trajectory traverses the same
physical environment. Foundation models avoid this via large, geographically
diverse datasets; a single repeatedly-traversed route will still overlap spatially.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

TRAJ_DATA_FILE = "traj_data.pkl"
TRAJ_NAMES_FILE = "traj_names.txt"


def list_trajectories(data_dir: str | Path) -> List[str]:
    """Return sorted names of subfolders that contain ``traj_data.pkl``."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    return sorted(
        d.name
        for d in data_dir.iterdir()
        if d.is_dir() and (d / TRAJ_DATA_FILE).exists()
    )


def split_trajectories(
    traj_names: List[str],
    split: float = 0.8,
    seed: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """Shuffle ``traj_names`` and partition them into (train, test).

    Mirrors upstream ``data_split.py``: ``split_index = int(split * len)``,
    ``train = shuffled[:split_index]``, ``test = shuffled[split_index:]``.
    A local RNG keeps the split reproducible (when ``seed`` is set) without
    perturbing global random state used elsewhere in training.
    """
    if not 0.0 <= split <= 1.0:
        raise ValueError(f"split must be in [0, 1], got {split}")

    shuffled = list(traj_names)
    random.Random(seed).shuffle(shuffled)

    split_index = int(split * len(shuffled))
    return shuffled[:split_index], shuffled[split_index:]


def write_split(
    data_dir: str | Path,
    train_names: List[str],
    test_names: List[str],
) -> Tuple[Path, Path]:
    """Persist train/test names into ``<data_dir>/{train,test}/traj_names.txt``.

    The split files live inside the dataset folder so training only needs the
    ``data_folder`` path. Existing split directories are cleared first so a
    regenerated split never leaves stale names behind. Returns (train_dir, test_dir).
    """
    base = Path(data_dir)
    train_dir = base / "train"
    test_dir = base / "test"

    for dir_path, names in ((train_dir, train_names), (test_dir, test_names)):
        if dir_path.exists():
            shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True)
        with (dir_path / TRAJ_NAMES_FILE).open("w", encoding="utf-8") as f:
            for name in names:
                f.write(name + "\n")

    return train_dir, test_dir


def read_traj_names(split_folder: str | Path) -> List[str]:
    """Read trajectory names from ``<split_folder>/traj_names.txt``.

    Mirrors the GNM/ViNT dataset loader: split on newlines and drop blank lines.
    """
    traj_names_file = Path(split_folder) / TRAJ_NAMES_FILE
    if not traj_names_file.exists():
        raise FileNotFoundError(f"Missing split file: {traj_names_file}")
    lines = traj_names_file.read_text(encoding="utf-8").split("\n")
    return [line for line in lines if line.strip()]


def ensure_split(
    data_dir: str | Path,
    split: float = 0.8,
    seed: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """Return ``(train, test)`` names, generating the split on first use.

    When ``<data_dir>/{train,test}/traj_names.txt`` already exist they are read
    back verbatim, so the partition stays fixed across runs. Otherwise the
    trajectories under ``data_dir`` are listed, shuffled, split by ``split`` and
    persisted. This is the only entry point that creates a split: it runs as part
    of training, so a split can never be produced outside (or drift from) the
    data a run trains on.
    """
    base = Path(data_dir)
    train_file = base / "train" / TRAJ_NAMES_FILE
    test_file = base / "test" / TRAJ_NAMES_FILE
    if not (train_file.exists() and test_file.exists()):
        traj_names = list_trajectories(base)
        if not traj_names:
            raise ValueError(f"No trajectories (with {TRAJ_DATA_FILE}) found in {base}")
        train_names, test_names = split_trajectories(traj_names, split=split, seed=seed)
        write_split(base, train_names, test_names)
    return read_traj_names(base / "train"), read_traj_names(base / "test")
