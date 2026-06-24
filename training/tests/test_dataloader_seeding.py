"""DataLoader worker seeding: workers must not share numpy's RNG state.

``EdgeNavigationDataset.__getitem__`` draws the goal time with ``np.random``.
Without per-worker reseeding, every forked worker inherits the same numpy state
and produces identical goal-time sequences. ``_seed_worker`` reseeds numpy from
each worker's distinct torch seed.
"""

from __future__ import annotations

import numpy as np
import torch

from train import _seed_worker


def test_seed_worker_reseeds_numpy_per_torch_seed() -> None:
    torch.manual_seed(123)
    _seed_worker(0)
    a = np.random.rand(4)

    torch.manual_seed(123)
    _seed_worker(0)
    b = np.random.rand(4)

    np.testing.assert_array_equal(a, b)  # same torch seed -> identical numpy stream

    torch.manual_seed(999)
    _seed_worker(0)
    c = np.random.rand(4)

    assert not np.array_equal(a, c)  # distinct worker seeds -> distinct streams (no dup)
