"""Tests for the clip_embeddings.bin writer shared by record_episode and create_topomap.

The binary layout MUST match what the C++ reader expects
(``pfoe/src/Episode.cpp``)::

    int32 N, int32 dim, float32[N][dim]   # little-endian, row-major

so these tests pin that contract.
"""

import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import record_episode


def read_embeddings_bin(path):
    with open(path, "rb") as f:
        n, dim = struct.unpack("<ii", f.read(8))
        data = np.frombuffer(f.read(), dtype="<f4")
    return n, dim, data.reshape(n, dim)


def test_roundtrip_preserves_values(tmp_path):
    emb = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = tmp_path / "clip_embeddings.bin"
    record_episode.write_embeddings_bin(out, emb)
    n, dim, back = read_embeddings_bin(out)
    assert (n, dim) == (2, 3)
    assert np.array_equal(back, emb)


def test_byte_length_matches_header_plus_payload(tmp_path):
    emb = np.zeros((5, 512), dtype=np.float32)
    out = tmp_path / "clip_embeddings.bin"
    record_episode.write_embeddings_bin(out, emb)
    assert out.stat().st_size == 8 + 5 * 512 * 4


def test_header_is_little_endian_n_dim(tmp_path):
    emb = np.zeros((3, 4), dtype=np.float32)
    out = tmp_path / "clip_embeddings.bin"
    record_episode.write_embeddings_bin(out, emb)
    with open(out, "rb") as f:
        head = f.read(8)
    assert head == struct.pack("<ii", 3, 4)


def test_non_float32_input_is_cast(tmp_path):
    emb = np.arange(4, dtype=np.float64).reshape(2, 2)
    out = tmp_path / "clip_embeddings.bin"
    record_episode.write_embeddings_bin(out, emb)
    _, _, back = read_embeddings_bin(out)
    assert back.dtype == np.float32
    assert np.array_equal(back, emb.astype(np.float32))


def test_rejects_non_2d_input(tmp_path):
    out = tmp_path / "clip_embeddings.bin"
    try:
        record_episode.write_embeddings_bin(out, np.zeros(4, dtype=np.float32))
    except ValueError:
        return
    raise AssertionError("expected ValueError for 1-D embeddings")
