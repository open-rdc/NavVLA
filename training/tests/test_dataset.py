"""Tests for EdgeNavigationDataset helpers (shared CLIP text encoder).

The text encoder is loaded once per process and shared by every dataset
instance, and its unused visual tower is freed, to keep language-modality
training from holding one full CLIP per dataset per DataLoader worker.
"""

from __future__ import annotations

import sys
import types

import torch
import torch.nn as nn

from training.data.dataset import _load_clip_text_encoder


def _install_fake_clip(monkeypatch) -> None:
    """Stub the ``clip`` module so tests never download real weights."""

    class FakeVisual(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 2, 1)
            self.heavy = nn.Linear(64, 64)  # stand-in for the unused visual tower

    class FakeCLIP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = FakeVisual()
            self.token_embedding = nn.Embedding(16, 4)

        @property
        def dtype(self) -> torch.dtype:
            return self.visual.conv1.weight.dtype  # mirrors openai/CLIP

        def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
            return self.token_embedding(tokens).type(self.dtype).sum(dim=1)

    fake = types.ModuleType("clip")
    fake.load = lambda name, device="cpu": (FakeCLIP(), None)
    fake.tokenize = lambda text, truncate=True: torch.zeros(1, 3, dtype=torch.long)
    monkeypatch.setitem(sys.modules, "clip", fake)


def test_clip_text_encoder_is_shared_across_calls(monkeypatch) -> None:
    _install_fake_clip(monkeypatch)
    _load_clip_text_encoder.cache_clear()

    first = _load_clip_text_encoder("ViT-B/32")
    second = _load_clip_text_encoder("ViT-B/32")

    assert first is second  # one model shared by every dataset instance in a process
    _load_clip_text_encoder.cache_clear()


def test_clip_text_encoder_frees_visual_tower_but_still_encodes(monkeypatch) -> None:
    _install_fake_clip(monkeypatch)
    _load_clip_text_encoder.cache_clear()

    model = _load_clip_text_encoder("ViT-B/32")

    # the heavy visual tower is dropped (replaced by a non-Module dtype shim)
    assert not isinstance(model.visual, nn.Module)
    assert "visual" not in dict(model.named_modules())
    # CLIP.dtype still resolves and text encoding still works
    assert model.dtype == torch.float32
    out = model.encode_text(torch.zeros(1, 3, dtype=torch.long))
    assert out.shape[0] == 1
    _load_clip_text_encoder.cache_clear()
