"""Tests for the LR scheduler used in OmniVLA-edge fine-tuning.

OmniVLA-edge is the ViNT-based 'edge' model; the OmniVLA paper states its training
settings follow "the default setting in the original code for each model type". For
this model type that is LogoNav (config/LogoNav.yaml in NHirose/Learning-to-Drive-
Anywhere-with-MBRA): ``optimizer: adamw``, ``scheduler: cosine`` with ``warmup``.
We implement cosine annealing (T_max=epochs) with a linear warmup, stepped per epoch.
"""

from __future__ import annotations

import math

import pytest
import torch

from training.optim import build_scheduler


def _dummy_optimizer(lr: float = 1.0) -> torch.optim.Optimizer:
    param = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.SGD([param], lr=lr)


def _lr_trace(scheduler, optimizer, n_epochs: int) -> list[float]:
    lrs = []
    for _ in range(n_epochs):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return lrs


def test_constant_scheduler_keeps_lr():
    opt = _dummy_optimizer(lr=1e-4)
    sched = build_scheduler(opt, scheduler_type="constant", total_epochs=10)
    lrs = _lr_trace(sched, opt, 10)
    assert all(lr == pytest.approx(1e-4) for lr in lrs)


def test_cosine_no_warmup_starts_at_peak_and_decays():
    opt = _dummy_optimizer(lr=1.0)
    sched = build_scheduler(opt, scheduler_type="cosine", total_epochs=10, warmup_epochs=0)
    lrs = _lr_trace(sched, opt, 10)
    assert lrs[0] == pytest.approx(1.0, abs=1e-6)  # epoch 0 -> cos(0)=1
    for a, b in zip(lrs[:-1], lrs[1:]):
        assert b <= a + 1e-9
    assert lrs[-1] < lrs[0]
    assert lrs[-1] < 0.1  # well decayed by the final epoch


def test_cosine_with_warmup_ramps_then_decays():
    opt = _dummy_optimizer(lr=1.0)
    sched = build_scheduler(opt, scheduler_type="cosine", total_epochs=10, warmup_epochs=3)
    lrs = _lr_trace(sched, opt, 10)
    # linear warmup: (epoch+1)/warmup_epochs for epoch < warmup_epochs
    assert lrs[0] == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert lrs[1] == pytest.approx(2.0 / 3.0, abs=1e-6)
    assert lrs[2] == pytest.approx(1.0, abs=1e-6)   # peak reached at end of warmup
    # after warmup it decays
    assert lrs[9] < lrs[3]


def test_constant_with_warmup():
    opt = _dummy_optimizer(lr=2.0)
    sched = build_scheduler(opt, scheduler_type="constant", total_epochs=10, warmup_epochs=2)
    lrs = _lr_trace(sched, opt, 5)
    assert lrs[0] == pytest.approx(1.0, abs=1e-6)   # 2.0 * 1/2
    assert lrs[1] == pytest.approx(2.0, abs=1e-6)   # 2.0 * 2/2
    assert lrs[2] == pytest.approx(2.0, abs=1e-6)   # flat thereafter
    assert lrs[4] == pytest.approx(2.0, abs=1e-6)


def test_unknown_scheduler_type_raises():
    opt = _dummy_optimizer()
    with pytest.raises(ValueError):
        build_scheduler(opt, scheduler_type="nope", total_epochs=10)
