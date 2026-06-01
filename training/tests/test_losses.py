"""Tests for the OmniVLA-edge fine-tuning action loss.

The loss mirrors OmniVLA's in-package nav training (vla-scripts/train_omnivla.py:404):
``MSE(pred, label)`` over all action dims plus a ``0.1 * MSE`` trajectory-smoothness
term computed on the predicted consecutive waypoints. (The object-pose term in the
reference is omitted: it needs an object-position label NavVLA does not provide.)
"""

from __future__ import annotations

import pytest
import torch

from training.losses import compute_action_loss


def test_action_term_is_mse_over_all_dims():
    label = torch.zeros(2, 4, 4)
    pred = torch.full((2, 4, 4), 0.5)  # constant across time -> zero smoothness
    total, parts = compute_action_loss(pred, label)
    assert parts["action_loss"].item() == pytest.approx(0.25, abs=1e-6)  # 0.5^2
    assert parts["smooth_loss"].item() == pytest.approx(0.0, abs=1e-6)
    assert total.item() == pytest.approx(0.25, abs=1e-6)


def test_perfect_constant_trajectory_is_zero():
    label = torch.full((2, 4, 4), 0.3)  # constant trajectory -> no smoothness penalty
    total, parts = compute_action_loss(label.clone(), label)
    assert parts["action_loss"].item() == pytest.approx(0.0, abs=1e-6)
    assert parts["smooth_loss"].item() == pytest.approx(0.0, abs=1e-6)
    assert total.item() == pytest.approx(0.0, abs=1e-6)


def test_smoothness_penalizes_changing_waypoints():
    label = torch.zeros(1, 3, 4)
    pred = torch.zeros(1, 3, 4)
    pred[0, :, 0] = torch.tensor([0.0, 1.0, 2.0])  # consecutive diff of 1.0 in dim 0
    _, parts = compute_action_loss(pred, label, smoothness_weight=0.1)
    # MSE over pred[:, :-1] vs pred[:, 1:]: two diffs of 1.0 among 8 elements -> 2/8
    assert parts["smooth_loss"].item() == pytest.approx(0.25, abs=1e-6)


def test_smoothness_weight_scales_total():
    label = torch.zeros(1, 3, 4)
    pred = torch.zeros(1, 3, 4)
    pred[0, :, 0] = torch.tensor([0.0, 1.0, 2.0])
    total0, parts = compute_action_loss(pred, label, smoothness_weight=0.0)
    total1, _ = compute_action_loss(pred, label, smoothness_weight=0.1)
    assert total0.item() == pytest.approx(parts["action_loss"].item(), abs=1e-6)
    assert total1.item() == pytest.approx(parts["action_loss"].item() + 0.1 * 0.25, abs=1e-6)


def test_single_step_has_no_smoothness():
    label = torch.zeros(1, 1, 4)
    pred = torch.ones(1, 1, 4)
    total, parts = compute_action_loss(pred, label)
    assert parts["smooth_loss"].item() == pytest.approx(0.0, abs=1e-6)
    assert torch.isfinite(torch.tensor(total.item()))


def test_works_without_angle_dims():
    label = torch.zeros(2, 4, 2)
    pred = torch.full((2, 4, 2), 0.5)
    _, parts = compute_action_loss(pred, label)
    assert parts["action_loss"].item() == pytest.approx(0.25, abs=1e-6)


def test_shape_mismatch_raises():
    pred = torch.zeros(2, 4, 4)
    label = torch.zeros(2, 5, 4)
    with pytest.raises(ValueError):
        compute_action_loss(pred, label)


def test_total_is_differentiable():
    label = torch.randn(2, 4, 4)
    pred = (label + 0.1).detach().clone().requires_grad_(True)
    total, _ = compute_action_loss(pred, label)
    total.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
