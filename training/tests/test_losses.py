"""Tests for the OmniVLA-edge fine-tuning action loss.

GNM/ViNT design: a single MSE over the full action tensor (normalized ego
waypoints plus, when ``learn_angle``, the unit cos/sin heading). The per-dim
scale is handled by the dataset normalization, not by separate weights, and
there is NO smoothness term (that is an OmniVLA-specific addition).
"""

from __future__ import annotations

import pytest
import torch

from training.losses import compute_action_loss


def test_action_loss_is_mse_over_all_dims():
    label = torch.zeros(2, 4, 4)
    pred = torch.full((2, 4, 4), 0.5)
    loss = compute_action_loss(pred, label)
    assert loss.item() == pytest.approx(0.25, abs=1e-6)  # 0.5^2


def test_perfect_prediction_is_zero():
    label = torch.full((2, 4, 4), 0.3)
    loss = compute_action_loss(label.clone(), label)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_changing_waypoints_are_not_penalized_for_smoothness():
    # A trajectory that changes over time must only be scored on its deviation
    # from the label -- ViNT has no smoothness term penalizing consecutive steps.
    label = torch.zeros(1, 3, 4)
    pred = torch.zeros(1, 3, 4)
    pred[0, :, 0] = torch.tensor([0.0, 1.0, 2.0])
    loss = compute_action_loss(pred, label)
    # pure MSE vs the zero label: (0 + 1 + 4) over 12 elements
    assert loss.item() == pytest.approx(5.0 / 12.0, abs=1e-6)


def test_works_without_angle_dims():
    label = torch.zeros(2, 4, 2)
    pred = torch.full((2, 4, 2), 0.5)
    loss = compute_action_loss(pred, label)
    assert loss.item() == pytest.approx(0.25, abs=1e-6)


def test_single_step_is_finite():
    label = torch.zeros(1, 1, 4)
    pred = torch.ones(1, 1, 4)
    loss = compute_action_loss(pred, label)
    assert loss.item() == pytest.approx(1.0, abs=1e-6)
    assert torch.isfinite(loss)


def test_shape_mismatch_raises():
    pred = torch.zeros(2, 4, 4)
    label = torch.zeros(2, 5, 4)
    with pytest.raises(ValueError):
        compute_action_loss(pred, label)


def test_loss_is_differentiable():
    label = torch.randn(2, 4, 4)
    pred = (label + 0.1).detach().clone().requires_grad_(True)
    loss = compute_action_loss(pred, label)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
