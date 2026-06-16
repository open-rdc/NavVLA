"""Action loss for OmniVLA-edge fine-tuning."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_action_loss(
    action_pred: torch.Tensor,
    action_label: torch.Tensor,
) -> torch.Tensor:
    """Compute the GNM/ViNT-style action loss.

    A single MSE over the full action tensor ``[B, len_traj, 2 or 4]``
    (normalized ego waypoints plus, when ``learn_angle``, the unit cos/sin
    heading) -- the same combined L2 GNM/ViNT use. The per-dimension scale is
    handled by the dataset normalization, not by separate weights, and there is
    no smoothness term (that is OmniVLA-specific).

    Args:
        action_pred: Predicted actions ``[B, len_traj, 2 or 4]``.
        action_label: Ground-truth actions, same shape as ``action_pred``.

    Returns:
        The scalar MSE loss.
    """
    if action_pred.shape != action_label.shape:
        raise ValueError(
            f"action_pred and action_label must match: {tuple(action_pred.shape)} "
            f"vs {tuple(action_label.shape)}"
        )
    return F.mse_loss(action_pred, action_label)
