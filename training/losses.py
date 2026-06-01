"""Action loss for OmniVLA-edge fine-tuning."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def compute_action_loss(
    action_pred: torch.Tensor,
    action_label: torch.Tensor,
    *,
    smoothness_weight: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the OmniVLA-edge action loss.

    Args:
        action_pred: Predicted actions ``[B, len_traj, 2 or 4]``.
        action_label: Ground-truth actions, same shape as ``action_pred``.
        smoothness_weight: Weight of the trajectory-smoothness term (reference: 0.1).

    Returns:
        ``(total_loss, parts)`` where ``parts`` holds ``loss``, ``action_loss`` and
        ``smooth_loss`` tensors.
    """
    if action_pred.shape != action_label.shape:
        raise ValueError(
            f"action_pred and action_label must match: {tuple(action_pred.shape)} "
            f"vs {tuple(action_label.shape)}"
        )

    action_loss = F.mse_loss(action_pred, action_label)

    if action_pred.shape[1] >= 2:
        smooth_loss = F.mse_loss(action_pred[:, :-1], action_pred[:, 1:])
    else:
        smooth_loss = torch.zeros((), device=action_pred.device, dtype=action_pred.dtype)

    total = action_loss + smoothness_weight * smooth_loss
    return total, {"loss": total, "action_loss": action_loss, "smooth_loss": smooth_loss}
