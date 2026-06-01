"""Learning-rate scheduler builders for OmniVLA-edge fine-tuning."""

from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import LambdaLR

CONSTANT = "constant"
COSINE = "cosine"


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_type: str = COSINE,
    total_epochs: int,
    warmup_epochs: int = 0,
) -> LambdaLR:
    """Build a per-epoch LR scheduler.

    Args:
        optimizer: Optimizer whose base LR(s) are scaled by the schedule.
        scheduler_type: ``"cosine"`` or ``"constant"``.
        total_epochs: Total number of epochs (cosine ``T_max``).
        warmup_epochs: If > 0, linearly warm the LR from 0 to the peak over this
            many epochs before the main schedule applies.

    Returns:
        A ``LambdaLR`` to be ``.step()``-ed once per epoch.
    """
    total_epochs = max(1, int(total_epochs))
    warmup_epochs = max(0, int(warmup_epochs))

    def warmup_factor(epoch: int) -> float:
        return float(epoch + 1) / float(warmup_epochs)

    if scheduler_type == CONSTANT:
        def lr_lambda(epoch: int) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return warmup_factor(epoch)
            return 1.0
    elif scheduler_type == COSINE:
        def lr_lambda(epoch: int) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return warmup_factor(epoch)
            denom = max(1, total_epochs - warmup_epochs)
            progress = float(epoch - warmup_epochs) / float(denom)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    else:
        raise ValueError(
            f"Unknown scheduler_type: {scheduler_type!r} "
            f"(expected {CONSTANT!r} or {COSINE!r})"
        )

    return LambdaLR(optimizer, lr_lambda)
