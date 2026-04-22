"""Train phase for OmniVLA-edge fine-tuning."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Train:
    """Run one train phase for OmniVLA-edge."""

    def __init__(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> None:
        self.model = model
        self.loader = loader
        self.optimizer = optimizer
        self.device = device

    def run(self, max_steps: Optional[int] = None) -> Dict[str, float]:
        """Run one OmniVLA-edge train epoch."""
        self.model.train()

        total_loss = 0.0
        total_action_loss = 0.0
        total_batches = 0

        for step, raw_batch in enumerate(self.loader, start=1):
            if max_steps is not None and step > max_steps:
                break

            batch = {key: value.to(self.device) for key, value in raw_batch.items()}
            self.optimizer.zero_grad(set_to_none=True)

            action_pred, _, _ = self.model(
                batch["obs_images"],
                batch["goal_pose"].float(),
                batch["map_images"],
                batch["goal_image"],
                batch["goal_mask"].long(),
                batch["feat_text"].float(),
                batch["current_img"],
            )
            action_loss = F.l1_loss(action_pred, batch["actions"].float())
            action_loss.backward()
            self.optimizer.step()

            loss_value = float(action_loss.detach().cpu())
            total_loss += loss_value
            total_action_loss += loss_value
            total_batches += 1

        if total_batches == 0:
            raise RuntimeError("Train loader produced no batches.")

        return {
            "loss": total_loss / total_batches,
            "action_loss": total_action_loss / total_batches,
        }
