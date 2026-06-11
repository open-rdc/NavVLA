"""Train phase for OmniVLA-edge fine-tuning."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


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
        total_dist_loss = 0.0
        total_batches = 0

        total_steps = len(self.loader) if max_steps is None else min(len(self.loader), max_steps)
        progress = tqdm(self.loader, total=total_steps, desc="train", leave=False)
        for step, raw_batch in enumerate(progress, start=1):
            if max_steps is not None and step > max_steps:
                break

            batch = {key: value.to(self.device) for key, value in raw_batch.items()}
            self.optimizer.zero_grad(set_to_none=True)

            action_pred, dist_pred, _ = self.model(
                batch["obs_images"],
                batch["goal_pose"].float(),
                batch["map_images"],
                batch["goal_image"],
                batch["goal_mask"].long(),
                batch["feat_text"].float(),
                batch["current_img"],
            )
            action_loss = F.l1_loss(action_pred, batch["actions"].float())
            dist_loss = F.l1_loss(dist_pred.squeeze(-1), batch["dist_to_goal"].float())
            loss = action_loss + dist_loss
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_action_loss += float(action_loss.detach().cpu())
            total_dist_loss += float(dist_loss.detach().cpu())
            total_batches += 1
            progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")

        if total_batches == 0:
            raise RuntimeError("Train loader produced no batches.")

        return {
            "loss": total_loss / total_batches,
            "action_loss": total_action_loss / total_batches,
            "dist_loss": total_dist_loss / total_batches,
        }
