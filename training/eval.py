"""Test phase for OmniVLA-edge fine-tuning."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.losses import compute_action_loss


class Test:
    """Run one test phase for OmniVLA-edge."""

    def __init__(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
        *,
        smoothness_weight: float = 0.1,
    ) -> None:
        self.model = model
        self.loader = loader
        self.device = device
        self.smoothness_weight = float(smoothness_weight)

    def run(self, max_steps: Optional[int] = None) -> Dict[str, float]:
        """Run one OmniVLA-edge test epoch."""
        self.model.eval()

        total_loss = 0.0
        total_action = 0.0
        total_smooth = 0.0
        total_batches = 0

        total_steps = len(self.loader) if max_steps is None else min(len(self.loader), max_steps)
        progress = tqdm(self.loader, total=total_steps, desc="eval", leave=False)
        with torch.no_grad():
            for step, raw_batch in enumerate(progress, start=1):
                if max_steps is not None and step > max_steps:
                    break

                batch = {key: value.to(self.device) for key, value in raw_batch.items()}
                action_pred, _, _ = self.model(
                    batch["obs_images"],
                    batch["goal_pose"].float(),
                    batch["map_images"],
                    batch["goal_image"],
                    batch["goal_mask"].long(),
                    batch["feat_text"].float(),
                    batch["current_img"],
                )
                loss, parts = compute_action_loss(
                    action_pred,
                    batch["actions"].float(),
                    smoothness_weight=self.smoothness_weight,
                )
                total_loss += float(loss.detach().cpu())
                total_action += float(parts["action_loss"].detach().cpu())
                total_smooth += float(parts["smooth_loss"].detach().cpu())
                total_batches += 1
                progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")

        if total_batches == 0:
            raise RuntimeError("Test loader produced no batches.")

        return {
            "loss": total_loss / total_batches,
            "action_loss": total_action / total_batches,
            "smooth_loss": total_smooth / total_batches,
        }
