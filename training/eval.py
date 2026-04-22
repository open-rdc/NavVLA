"""Test phase for OmniVLA-edge fine-tuning."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Test:
    """Run one test phase for OmniVLA-edge."""

    def __init__(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> None:
        self.model = model
        self.loader = loader
        self.device = device

    def run(self, max_steps: Optional[int] = None) -> Dict[str, float]:
        """Run one OmniVLA-edge test epoch."""
        self.model.eval()

        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for step, raw_batch in enumerate(self.loader, start=1):
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
                loss = F.l1_loss(action_pred, batch["actions"].float())
                total_loss += float(loss.detach().cpu())
                total_batches += 1

        if total_batches == 0:
            raise RuntimeError("Test loader produced no batches.")

        return {"loss": total_loss / total_batches}
