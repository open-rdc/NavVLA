"""Train phase for OmniVLA-edge fine-tuning."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.losses import compute_action_loss


class Train:
    """Run one train phase for OmniVLA-edge."""

    def __init__(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        *,
        smoothness_weight: float = 0.1,
        max_grad_norm: Optional[float] = None,
    ) -> None:
        self.model = model
        self.loader = loader
        self.optimizer = optimizer
        self.device = device
        self.smoothness_weight = float(smoothness_weight)
        self.max_grad_norm = max_grad_norm

    def run(
        self,
        max_steps: Optional[int] = None,
        writer: Optional[Any] = None,
        global_step: int = 0,
    ) -> Tuple[Dict[str, float], int]:
        """Run one OmniVLA-edge train epoch.

        Returns the epoch-averaged metrics and the updated global step counter.
        """
        self.model.train()

        total_loss = 0.0
        total_action = 0.0
        total_smooth = 0.0
        total_batches = 0

        total_steps = len(self.loader) if max_steps is None else min(len(self.loader), max_steps)
        progress = tqdm(self.loader, total=total_steps, desc="train", leave=False)
        for step, raw_batch in enumerate(progress, start=1):
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
            loss, parts = compute_action_loss(
                action_pred,
                batch["actions"].float(),
                smoothness_weight=self.smoothness_weight,
            )
            loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            global_step += 1

            loss_value = float(loss.detach().cpu())
            action_value = float(parts["action_loss"].detach().cpu())
            smooth_value = float(parts["smooth_loss"].detach().cpu())
            total_loss += loss_value
            total_action += action_value
            total_smooth += smooth_value
            total_batches += 1

            if writer is not None:
                writer.add_scalar("loss/train_step", loss_value, global_step)
                writer.add_scalar("loss/train_step_action", action_value, global_step)
                writer.add_scalar("loss/train_step_smooth", smooth_value, global_step)
                writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], global_step)

            progress.set_postfix(loss=f"{loss_value:.4f}")

        if total_batches == 0:
            raise RuntimeError("Train loader produced no batches.")

        metrics = {
            "loss": total_loss / total_batches,
            "action_loss": total_action / total_batches,
            "smooth_loss": total_smooth / total_batches,
        }
        return metrics, global_step
