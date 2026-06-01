"""Integration tests for the Train phase (loss, grad clip, per-step logging).

The LR scheduler is stepped per epoch by the training loop, not inside Train, so
Train no longer owns a scheduler; it logs the current optimizer LR each step.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from training.train import Train


LEN_TRAJ = 4
NUM_PARAMS = 4


class DummyModel(nn.Module):
    """Minimal model matching the OmniVLA_edge.forward signature."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, LEN_TRAJ * NUM_PARAMS)

    def forward(
        self, obs_img, goal_pose, map_images, goal_img, goal_mask, feat_text, current_img
    ):
        b = obs_img.shape[0]
        x = obs_img.reshape(b, -1)[:, :4]
        out = self.lin(x).reshape(b, LEN_TRAJ, NUM_PARAMS)
        return out, None, None


def _make_batch(batch_size: int = 2) -> dict:
    return {
        "obs_images": torch.randn(batch_size, 4),
        "goal_pose": torch.randn(batch_size, 4),
        "map_images": torch.randn(batch_size, 4),
        "goal_image": torch.randn(batch_size, 4),
        "goal_mask": torch.zeros(batch_size, dtype=torch.long),
        "feat_text": torch.randn(batch_size, 4),
        "current_img": torch.randn(batch_size, 4),
        "actions": torch.randn(batch_size, LEN_TRAJ, NUM_PARAMS),
    }


class StubWriter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float, int]] = []

    def add_scalar(self, tag, value, step):
        self.calls.append((tag, float(value), int(step)))


def _loader(n_batches: int = 3) -> list:
    return [_make_batch() for _ in range(n_batches)]


def test_run_returns_metrics_and_global_step():
    model = DummyModel()
    loader = _loader(3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Train(model=model, loader=loader, optimizer=opt, device=torch.device("cpu"))
    metrics, global_step = trainer.run(global_step=0)
    assert {"loss", "action_loss", "smooth_loss"} <= set(metrics.keys())
    assert global_step == len(loader)


def test_per_step_scalars_and_lr_are_logged():
    model = DummyModel()
    loader = _loader(4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    writer = StubWriter()
    trainer = Train(model=model, loader=loader, optimizer=opt, device=torch.device("cpu"))
    _, global_step = trainer.run(global_step=10, writer=writer)
    step_losses = [c for c in writer.calls if c[0] == "loss/train_step"]
    lr_logs = [c for c in writer.calls if c[0] == "lr"]
    assert len(step_losses) == len(loader)
    assert [c[2] for c in step_losses] == [11, 12, 13, 14]
    assert len(lr_logs) == len(loader)
    assert all(lr[1] == 1e-3 for lr in lr_logs)
    assert global_step == 14


def test_grad_clipping_keeps_training_finite():
    model = DummyModel()
    loader = _loader(3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-1)
    trainer = Train(
        model=model,
        loader=loader,
        optimizer=opt,
        device=torch.device("cpu"),
        max_grad_norm=1.0,
    )
    metrics, _ = trainer.run(global_step=0)
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    for p in model.parameters():
        assert torch.isfinite(p).all()
