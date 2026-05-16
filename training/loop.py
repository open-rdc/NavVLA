"""Train/test cycle orchestration for OmniVLA-edge fine-tuning."""

from __future__ import annotations

from datetime import datetime
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from OmniVLA.inference.model_omnivla_edge import OmniVLA_edge
from training.eval import Test
from training.train import Train


def resolve_path(navvla_root: Path, raw_path: object) -> Path:
    path = Path(str(raw_path))
    return path if path.is_absolute() else (navvla_root / path).resolve()


def make_timestamped_run_dir(run_root_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = run_root_dir / timestamp
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = run_root_dir / f"{timestamp}_{suffix:02d}"
    return run_dir


def extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Checkpoint must be a state_dict or a dict containing state_dict/model_state_dict.")


def save_checkpoint(
    checkpoint_path: Path,
    model_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_cfg: Dict[str, object],
    network_cfg: Dict[str, object],
    dataset_cfg: Dict[str, object],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    torch.save(
        {
            "epoch": epoch,
            "state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "train_cfg": train_cfg,
            "network_cfg": network_cfg,
            "dataset_cfg": dataset_cfg,
        },
        checkpoint_path,
    )
    torch.save(state_dict, model_path)


def validate_config(
    train_cfg: Dict[str, object],
    network_cfg: Dict[str, object],
    dataset_cfg: Dict[str, object],
) -> None:
    sections = (
        (
            "training/config/train.yaml",
            train_cfg,
            (
                "weights_path",
                "run_root_dir",
                "epochs",
                "batch_size",
                "learning_rate",
                "weight_decay",
                "num_workers",
                "seed",
                "eval_freq",
            ),
        ),
        (
            "training/config/network.yaml",
            network_cfg,
            (
                "context_size",
                "len_traj_pred",
                "learn_angle",
                "obs_encoder",
                "obs_encoding_size",
                "late_fusion",
                "mha_num_attention_heads",
                "mha_num_attention_layers",
                "mha_ff_dim_factor",
            ),
        ),
        (
            "training/config/dataset.yaml",
            dataset_cfg,
            (
                "image_size",
                "context_type",
                "normalize",
                "datasets",
            ),
        ),
    )
    for section, cfg, keys in sections:
        missing = [key for key in keys if key not in cfg]
        if missing:
            raise ValueError(f"Missing required keys in {section}: {missing}")


def main_loop(
    navvla_root: Path,
    train_cfg: Dict[str, object],
    network_cfg: Dict[str, object],
    dataset_cfg: Dict[str, object],
    train_loader: DataLoader,
    train_eval_dataloaders: Dict[str, DataLoader],
    test_dataloaders: Dict[str, DataLoader],
) -> int:
    validate_config(train_cfg, network_cfg, dataset_cfg)
    weights_path = resolve_path(navvla_root, train_cfg["weights_path"])

    run_root_dir = Path(str(train_cfg["run_root_dir"]))
    run_root_dir = run_root_dir if run_root_dir.is_absolute() else (navvla_root / run_root_dir).resolve()
    run_dir = make_timestamped_run_dir(run_root_dir)

    print(f"[NavVLA] OmniVLA-edge weights: {weights_path}")
    print(f"[NavVLA] Run root directory: {run_root_dir}")
    print(f"[NavVLA] Run directory: {run_dir}")

    seed = int(train_cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OmniVLA_edge(
        context_size=int(network_cfg["context_size"]),
        len_traj_pred=int(network_cfg["len_traj_pred"]),
        learn_angle=bool(network_cfg["learn_angle"]),
        obs_encoder=str(network_cfg["obs_encoder"]),
        obs_encoding_size=int(network_cfg["obs_encoding_size"]),
        late_fusion=bool(network_cfg["late_fusion"]),
        mha_num_attention_heads=int(network_cfg["mha_num_attention_heads"]),
        mha_num_attention_layers=int(network_cfg["mha_num_attention_layers"]),
        mha_ff_dim_factor=int(network_cfg["mha_ff_dim_factor"]),
    )
    if not weights_path.exists():
        raise FileNotFoundError(f"OmniVLA-edge weights not found: {weights_path}")
    
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    start_epoch = 1
    resume_from = train_cfg.get("resume_from")
    if resume_from:
        resume_path = resolve_path(navvla_root, resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(extract_state_dict(resume_checkpoint), strict=True)
        if isinstance(resume_checkpoint, dict) and "optimizer_state_dict" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        if isinstance(resume_checkpoint, dict) and "epoch" in resume_checkpoint:
            start_epoch = int(resume_checkpoint["epoch"]) + 1
        print(f"[NavVLA] Resumed from {resume_path} at epoch={start_epoch - 1}")

    Trainer = Train(model=model, loader=train_loader, optimizer=optimizer, device=device)
    TrainEvaluators = {
        dataset_type: Test(model=model, loader=loader, device=device)
        for dataset_type, loader in train_eval_dataloaders.items()
    }
    Testers = {
        dataset_type: Test(model=model, loader=loader, device=device)
        for dataset_type, loader in test_dataloaders.items()
    }

    max_train_steps = train_cfg.get("max_train_steps")
    max_test_steps = train_cfg.get("max_test_steps")
    eval_freq = int(train_cfg["eval_freq"])
    best_loss = float("inf")
    if SummaryWriter is None:
        raise ImportError("TensorBoard is not available. Install it with: pip install tensorboard")
    tensorboard_dir = train_cfg.get("tensorboard_log_dir", run_dir / "tensorboard")
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    print(f"[NavVLA] TensorBoard: tensorboard --logdir {tensorboard_dir}")

    total_epochs = int(train_cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs - start_epoch + 1
    )
    if start_epoch > total_epochs:
        print(f"[NavVLA] Nothing to train: start_epoch={start_epoch} > epochs={total_epochs}")
        writer.close()
        return 0

    for epoch in range(start_epoch, total_epochs + 1):
        train_metrics = Trainer.run(
            max_steps=None if max_train_steps is None else int(max_train_steps)
        )
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"[NavVLA] epoch={epoch} train={train_metrics} lr={current_lr:.2e}")
        writer.add_scalar("loss/train_total", train_metrics["loss"], epoch)
        writer.add_scalar("loss/train_action", train_metrics["action_loss"], epoch)
        writer.add_scalar("loss/train_dist", train_metrics["dist_loss"], epoch)
        writer.add_scalar("lr", current_lr, epoch)
        writer.flush()

        train_eval_losses = []
        for dataset_type, evaluator in TrainEvaluators.items():
            metrics = evaluator.run(max_steps=None if max_test_steps is None else int(max_test_steps))
            train_eval_losses.append(metrics["loss"])
            print(f"[NavVLA] epoch={epoch} train[{dataset_type}]={metrics}")
            writer.add_scalar(f"loss/train/{dataset_type}", metrics["loss"], epoch)
        if train_eval_losses:
            writer.add_scalar("loss/train_datasets_total", float(np.mean(train_eval_losses)), epoch)
            writer.flush()

        if epoch % eval_freq == 0:
            eval_losses = []
            for dataset_type, tester in Testers.items():
                test_metrics = tester.run(
                    max_steps=None if max_test_steps is None else int(max_test_steps)
                )
                eval_losses.append(test_metrics["loss"])
                print(f"[NavVLA] epoch={epoch} test[{dataset_type}]={test_metrics}")
                writer.add_scalar(f"loss/eval/{dataset_type}", test_metrics["loss"], epoch)
            if eval_losses:
                writer.add_scalar("loss/eval_total", float(np.mean(eval_losses)), epoch)
                writer.flush()

            if eval_losses:
                candidate_loss = float(np.mean(eval_losses))
            elif train_eval_losses:
                candidate_loss = float(np.mean(train_eval_losses))
            else:
                candidate_loss = train_metrics["loss"]

            if candidate_loss < best_loss:
                best_loss = candidate_loss
                save_checkpoint(
                    checkpoint_path=run_dir / "checkpoint_best.pth",
                    model_path=run_dir / "model_best.pth",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    train_cfg=train_cfg,
                    network_cfg=network_cfg,
                    dataset_cfg=dataset_cfg,
                )
                print(f"[NavVLA] best model updated: loss={best_loss:.6f} epoch={epoch}")

        save_checkpoint(
            checkpoint_path=run_dir / "checkpoint_latest.pth",
            model_path=run_dir / "model_latest.pth",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_cfg=train_cfg,
            network_cfg=network_cfg,
            dataset_cfg=dataset_cfg,
        )

    writer.close()
    return 0
