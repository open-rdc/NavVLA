"""Train/test cycle orchestration for OmniVLA-edge fine-tuning."""

from __future__ import annotations

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
                "device",
                "seed",
                "save_freq",
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
    weights_path = Path(str(train_cfg["weights_path"]))
    weights_path = (weights_path if weights_path.is_absolute() else (navvla_root / weights_path).resolve())

    run_dir = Path(str(train_cfg["run_root_dir"]))
    run_dir = run_dir if run_dir.is_absolute() else (navvla_root / run_dir).resolve()

    print(f"[NavVLA] OmniVLA-edge weights: {weights_path}")
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
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

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
    save_freq = int(train_cfg["save_freq"])
    eval_freq = int(train_cfg["eval_freq"])
    if SummaryWriter is None:
        raise ImportError("TensorBoard is not available. Install it with: pip install tensorboard")
    tensorboard_dir = train_cfg.get("tensorboard_log_dir", run_dir / "tensorboard")
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    print(f"[NavVLA] TensorBoard: {tensorboard_dir}")

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_metrics = Trainer.run(
            max_steps=None if max_train_steps is None else int(max_train_steps)
        )
        print(f"[NavVLA] epoch={epoch} train={train_metrics}")
        writer.add_scalar("loss/train_total", train_metrics["loss"], epoch)

        train_eval_losses = []
        for dataset_type, evaluator in TrainEvaluators.items():
            metrics = evaluator.run(max_steps=None if max_test_steps is None else int(max_test_steps))
            train_eval_losses.append(metrics["loss"])
            print(f"[NavVLA] epoch={epoch} train[{dataset_type}]={metrics}")
            writer.add_scalar(f"loss/train/{dataset_type}", metrics["loss"], epoch)
        if train_eval_losses:
            writer.add_scalar("loss/train_datasets_total", float(np.mean(train_eval_losses)), epoch)

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

        if epoch % save_freq == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = run_dir / "model_latest.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[NavVLA] saved={checkpoint_path}")

        writer.close()

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model_latest.pth")
    return 0
