"""Train/test cycle orchestration for OmniVLA-edge fine-tuning."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from OmniVLA.inference.model_omnivla_edge import OmniVLA_edge
from training.data.dataset import EdgeTensorDataset, collate_edge_samples
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
                "train_data",
                "test_data",
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
) -> int:
    """Run OmniVLA-edge fine-tuning with train/test phases."""
    validate_config(train_cfg, network_cfg, dataset_cfg)
    weights_path = Path(str(train_cfg["weights_path"]))
    weights_path = (
        weights_path if weights_path.is_absolute() else (navvla_root / weights_path).resolve()
    )
    train_data = Path(str(train_cfg["train_data"]))
    train_data = train_data if train_data.is_absolute() else (navvla_root / train_data).resolve()
    test_data = Path(str(train_cfg["test_data"]))
    test_data = test_data if test_data.is_absolute() else (navvla_root / test_data).resolve()
    run_dir = Path(str(train_cfg["run_root_dir"]))
    run_dir = run_dir if run_dir.is_absolute() else (navvla_root / run_dir).resolve()

    print(f"[NavVLA] OmniVLA-edge weights: {weights_path}")
    print(f"[NavVLA] Train data: {train_data}")
    print(f"[NavVLA] Test data: {test_data}")
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

    train_loader = torch.utils.data.DataLoader(
        EdgeTensorDataset(train_data),
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        collate_fn=collate_edge_samples,
    )
    test_loader = torch.utils.data.DataLoader(
        EdgeTensorDataset(test_data),
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
        collate_fn=collate_edge_samples,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    Trainer = Train(model=model, loader=train_loader, optimizer=optimizer, device=device)
    Tester = Test(model=model, loader=test_loader, device=device)

    max_train_steps = train_cfg.get("max_train_steps")
    max_test_steps = train_cfg.get("max_test_steps")
    save_freq = int(train_cfg["save_freq"])
    eval_freq = int(train_cfg["eval_freq"])

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_metrics = Trainer.run(
            max_steps=None if max_train_steps is None else int(max_train_steps)
        )
        print(f"[NavVLA] epoch={epoch} train={train_metrics}")

        if epoch % eval_freq == 0:
            test_metrics = Tester.run(max_steps=None if max_test_steps is None else int(max_test_steps))
            print(f"[NavVLA] epoch={epoch} test={test_metrics}")

        if epoch % save_freq == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = run_dir / "model_latest.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[NavVLA] saved={checkpoint_path}")

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model_latest.pth")
    return 0
