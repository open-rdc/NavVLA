#!/usr/bin/env python3
"""Training entrypoint for NavVLA."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader
import yaml

from training.data.dataset import EdgeNavigationDataset, collate_edge_samples
from training.loop import main_loop


def create_dataloaders(
    train_cfg: dict,
    network_cfg: dict,
    dataset_cfg: dict,
) -> tuple[DataLoader, dict[str, DataLoader], dict[str, DataLoader]]:
    """Build train/test dataloaders from dataset.yaml datasets."""
    datasets_cfg = dataset_cfg.get("datasets")
    if not isinstance(datasets_cfg, dict) or not datasets_cfg:
        raise ValueError("training/config/dataset.yaml must contain a non-empty datasets mapping.")

    train_datasets = []
    train_eval_datasets = {}
    test_datasets = {}
    for dataset_name, data_config in datasets_cfg.items():
        if not isinstance(data_config, dict):
            raise ValueError(f"Dataset config must be a mapping: {dataset_name}")
        required_data_keys = (
            "data_folder",
            "waypoint_spacing",
            "end_slack",
            "goals_per_obs",
            "modality_id",
        )
        missing = [key for key in required_data_keys if key not in data_config]
        if missing:
            raise ValueError(f"Missing required keys for dataset {dataset_name}: {missing}")
        for data_split_type in ("train", "test"):
            if data_split_type not in data_config:
                continue
            dataset = EdgeNavigationDataset(
                data_folder=data_config["data_folder"],
                data_split_folder=data_config[data_split_type],
                dataset_name=str(dataset_name),
                image_size=tuple(dataset_cfg["image_size"]),
                waypoint_spacing=int(data_config["waypoint_spacing"]),
                len_traj_pred=int(network_cfg["len_traj_pred"]),
                learn_angle=bool(network_cfg["learn_angle"]),
                context_size=int(network_cfg["context_size"]),
                context_type=str(dataset_cfg["context_type"]),
                end_slack=int(data_config["end_slack"]),
                goals_per_obs=int(data_config["goals_per_obs"]),
                normalize=bool(dataset_cfg["normalize"]),
                modality_id=int(data_config["modality_id"]),
                metric_waypoint_spacing=float(data_config.get("metric_waypoint_spacing", 1.0)),
                clip_image_size=tuple(dataset_cfg.get("clip_image_size", (224, 224))),
                clip_model=str(dataset_cfg.get("clip_model", "ViT-B/32")),
            )
            if data_split_type == "train":
                train_datasets.append(dataset)
                train_eval_datasets[f"{dataset_name}_{data_split_type}"] = dataset
            else:
                test_datasets[f"{dataset_name}_{data_split_type}"] = dataset

    if not train_datasets:
        raise ValueError("No train datasets were configured in dataset.yaml.")

    train_loader = DataLoader(
        ConcatDataset(train_datasets),
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        collate_fn=collate_edge_samples,
        pin_memory=torch.cuda.is_available(),
    )
    train_eval_dataloaders = {
        dataset_type: DataLoader(
            dataset,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(train_cfg["num_workers"]),
            collate_fn=collate_edge_samples,
            pin_memory=torch.cuda.is_available(),
        )
        for dataset_type, dataset in train_eval_datasets.items()
    }
    test_dataloaders = {
        dataset_type: DataLoader(
            dataset,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(train_cfg["num_workers"]),
            collate_fn=collate_edge_samples,
            pin_memory=torch.cuda.is_available(),
        )
        for dataset_type, dataset in test_datasets.items()
    }
    for dataset_type, dataset in test_datasets.items():
        print(f"[NavVLA] Loaded {len(dataset)} test samples for {dataset_type}")
    return train_loader, train_eval_dataloaders, test_dataloaders


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="training/config/train.yaml",
        help="Path to train config YAML.",
    )
    parser.add_argument(
        "--network-config",
        default="training/config/network.yaml",
        help="Path to network config YAML. Defaults to sibling network.yaml.",
    )
    parser.add_argument(
        "--dataset-config",
        default="training/config/dataset.yaml",
        help="Path to dataset config YAML. Defaults to sibling dataset.yaml.",
    )
    args = parser.parse_args()

    navvla_root = Path(__file__).resolve().parent
    train_cfg_path = (navvla_root / args.config).resolve()
    network_cfg_path = (
        (navvla_root / args.network_config).resolve()
        if args.network_config
        else train_cfg_path.with_name("network.yaml")
    )
    dataset_cfg_path = (
        (navvla_root / args.dataset_config).resolve()
        if args.dataset_config
        else train_cfg_path.with_name("dataset.yaml")
    )

    with train_cfg_path.open("r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f) or {}
    with network_cfg_path.open("r", encoding="utf-8") as f:
        network_cfg = yaml.safe_load(f) or {}
    with dataset_cfg_path.open("r", encoding="utf-8") as f:
        dataset_cfg = yaml.safe_load(f) or {}

    if not isinstance(train_cfg, dict):
        raise ValueError(f"YAML root must be a mapping: {train_cfg_path}")
    if not isinstance(network_cfg, dict):
        raise ValueError(f"YAML root must be a mapping: {network_cfg_path}")
    if not isinstance(dataset_cfg, dict):
        raise ValueError(f"YAML root must be a mapping: {dataset_cfg_path}")

    train_loader, train_eval_dataloaders, test_dataloaders = create_dataloaders(
        train_cfg=train_cfg,
        network_cfg=network_cfg,
        dataset_cfg=dataset_cfg,
    )

    return main_loop(
        navvla_root=navvla_root,
        train_cfg=train_cfg,
        network_cfg=network_cfg,
        dataset_cfg=dataset_cfg,
        train_loader=train_loader,
        train_eval_dataloaders=train_eval_dataloaders,
        test_dataloaders=test_dataloaders,
    )


if __name__ == "__main__":
    raise SystemExit(main())
