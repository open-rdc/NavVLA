#!/usr/bin/env python3
"""Training entrypoint for NavVLA."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from training.loop import main_loop


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

    return main_loop(
        navvla_root=navvla_root,
        train_cfg=train_cfg,
        network_cfg=network_cfg,
        dataset_cfg=dataset_cfg,
    )


if __name__ == "__main__":
    raise SystemExit(main())
