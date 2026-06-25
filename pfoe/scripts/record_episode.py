#!/usr/bin/env python3
"""Offline CLIP encoding tool for episode images.

Usage:
    python3 pfoe/scripts/record_episode.py \
        --data_folder data/real/3f/navvla_3f_linestop \
        --traj_name   episode01 \
        --clip_model  ViT-B/32
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image as PILImage


def write_embeddings_bin(out_file, embeddings) -> None:
    """Write embeddings as ``int32 N, int32 dim, float32[N][dim]`` (little-endian).

    Matches the layout read by ``pfoe/src/Episode.cpp``.
    """
    arr = np.ascontiguousarray(embeddings, dtype="<f4")
    if arr.ndim != 2:
        raise ValueError(f"embeddings must be 2-D [N, dim], got shape {arr.shape}")
    n, dim = arr.shape
    with open(out_file, "wb") as f:
        f.write(struct.pack("<ii", n, dim))
        f.write(arr.tobytes())


def encode_episode(data_folder: Path, traj_name: str, clip_model: str) -> None:
    import torch
    import clip

    traj_dir = data_folder / traj_name
    prompt_file = traj_dir / "traj_prompt.txt"
    out_file = traj_dir / "clip_embeddings.bin"

    # collect frames sorted numerically
    jpg_files = sorted(traj_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    if not jpg_files:
        print(f"[ERROR] No jpg files found in {traj_dir}", file=sys.stderr)
        sys.exit(1)

    n = len(jpg_files)
    print(f"Found {n} frames in {traj_dir}")

    # validate prompt file line count
    if prompt_file.exists():
        with open(prompt_file) as f:
            prompt_lines = [l.rstrip() for l in f if l.strip()]
        if len(prompt_lines) != n:
            print(
                f"[WARN] traj_prompt.txt has {len(prompt_lines)} lines "
                f"but found {n} images — counts differ"
            )
    else:
        print(f"[WARN] {prompt_file} not found")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(clip_model, device=device)
    model.eval()
    print(f"CLIP model={clip_model}, device={device}")

    embeddings = []
    for i, jpg in enumerate(jpg_files):
        img = PILImage.open(jpg).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(tensor).squeeze().cpu().numpy().astype(np.float32)
        embeddings.append(feat)
        if (i + 1) % 20 == 0:
            print(f"  encoded {i + 1}/{n}")

    dim = embeddings[0].shape[0]
    print(f"Writing {out_file}  (N={n}, dim={dim})")
    write_embeddings_bin(out_file, np.stack(embeddings))

    print("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", required=True, type=Path)
    parser.add_argument("--traj_name",   required=True)
    parser.add_argument("--clip_model",  default="ViT-B/32")
    args = parser.parse_args()
    encode_episode(args.data_folder, args.traj_name, args.clip_model)


if __name__ == "__main__":
    main()
