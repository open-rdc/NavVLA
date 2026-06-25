#!/usr/bin/env python3

import argparse
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np


NODE_INTERVAL = 10
TRAJ_NAME = 'traj_episode'


def encode_topomap_embeddings(dataset_dir: Path, clip_model: str) -> None:
    """Generate clip_embeddings.bin for the assembled topomap via the canonical encoder."""
    pfoe_scripts = Path(__file__).resolve().parents[2] / 'pfoe' / 'scripts'
    sys.path.insert(0, str(pfoe_scripts))
    import record_episode

    record_episode.encode_episode(dataset_dir, TRAJ_NAME, clip_model)


def create_topomap(dataset_dir: Path, clip_model: str = 'ViT-B/32') -> None:
    episode_dirs = sorted(
        d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith('episode')
    )
    if not episode_dirs:
        raise FileNotFoundError(f'No episode directories found in {dataset_dir}')

    topomap_dir = dataset_dir / TRAJ_NAME
    if topomap_dir.exists():
        shutil.rmtree(topomap_dir)
    topomap_dir.mkdir(parents=True)

    positions = []
    yaws = []
    prompts = []
    node_index = 0

    for episode_dir in episode_dirs:
        jpg_files = sorted(episode_dir.glob('*.jpg'), key=lambda p: int(p.stem))
        with (episode_dir / 'traj_data.pkl').open('rb') as f:
            traj_data = pickle.load(f)
        episode_positions = np.asarray(traj_data['position'], dtype=np.float32)
        episode_yaws = np.asarray(traj_data['yaw'], dtype=np.float32)
        prompt_path = episode_dir / 'traj_prompt.txt'
        episode_prompts = (
            prompt_path.read_text(encoding='utf-8').splitlines() if prompt_path.exists() else []
        )

        node_count = min(len(jpg_files), len(episode_positions))
        for frame_index in range(0, node_count, NODE_INTERVAL):
            shutil.copy(jpg_files[frame_index], topomap_dir / f'{node_index}.jpg')
            positions.append(episode_positions[frame_index])
            yaws.append(episode_yaws[frame_index])
            prompts.append(
                episode_prompts[frame_index] if frame_index < len(episode_prompts) else ''
            )
            node_index += 1

    traj_data = {
        'position': np.asarray(positions, dtype=np.float32),
        'yaw': np.asarray(yaws, dtype=np.float32),
    }
    with (topomap_dir / 'traj_data.pkl').open('wb') as f:
        pickle.dump(traj_data, f)

    (topomap_dir / 'traj_prompt.txt').write_text('\n'.join(prompts) + '\n', encoding='utf-8')

    encode_topomap_embeddings(dataset_dir, clip_model)

    print(f'Created {node_index} nodes from {len(episode_dirs)} episodes at {topomap_dir}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=Path)
    parser.add_argument('--clip_model', default='ViT-B/32')
    args = parser.parse_args()
    create_topomap(args.dataset_dir, args.clip_model)


if __name__ == '__main__':
    main()
