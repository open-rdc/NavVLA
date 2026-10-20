#!/bin/bash
#SBATCH -p research
#SBATCH -t 24:00:00
#SBATCH --gpus-per-node=1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -J omni_ftv2
#SBATCH -o omnivla_fisheye_tsudanuma_v2_%j.out
# 【H100(津田沼 / gpumng2.cle.it-chiba.ac.jp)用】
# OmniVLA-edge を魚眼カメラの津田沼(コース+ランダム)だけで 30 エポック学習する。注釈は 2026-10-20 に見直しが終わったもの。設定は fisheye_v1 と揃える。
# データは ~/navvla_data/fisheye_tsudanuma_v2(画像は v1 とハードリンクで共有し traj_prompt.txt だけ差し替え)。分割は v1 と同じ(test=tsudanuma_20261010_134655)。
# 24 時間で終わらなければ、同じスクリプトを投入し直すと checkpoint_last.pth から続く。
#
# 使い方: `sbatch training/gpgpu_train_fisheye_tsudanuma_v2_h100.sbatch`
set -euo pipefail
cd $HOME/NavVLA
RUN_DIR=$HOME/NavVLA/training/runs/fisheye_tsudanuma_v2_omnivla
if [ -e "$RUN_DIR/model_final.pth" ]; then
  echo "学習は完了済み ($RUN_DIR/model_final.pth)。何もしない"
  exit 0
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$HOME/navvla_venv/bin/python train.py \
  --config training/config/fisheye_tsudanuma_v2/train.yaml \
  --network-config training/config/fisheye_tsudanuma_v2/network.yaml \
  --dataset-config training/config/fisheye_tsudanuma_v2/dataset.yaml
echo "OMNI_FTV2_DONE $(date)"
