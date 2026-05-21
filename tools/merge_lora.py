from __future__ import annotations
import argparse
from pathlib import Path
import torch
import yaml
from peft import PeftModel
from OmniVLA.inference.model_omnivla_edge import OmniVLA_edge


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base",        required=True, help="ベースモデル重みパス (.pth)")
    parser.add_argument("--adapter",     required=True, help="LoRA アダプタディレクトリ")
    parser.add_argument("--out",         required=True, help="マージ済み重みの出力パス (.pth)")
    parser.add_argument("--network-cfg", default="training/config/network.yaml",
                        help="ネットワーク設定 YAML（デフォルト: training/config/network.yaml）")
    args = parser.parse_args()

    with open(args.network_cfg) as f:
        net = yaml.safe_load(f)

    model = OmniVLA_edge(
        context_size=int(net["context_size"]),
        len_traj_pred=int(net["len_traj_pred"]),
        learn_angle=bool(net["learn_angle"]),
        obs_encoder=str(net["obs_encoder"]),
        obs_encoding_size=int(net["obs_encoding_size"]),
        late_fusion=bool(net["late_fusion"]),
        mha_num_attention_heads=int(net["mha_num_attention_heads"]),
        mha_num_attention_layers=int(net["mha_num_attention_layers"]),
        mha_ff_dim_factor=int(net["mha_ff_dim_factor"]),
    )
    checkpoint = torch.load(args.base, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=True)

    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"[merge_lora] saved → {args.out}")


if __name__ == "__main__":
    main()
