#!/usr/bin/env python3
"""Create the ActionDiT backbone from a strictly local Wan2.2 asset bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from wm3d_wam.assets import load_local_wan_components
from wm3d_wam.vendor.fastwam.wan22.action_dit import ActionDiT


def _interpolate_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).float()
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)


def _resize_tensor(src: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    if tuple(src.shape) == target_shape:
        return src
    out = src.float()
    while out.ndim < len(target_shape):
        out = out.unsqueeze(0)
    while out.ndim > len(target_shape):
        if out.shape[0] != 1:
            raise ValueError(
                f"cannot reduce {tuple(src.shape)} to target rank {len(target_shape)}"
            )
        out = out.squeeze(0)
    for dimension, new_size in enumerate(target_shape):
        if out.shape[dimension] == new_size:
            continue
        permutation = [index for index in range(out.ndim) if index != dimension]
        permutation.append(dimension)
        inverse = [0] * out.ndim
        for index, value in enumerate(permutation):
            inverse[value] = index
        out = out.permute(*permutation).contiguous()
        out = _interpolate_last_dim(out, new_size)
        out = out.permute(*inverse).contiguous()
    if tuple(out.shape) != target_shape:
        raise RuntimeError("ActionDiT interpolation produced the wrong shape")
    return out.to(dtype=src.dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)
    config = yaml.safe_load(Path(args.model_config).read_text(encoding="utf-8"))
    video_config = dict(config["video_expert"])
    video_config.pop("_target_", None)
    action_config = dict(config["action_expert"])
    action_config.pop("_target_", None)
    action_config.pop("codec_config", None)
    action_config["action_dim"] = 1
    action_config["use_gradient_checkpointing"] = False

    components = load_local_wan_components(
        asset_root=args.asset_root,
        dit_config=video_config,
        device=args.device,
        dtype=dtype,
        load_dit=True,
        load_vae=False,
        load_text_encoder=False,
    )
    if components.dit is None:
        raise RuntimeError("local Wan DiT was not loaded")
    video = components.dit
    action = ActionDiT(**action_config).to(device=args.device, dtype=dtype)
    action_state = action.state_dict()
    video_state = video.state_dict()
    keys = ActionDiT.backbone_key_set(action_state.keys())
    converted: dict[str, torch.Tensor] = {}
    copied = 0
    interpolated = 0
    for key in sorted(keys):
        if key not in video_state:
            raise ValueError(f"Wan DiT does not contain ActionDiT backbone key {key!r}")
        source = video_state[key]
        target = action_state[key]
        if source.shape == target.shape:
            value = source
            copied += 1
        else:
            value = _resize_tensor(source, tuple(target.shape))
            if source.ndim >= 2 and source.shape[-1] != target.shape[-1]:
                value = value.float() * (
                    float(source.shape[-1]) / float(target.shape[-1])
                ) ** 0.5
            interpolated += 1
        converted[key] = value.detach().to(
            device="cpu", dtype=target.dtype
        ).contiguous()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy": {
                "alpha_scaling": True,
                "interpolation": "sequential_1d_linear_align_corners_true",
            },
            "backbone_state_dict": converted,
            "meta": {
                "hidden_dim": int(action_config["hidden_dim"]),
                "ffn_dim": int(action_config["ffn_dim"]),
                "num_layers": int(action_config["num_layers"]),
                "num_heads": int(action_config["num_heads"]),
                "attn_head_dim": int(action_config["attn_head_dim"]),
                "text_dim": int(action_config["text_dim"]),
                "freq_dim": int(action_config["freq_dim"]),
                "eps": float(action_config["eps"]),
            },
        },
        output,
    )
    print(
        {
            "output": str(output),
            "copied": copied,
            "interpolated": interpolated,
            "backbone_keys": len(converted),
        }
    )


if __name__ == "__main__":
    main()
