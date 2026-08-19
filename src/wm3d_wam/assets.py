"""Strict local-only model asset loading for WM3D-WAM.

Provisioning is an explicit operator step.  Training workers never call a Hub
client, choose a mirror, or download a missing file at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from safetensors import safe_open

from wm3d_wam.vendor.fastwam.wan22.helpers.state_dict_converters import (
    wan_video_vae_state_dict_converter,
)
from wm3d_wam.vendor.fastwam.wan22.wan_video_dit import WanVideoDiT
from wm3d_wam.vendor.fastwam.wan22.wan_video_text_encoder import (
    HuggingfaceTokenizer,
    WanTextEncoder,
)
from wm3d_wam.vendor.fastwam.wan22.wan_video_vae import WanVideoVAE38


class LocalAssetError(RuntimeError):
    pass


@dataclass(frozen=True)
class LocalWanAssetPaths:
    root: Path
    dit_shards: tuple[Path, ...]
    vae: Path
    text_encoder: Path
    tokenizer: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "LocalWanAssetPaths":
        root = Path(root).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise LocalAssetError(f"Wan asset root is not a directory: {root}")
        dit_shards = tuple(sorted(root.glob("diffusion_pytorch_model-*.safetensors")))
        if not dit_shards:
            single = root / "diffusion_pytorch_model.safetensors"
            dit_shards = (single,) if single.is_file() else ()
        vae = root / "Wan2.2_VAE.pth"
        text = root / "models_t5_umt5-xxl-enc-bf16.pth"
        tokenizer = root / "google/umt5-xxl"
        missing = [
            str(path)
            for path in (*dit_shards, vae, text, tokenizer)
            if not path.exists()
        ]
        if not dit_shards:
            missing.append(str(root / "diffusion_pytorch_model*.safetensors"))
        if missing:
            raise LocalAssetError(f"Wan asset bundle is incomplete: {missing}")
        if any(path.is_symlink() or not path.is_file() for path in dit_shards):
            raise LocalAssetError("Wan DiT shards must be regular local files")
        if vae.is_symlink() or not vae.is_file():
            raise LocalAssetError("Wan VAE must be a regular local file")
        if text.is_symlink() or not text.is_file():
            raise LocalAssetError("Wan text encoder must be a regular local file")
        if tokenizer.is_symlink() or not tokenizer.is_dir():
            raise LocalAssetError("Wan tokenizer must be a regular local directory")
        return cls(root, dit_shards, vae, text, tokenizer)


@dataclass
class LocalWanComponents:
    dit: Optional[WanVideoDiT]
    vae: Optional[WanVideoVAE38]
    text_encoder: Optional[WanTextEncoder]
    tokenizer: Optional[HuggingfaceTokenizer]
    paths: LocalWanAssetPaths


def _load_safetensor_shards(
    paths: Iterable[Path], *, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for path in paths:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            overlap = set(state).intersection(handle.keys())
            if overlap:
                raise LocalAssetError(
                    f"duplicate keys across Wan shards: {sorted(overlap)[:8]}"
                )
            for key in handle.keys():
                state[key] = handle.get_tensor(key).to(dtype=dtype)
    if not state:
        raise LocalAssetError("Wan DiT shards contain no tensors")
    return state


def _unwrap_torch_state(value: object) -> dict[str, torch.Tensor]:
    if not isinstance(value, dict):
        raise LocalAssetError(f"checkpoint root must be a dict, got {type(value)}")
    for key in ("state_dict", "model", "module", "model_state"):
        nested = value.get(key)
        if isinstance(nested, dict):
            value = nested
            break
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        raise LocalAssetError("checkpoint state dict is malformed")
    return value  # type: ignore[return-value]


def _load_torch_state(path: Path, *, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    value = torch.load(str(path), map_location="cpu", weights_only=True)
    state = _unwrap_torch_state(value)
    return {
        key: tensor.to(dtype=dtype) if isinstance(tensor, torch.Tensor) else tensor
        for key, tensor in state.items()
    }


def _assign_state(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
    *,
    name: str,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.nn.Module:
    incompatible = model.load_state_dict(state, strict=False, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise LocalAssetError(
            f"{name} checkpoint mismatch: missing={incompatible.missing_keys[:8]}, "
            f"unexpected={incompatible.unexpected_keys[:8]}"
        )
    return model.to(device=device, dtype=dtype)


def _meta_model(model_type, *args, **kwargs):
    try:
        with torch.device("meta"):
            return model_type(*args, **kwargs)
    except Exception as exc:
        raise LocalAssetError(
            f"{model_type.__name__} cannot be constructed on meta device: {exc}"
        ) from exc


def load_local_wan_components(
    *,
    asset_root: str | Path,
    dit_config: dict[str, Any],
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
    tokenizer_max_len: int = 128,
    load_dit: bool = True,
    load_vae: bool = True,
    load_text_encoder: bool = True,
) -> LocalWanComponents:
    """Materialize requested Wan2.2 components from one complete local bundle."""

    paths = LocalWanAssetPaths.from_root(asset_root)
    dit = None
    vae = None
    text_encoder = None
    tokenizer = None
    if load_dit:
        state = _load_safetensor_shards(paths.dit_shards, dtype=dtype)
        dit = _meta_model(WanVideoDiT, **dict(dit_config))
        dit = _assign_state(
            dit, state, name="Wan2.2 DiT", device=device, dtype=dtype
        )
        del state
    if load_vae:
        raw = _load_torch_state(paths.vae, dtype=dtype)
        state = wan_video_vae_state_dict_converter(raw)
        vae = _meta_model(WanVideoVAE38)
        vae = _assign_state(
            vae, state, name="Wan2.2 VAE", device=device, dtype=dtype
        )
        del raw, state
    if load_text_encoder:
        state = _load_torch_state(paths.text_encoder, dtype=dtype)
        text_encoder = _meta_model(WanTextEncoder)
        text_encoder = _assign_state(
            text_encoder,
            state,
            name="Wan UMT5 encoder",
            device=device,
            dtype=dtype,
        )
        tokenizer = HuggingfaceTokenizer(
            name=str(paths.tokenizer),
            seq_len=int(tokenizer_max_len),
            clean="whitespace",
        )
        del state
    return LocalWanComponents(
        dit=dit,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        paths=paths,
    )


@torch.no_grad()
def encode_local_wan_prompts(
    components: LocalWanComponents,
    prompts: list[str],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if components.text_encoder is None or components.tokenizer is None:
        raise LocalAssetError("text encoder/tokenizer were not loaded")
    ids, mask = components.tokenizer(
        prompts, return_mask=True, add_special_tokens=True
    )
    ids = ids.to(device=device)
    mask = mask.to(device=device, dtype=torch.bool)
    context = components.text_encoder(ids, mask).to(device=device, dtype=dtype)
    context = context * mask.unsqueeze(-1).to(dtype=context.dtype)
    return context, mask
