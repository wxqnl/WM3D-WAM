"""VGGT backbone adapter for the GAM shallow-predict-deep contract.

The official VGGT aggregator is a stack of 24 frame/global block pairs.  GAM
needs to stop the geometry model at a shallow layer, predict future shallow
tokens and an action token, then resume the trainable deep stack.  This module
implements that split without modifying the pinned VGGT source tree.

Temporal policy
---------------
Shallow encoding treats each timestep as an independent multi-camera VGGT
scene, matching the GAM paper. Deep propagation supports either independent
timesteps or cross-timestep global attention with a strict block-causal mask.
Both modes are bit-identical to official VGGT for ``H=1``.
"""

from __future__ import annotations

import inspect
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint


try:
    from torch.nn.attention.flex_attention import (
        create_block_mask as _create_block_mask,
        flex_attention as _flex_attention_raw,
    )

    _flex_attention = torch.compile(_flex_attention_raw)
    _HAS_FLEX_ATTENTION = True
except Exception:
    _create_block_mask = None
    _flex_attention = None
    _HAS_FLEX_ATTENTION = False


def _make_temporal_causal_mask_mod(token_count: int, views: int):
    def mask_mod(batch_idx, head_idx, query_idx, key_value_idx):
        del batch_idx, head_idx
        query_timestep = (query_idx // token_count) // views
        key_timestep = (key_value_idx // token_count) // views
        return query_timestep >= key_timestep

    return mask_mod


def _cuda_profile_mark(profile: Optional[Dict[str, object]], name: str) -> None:
    if profile is None or not torch.cuda.is_available():
        return
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    profile.setdefault("_cuda_marks", []).append((name, event))


def _ensure_local_vggt_on_path(source_root: str) -> Path:
    root = Path(source_root).expanduser().resolve(strict=True)
    for name, module in tuple(sys.modules.items()):
        if name != "vggt" and not name.startswith("vggt."):
            continue
        module_path = getattr(module, "__file__", None)
        if module_path is None:
            continue
        try:
            Path(module_path).resolve(strict=True).relative_to(root)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                "A VGGT module from a different source tree is already loaded: "
                f"{name}={module_path}; requested root={root}."
            ) from exc
    sys.path[:] = [entry for entry in sys.path if entry != str(root)]
    sys.path.insert(0, str(root))
    return root


class VGGTEncoder(nn.Module):
    """Expose VGGT through the Stage-1 GAM backbone API.

    ``split_layer`` is the first deep frame/global pair.  With the recommended
    value 4, shallow encoding runs pairs 0--3 and deep propagation runs 4--23.
    VGGT's four DPT taps (4, 11, 17, 23) therefore all remain downstream of the
    predicted action token.
    """

    PATCH_SIZE = 14
    DPT_LAYER_INDICES = (4, 11, 17, 23)

    def __init__(
        self,
        ckpt_path: str,
        source_root: Optional[str] = None,
        encoder_input_size: int = 224,
        split_layer: int = 4,
        normalization_stat_path: Optional[str] = None,
        freeze_backbone: bool = True,
        n_action_steps: int = 0,
        views_per_timestep: int = 2,
        action_steps_per_token: int = 1,
        use_temporal_embed: bool = False,
        action_input_rate: float = 0.4,
        action_only_frame_attn: bool = False,
        deep_time_mode: str = "independent",
        local_files_only: bool = True,
        logger=None,
    ):
        super().__init__()
        if not ckpt_path:
            raise ValueError("stage_1.ckpt_path is required for the VGGT backbone.")
        source_root = source_root or os.environ.get("VGGT_SOURCE_ROOT")
        if not source_root:
            raise ValueError(
                "VGGT source root is required. Set stage_1.source_root or "
                "the VGGT_SOURCE_ROOT environment variable."
            )
        if normalization_stat_path:
            raise ValueError(
                "VGGT shallow-token normalization stats are not supported; "
                "set stage_1.normalization_stat_path=null."
            )
        if use_temporal_embed:
            raise ValueError(
                "VGGTEncoder does not add a second temporal embedding. "
                "Temporal information is modeled by GAMFuturePredictor."
            )
        if action_only_frame_attn:
            raise ValueError("action_only_frame_attn is DA3-specific and unsupported by VGGTEncoder.")
        deep_time_mode = str(deep_time_mode).lower()
        if deep_time_mode not in {"independent", "causal"}:
            raise ValueError(
                "VGGTEncoder deep_time_mode must be 'independent' or 'causal'."
            )

        self.encoder_input_size = int(encoder_input_size)
        self.patch_size = self.PATCH_SIZE
        if self.encoder_input_size % self.patch_size:
            raise ValueError(
                f"encoder_input_size={self.encoder_input_size} must be divisible "
                f"by patch_size={self.patch_size}."
            )
        self.h_patches = self.encoder_input_size // self.patch_size
        self.w_patches = self.h_patches
        self.num_patches = self.h_patches * self.w_patches
        self.num_register_tokens = 4
        self.embed_dim = 1024
        self.hidden_size = 2 * self.embed_dim
        self.views_per_timestep = int(views_per_timestep)
        self.n_action_steps = int(n_action_steps)
        self.action_steps_per_token = int(action_steps_per_token)
        self.action_input_rate = float(action_input_rate)
        self.deep_time_mode = deep_time_mode

        source_path = _ensure_local_vggt_on_path(source_root)
        from vggt.models.vggt import VGGT

        source_file = Path(inspect.getsourcefile(VGGT) or "").resolve(strict=True)
        try:
            source_file.relative_to(source_path)
        except ValueError as exc:
            raise RuntimeError(
                f"VGGT resolved outside requested source tree: {source_file}"
            ) from exc

        checkpoint = Path(ckpt_path).expanduser().resolve(strict=True)
        if checkpoint.is_dir():
            model = VGGT.from_pretrained(
                str(checkpoint), local_files_only=bool(local_files_only)
            )
        else:
            model = VGGT()
            if checkpoint.suffix == ".safetensors":
                from safetensors.torch import load_file

                state = load_file(str(checkpoint), device="cpu")
            else:
                load_kwargs = {"map_location": "cpu"}
                if "weights_only" in inspect.signature(torch.load).parameters:
                    load_kwargs["weights_only"] = True
                state = torch.load(str(checkpoint), **load_kwargs)
            if isinstance(state, dict):
                for key in ("state_dict", "model", "module"):
                    nested = state.get(key)
                    if isinstance(nested, dict):
                        state = nested
                        break
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    "VGGT checkpoint mismatch: "
                    f"missing={missing[:8]}, unexpected={unexpected[:8]}"
                )

        # WM3D-WAM keeps geometry inside the model, so camera, depth, and point
        # heads remain attached. Tracking is outside the v1 training contract.
        model.track_head = None
        missing_geometry_heads = [
            name
            for name in ("camera_head", "depth_head", "point_head")
            if getattr(model, name, None) is None
        ]
        if missing_geometry_heads:
            raise RuntimeError(
                "The selected VGGT checkpoint is missing required geometry heads: "
                f"{missing_geometry_heads}"
            )
        self.vggt_model = model
        self.vggt_source_root = str(source_path)
        self.vggt_source_file = str(source_file)
        self.model_snapshot_path = str(checkpoint)

        aggregator = self.aggregator
        self.block_count = int(aggregator.depth)
        self.out_layers = list(self.DPT_LAYER_INDICES)
        self.split_layer = int(split_layer)
        if not (0 < self.split_layer <= min(self.out_layers)):
            raise ValueError(
                f"split_layer must be in [1, {min(self.out_layers)}] so all VGGT "
                f"DPT taps remain deep; got {self.split_layer}."
            )
        self.shallow_target_layer = self.split_layer - 1
        self.default_freeze_blocks_before = self.split_layer
        self._deep_flex_block_mask_cache: Dict[tuple, object] = {}
        if int(aggregator.patch_start_idx) != 1 + self.num_register_tokens:
            raise RuntimeError(
                "Unexpected VGGT special-token layout: "
                f"patch_start_idx={aggregator.patch_start_idx}."
            )
        if len(aggregator.frame_blocks) != self.block_count or len(
            aggregator.global_blocks
        ) != self.block_count:
            raise RuntimeError("VGGT frame/global block counts do not match aggregator.depth.")

        # The generic GAM batch helper applies (x - encoder_mean)/encoder_std.
        # VGGT performs ImageNet normalization inside token preparation, so keep
        # the outer transform as identity to normalize exactly once.
        self.register_buffer(
            "encoder_mean", torch.zeros(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "encoder_std", torch.ones(1, 3, 1, 1), persistent=False
        )

        self._set_backbone_trainable(not freeze_backbone)
        for head in (self.camera_head, self.dpt_head, self.point_head):
            head.eval()
            for parameter in head.parameters():
                parameter.requires_grad = False
            head.float()

        if logger is not None:
            logger.info(
                "VGGT loaded from %s (source=%s, split=%d, deep_time_mode=%s)",
                checkpoint,
                source_path,
                self.split_layer,
                self.deep_time_mode,
            )

    @property
    def aggregator(self):
        return self.vggt_model.aggregator

    @property
    def dpt_head(self):
        return self.vggt_model.depth_head

    @property
    def camera_head(self):
        return self.vggt_model.camera_head

    @property
    def point_head(self):
        return self.vggt_model.point_head

    def _set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.vggt_model.parameters():
            parameter.requires_grad = False
        if trainable:
            for idx in range(self.split_layer, self.block_count):
                for parameter in self.aggregator.frame_blocks[idx].parameters():
                    parameter.requires_grad = True
                for parameter in self.aggregator.global_blocks[idx].parameters():
                    parameter.requires_grad = True

    def freeze_blocks_before(self, block_idx: int) -> None:
        """Freeze all VGGT parameters except frame/global pairs at or above idx."""
        block_idx = int(block_idx)
        if not (0 <= block_idx <= self.block_count):
            raise ValueError(
                f"freeze_blocks_before must be in [0, {self.block_count}], got {block_idx}."
            )
        for parameter in self.vggt_model.parameters():
            parameter.requires_grad = False
        for idx in range(max(block_idx, self.split_layer), self.block_count):
            for parameter in self.aggregator.frame_blocks[idx].parameters():
                parameter.requires_grad = True
            for parameter in self.aggregator.global_blocks[idx].parameters():
                parameter.requires_grad = True

    def normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        return images

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def denormalize(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def project_proprio(self, proprio: Optional[torch.Tensor]) -> None:
        return None

    @staticmethod
    def _slice_expand_and_flatten(
        token_tensor: torch.Tensor, batch_size: int, views: int
    ) -> torch.Tensor:
        query = token_tensor[:, 0:1].expand(
            batch_size, 1, *token_tensor.shape[2:]
        )
        others = token_tensor[:, 1:2].expand(
            batch_size, max(views - 1, 0), *token_tensor.shape[2:]
        )
        return torch.cat([query, others], dim=1).reshape(
            batch_size * views, *token_tensor.shape[2:]
        )

    def _build_positions(
        self,
        scene_batch: int,
        views: int,
        grid_h: int,
        grid_w: int,
        device: torch.device,
        *,
        with_action: bool,
    ) -> Optional[torch.Tensor]:
        aggregator = self.aggregator
        if aggregator.rope is None:
            return None
        pos = aggregator.position_getter(
            scene_batch * views, grid_h, grid_w, device=device
        )
        pos = pos + 1
        special_count = int(aggregator.patch_start_idx)
        special = torch.zeros(
            scene_batch * views,
            special_count,
            2,
            device=device,
            dtype=pos.dtype,
        )
        pos = torch.cat([special, pos], dim=1)
        if with_action:
            action_pos = torch.zeros(
                scene_batch * views, 1, 2, device=device, dtype=pos.dtype
            )
            pos = torch.cat(
                [pos[:, :special_count], action_pos, pos[:, special_count:]], dim=1
            )
        return pos

    def _prepare_scene_tokens(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        scene_batch, views, channels, height, width = images.shape
        if channels != 3:
            raise ValueError(f"Expected RGB input, got {channels} channels.")
        if height != self.encoder_input_size or width != self.encoder_input_size:
            raise ValueError(
                f"VGGTEncoder expects {self.encoder_input_size}x{self.encoder_input_size}, "
                f"got {height}x{width}."
            )
        aggregator = self.aggregator
        normalized = (
            images - aggregator._resnet_mean.to(device=images.device, dtype=images.dtype)
        ) / aggregator._resnet_std.to(device=images.device, dtype=images.dtype)
        patch_tokens = aggregator.patch_embed(
            normalized.reshape(scene_batch * views, channels, height, width)
        )
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        camera = self._slice_expand_and_flatten(
            aggregator.camera_token, scene_batch, views
        )
        registers = self._slice_expand_and_flatten(
            aggregator.register_token, scene_batch, views
        )
        tokens = torch.cat([camera, registers, patch_tokens], dim=1)
        pos = self._build_positions(
            scene_batch,
            views,
            height // self.patch_size,
            width // self.patch_size,
            images.device,
            with_action=False,
        )
        return tokens.reshape(scene_batch, views, tokens.shape[1], tokens.shape[2]), pos

    @staticmethod
    def _apply_block(
        block: nn.Module,
        tokens: torch.Tensor,
        pos: Optional[torch.Tensor],
        gradient_checkpointing: bool,
    ) -> torch.Tensor:
        if (
            gradient_checkpointing
            and torch.is_grad_enabled()
            and tokens.requires_grad
        ):
            return torch_checkpoint(
                lambda value: block(value, pos=pos),
                tokens,
                use_reentrant=False,
            )
        return block(tokens, pos=pos)

    def _run_pairs(
        self,
        tokens: torch.Tensor,
        pos: Optional[torch.Tensor],
        start_layer: int,
        *,
        gradient_checkpointing: bool,
        capture_layers: bool,
    ) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
        scene_batch, views, token_count, channels = tokens.shape
        cached: List[Optional[torch.Tensor]] = [None] * self.block_count
        current = tokens
        for layer_idx in range(int(start_layer), self.block_count):
            frame_in = current.reshape(
                scene_batch * views, token_count, channels
            )
            frame_pos = None
            if pos is not None:
                frame_pos = pos.reshape(
                    scene_batch * views, token_count, pos.shape[-1]
                )
            frame_out = self._apply_block(
                self.aggregator.frame_blocks[layer_idx],
                frame_in,
                frame_pos,
                gradient_checkpointing,
            )
            frame_view = frame_out.reshape(
                scene_batch, views, token_count, channels
            )

            global_in = frame_view.reshape(
                scene_batch, views * token_count, channels
            )
            global_pos = None
            if pos is not None:
                global_pos = pos.reshape(
                    scene_batch, views * token_count, pos.shape[-1]
                )
            global_out = self._apply_block(
                self.aggregator.global_blocks[layer_idx],
                global_in,
                global_pos,
                gradient_checkpointing,
            )
            current = global_out.reshape(
                scene_batch, views, token_count, channels
            )
            if capture_layers and layer_idx in self.out_layers:
                cached[layer_idx] = torch.cat([frame_view, current], dim=-1)
        return current, cached

    def _causal_block_mask(
        self,
        *,
        steps: int,
        views: int,
        token_count: int,
        device: torch.device,
    ):
        if not _HAS_FLEX_ATTENTION:
            raise RuntimeError(
                "Cross-timestep causal VGGT propagation requires "
                "torch.nn.attention.flex_attention (PyTorch >= 2.5)."
            )
        key = (int(steps), int(views), int(token_count), str(device))
        cached = self._deep_flex_block_mask_cache.get(key)
        if cached is None:
            sequence_length = int(steps) * int(views) * int(token_count)
            cached = _create_block_mask(
                _make_temporal_causal_mask_mod(
                    token_count=int(token_count), views=int(views)
                ),
                B=None,
                H=None,
                Q_LEN=sequence_length,
                KV_LEN=sequence_length,
                device=device,
            )
            self._deep_flex_block_mask_cache[key] = cached
        return cached

    @staticmethod
    def _run_global_block_flex(
        tokens: torch.Tensor,
        block: nn.Module,
        pos: Optional[torch.Tensor],
        block_mask,
    ) -> torch.Tensor:
        if not _HAS_FLEX_ATTENTION:
            raise RuntimeError("FlexAttention is unavailable.")
        if float(getattr(block, "sample_drop_ratio", 0.0)) != 0.0:
            raise RuntimeError(
                "Causal VGGT propagation assumes the official zero drop-path blocks."
            )
        residual = tokens
        hidden = block.norm1(tokens)
        attention = block.attn
        if block.training and float(attention.attn_drop.p) != 0.0:
            raise RuntimeError(
                "Causal VGGT propagation currently requires attention dropout=0."
            )
        batch_size, sequence_length, channels = hidden.shape
        qkv = (
            attention.qkv(hidden)
            .reshape(
                batch_size,
                sequence_length,
                3,
                attention.num_heads,
                attention.head_dim,
            )
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv.unbind(0)
        query = attention.q_norm(query)
        key = attention.k_norm(key)
        if attention.rope is not None:
            query = attention.rope(query, pos)
            key = attention.rope(key, pos)
        # Under bf16 autocast VGGT's q/k LayerNorms may retain fp32 while the
        # unnormalized value tensor is bf16. SDPA autocast handles this
        # implicitly; FlexAttention validates equal dtypes before dispatch.
        query = query.to(dtype=value.dtype)
        key = key.to(dtype=value.dtype)
        attended = _flex_attention(
            query, key, value, block_mask=block_mask
        )
        attended = attended.transpose(1, 2).reshape(
            batch_size, sequence_length, channels
        )
        attended = attention.proj(attended)
        attended = attention.proj_drop(attended)
        tokens = residual + block.ls1(attended)
        tokens = tokens + block.ls2(block.mlp(block.norm2(tokens)))
        return tokens

    def _run_causal_deep_pairs(
        self,
        tokens: torch.Tensor,
        pos: Optional[torch.Tensor],
        *,
        batch_size: int,
        steps: int,
        views: int,
        gradient_checkpointing: bool,
        capture_layers: bool,
    ) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
        """Run VGGT frame attention per view and global attention causally in time."""
        scene_batch, views_in, token_count, channels = tokens.shape
        if scene_batch != int(batch_size) * int(steps) or views_in != int(views):
            raise ValueError("Invalid scene layout for causal VGGT deep propagation.")
        current = tokens.reshape(
            batch_size, steps, views, token_count, channels
        )
        pos_5d = None
        if pos is not None:
            pos_5d = pos.reshape(
                batch_size, steps, views, token_count, pos.shape[-1]
            )
        cached: List[Optional[torch.Tensor]] = [None] * self.block_count
        block_mask = None
        if steps > 1:
            block_mask = self._causal_block_mask(
                steps=steps,
                views=views,
                token_count=token_count,
                device=tokens.device,
            )
        for layer_idx in range(self.split_layer, self.block_count):
            frame_in = current.reshape(
                batch_size * steps * views, token_count, channels
            )
            frame_pos = None if pos_5d is None else pos_5d.reshape(
                batch_size * steps * views, token_count, pos_5d.shape[-1]
            )
            frame_out = self._apply_block(
                self.aggregator.frame_blocks[layer_idx],
                frame_in,
                frame_pos,
                gradient_checkpointing,
            )
            frame_view = frame_out.reshape(
                batch_size, steps, views, token_count, channels
            )

            global_in = frame_view.reshape(
                batch_size, steps * views * token_count, channels
            )
            global_pos = None if pos_5d is None else pos_5d.reshape(
                batch_size, steps * views * token_count, pos_5d.shape[-1]
            )
            global_block = self.aggregator.global_blocks[layer_idx]
            if block_mask is None:
                global_out = self._apply_block(
                    global_block,
                    global_in,
                    global_pos,
                    gradient_checkpointing,
                )
            elif (
                gradient_checkpointing
                and torch.is_grad_enabled()
                and global_in.requires_grad
            ):
                global_out = torch_checkpoint(
                    lambda value, block=global_block: self._run_global_block_flex(
                        value, block, global_pos, block_mask
                    ),
                    global_in,
                    use_reentrant=False,
                )
            else:
                global_out = self._run_global_block_flex(
                    global_in, global_block, global_pos, block_mask
                )
            current = global_out.reshape(
                batch_size, steps, views, token_count, channels
            )
            if capture_layers and layer_idx in self.out_layers:
                cached[layer_idx] = torch.cat(
                    [frame_view, current], dim=-1
                ).reshape(
                    scene_batch, views, token_count, 2 * channels
                )
        return current.reshape(
            scene_batch, views, token_count, channels
        ), cached

    @staticmethod
    def _merge_cached(
        prefix: List[Optional[torch.Tensor]],
        suffix: List[Optional[torch.Tensor]],
    ) -> List[Optional[torch.Tensor]]:
        return [
            right if right is not None else left
            for left, right in zip(prefix, suffix)
        ]

    def _reshape_as_scenes(
        self, images: torch.Tensor, timesteps: int, views: int
    ) -> torch.Tensor:
        if images.ndim == 4:
            images = images.unsqueeze(1)
        batch_size, total_views = images.shape[:2]
        if total_views != int(timesteps) * int(views):
            raise ValueError(
                f"Expected T*V={int(timesteps) * int(views)} views, got {total_views}."
            )
        return images.reshape(
            batch_size * int(timesteps),
            int(views),
            *images.shape[2:],
        )

    @torch.no_grad()
    def encode_shallow_visual_slots(
        self,
        images: torch.Tensor,
        T: int,
        V: int,
        target_layer: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if target_layer is None:
            target_layer = self.shallow_target_layer
        if int(target_layer) != self.shallow_target_layer:
            raise ValueError(
                "VGGT split-resume requires target_layer="
                f"{self.shallow_target_layer}, got {target_layer}."
            )
        if images.ndim == 4:
            images = images.unsqueeze(1)
        batch_size = images.shape[0]
        scenes = self._reshape_as_scenes(images, int(T), int(V))
        tokens, pos = self._prepare_scene_tokens(scenes)
        tokens, _ = self._run_shallow_pairs(tokens, pos)
        visual = tokens.reshape(
            batch_size,
            int(T),
            int(V),
            tokens.shape[2],
            tokens.shape[3],
        )
        return {
            "visual_tokens": visual,
            "raw": visual.reshape(
                batch_size, int(T) * int(V), visual.shape[-2], visual.shape[-1]
            ),
            "layer": torch.tensor(
                self.shallow_target_layer, device=visual.device
            ),
        }

    def _run_shallow_pairs(
        self, tokens: torch.Tensor, pos: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
        scene_batch, views, token_count, channels = tokens.shape
        current = tokens
        cached: List[Optional[torch.Tensor]] = [None] * self.block_count
        for layer_idx in range(self.split_layer):
            frame_pos = None if pos is None else pos.reshape(
                scene_batch * views, token_count, pos.shape[-1]
            )
            frame = self.aggregator.frame_blocks[layer_idx](
                current.reshape(scene_batch * views, token_count, channels),
                pos=frame_pos,
            ).reshape(scene_batch, views, token_count, channels)
            global_pos = None if pos is None else pos.reshape(
                scene_batch, views * token_count, pos.shape[-1]
            )
            current = self.aggregator.global_blocks[layer_idx](
                frame.reshape(scene_batch, views * token_count, channels),
                pos=global_pos,
            ).reshape(scene_batch, views, token_count, channels)
            if layer_idx in self.out_layers:
                cached[layer_idx] = torch.cat([frame, current], dim=-1)
        return current, cached

    def _decode_cached(
        self,
        cached: List[Optional[torch.Tensor]],
        *,
        batch_size: int,
        views: int,
        patch_start_idx: int,
        frames_chunk_size: Optional[int],
    ) -> Dict[str, torch.Tensor]:
        required = list(self.out_layers)
        missing = [idx for idx in required if cached[idx] is None]
        if missing:
            raise RuntimeError(f"Missing VGGT DPT features at layers {missing}.")
        self.dpt_head.float()
        self.point_head.float()
        self.camera_head.float()
        cached_float = [
            value.float() if value is not None else None for value in cached
        ]
        dummy_images = torch.zeros(
            batch_size,
            views,
            3,
            self.encoder_input_size,
            self.encoder_input_size,
            device=cached_float[required[0]].device,
            dtype=torch.float32,
        )
        with torch.autocast(
            device_type=dummy_images.device.type, enabled=False
        ):
            depth, depth_conf = self.dpt_head(
                cached_float,
                images=dummy_images,
                patch_start_idx=int(patch_start_idx),
                frames_chunk_size=frames_chunk_size,
            )
            world_points, world_points_conf = self.point_head(
                cached_float,
                images=dummy_images,
                patch_start_idx=int(patch_start_idx),
                frames_chunk_size=frames_chunk_size,
            )
            pose_enc_list = self.camera_head(cached_float)
        if depth.shape[-1] != 1:
            raise RuntimeError(f"Expected singleton VGGT depth channel, got {depth.shape}.")
        if world_points.shape[-1] != 3:
            raise RuntimeError(
                f"Expected three VGGT world-point channels, got {world_points.shape}."
            )
        return {
            "depth": depth.squeeze(-1).reshape(
                batch_size * views, *depth.shape[2:4]
            ),
            "depth_conf": depth_conf.reshape(
                batch_size * views, *depth_conf.shape[2:]
            ),
            "world_points": world_points.reshape(
                batch_size * views, *world_points.shape[2:]
            ),
            "world_points_conf": world_points_conf.reshape(
                batch_size * views, *world_points_conf.shape[2:]
            ),
            "pose_enc": pose_enc_list[-1].reshape(batch_size * views, -1),
        }

    def _propagate_impl(
        self,
        visual_tokens: torch.Tensor,
        action_tokens: Optional[torch.Tensor],
        *,
        decode_visuals: bool,
        dpt_chunk_size: Optional[int],
        gradient_checkpointing: bool,
        return_multi_level: bool,
        step_valid_mask: Optional[torch.Tensor],
        deep_temporal_causal_mask: bool,
        profile: Optional[Dict[str, object]],
    ) -> Dict[str, torch.Tensor]:
        _cuda_profile_mark(profile, "deep_start")
        if visual_tokens.ndim != 5:
            raise ValueError(
                "visual_tokens must have shape (B, steps, V, tokens, dim), "
                f"got {tuple(visual_tokens.shape)}."
            )
        batch_size, steps, views, token_count, channels = visual_tokens.shape
        expected_tokens = 1 + self.num_register_tokens + self.num_patches
        if token_count != expected_tokens or channels != self.embed_dim:
            raise ValueError(
                "Unexpected VGGT shallow tokens: "
                f"got N={token_count}, D={channels}; expected "
                f"N={expected_tokens}, D={self.embed_dim}."
            )
        if views != self.views_per_timestep:
            raise ValueError(
                f"Expected {self.views_per_timestep} views, got {views}."
            )
        scene_batch = batch_size * steps
        current = visual_tokens.reshape(
            scene_batch, views, token_count, channels
        )
        with_action = action_tokens is not None
        action_index = int(self.aggregator.patch_start_idx)
        if with_action:
            if action_tokens.shape != (
                batch_size,
                steps,
                views,
                channels,
            ):
                raise ValueError(
                    f"action_tokens shape {tuple(action_tokens.shape)} != "
                    f"{(batch_size, steps, views, channels)}."
                )
            action = action_tokens.reshape(
                scene_batch, views, 1, channels
            ).to(dtype=current.dtype)
            current = torch.cat(
                [
                    current[:, :, :action_index],
                    action,
                    current[:, :, action_index:],
                ],
                dim=2,
            )
        grid = int(math.isqrt(self.num_patches))
        pos = self._build_positions(
            scene_batch,
            views,
            grid,
            grid,
            current.device,
            with_action=with_action,
        )
        _cuda_profile_mark(profile, "deep_prep_done")
        use_temporal_causal = bool(deep_temporal_causal_mask) or (
            self.deep_time_mode == "causal"
        )
        if use_temporal_causal:
            current, cached = self._run_causal_deep_pairs(
                current,
                pos,
                batch_size=batch_size,
                steps=steps,
                views=views,
                gradient_checkpointing=gradient_checkpointing,
                capture_layers=bool(decode_visuals or return_multi_level),
            )
        else:
            current, cached = self._run_pairs(
                current,
                pos,
                start_layer=self.split_layer,
                gradient_checkpointing=gradient_checkpointing,
                capture_layers=bool(decode_visuals or return_multi_level),
            )
        _cuda_profile_mark(profile, "deep_blocks_done")

        result: Dict[str, torch.Tensor] = {}
        if with_action:
            deep_visual = torch.cat(
                [current[:, :, :action_index], current[:, :, action_index + 1 :]],
                dim=2,
            )
        else:
            deep_visual = current
        deep_visual = deep_visual.reshape(
            batch_size, steps, views, expected_tokens, channels
        )
        if step_valid_mask is not None:
            valid_steps = step_valid_mask.to(
                device=deep_visual.device, dtype=deep_visual.dtype
            )
            while valid_steps.ndim > 2:
                valid_steps = valid_steps.any(dim=-1)
            if valid_steps.shape != (batch_size, steps):
                raise ValueError(
                    f"step_valid_mask shape {tuple(valid_steps.shape)} != "
                    f"{(batch_size, steps)}."
                )
            deep_visual = deep_visual * valid_steps[:, :, None, None, None]
        # Expose the actual pair-23 visual state.  WM3D-WAM uses predicted
        # future slots from this tensor as geometry K/V; DPT outputs remain
        # auxiliary supervision and are never substituted for these tokens.
        result["deep_visual_tokens"] = deep_visual
        if with_action:
            final_action = current[:, :, action_index].reshape(
                batch_size, steps, views, channels
            )
            if step_valid_mask is not None:
                valid = step_valid_mask.to(
                    device=final_action.device, dtype=final_action.dtype
                )
                while valid.ndim > 2:
                    valid = valid.any(dim=-1)
                if valid.shape != (batch_size, steps):
                    raise ValueError(
                        f"step_valid_mask shape {tuple(valid.shape)} != "
                        f"{(batch_size, steps)}."
                    )
                final_action = final_action * valid[:, :, None, None]
            result["action_tokens"] = final_action.reshape(
                batch_size * steps * views, channels
            )

        patch_start_idx = int(self.aggregator.patch_start_idx) + int(with_action)
        if decode_visuals:
            decoded = self._decode_cached(
                cached,
                batch_size=scene_batch,
                views=views,
                patch_start_idx=patch_start_idx,
                frames_chunk_size=dpt_chunk_size,
            )
            result.update(decoded)
        _cuda_profile_mark(profile, "deep_dpt_done")

        if return_multi_level:
            levels = []
            total_views = steps * views
            for layer_idx in self.out_layers:
                feature = cached[layer_idx].reshape(
                    batch_size,
                    total_views,
                    current.shape[2],
                    2 * channels,
                )
                levels.append(
                    (
                        feature[:, :, patch_start_idx:],
                        feature[:, :, 0],
                    )
                )
            result["level_feats"] = levels
        if not with_action:
            result["aggregated_tokens"] = cached
        _cuda_profile_mark(profile, "deep_final_done")
        return result

    @torch.no_grad()
    def propagate_shallow_with_actions(
        self,
        visual_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        decode_visuals: bool = True,
        step_valid_mask: Optional[torch.Tensor] = None,
        deep_temporal_causal_mask: bool = False,
    ) -> Dict[str, torch.Tensor]:
        return self._propagate_impl(
            visual_tokens,
            action_tokens,
            decode_visuals=decode_visuals,
            dpt_chunk_size=None,
            gradient_checkpointing=False,
            return_multi_level=False,
            step_valid_mask=step_valid_mask,
            deep_temporal_causal_mask=deep_temporal_causal_mask,
            profile=None,
        )

    def propagate_shallow_with_actions_grad(
        self,
        visual_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        decode_visuals: bool = True,
        dpt_chunk_size: int = 1,
        gradient_checkpointing: bool = False,
        return_multi_level: bool = False,
        step_valid_mask: Optional[torch.Tensor] = None,
        deep_temporal_causal_mask: bool = False,
        profile: Optional[Dict[str, object]] = None,
    ) -> Dict[str, torch.Tensor]:
        return self._propagate_impl(
            visual_tokens,
            action_tokens,
            decode_visuals=decode_visuals,
            dpt_chunk_size=dpt_chunk_size,
            gradient_checkpointing=gradient_checkpointing,
            return_multi_level=return_multi_level,
            step_valid_mask=step_valid_mask,
            deep_temporal_causal_mask=deep_temporal_causal_mask,
            profile=profile,
        )

    @torch.no_grad()
    def propagate_shallow_without_actions(
        self,
        visual_tokens: torch.Tensor,
        *,
        decode_visuals: bool = True,
        dpt_chunk_size: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Parity-test helper: resume VGGT without inserting an action token."""
        return self._propagate_impl(
            visual_tokens,
            None,
            decode_visuals=decode_visuals,
            dpt_chunk_size=dpt_chunk_size,
            gradient_checkpointing=False,
            return_multi_level=False,
            step_valid_mask=None,
            deep_temporal_causal_mask=False,
            profile=None,
        )

    @torch.no_grad()
    def decode_depth_from_shallow_visual_slots(
        self,
        visual_tokens: torch.Tensor,
        *,
        frames_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Finish an independent VGGT pass from already-computed shallow slots."""
        if visual_tokens.ndim != 5:
            raise ValueError(
                "visual_tokens must have shape (B, T, V, tokens, dim), "
                f"got {tuple(visual_tokens.shape)}."
            )
        batch_size, timesteps, views, token_count, channels = visual_tokens.shape
        expected_tokens = 1 + self.num_register_tokens + self.num_patches
        if token_count != expected_tokens or channels != self.embed_dim:
            raise ValueError(
                "Unexpected VGGT shallow tokens: "
                f"got N={token_count}, D={channels}; expected "
                f"N={expected_tokens}, D={self.embed_dim}."
            )
        if views != self.views_per_timestep:
            raise ValueError(
                f"Expected {self.views_per_timestep} views, got {views}."
            )

        scene_batch = batch_size * timesteps
        current = visual_tokens.reshape(
            scene_batch, views, token_count, channels
        )
        grid = int(math.isqrt(self.num_patches))
        pos = self._build_positions(
            scene_batch,
            views,
            grid,
            grid,
            current.device,
            with_action=False,
        )
        _, cached = self._run_pairs(
            current,
            pos,
            start_layer=self.split_layer,
            gradient_checkpointing=False,
            capture_layers=True,
        )
        cached_sequence = [
            None
            if value is None
            else value.reshape(
                batch_size,
                timesteps * views,
                value.shape[-2],
                value.shape[-1],
            )
            for value in cached
        ]
        decoded = self._decode_cached(
            cached_sequence,
            batch_size=batch_size,
            views=timesteps * views,
            patch_start_idx=int(self.aggregator.patch_start_idx),
            frames_chunk_size=frames_chunk_size,
        )
        return decoded["depth"]

    @torch.no_grad()
    def encode_all_levels(
        self, images: torch.Tensor
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        if images.ndim == 4:
            images = images.unsqueeze(1)
        batch_size, total_views = images.shape[:2]
        views = self.views_per_timestep
        if total_views % views:
            if total_views == 1:
                views = 1
            else:
                raise ValueError(
                    f"Cannot split {total_views} views into groups of {views}."
                )
        timesteps = total_views // views
        scenes = self._reshape_as_scenes(images, timesteps, views)
        tokens, pos = self._prepare_scene_tokens(scenes)
        _, cached = self._run_pairs(
            tokens,
            pos,
            start_layer=0,
            gradient_checkpointing=False,
            capture_layers=True,
        )
        levels: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        patch_start = int(self.aggregator.patch_start_idx)
        for level, layer_idx in enumerate(self.out_layers):
            feature = cached[layer_idx].reshape(
                batch_size * total_views,
                cached[layer_idx].shape[2],
                cached[layer_idx].shape[3],
            )
            levels[level] = (feature[:, patch_start:], feature[:, 0])
        return levels

    def decode_depth(
        self,
        features_per_level: List[Tuple[torch.Tensor, torch.Tensor]],
        batch_size: Optional[int] = None,
        views_per_sequence: Optional[int] = None,
        frames_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        return self.decode_depth_full(
            features_per_level,
            batch_size=batch_size,
            views_per_sequence=views_per_sequence,
            frames_chunk_size=frames_chunk_size,
        )["depth"]

    def decode_depth_full(
        self,
        features_per_level: List[Tuple[torch.Tensor, torch.Tensor]],
        batch_size: Optional[int] = None,
        views_per_sequence: Optional[int] = None,
        frames_chunk_size: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if len(features_per_level) != len(self.out_layers):
            raise ValueError(
                f"Expected {len(self.out_layers)} VGGT levels, "
                f"got {len(features_per_level)}."
            )
        first = features_per_level[0][0]
        if batch_size is None:
            batch_size = 1
        if views_per_sequence is None:
            flat = first.shape[0] if first.ndim == 3 else first.shape[0] * first.shape[1]
            if flat % int(batch_size):
                raise ValueError(
                    f"Cannot reshape {flat} features into batch {batch_size}."
                )
            views_per_sequence = flat // int(batch_size)
        cached: List[Optional[torch.Tensor]] = [None] * self.block_count
        for (patches, _camera), layer_idx in zip(
            features_per_level, self.out_layers
        ):
            if patches.ndim == 3:
                patches = patches.reshape(
                    int(batch_size),
                    int(views_per_sequence),
                    patches.shape[-2],
                    patches.shape[-1],
                )
            cached[layer_idx] = patches
        return self._decode_cached(
            cached,
            batch_size=int(batch_size),
            views=int(views_per_sequence),
            patch_start_idx=0,
            frames_chunk_size=frames_chunk_size,
        )
