"""Production Stage A/B/C trainer for online WM3D-WAM."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn

from wm3d_wam.assets import (
    LocalWanComponents,
    encode_local_wan_prompts,
    load_local_wan_components,
)
from wm3d_wam.data.hierarchical_sampler import RecoverableHierarchicalSampler
from wm3d_wam.data.online_dataset import (
    OnlineRobotDataset,
    OnlineTrainingSample,
    build_online_dataloader,
)
from wm3d_wam.models.factory import (
    build_online_geometry_core,
    build_wan_action_mot,
    load_yaml_mapping,
)
from wm3d_wam.models.system import WM3DWAMSystem

from .checkpointing import (
    LOCAL_FSDP_SCHEMA,
    load_checkpoint,
    load_model_only,
    prune_completed_checkpoints,
    resolve_checkpoint,
    save_checkpoint,
)
from .distributed import (
    DistributedContext,
    distributed_barrier,
    reduce_metrics,
    seed_everything,
    wrap_full_shard,
)
from .geometry_pipeline import WorldCorePretrainingPipeline
from .parameter_groups import (
    TrainingStage,
    configure_world_core_pretraining_parameter_groups,
    configure_stage_parameter_groups,
)
from .pipeline import WM3DWAMTrainingPipeline


PHASES = tuple(stage.value for stage in TrainingStage)


class TrainerError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrainingPaths:
    data_profile: Path
    source_contracts: Path
    normalization: Path
    episode_splits: Path
    wan_assets: Path
    action_backbone: Path
    vggt_checkpoint: Path
    vggt_source_root: Path
    model_config: Path
    geometry_config: Path

    @classmethod
    def resolve(
        cls,
        *,
        data_profile: str | Path,
        source_contracts: str | Path,
        normalization: str | Path,
        episode_splits: str | Path,
        wan_assets: str | Path,
        action_backbone: str | Path,
        vggt_checkpoint: str | Path,
        vggt_source_root: str | Path,
        model_config: str | Path,
        geometry_config: str | Path,
    ) -> "TrainingPaths":
        values = {
            name: Path(value).expanduser().resolve(strict=True)
            for name, value in {
                "data_profile": data_profile,
                "source_contracts": source_contracts,
                "normalization": normalization,
                "episode_splits": episode_splits,
                "wan_assets": wan_assets,
                "action_backbone": action_backbone,
                "vggt_checkpoint": vggt_checkpoint,
                "vggt_source_root": vggt_source_root,
                "model_config": model_config,
                "geometry_config": geometry_config,
            }.items()
        }
        return cls(**values)


@dataclass(frozen=True)
class TrainerOptions:
    phase: str
    max_steps: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    num_workers: int
    prefetch_factor: int
    seed: int
    warmup_steps: int
    max_grad_norm: float
    log_interval: int
    validation_interval: int
    validation_samples_per_rank: int
    checkpoint_interval: int
    keep_last_checkpoints: int
    output_dir: Path
    resume: Path | None = None
    initialize_from: Path | None = None
    prompt_cache_entries: int = 64
    stop_after_step: int | None = None

    def validate(self) -> None:
        if self.phase not in PHASES:
            raise TrainerError(f"unknown training phase {self.phase!r}")
        for name in (
            "max_steps",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "log_interval",
            "checkpoint_interval",
        ):
            if int(getattr(self, name)) <= 0:
                raise TrainerError(f"{name} must be positive")
        if self.num_workers < 0 or self.prefetch_factor <= 0:
            raise TrainerError("DataLoader worker configuration is invalid")
        if self.warmup_steps < 0 or self.warmup_steps >= self.max_steps:
            if not (self.max_steps == 1 and self.warmup_steps == 0):
                raise TrainerError("warmup_steps must be in [0,max_steps)")
        if not np_is_finite_positive(self.max_grad_norm):
            raise TrainerError("max_grad_norm must be finite and positive")
        if self.validation_interval < 0 or self.validation_samples_per_rank < 0:
            raise TrainerError("validation controls must be non-negative")
        if self.keep_last_checkpoints < 0:
            raise TrainerError("keep_last_checkpoints must be non-negative")
        if bool(self.validation_interval) != bool(self.validation_samples_per_rank):
            raise TrainerError(
                "validation interval and sample count must both be zero or both positive"
            )
        if (
            self.validation_samples_per_rank
            and self.validation_samples_per_rank % self.micro_batch_size
        ):
            raise TrainerError(
                "validation_samples_per_rank must contain complete micro-batches"
            )
        if self.resume is not None and self.initialize_from is not None:
            raise TrainerError(
                "exact resume and cross-phase initialization are mutually exclusive"
            )
        if self.stop_after_step is not None and not (
            0 < self.stop_after_step <= self.max_steps
        ):
            raise TrainerError("stop_after_step must be in (0,max_steps]")


def np_is_finite_positive(value: float) -> bool:
    return math.isfinite(float(value)) and float(value) > 0.0


class PromptEncoderCache:
    """Small in-memory cache around the frozen local UMT5 encoder."""

    def __init__(
        self,
        *,
        components: LocalWanComponents,
        device: torch.device,
        dtype: torch.dtype,
        max_entries: int,
    ) -> None:
        if components.text_encoder is None or components.tokenizer is None:
            raise TrainerError("frozen text components are incomplete")
        if max_entries <= 0:
            raise TrainerError("prompt cache must retain at least one entry")
        self.components = components
        self.components.text_encoder.eval().requires_grad_(False)
        self.device = device
        self.dtype = dtype
        self.max_entries = int(max_entries)
        self.cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()

    @torch.no_grad()
    def encode(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        key = prompt.strip()
        cached = self.cache.pop(key, None)
        if cached is not None:
            self.cache[key] = cached
            return cached
        context, mask = encode_local_wan_prompts(
            self.components,
            [key],
            device=self.device,
            dtype=self.dtype,
        )
        value = (context.detach(), mask.detach())
        self.cache[key] = value
        while len(self.cache) > self.max_entries:
            self.cache.popitem(last=False)
        return value

    @torch.no_grad()
    def encode_batch(self, prompts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if not prompts:
            raise TrainerError("prompt micro-batch is empty")
        encoded = [self.encode(prompt) for prompt in prompts]
        return (
            torch.cat([value[0] for value in encoded], dim=0),
            torch.cat([value[1] for value in encoded], dim=0),
        )


def _video_config(model_config: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(model_config["video_expert"])
    value.pop("_target_", None)
    return value


@dataclass
class BuiltTrainingModel:
    pipeline: nn.Module
    text_components: LocalWanComponents
    is_geometry_only: bool


def build_training_model(
    *,
    phase: str,
    paths: TrainingPaths,
    device: torch.device,
    dtype: torch.dtype,
) -> BuiltTrainingModel:
    model_config = load_yaml_mapping(paths.model_config)
    geometry_config = load_yaml_mapping(paths.geometry_config)
    if phase == TrainingStage.WORLD_CORE_PRETRAIN.value:
        text = load_local_wan_components(
            asset_root=paths.wan_assets,
            dit_config={},
            device=device,
            dtype=dtype,
            load_dit=False,
            load_vae=False,
            load_text_encoder=True,
        )
        geometry = build_online_geometry_core(
            geometry_config=geometry_config,
            action_codec_config=model_config["action_expert"]["codec_config"],
            vggt_checkpoint=paths.vggt_checkpoint,
            vggt_source_root=paths.vggt_source_root,
            views_per_timestep=3,
            device=device,
            dtype=dtype,
        )
        return BuiltTrainingModel(
            pipeline=WorldCorePretrainingPipeline(geometry),
            text_components=text,
            is_geometry_only=True,
        )

    components = load_local_wan_components(
        asset_root=paths.wan_assets,
        dit_config=_video_config(model_config),
        device=device,
        dtype=dtype,
        load_dit=True,
        load_vae=True,
        load_text_encoder=True,
    )
    if components.dit is None or components.vae is None:
        raise TrainerError("full phase did not load Wan DiT/VAE")
    wan_action = build_wan_action_mot(
        video_expert=components.dit,
        model_config=model_config,
        action_backbone_path=paths.action_backbone,
        device=device,
        dtype=dtype,
    )
    geometry = build_online_geometry_core(
        geometry_config=geometry_config,
        action_codec_config=model_config["action_expert"]["codec_config"],
        vggt_checkpoint=paths.vggt_checkpoint,
        vggt_source_root=paths.vggt_source_root,
        views_per_timestep=3,
        device=device,
        dtype=dtype,
    )
    system = WM3DWAMSystem(
        geometry_core=geometry,
        wan_action=wan_action,
        video_vae=components.vae,
    )
    # The prompt cache owns only the frozen text objects.  Clear duplicate
    # component references to the train graph before dropping this container.
    text = LocalWanComponents(
        dit=None,
        vae=None,
        text_encoder=components.text_encoder,
        tokenizer=components.tokenizer,
        paths=components.paths,
    )
    return BuiltTrainingModel(
        pipeline=WM3DWAMTrainingPipeline(system=system),
        text_components=text,
        is_geometry_only=False,
    )


def _program_mix(phase: str) -> dict[str, float]:
    if phase == TrainingStage.WORLD_CORE_PRETRAIN.value:
        return {"world_core_pretrain": 1.0}
    if phase in {
        TrainingStage.WAN_ACTION_WARMUP.value,
        TrainingStage.WAN_ACTION_MAIN.value,
    }:
        return {
            "action_only": 0.50,
            "forward_world": 0.30,
            "joint_world_action": 0.20,
        }
    return {
        "action_only": 0.40,
        "forward_world": 0.30,
        "joint_world_action": 0.30,
    }


def _configure_groups(model: BuiltTrainingModel, phase: str) -> list[dict[str, object]]:
    if model.is_geometry_only:
        pipeline = model.pipeline
        assert isinstance(pipeline, WorldCorePretrainingPipeline)
        return configure_world_core_pretraining_parameter_groups(pipeline.geometry_core)
    pipeline = model.pipeline
    assert isinstance(pipeline, WM3DWAMTrainingPipeline)
    return configure_stage_parameter_groups(pipeline.system, TrainingStage(phase))


def _optimizer(groups: list[dict[str, object]]) -> torch.optim.AdamW:
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1.0e-8)
    # DCP must see the same optimizer key set at save and restore time. AdamW
    # normally creates slots lazily only for parameters that received a grad,
    # which makes strict restoration impossible when a routed program leaves
    # some trainable parameters unused in the first checkpointed step. Create
    # the standard zero/step=0 slots eagerly without performing an update.
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state[parameter]
            if state:
                continue
            state["step"] = (
                torch.zeros((), dtype=torch.float32, device=parameter.device)
                if bool(group.get("capturable")) or bool(group.get("fused"))
                else torch.tensor(0.0, dtype=torch.float32)
            )
            state["exp_avg"] = torch.zeros_like(
                parameter, memory_format=torch.preserve_format
            )
            state["exp_avg_sq"] = torch.zeros_like(
                parameter, memory_format=torch.preserve_format
            )
            if bool(group.get("amsgrad")):
                state["max_exp_avg_sq"] = torch.zeros_like(
                    parameter, memory_format=torch.preserve_format
                )
    return optimizer


def _scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / remaining))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _initialize_cross_phase(
    *,
    pipeline: nn.Module,
    current_geometry_only: bool,
    checkpoint: Path,
    physical_cuda_devices: tuple[int, ...],
) -> Mapping[str, Any]:
    source = resolve_checkpoint(checkpoint)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    source_phase = str(metadata.get("phase", ""))
    source_geometry_only = source_phase == TrainingStage.WORLD_CORE_PRETRAIN.value
    if current_geometry_only and not source_geometry_only:
        raise TrainerError("a full Wan/Action checkpoint cannot initialize Stage A")
    if source_geometry_only and not current_geometry_only:
        if not isinstance(pipeline, WM3DWAMTrainingPipeline):
            raise TrainerError("full pipeline type is inconsistent")
        geometry_view = WorldCorePretrainingPipeline(pipeline.system.geometry_core)
        load_model_only(
            path_or_root=source,
            model=geometry_view,
            expected_physical_cuda_devices=physical_cuda_devices,
        )
    else:
        load_model_only(
            path_or_root=source,
            model=pipeline,
            expected_physical_cuda_devices=physical_cuda_devices,
        )
    return metadata


def _forward_sample(
    model: nn.Module,
    *,
    sample: OnlineTrainingSample,
    prompt_cache: PromptEncoderCache,
    device: torch.device,
    dtype: torch.dtype,
    gradient_checkpointing: bool,
):
    window = sample.window.to(device=device, dtype=dtype)
    context, context_mask = prompt_cache.encode_batch(sample.task_texts)
    with torch.autocast(device_type="cuda", dtype=dtype):
        if sample.request.program == "world_core_pretrain":
            output = model(
                window=window,
                context=context,
                context_mask=context_mask,
                gradient_checkpointing=gradient_checkpointing,
            )
        else:
            output = model(
                program=sample.request.program,
                window=window,
                context=context,
                context_mask=context_mask,
                geometry_gradient_checkpointing=gradient_checkpointing,
            )
    return output


def _next_sample_synchronized(
    iterator,
    *,
    context: DistributedContext,
    purpose: str,
) -> OnlineTrainingSample:
    """Make raw-data readiness a collective boundary before FSDP forward.

    If one worker exhausts deterministic decode retries, every rank must learn
    that fact before any successful rank enters a parameter all-gather. This
    keeps the original data error visible and prevents an NCCL mismatch hang.
    """

    sample: OnlineTrainingSample | None = None
    error: str | None = None
    try:
        value = next(iterator)
        if not isinstance(value, OnlineTrainingSample):
            raise TrainerError(
                f"{purpose} loader returned {type(value).__name__}, "
                "expected OnlineTrainingSample"
            )
        sample = value
    except Exception as exc:  # synchronized and re-raised below on every rank
        error = f"{type(exc).__name__}: {exc}"

    local_status = {"rank": context.rank, "error": error}
    statuses: list[Mapping[str, object]] = [local_status]
    if context.world_size > 1:
        gathered: list[object] = [None] * context.world_size
        dist.all_gather_object(gathered, local_status)
        statuses = [
            value if isinstance(value, Mapping) else {"rank": -1, "error": repr(value)}
            for value in gathered
        ]
    failures = [
        f"rank {status.get('rank')}: {status.get('error')}"
        for status in statuses
        if status.get("error") is not None
    ]
    if failures:
        raise TrainerError(f"{purpose} data readiness failed; " + " | ".join(failures))
    if sample is None:
        raise TrainerError(f"{purpose} loader produced no sample")
    return sample


@torch.no_grad()
def _validate(
    *,
    model: nn.Module,
    loader,
    prompt_cache: PromptEncoderCache,
    context: DistributedContext,
    dtype: torch.dtype,
    global_step: int,
    validation_seed: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    # Validation flow noise is deterministic and cannot advance training RNG.
    with torch.random.fork_rng(devices=[context.local_rank]):
        torch.manual_seed(validation_seed + global_step)
        torch.cuda.manual_seed(validation_seed + global_step)
        iterator = iter(loader)
        for _ in range(len(loader)):
            sample = _next_sample_synchronized(
                iterator,
                context=context,
                purpose="validation",
            )
            output = _forward_sample(
                model,
                sample=sample,
                prompt_cache=prompt_cache,
                device=context.device,
                dtype=dtype,
                gradient_checkpointing=False,
            )
            batch_size = sample.batch_size
            for name, value in output.detached_metrics().items():
                sums[name] = sums.get(name, 0.0) + float(value) * batch_size
            count += batch_size
    if was_training:
        model.train()
    local = {name: value / max(1, count) for name, value in sums.items()}
    local["samples_per_rank"] = float(count)
    return reduce_metrics(local, device=context.device)


def _clip_grad_norm(model: nn.Module, max_norm: float) -> float:
    if hasattr(model, "clip_grad_norm_"):
        value = model.clip_grad_norm_(max_norm)  # type: ignore[attr-defined]
    else:
        value = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            max_norm=max_norm,
            error_if_nonfinite=True,
        )
    scalar = float(value.detach().float())
    if not math.isfinite(scalar):
        raise FloatingPointError("pre-clipping gradient norm is non-finite")
    return scalar


def train(
    *,
    context: DistributedContext,
    paths: TrainingPaths,
    options: TrainerOptions,
) -> Mapping[str, object]:
    options.validate()
    dtype = torch.bfloat16
    output_dir = options.output_dir.expanduser().resolve()
    checkpoint_root = output_dir / "checkpoints"
    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    distributed_barrier()

    train_dataset = OnlineRobotDataset(
        data_profile_path=paths.data_profile,
        source_contract_path=paths.source_contracts,
        normalization_path=paths.normalization,
        split_root=paths.episode_splits,
        split="train",
    )
    validation_dataset = None
    if options.validation_interval:
        validation_dataset = OnlineRobotDataset(
            data_profile_path=paths.data_profile,
            source_contract_path=paths.source_contracts,
            normalization_path=paths.normalization,
            split_root=paths.episode_splits,
            split="val",
            max_decode_retries=4,
        )

    # Identical initialization on every rank; FSDP syncs rank 0 as an
    # additional guard.  Rank-specific flow noise is seeded after wrapping.
    seed_everything(options.seed, rank_offset=False)
    built = build_training_model(
        phase=options.phase,
        paths=paths,
        device=context.device,
        dtype=dtype,
    )
    initialized_from: Mapping[str, Any] | None = None
    deferred_local_initialization: Path | None = None
    if options.initialize_from is not None:
        source = resolve_checkpoint(options.initialize_from)
        source_metadata = json.loads(
            (source / "metadata.json").read_text(encoding="utf-8")
        )
        if source_metadata.get("schema") == LOCAL_FSDP_SCHEMA:
            if built.is_geometry_only:
                raise TrainerError(
                    "a full rank-local checkpoint cannot initialize Stage A"
                )
            # Rank-local flat shards can only be loaded after the destination
            # has the identical FSDP hierarchy and device mesh.
            deferred_local_initialization = source
            initialized_from = source_metadata
        else:
            initialized_from = _initialize_cross_phase(
                pipeline=built.pipeline,
                current_geometry_only=built.is_geometry_only,
                checkpoint=source,
                physical_cuda_devices=context.physical_cuda_devices,
            )
    groups = _configure_groups(built, options.phase)
    parameter_group_descriptions = {
        str(group["name"]): {
            "parameters": sum(parameter.numel() for parameter in group["params"]),
            "learning_rate": float(group["lr"]),
        }
        for group in groups
    }
    if built.is_geometry_only:
        assert isinstance(built.pipeline, WorldCorePretrainingPipeline)
        geometry_core = built.pipeline.geometry_core
    else:
        assert isinstance(built.pipeline, WM3DWAMTrainingPipeline)
        geometry_core = built.pipeline.system.geometry_core
    frozen_geometry_heads = (
        geometry_core.encoder.dpt_head,
        geometry_core.encoder.point_head,
        geometry_core.encoder.camera_head,
    )
    # Official VGGT decoding runs these frozen heads in explicit FP32 with
    # autocast disabled. Keeping them outside FSDP prevents that dtype change
    # from mutating sharded parameter views while preserving gradients into
    # the trainable deep VGGT blocks.
    for head in frozen_geometry_heads:
        head.float()
    wrapped = wrap_full_shard(
        built.pipeline,
        context=context,
        ignored_modules=frozen_geometry_heads,
        use_device_mesh=built.is_geometry_only,
    ).train()
    if deferred_local_initialization is not None:
        initialized_from = _initialize_cross_phase(
            pipeline=wrapped,
            current_geometry_only=False,
            checkpoint=deferred_local_initialization,
            physical_cuda_devices=context.physical_cuda_devices,
        )
    optimizer = _optimizer(groups)
    scheduler = _scheduler(
        optimizer,
        total_steps=options.max_steps,
        warmup_steps=options.warmup_steps,
    )
    global_step = 0
    next_local_index = 0
    resume_metadata: Mapping[str, Any] | None = None
    resumed_checkpoint: Path | None = None
    if options.resume is not None:
        resumed = load_checkpoint(
            path_or_root=options.resume,
            model=wrapped,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_phase=options.phase,
            expected_seed=options.seed,
            expected_gradient_accumulation_steps=options.gradient_accumulation_steps,
            expected_micro_batch_size=options.micro_batch_size,
            expected_physical_cuda_devices=context.physical_cuda_devices,
        )
        if int(resumed.metadata.get("phase_total_steps", -1)) != options.max_steps:
            raise TrainerError("resume phase_total_steps differs from this run")
        global_step = resumed.global_step
        next_local_index = resumed.next_local_sample_index
        resume_metadata = resumed.metadata
        resumed_checkpoint = resumed.checkpoint_path
    else:
        seed_everything(options.seed, rank_offset=True)

    train_sampler = RecoverableHierarchicalSampler(
        episode_counts=train_dataset.episode_counts,
        contracts=train_dataset.contracts,
        profile_weights=train_dataset.profile_weights,
        program_mix=_program_mix(options.phase),
        seed=options.seed,
        rank=context.rank,
        world_size=context.world_size,
        micro_batch_size=options.micro_batch_size,
        start_local_index=next_local_index,
    )
    train_loader = build_online_dataloader(
        train_dataset,
        train_sampler,
        num_workers=options.num_workers,
        micro_batch_size=options.micro_batch_size,
        prefetch_factor=options.prefetch_factor,
    )
    train_iterator = iter(train_loader)
    validation_loader = None
    if validation_dataset is not None:
        validation_sampler = RecoverableHierarchicalSampler(
            episode_counts=validation_dataset.episode_counts,
            contracts=validation_dataset.contracts,
            profile_weights=validation_dataset.profile_weights,
            program_mix=_program_mix(options.phase),
            seed=options.seed + 10_000,
            rank=context.rank,
            world_size=context.world_size,
            micro_batch_size=options.micro_batch_size,
            start_local_index=0,
            num_local_samples=options.validation_samples_per_rank,
        )
        validation_loader = build_online_dataloader(
            validation_dataset,
            validation_sampler,
            num_workers=max(0, min(options.num_workers, 2)),
            micro_batch_size=options.micro_batch_size,
            prefetch_factor=options.prefetch_factor,
        )
    prompt_cache = PromptEncoderCache(
        components=built.text_components,
        device=context.device,
        dtype=dtype,
        max_entries=options.prompt_cache_entries,
    )

    run_description: dict[str, object] = {
        "schema": "wm3d_wam_training_run_v1",
        "phase": options.phase,
        "phase_total_steps": options.max_steps,
        "seed": options.seed,
        "world_size": context.world_size,
        "physical_cuda_devices": list(context.physical_cuda_devices),
        "gradient_accumulation_steps": options.gradient_accumulation_steps,
        "micro_batch_size": options.micro_batch_size,
        "effective_global_batch": context.world_size
        * options.micro_batch_size
        * options.gradient_accumulation_steps,
        "checkpoint_interval": options.checkpoint_interval,
        "keep_last_checkpoints": options.keep_last_checkpoints,
        "train_episode_counts": train_dataset.episode_counts,
        "program_mix": _program_mix(options.phase),
        "parameter_groups": parameter_group_descriptions,
        "initialized_from_phase": (
            initialized_from.get("phase") if initialized_from is not None else None
        ),
        "resumed_from": str(resumed_checkpoint)
        if resumed_checkpoint is not None
        else None,
        "resume_global_step": global_step if resume_metadata is not None else None,
        "resume_saved_world_size": (
            int(resume_metadata["world_size"]) if resume_metadata is not None else None
        ),
        "resume_saved_effective_global_batch": (
            int(resume_metadata["effective_global_batch"])
            if resume_metadata is not None
            and "effective_global_batch" in resume_metadata
            else None
        ),
        "resume_resharded": (
            int(resume_metadata["world_size"]) != context.world_size
            if resume_metadata is not None
            else False
        ),
        "resume_next_local_sample_index": (
            next_local_index if resume_metadata is not None else None
        ),
    }
    if context.is_main:
        run_path = output_dir / "run.json"
        if options.resume is None:
            if run_path.exists():
                raise TrainerError(
                    f"refusing to overwrite existing run metadata {run_path}"
                )
            _write_json(run_path, run_description)
        elif run_path.is_file():
            existing = json.loads(run_path.read_text(encoding="utf-8"))
            for name in (
                "phase",
                "phase_total_steps",
                "seed",
                "world_size",
                "physical_cuda_devices",
                "gradient_accumulation_steps",
                "micro_batch_size",
            ):
                if existing.get(name) != run_description[name]:
                    raise TrainerError(
                        f"resume run metadata {name} mismatch: "
                        f"saved={existing.get(name)!r}, runtime={run_description[name]!r}"
                    )
        else:
            # A canonical Stage-A checkpoint may deliberately resume into a
            # new output directory and a different device mesh. Preserve the
            # source run and write an unambiguous continuation record here.
            _write_json(run_path, run_description)
    distributed_barrier()

    metrics_path = output_dir / "metrics.jsonl"
    local_samples_committed = next_local_index
    last_checkpoint: Path | None = None
    torch.cuda.reset_peak_memory_stats(context.device)
    invocation_target = (
        options.max_steps
        if options.stop_after_step is None
        else options.stop_after_step
    )
    if invocation_target < global_step:
        raise TrainerError(
            f"stop_after_step={invocation_target} precedes resumed step {global_step}"
        )
    while global_step < invocation_target:
        step_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        accumulated: dict[str, float] = {}
        program_counts: dict[str, int] = {}
        source_counts: dict[str, int] = {}
        retry_count = 0
        for micro_step in range(options.gradient_accumulation_steps):
            sample = _next_sample_synchronized(
                train_iterator,
                context=context,
                purpose="training",
            )
            sync_context = (
                wrapped.no_sync()
                if micro_step + 1 < options.gradient_accumulation_steps
                and hasattr(wrapped, "no_sync")
                else nullcontext()
            )
            with sync_context:
                output = _forward_sample(
                    wrapped,
                    sample=sample,
                    prompt_cache=prompt_cache,
                    device=context.device,
                    dtype=dtype,
                    gradient_checkpointing=True,
                )
                weighted_loss = output.total_loss * sample.quality_weight
                scaled_loss = weighted_loss / options.gradient_accumulation_steps
                scaled_loss.backward()
            for name, value in output.detached_metrics().items():
                accumulated[name] = accumulated.get(name, 0.0) + float(value)
            accumulated["loss_weighted"] = accumulated.get(
                "loss_weighted", 0.0
            ) + float(weighted_loss.detach())
            for request in sample.requests:
                program_counts[request.program] = (
                    program_counts.get(request.program, 0) + 1
                )
                source_counts[request.source] = source_counts.get(request.source, 0) + 1
            retry_count += sample.decode_retry_count
        grad_norm = _clip_grad_norm(wrapped, options.max_grad_norm)
        optimizer.step()
        scheduler.step()
        global_step += 1
        local_samples_committed += (
            options.gradient_accumulation_steps * options.micro_batch_size
        )
        torch.cuda.synchronize(context.device)
        step_seconds = time.perf_counter() - step_start
        local_metrics = {
            name: value / options.gradient_accumulation_steps
            for name, value in accumulated.items()
        }
        local_metrics.update(
            {
                "grad_norm_preclip": grad_norm,
                "step_seconds": step_seconds,
                "samples_per_second_global": (
                    context.world_size
                    * options.micro_batch_size
                    * options.gradient_accumulation_steps
                    / max(step_seconds, 1.0e-12)
                ),
                "decode_retries_per_step": float(retry_count),
                "peak_memory_gib": torch.cuda.max_memory_allocated(context.device)
                / (1024**3),
            }
        )
        reduced = reduce_metrics(local_metrics, device=context.device)
        if context.is_main and (
            global_step == 1 or global_step % options.log_interval == 0
        ):
            record: dict[str, object] = {
                "kind": "train",
                "step": global_step,
                **reduced,
                "learning_rates": {
                    str(group.get("name", index)): float(group["lr"])
                    for index, group in enumerate(optimizer.param_groups)
                },
                "rank0_program_counts": program_counts,
                "rank0_source_counts": source_counts,
            }
            _append_jsonl(metrics_path, record)
            print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)

        if validation_loader is not None and (
            global_step % options.validation_interval == 0
            or global_step == invocation_target
        ):
            validation_metrics = _validate(
                model=wrapped,
                loader=validation_loader,
                prompt_cache=prompt_cache,
                context=context,
                dtype=dtype,
                global_step=global_step,
                validation_seed=options.seed + 20_000,
            )
            if context.is_main:
                record = {
                    "kind": "validation",
                    "step": global_step,
                    **validation_metrics,
                }
                _append_jsonl(metrics_path, record)
                print(
                    json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True
                )

        if (
            global_step % options.checkpoint_interval == 0
            or global_step == invocation_target
        ):
            last_checkpoint = save_checkpoint(
                root=checkpoint_root,
                model=wrapped,
                optimizer=optimizer,
                scheduler=scheduler,
                phase=options.phase,
                global_step=global_step,
                next_local_sample_index=local_samples_committed,
                seed=options.seed,
                gradient_accumulation_steps=options.gradient_accumulation_steps,
                micro_batch_size=options.micro_batch_size,
                extra_metadata={
                    "phase_total_steps": options.max_steps,
                    "effective_global_batch": context.world_size
                    * options.micro_batch_size
                    * options.gradient_accumulation_steps,
                    "physical_cuda_devices": list(context.physical_cuda_devices),
                },
            )
            if context.is_main and options.keep_last_checkpoints:
                prune_completed_checkpoints(
                    checkpoint_root,
                    keep_last=options.keep_last_checkpoints,
                )
            distributed_barrier()

    completed = global_step == options.max_steps
    result: dict[str, object] = {
        "phase": options.phase,
        "global_step": global_step,
        "next_local_sample_index": local_samples_committed,
        "last_checkpoint": str(last_checkpoint) if last_checkpoint else None,
        "peak_memory_gib": torch.cuda.max_memory_allocated(context.device) / (1024**3),
        "status": "completed" if completed else "paused",
    }
    if context.is_main:
        _write_json(
            output_dir / ("completed.json" if completed else "paused.json"),
            result,
        )
    distributed_barrier()
    return result
