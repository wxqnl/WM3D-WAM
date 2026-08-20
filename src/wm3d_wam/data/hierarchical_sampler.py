"""Counter-based, exactly recoverable hierarchical training sampler."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterator, Mapping, Sequence

import numpy as np
from torch.utils.data import Sampler

from .source_contracts import SourceContract, SourceContractError


PROGRAM_FAMILY_MIX: Mapping[str, Mapping[str, float]] = {
    "world_core_pretrain": {"oxe": 0.70, "robocasa": 0.30},
    "action_only": {"oxe": 0.70, "robocasa": 0.30},
    "forward_world": {"oxe": 0.40, "robocasa": 0.60},
    "joint_world_action": {"oxe": 0.60, "robocasa": 0.40},
}


@dataclass(frozen=True)
class WindowRequest:
    global_sample_index: int
    local_sample_index: int
    program: str
    family: str
    source: str
    episode_index: int
    anchor_fraction: float
    target_view_fraction: float
    retry_stride: int


def _normalized_choice(
    rng: np.random.Generator,
    names: Sequence[str],
    weights: Sequence[float],
) -> str:
    if not names or len(names) != len(weights):
        raise SourceContractError("weighted choice received an empty or misaligned table")
    probabilities = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(probabilities).all() or bool((probabilities < 0.0).any()):
        raise SourceContractError("weighted choice contains invalid probabilities")
    total = float(probabilities.sum())
    if total <= 0.0:
        raise SourceContractError("weighted choice probabilities sum to zero")
    probabilities /= total
    return str(rng.choice(np.asarray(names, dtype=object), p=probabilities))


def _waterfill_cap(weights: Mapping[str, float], cap: float) -> dict[str, float]:
    """Normalize source weights with a feasible upper cap."""

    if not weights:
        return {}
    names = sorted(weights)
    raw = np.asarray([float(weights[name]) for name in names], dtype=np.float64)
    if not np.isfinite(raw).all() or bool((raw <= 0.0).any()):
        raise SourceContractError("source sampling weights must be finite and positive")
    feasible_cap = max(float(cap), 1.0 / len(names))
    remaining = set(range(len(names)))
    result = np.zeros_like(raw)
    mass = 1.0
    while remaining:
        denom = float(raw[list(remaining)].sum())
        proposal = {index: mass * raw[index] / denom for index in remaining}
        clipped = [index for index, value in proposal.items() if value > feasible_cap + 1.0e-12]
        if not clipped:
            for index, value in proposal.items():
                result[index] = value
            break
        for index in clipped:
            result[index] = feasible_cap
            mass -= feasible_cap
            remaining.remove(index)
    result /= result.sum()
    return {name: float(result[index]) for index, name in enumerate(names)}


def source_sampling_weights(
    contracts: Sequence[SourceContract],
    profile_weights: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    """Apply the frozen OXE cap and fixed RoboCasa 10/60/30 mixture."""

    approved = {contract.name: contract for contract in contracts if contract.trainable}
    oxe_raw = {
        name: float(profile_weights[name])
        for name, contract in approved.items()
        if contract.family == "oxe"
    }
    oxe = _waterfill_cap(oxe_raw, 0.20)
    robo_target = {
        "robocasa_atomic": 0.10,
        "robocasa_composite": 0.60,
        "robocasa_mg": 0.30,
    }
    robo = {name: weight for name, weight in robo_target.items() if name in approved}
    if robo:
        total = sum(robo.values())
        robo = {name: weight / total for name, weight in robo.items()}
    return {"oxe": oxe, "robocasa": robo}


class RecoverableHierarchicalSampler(Sampler[WindowRequest]):
    """Sample by program → family → source → episode → window → view.

    A request is a pure function of ``seed``, global sample index, rank, and
    world size.  DataLoader prefetch therefore cannot advance checkpointed
    state: the trainer records only the number of samples it has committed.
    """

    def __init__(
        self,
        *,
        episode_counts: Mapping[str, int],
        contracts: Sequence[SourceContract],
        profile_weights: Mapping[str, float],
        program_mix: Mapping[str, float],
        seed: int,
        rank: int,
        world_size: int,
        micro_batch_size: int = 1,
        start_local_index: int = 0,
        num_local_samples: int | None = None,
    ) -> None:
        if rank < 0 or world_size <= 0 or rank >= world_size:
            raise SourceContractError("invalid distributed sampler rank/world_size")
        if micro_batch_size <= 0:
            raise SourceContractError("micro_batch_size must be positive")
        if start_local_index < 0 or (num_local_samples is not None and num_local_samples < 0):
            raise SourceContractError("sampler indices must be non-negative")
        if start_local_index % int(micro_batch_size):
            raise SourceContractError(
                "start_local_index must align to the micro-batch boundary"
            )
        if num_local_samples is not None and num_local_samples % int(micro_batch_size):
            raise SourceContractError(
                "num_local_samples must contain complete micro-batches"
            )
        self.episode_counts = {name: int(count) for name, count in episode_counts.items()}
        if not self.episode_counts or any(count <= 0 for count in self.episode_counts.values()):
            raise SourceContractError("every sampled source requires at least one episode")
        self.contracts = {contract.name: contract for contract in contracts if contract.trainable}
        if set(self.episode_counts) - set(self.contracts):
            raise SourceContractError("episode catalog contains a source without an approved contract")
        self.source_weights = source_sampling_weights(
            tuple(self.contracts.values()), profile_weights
        )
        self.program_mix = {str(name): float(weight) for name, weight in program_mix.items()}
        if not self.program_mix or any(
            name not in PROGRAM_FAMILY_MIX or not np.isfinite(weight) or weight <= 0.0
            for name, weight in self.program_mix.items()
        ):
            raise SourceContractError("program_mix contains an invalid program or weight")
        for program in self.program_mix:
            for family, weight in PROGRAM_FAMILY_MIX[program].items():
                if weight > 0.0 and not self._sources(program, family):
                    raise SourceContractError(
                        f"program {program!r} has no approved {family!r} source"
                    )
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.micro_batch_size = int(micro_batch_size)
        self.start_local_index = int(start_local_index)
        self.num_local_samples = (
            None if num_local_samples is None else int(num_local_samples)
        )

    def _sources(self, program: str, family: str) -> tuple[str, ...]:
        return tuple(
            name
            for name in sorted(self.episode_counts)
            if self.contracts[name].family == family
            and program in self.contracts[name].programs
        )

    def request_at(self, local_index: int) -> WindowRequest:
        if local_index < 0:
            raise SourceContractError("local sample index must be non-negative")
        global_index = int(local_index) * self.world_size + self.rank
        # Routed FSDP modules and compiled attention shapes must be reached in
        # the same order on every rank. Choose the program and source schema
        # from the rank-independent local optimizer-sample index, then choose
        # episode/window details from the unique global index.
        route_index = int(local_index) // self.micro_batch_size
        route_seed_sequence = np.random.SeedSequence(
            [
                self.seed & 0xFFFFFFFF,
                route_index & 0xFFFFFFFF,
                route_index >> 32,
                0x50524F47,
            ]
        )
        route_rng = np.random.Generator(np.random.PCG64(route_seed_sequence))
        programs = tuple(sorted(self.program_mix))
        program = _normalized_choice(
            route_rng,
            programs,
            [self.program_mix[name] for name in programs],
        )
        family_mix = PROGRAM_FAMILY_MIX[program]
        families = tuple(sorted(family_mix))
        family = _normalized_choice(
            route_rng, families, [family_mix[name] for name in families]
        )
        sources = self._sources(program, family)
        source = _normalized_choice(
            route_rng,
            sources,
            [self.source_weights[family][name] for name in sources],
        )
        target_view_fraction = float(route_rng.random())
        seed_sequence = np.random.SeedSequence(
            [self.seed & 0xFFFFFFFF, global_index & 0xFFFFFFFF, global_index >> 32]
        )
        rng = np.random.Generator(np.random.PCG64(seed_sequence))
        episode_count = self.episode_counts[source]
        # A source-local odd retry stride avoids repeatedly testing adjacent
        # corrupt files while staying deterministic.
        retry_stride = int(rng.integers(1, max(2, episode_count)))
        if retry_stride % 2 == 0:
            retry_stride += 1
        return WindowRequest(
            global_sample_index=global_index,
            local_sample_index=int(local_index),
            program=program,
            family=family,
            source=source,
            episode_index=int(rng.integers(0, episode_count)),
            anchor_fraction=float(rng.random()),
            target_view_fraction=target_view_fraction,
            retry_stride=retry_stride,
        )

    def __iter__(self) -> Iterator[WindowRequest]:
        local = self.start_local_index
        stop = (
            None
            if self.num_local_samples is None
            else self.start_local_index + self.num_local_samples
        )
        while stop is None or local < stop:
            yield self.request_at(local)
            local += 1

    def __len__(self) -> int:
        if self.num_local_samples is None:
            raise TypeError("an unbounded training sampler has no finite length")
        return self.num_local_samples

    def state_dict(self, *, committed_local_samples: int) -> dict[str, int]:
        if committed_local_samples < 0:
            raise SourceContractError("committed sample count must be non-negative")
        return {
            "seed": self.seed,
            "rank": self.rank,
            "world_size": self.world_size,
            "micro_batch_size": self.micro_batch_size,
            "next_local_index": self.start_local_index + int(committed_local_samples),
        }

    @staticmethod
    def validate_resume_state(
        state: Mapping[str, object],
        *,
        seed: int,
        rank: int,
        world_size: int,
        micro_batch_size: int = 1,
    ) -> int:
        expected = {
            "seed": int(seed),
            "rank": int(rank),
            "world_size": int(world_size),
            "micro_batch_size": int(micro_batch_size),
        }
        for name, value in expected.items():
            if int(state.get(name, -1)) != value:
                raise SourceContractError(
                    f"sampler resume {name} mismatch: checkpoint={state.get(name)!r}, runtime={value}"
                )
        next_index = int(state.get("next_local_index", -1))
        if next_index < 0:
            raise SourceContractError("sampler checkpoint has no valid next_local_index")
        if next_index % int(micro_batch_size):
            raise SourceContractError("sampler checkpoint cursor is not micro-batch aligned")
        return next_index

    def describe_request(self, request: WindowRequest) -> dict[str, object]:
        return asdict(request)
