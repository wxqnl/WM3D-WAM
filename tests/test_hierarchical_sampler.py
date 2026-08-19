from dataclasses import replace
from pathlib import Path

from wm3d_wam.data.hierarchical_sampler import (
    PROGRAM_FAMILY_MIX,
    RecoverableHierarchicalSampler,
    source_sampling_weights,
)
from wm3d_wam.data.source_contracts import SourceContractRegistry


ROOT = Path(__file__).resolve().parents[1]


def _fixture():
    registry = SourceContractRegistry.load(
        ROOT / "configs/data/source_contracts_v1.yaml"
    )
    contracts = tuple(contract for contract in registry.sources.values() if contract.trainable)
    counts = {contract.name: 101 for contract in contracts}
    profile = {contract.name: float(index + 1) for index, contract in enumerate(contracts)}
    return contracts, counts, profile


def test_request_is_counter_based_and_resume_does_not_depend_on_prefetch():
    contracts, counts, profile = _fixture()
    kwargs = dict(
        episode_counts=counts,
        contracts=contracts,
        profile_weights=profile,
        program_mix={"action_only": 0.5, "forward_world": 0.3, "joint_world_action": 0.2},
        seed=123,
        rank=2,
        world_size=4,
    )
    sampler = RecoverableHierarchicalSampler(**kwargs)
    expected = sampler.request_at(17)
    state = sampler.state_dict(committed_local_samples=17)
    resumed_index = RecoverableHierarchicalSampler.validate_resume_state(
        state, seed=123, rank=2, world_size=4
    )
    resumed = RecoverableHierarchicalSampler(
        **kwargs, start_local_index=resumed_index
    )
    assert next(iter(resumed)) == expected
    assert expected.global_sample_index == 17 * 4 + 2


def test_distributed_ranks_never_share_a_global_sample_index():
    contracts, counts, profile = _fixture()
    values = set()
    for rank in range(4):
        sampler = RecoverableHierarchicalSampler(
            episode_counts=counts,
            contracts=contracts,
            profile_weights=profile,
            program_mix={"geometry_pretrain": 1.0},
            seed=7,
            rank=rank,
            world_size=4,
        )
        for local in range(20):
            value = sampler.request_at(local).global_sample_index
            assert value not in values
            values.add(value)


def test_distributed_ranks_share_route_schema_at_each_local_step():
    contracts, counts, profile = _fixture()
    samplers = [
        RecoverableHierarchicalSampler(
            episode_counts=counts,
            contracts=contracts,
            profile_weights=profile,
            program_mix={
                "action_only": 0.5,
                "forward_world": 0.3,
                "joint_world_action": 0.2,
            },
            seed=19,
            rank=rank,
            world_size=4,
        )
        for rank in range(4)
    ]
    for local in range(100):
        requests = [sampler.request_at(local) for sampler in samplers]
        assert len({request.program for request in requests}) == 1
        assert len({request.family for request in requests}) == 1
        assert len({request.source for request in requests}) == 1
        assert len({request.target_view_fraction for request in requests}) == 1
        assert len({request.global_sample_index for request in requests}) == 4


def test_source_weight_contract_keeps_robocasa_10_60_30():
    contracts, _, profile = _fixture()
    weights = source_sampling_weights(contracts, profile)
    assert weights["robocasa"] == {
        "robocasa_atomic": 0.1,
        "robocasa_composite": 0.6,
        "robocasa_mg": 0.3,
    }
    assert abs(sum(weights["oxe"].values()) - 1.0) < 1.0e-12
