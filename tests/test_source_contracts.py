from pathlib import Path

import numpy as np
import pytest

from wm3d_wam.data.source_contracts import (
    NormalizationRegistry,
    NormalizationStat,
    SourceContractError,
    SourceContractRegistry,
    denormalize_action_values,
    pack_robot_arrays,
)


ROOT = Path(__file__).resolve().parents[1]


def _identity_normalization(source: str, action_dim: int, state_dim: int):
    rows = {
        (source, "action", index): NormalizationStat(offset=0.0, scale=1.0)
        for index in range(action_dim)
    }
    rows.update(
        {
            (source, "state", index): NormalizationStat(offset=0.0, scale=1.0)
            for index in range(state_dim)
        }
    )
    return NormalizationRegistry(rows=rows)


def test_registry_accounts_for_all_21_sources_and_hard_excludes_ambiguous_payloads():
    registry = SourceContractRegistry.load(
        ROOT / "configs/data/source_contracts_v1.yaml"
    )
    assert len(registry.sources) == 21
    assert {contract.name for contract in registry.approved("action_only")} == {
        "oxe_bridge",
        "oxe_droid",
        "oxe_furniture_bench",
        "oxe_bc_z",
        "robocasa_atomic",
        "robocasa_composite",
        "robocasa_mg",
    }
    with pytest.raises(SourceContractError, match="excluded"):
        registry.require("oxe_austin_buds", program="action_only")


def test_robocasa_groups_and_gripper_roundtrip():
    registry = SourceContractRegistry.load(
        ROOT / "configs/data/source_contracts_v1.yaml"
    )
    contract = registry.require("robocasa_atomic")
    normalization = _identity_normalization(
        contract.name, contract.raw_action_dim, contract.raw_state_dim
    )
    action = np.zeros((2, 12), dtype=np.float32)
    action[:, 4] = (-1.0, 1.0)
    action[:, 11] = (-1.0, 1.0)
    state = np.zeros((2, 16), dtype=np.float32)
    packed = pack_robot_arrays(
        action,
        state,
        contract=contract,
        normalization=normalization,
        max_groups=8,
        max_action_dim=16,
        max_state_dim=32,
    )
    assert packed.group_ids[:5].tolist() == [34, 35, 36, 31, 32]
    assert packed.action_values[:, 4, 0].tolist() == [0.0, 1.0]
    recovered = denormalize_action_values(
        packed.action_values,
        contract=contract,
        normalization=normalization,
    )
    np.testing.assert_allclose(recovered, action)


def test_bridge_pad_state_column_is_explicitly_ignored():
    registry = SourceContractRegistry.load(
        ROOT / "configs/data/source_contracts_v1.yaml"
    )
    contract = registry.require("oxe_bridge")
    normalization = _identity_normalization(
        contract.name, contract.raw_action_dim, contract.raw_state_dim
    )
    action = np.zeros((1, 7), dtype=np.float32)
    state = np.arange(8, dtype=np.float32)[None]
    packed = pack_robot_arrays(
        action,
        state,
        contract=contract,
        normalization=normalization,
        max_groups=8,
        max_action_dim=16,
        max_state_dim=32,
    )
    assert not bool((packed.state_values == 6.0).any())
    assert packed.state_value_mask[0, 0, :6].all()
    assert packed.state_value_mask[0, 1, 0]
