from wm3d_wam.data.episode_splits import (
    is_v1_eligible_manifest_record,
    materialize_source_split,
    requested_holdout_count,
)


def _records(count: int, *, grouped: bool = False):
    return [
        {
            "source": "source_a",
            "episode_id": f"source_a:{index:06d}",
            **({"parent_trajectory_id": f"parent:{index // 2:04d}"} if grouped else {}),
        }
        for index in range(count)
    ]


def test_split_counts_and_seed_are_deterministic():
    first = materialize_source_split(_records(1000), source="source_a", seed=17)
    second = materialize_source_split(_records(1000), source="source_a", seed=17)
    assert first == second
    assert len(first.validation) == 10
    assert len(first.test) == 10
    assert len(first.train) == 980
    assert requested_holdout_count(49) == 5
    assert requested_holdout_count(120) == 10


def test_parent_trajectory_is_never_split_across_partitions():
    split = materialize_source_split(
        _records(30, grouped=True), source="source_a", seed=3
    )
    membership = {}
    for name, values in (
        ("train", split.train),
        ("val", split.validation),
        ("test", split.test),
    ):
        for episode_id in values:
            parent = int(episode_id.rsplit(":", 1)[1]) // 2
            assert membership.setdefault(parent, name) == name


def test_manifest_eligibility_uses_recorded_clock_span():
    base = {
        "split": "train",
        "observation_clock": {"start_s": 0.0, "end_s": 4.32, "sample_count": 24},
    }
    assert is_v1_eligible_manifest_record(base)
    assert not is_v1_eligible_manifest_record(
        {**base, "observation_clock": {"start_s": 0.0, "end_s": 4.31, "sample_count": 24}}
    )
    assert not is_v1_eligible_manifest_record({**base, "split": "validation"})
