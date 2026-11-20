from research_sim.types import EpisodeSpec
from research_sim.warmup_snapshot import (
    _snapshot_split_matches, get_snapshot_key, get_snapshot_path, has_warmup_snapshot,
)


def test_train_prefix_can_be_reused_for_causal_history_label():
    assert _snapshot_split_matches("train", "history_8812800", True)


def test_history_prefix_reuse_is_opt_in_and_narrow():
    assert not _snapshot_split_matches("train", "history_8812800", False)
    assert _snapshot_split_matches("val", "history_8812800", True)
    assert _snapshot_split_matches("test", "history_8812800", True)
    assert not _snapshot_split_matches("history_8812800", "train", True)


def test_snapshot_key_distinguishes_stress_lead_in_start():
    common = {
        "score_start": 8985600,
        "score_end": 9072000,
        "followup_end": 9075600,
        "spot_arrival_multiplier": 4,
        "spot_arrival_seed": 917,
    }
    same_day = EpisodeSpec(**common)
    lead_in = EpisodeSpec(**common, spot_stress_start=8899200)
    assert get_snapshot_key(same_day) != get_snapshot_key(lead_in)
    assert "_from8899200" in get_snapshot_key(lead_in)


def test_old_completion_credit_snapshot_is_not_reused(tmp_path):
    spec = EpisodeSpec(score_start=100, score_end=200, followup_end=3800)
    current = get_snapshot_path(spec, str(tmp_path))
    old_name = current.name.replace("_acctcompletion_credit_cap_v2", "")
    (tmp_path / old_name).write_bytes(b"obsolete snapshot placeholder")
    assert "acctcompletion_credit_cap_v2" in current.name
    assert not current.exists()
    assert not has_warmup_snapshot(spec, str(tmp_path))


def test_snapshot_key_uses_environment_seed_not_policy_seed():
    common = {"score_start": 100, "score_end": 200, "followup_end": 3800}
    first_policy = EpisodeSpec(**common, environment_seed=7, policy_seed=42)
    second_policy = EpisodeSpec(**common, environment_seed=7, policy_seed=123)
    other_environment = EpisodeSpec(**common, environment_seed=8, policy_seed=42)

    assert get_snapshot_key(first_policy) == get_snapshot_key(second_policy)
    assert get_snapshot_key(first_policy) != get_snapshot_key(other_environment)


def test_long_snapshot_key_uses_deterministic_short_path(tmp_path):
    spec = EpisodeSpec(
        score_start=8985600,
        score_end=9072000,
        followup_end=9075600,
        split_id="train_june13_stress2x_gate_" + "long" * 80,
        spot_arrival_multiplier=2,
        spot_arrival_seed=4201,
        spot_stress_start=8899200,
        candidate_geometry_version="publication_v1",
        gpu_accounting_mode="total_demand",
        rule_version="3.1",
    )

    path = get_snapshot_path(spec, str(tmp_path))

    assert len(path.name.encode("utf-8")) <= 240
    assert path == get_snapshot_path(spec, str(tmp_path))
    assert path.name.endswith("_v3.1.pkl")


def test_long_snapshot_paths_remain_collision_resistant(tmp_path):
    common = {
        "score_start": 8985600,
        "score_end": 9072000,
        "followup_end": 9075600,
        "split_id": "x" * 300,
        "spot_arrival_multiplier": 2,
        "spot_arrival_seed": 4201,
        "spot_stress_start": 8899200,
        "candidate_geometry_version": "publication_v1",
        "gpu_accounting_mode": "total_demand",
        "rule_version": "3.1",
    }

    first = get_snapshot_path(EpisodeSpec(**common, environment_seed=1), str(tmp_path))
    second = get_snapshot_path(EpisodeSpec(**common, environment_seed=2), str(tmp_path))

    assert first != second
