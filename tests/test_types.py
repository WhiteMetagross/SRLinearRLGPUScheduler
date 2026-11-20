import pytest
from research_sim.types import (
    EpisodeSpec,
    InvalidConfigError,
    InvalidActionError,
    PhaseCursor,
    Phase,
    WorkerAllocation,
    Placement,
    Candidate,
    CandidateSet,
    DecisionContext
)

def test_episode_spec_validation():
    # Valid spec
    spec = EpisodeSpec()
    assert spec.score_start == 5270400
    assert spec.score_end == 10540800
    assert spec.followup_end >= spec.score_end + spec.L_H

    # Invalid bounds
    with pytest.raises(InvalidConfigError):
        EpisodeSpec(score_start=1000, score_end=500)

    # Invalid followup_end
    with pytest.raises(InvalidConfigError):
        EpisodeSpec(score_end=1000, followup_end=1000, L_H=100)

    # Invalid retry_mode
    with pytest.raises(InvalidConfigError):
        EpisodeSpec(retry_mode="bad_mode")

    # Invalid trace_mode
    with pytest.raises(InvalidConfigError):
        EpisodeSpec(trace_mode="bad_trace")

def test_placement_canonical_key():
    alloc1 = WorkerAllocation(node_id=3, card_ids=(0, 1))
    alloc2 = WorkerAllocation(node_id=1, card_ids=(2, 3))
    alloc3 = WorkerAllocation(node_id=3, card_ids=(4, 5))

    p1 = Placement.from_allocations([alloc1, alloc2, alloc3])
    # Sorted key should group by node_id: node 1 has 1 worker, node 3 has 2 workers
    assert p1.key == ((1, 1), (3, 2))
    assert not p1.is_deferral

    defer = Placement.make_deferral()
    assert defer.is_deferral
    assert defer.key == ()
