import os

import pytest

from research_sim.parallel import (
    configure_worker_threads,
    ordered_process_map,
    plan_parallelism,
)


def test_parallelism_plan_prevents_nested_cpu_oversubscription():
    plan = plan_parallelism(8, available_cpus=6, threads_per_worker=2)
    assert plan.worker_count == 3
    assert plan.limiting_factors == ("cpu",)


def test_parallelism_plan_applies_memory_cap():
    plan = plan_parallelism(
        "auto",
        available_cpus=16,
        memory_limit_gib=10.0,
        estimated_worker_gib=3.0,
    )
    assert plan.worker_count == 3
    assert plan.limiting_factors == ("memory",)


def test_parallelism_plan_rejects_partial_memory_information():
    with pytest.raises(ValueError, match="must be set together"):
        plan_parallelism(2, memory_limit_gib=8.0)


def test_worker_thread_limits_are_explicit(monkeypatch):
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.delenv(key, raising=False)
    configure_worker_threads(1)
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"


def test_ordered_process_map_preserves_input_order():
    assert ordered_process_map(
        abs,
        [-3, -1, 2],
        requested_workers=2,
        available_cpus=2,
    ) == [3, 1, 2]
