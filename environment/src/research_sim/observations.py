"""
observations.py: Causal, bounded observation construction and legal action masks.

Schema 3.0 removes raw scale disparities, exposes causal restart exposure and
queue pressure, and encodes candidate provenance categorically. It still
excludes future arrivals and trace-known remaining durations.
"""

from typing import Dict, Any, Optional
import numpy as np

from .types import DecisionContext


OBSERVATION_SCHEMA_VERSION = "3.0"

NODE_FEATURE_NAMES = (
    "capacity_fraction",
    "free_card_fraction",
    "hp_occupied_fraction",
    "spot_occupied_fraction",
    "eviction_score_fraction",
    "rollback_exposure_fraction",
    "running_spot_job_density",
)

CANDIDATE_SOURCES = (
    "defer",
    "fallback",
    "native_gfs",
    "tight_pack",
    "spread",
    "local_move",
    "failure_domain_min_nodes",
    "preserve_full_nodes",
    "consume_full_nodes",
)

CANDIDATE_SOURCE_ALIASES = {
    "deferral": "defer",
    "pack": "tight_pack",
}

JOB_FEATURE_NAMES = (
    "worker_fraction_of_nodes",
    "gpu_per_worker_fraction",
    "total_gpu_demand_fraction",
    "queue_wait_squashed",
    "estimated_duration_squashed",
)

GLOBAL_FEATURE_NAMES = (
    "score_progress",
    "score_time_remaining",
    "quota_fraction",
    "spot_usage_fraction",
    "verification_budget_fraction",
    "quota_window_progress",
    "queued_hp_count_squashed",
    "queued_spot_count_squashed",
    "causal_forecast_fraction",
    "queued_hp_gpu_demand_squashed",
    "largest_queued_hp_gang_fraction",
    "oldest_queued_hp_wait_squashed",
    "fully_free_node_fraction",
)

NODE_FEATURE_DIM = len(NODE_FEATURE_NAMES)
CANDIDATE_FEATURE_DIM = 4 + len(CANDIDATE_SOURCES)
JOB_FEATURE_DIM = len(JOB_FEATURE_NAMES)
GLOBAL_FEATURE_DIM = len(GLOBAL_FEATURE_NAMES)
SOURCE_FEATURE_OFFSET = 4


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _fraction(value: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return _clip01(float(value) / float(denominator))


def _squash(value: float, scale: float) -> float:
    """Bound a nonnegative count while preserving ordering above the scale."""
    value = max(0.0, float(value))
    scale = max(1.0, float(scale))
    return _clip01(value / (value + scale))


def _job_value(job: Any, key: str, default: Any = None) -> Any:
    if hasattr(job, "get"):
        return job.get(key, default)
    try:
        return job[key]
    except (KeyError, TypeError):
        return default


def _checkpoint_time(job: Any, now: int) -> int:
    if hasattr(job, "get_ckpt_time"):
        return int(job.get_ckpt_time())
    return int(_job_value(job, "ckpt_time", _job_value(job, "start_time", now)))


def _node_spot_state(node: Any, now: int) -> tuple[float, float]:
    """Return bounded rollback exposure and running Spot job density."""
    per_job: Dict[Any, Dict[str, Any]] = {}
    for jobs in node.node_gpu_dict.values():
        for job in jobs:
            job_type = _job_value(job, "type", _job_value(job, "job_type", None))
            if job_type != "Spot":
                continue
            job_id = _job_value(job, "job_name", id(job))
            rec = per_job.setdefault(job_id, {"job": job, "gpu_equivalents": 0.0})
            rec["gpu_equivalents"] += min(
                1.0,
                max(0.0, float(_job_value(job, "gpu_request", 1.0))),
            )

    capacity = max(1.0, float(node.num_gpus))
    exposure = 0.0
    for rec in per_job.values():
        job = rec["job"]
        checkpoint_interval = max(
            1.0,
            float(_job_value(job, "spot_duration", 3600.0)),
        )
        checkpoint_age = min(
            checkpoint_interval,
            max(0.0, float(now - _checkpoint_time(job, now))),
        )
        exposure += (
            rec["gpu_equivalents"] / capacity
        ) * (
            checkpoint_age / checkpoint_interval
        )

    return _clip01(exposure), _squash(len(per_job), capacity)


def _candidate_source_index(source: str) -> int:
    canonical = CANDIDATE_SOURCE_ALIASES.get(source, source)
    try:
        return CANDIDATE_SOURCES.index(canonical)
    except ValueError as exc:
        raise ValueError(
            f"Candidate source {source!r} is absent from observation schema "
            f"{OBSERVATION_SCHEMA_VERSION}."
        ) from exc


def build_observation(
    context: Optional[DecisionContext],
    cluster_view: Any,
    N: int,
    M: int,
    K: int,
    verifier_ready: bool,
    filter_mode: Optional[str] = None,
    observation_time: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Construct a fixed shape, causal, bounded observation dictionary."""
    node_list = cluster_view.node_list
    if len(node_list) != N:
        raise ValueError(f"Cluster node count {len(node_list)} does not match expected N={N}.")

    sorted_nodes = sorted(node_list, key=lambda node: node.node_name)
    max_node_capacity = max(1.0, max(float(node.num_gpus) for node in sorted_nodes))
    total_gpu_capacity = max(1.0, sum(float(node.num_gpus) for node in sorted_nodes))
    now = (
        int(context.time)
        if context is not None
        else int(observation_time or 0)
    )

    node_feat = np.zeros((N, NODE_FEATURE_DIM), dtype=np.float32)
    eviction_scores = getattr(cluster_view, "eviction_score", None) or {}
    for idx, node in enumerate(sorted_nodes):
        capacity = max(1.0, float(node.num_gpus))
        free = float(sum(
            1 for card in range(node.num_gpus)
            if not node.node_gpu_dict[card]
        ))
        hp_occupied = float(sum(
            1 for card in range(node.num_gpus)
            if node.node_type_galloc[card]["HP"] > 0
        ))
        spot_occupied = float(sum(
            1 for card in range(node.num_gpus)
            if node.node_type_galloc[card]["Spot"] > 0
        ))
        rollback_exposure, running_spot_density = _node_spot_state(node, now)
        node_feat[idx] = [
            _fraction(capacity, max_node_capacity),
            _fraction(free, capacity),
            _fraction(hp_occupied, capacity),
            _fraction(spot_occupied, capacity),
            _fraction(float(eviction_scores.get(node.node_name, 0.0)), 10000.0),
            rollback_exposure,
            running_spot_density,
        ]

    cand_counts = np.zeros((M + 1, N), dtype=np.float32)
    cand_feat = np.zeros((M + 1, CANDIDATE_FEATURE_DIM), dtype=np.float32)
    job_feat = np.zeros(JOB_FEATURE_DIM, dtype=np.float32)
    glob_feat = np.zeros(GLOBAL_FEATURE_DIM, dtype=np.float32)
    action_mask = np.zeros(2 * (M + 1), dtype=bool)

    if context is None:
        cand_feat[0, 0] = 1.0
        source_index = _candidate_source_index("defer")
        cand_feat[0, SOURCE_FEATURE_OFFSET + source_index] = 1.0
        action_mask[0] = True
        return {
            "node_features": node_feat,
            "candidate_node_counts": cand_counts,
            "candidate_features": cand_feat,
            "job_features": job_feat,
            "global_features": glob_feat,
            "action_mask": action_mask,
        }

    node_idx_map = {node.node_name: idx for idx, node in enumerate(sorted_nodes)}
    for candidate in context.candidates.candidates:
        slot = int(candidate.slot)
        if slot > M:
            continue
        cand_feat[slot, 0] = 1.0
        source_index = _candidate_source_index(candidate.source)
        cand_feat[slot, SOURCE_FEATURE_OFFSET + source_index] = 1.0
        if slot == 0:
            continue

        placement = candidate.placement
        nodes_used = len(placement.key)
        worker_num = len(placement.workers)
        gpu_request = len(placement.workers[0].card_ids) if placement.workers else 0.0
        cand_feat[slot, 1:4] = [
            _fraction(nodes_used, N),
            _fraction(worker_num, N),
            _fraction(gpu_request, max_node_capacity),
        ]
        for node_id, count in placement.key:
            if node_id in node_idx_map:
                cand_counts[slot, node_idx_map[node_id]] = float(count)

    job_info = context.job_info
    worker_num = max(0.0, float(job_info.get("worker_num", 1.0)))
    gpu_request = max(0.0, float(job_info.get("gpu_request", 1.0)))
    total_demand = worker_num * gpu_request
    service_horizon = max(1.0, float(job_info.get("service_horizon", 3600.0)))
    job_feat[:] = [
        _fraction(worker_num, N),
        _fraction(gpu_request, max_node_capacity),
        _fraction(total_demand, total_gpu_capacity),
        _squash(float(job_info.get("elapsed_queue_time", 0.0)), service_horizon),
        _squash(float(job_info.get("estimated_duration_seconds", 0.0)), service_horizon),
    ]

    score_start = float(job_info.get("score_start", context.time))
    score_end = float(job_info.get("score_end", context.time))
    score_span = max(0.0, score_end - score_start)
    if score_span > 0:
        score_progress = _fraction(float(context.time) - score_start, score_span)
        score_remaining = _fraction(score_end - float(context.time), score_span)
    else:
        score_progress = 0.0
        score_remaining = 0.0

    total_capacity = max(
        1.0,
        float(job_info.get("total_gpu_capacity", total_gpu_capacity)),
    )
    budget_capacity = max(
        1.0,
        float(job_info.get("verification_budget_capacity", context.budget_remaining)),
    )
    queued_hp_count = float(job_info.get("queued_hp_count", 0.0))
    queued_spot_count = float(job_info.get("queued_spot_count", 0.0))
    causal_forecast = float(
        job_info.get("causal_forecast", job_info.get("current_quota", 0.0))
    )
    glob_feat[:] = [
        score_progress,
        score_remaining,
        _fraction(float(job_info.get("current_quota", 0.0)), total_capacity),
        _fraction(float(job_info.get("spot_in_use", 0.0)), total_capacity),
        _fraction(float(context.budget_remaining), budget_capacity),
        _fraction(float(context.time % 300), 300.0),
        _squash(queued_hp_count, N),
        _squash(queued_spot_count, N),
        _fraction(causal_forecast, total_capacity),
        _squash(float(job_info.get("queued_hp_gpu_demand", 0.0)), total_capacity),
        _fraction(float(job_info.get("largest_queued_hp_gang", 0.0)), total_capacity),
        _squash(float(job_info.get("oldest_queued_hp_wait", 0.0)), service_horizon),
        _fraction(float(job_info.get("fully_free_node_count", 0.0)), N),
    ]

    minimum_budget_for_verify = 2 * K
    for slot in range(M + 1):
        if not bool(cand_feat[slot, 0] > 0.5):
            continue
        action_mask[2 * slot] = True
        can_verify = filter_mode in ("paired_check", "label_collection") or filter_mode is None
        if (
            slot != 0
            and K > 0
            and can_verify
            and verifier_ready
            and context.budget_remaining >= minimum_budget_for_verify
        ):
            action_mask[2 * slot + 1] = True

    if not bool(action_mask[0]):
        raise AssertionError("Deferral action must always be legal.")

    return {
        "node_features": node_feat,
        "candidate_node_counts": cand_counts,
        "candidate_features": cand_feat,
        "job_features": job_feat,
        "global_features": glob_feat,
        "action_mask": action_mask,
    }
