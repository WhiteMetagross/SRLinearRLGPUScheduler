"""
types.py: Typed immutable contracts, configurations, cursors, and errors for research_sim.
No pandas objects or future trace labels are exposed to decision policies.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Tuple, Dict, Set, Optional, List


NATIVE_GFS_SEQUENCE_LENGTH_HOURS = 672
NATIVE_GFS_MIN_HISTORY_SECONDS = NATIVE_GFS_SEQUENCE_LENGTH_HOURS * 3600


# ---------------------------------------------------------------------------
# Typed Exceptions
# ---------------------------------------------------------------------------

class ResearchSimError(Exception):
    """Base exception for research simulator errors."""
    pass


class InvalidActionError(ResearchSimError):
    """Raised when an action violates constraints or is invalid."""
    pass


class StaleStateError(ResearchSimError):
    """Raised when an action is submitted for an expired state version."""
    pass


class InvalidConfigError(ResearchSimError):
    """Raised when an episode specification or configuration is invalid."""
    pass


class ExhaustedReplayError(ResearchSimError):
    """Raised when advance is called on a finished replay."""
    pass


# ---------------------------------------------------------------------------
# Simulator Phases
# ---------------------------------------------------------------------------

class Phase(str, Enum):
    QUOTA = "QUOTA"
    CONTROL = "CONTROL"
    RUNNING = "RUNNING"
    ARRIVALS = "ARRIVALS"
    HP_SCAN = "HP_SCAN"
    SPOT_SCAN = "SPOT_SCAN"
    RECORD = "RECORD"
    TICK_END = "TICK_END"


# ---------------------------------------------------------------------------
# Episode Specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EpisodeSpec:
    node_filter: str = "A100-SXM4-80GB"
    calendar_anchor: str = "2024-03-01 00:00:00"
    warmup_start: int = 0
    score_start: int = 5270400      # 2024-05-01 00:00:00 (offset: 61 days)
    score_end: int = 10540800       # 2024-07-01 00:00:00 (offset: 122 days)
    followup_end: int = 10544400    # score_end + L_H (at least)
    retry_mode: str = "next_event"  # "every_second" or "next_event"
    trace_mode: str = "research_full_history"  # "research_full_history" or "source_compatible"
    forecast_mode: str = "native_online"      # "native_online" or "causal_capacity"
    rule_version: str = "1.0"
    split_id: str = "train"
    environment_seed: int = 20261002
    policy_seed: int = 42
    forecast_provider_id: str = "builtin"
    forecast_provider_version: str = "1.0"
    spot_arrival_multiplier: int = 1
    spot_arrival_seed: int = 0
    spot_stress_start: Optional[int] = None
    L_H: int = 3600                 # Service deadline limit in seconds (1 hour default)
    B_call: int = 2000              # Max solver calls per 300s quota window
    K: int = 4                      # Paired scenarios for verification
    M: int = 8                      # Maximum candidate placements
    candidate_geometry_version: str = "legacy_v1"
    gpu_accounting_mode: str = "auto"
    control_interval_seconds: int = 300
    control_queue_size: int = 64
    max_quota_delta_fraction: float = 1.0

    def __post_init__(self):
        if self.warmup_start != 0:
            raise InvalidConfigError("Replay must start at trace time zero to retain prior occupancy.")
        if not (self.warmup_start <= self.score_start < self.score_end):
            raise InvalidConfigError(
                f"Invalid time bounds: warmup_start ({self.warmup_start}) <= "
                f"score_start ({self.score_start}) < score_end ({self.score_end}) required."
            )
        if self.followup_end < self.score_end + self.L_H:
            raise InvalidConfigError(
                f"followup_end ({self.followup_end}) must be >= score_end + L_H "
                f"({self.score_end + self.L_H})."
            )
        if self.retry_mode not in ("every_second", "next_event"):
            raise InvalidConfigError(f"Unknown retry_mode: {self.retry_mode}")
        if self.trace_mode not in ("research_full_history", "source_compatible"):
            raise InvalidConfigError(f"Unknown trace_mode: {self.trace_mode}")
        if self.gpu_accounting_mode == "auto":
            object.__setattr__(
                self,
                "gpu_accounting_mode",
                "source_per_worker" if self.trace_mode == "source_compatible" else "total_demand",
            )
        if self.gpu_accounting_mode not in ("source_per_worker", "total_demand"):
            raise InvalidConfigError(
                f"Unknown gpu_accounting_mode: {self.gpu_accounting_mode}"
            )
        if self.forecast_mode not in ("native_online", "causal_capacity", "external_causal"):
            raise InvalidConfigError(f"Unknown forecast_mode: {self.forecast_mode}")
        if isinstance(self.environment_seed, bool) or int(self.environment_seed) != self.environment_seed:
            raise InvalidConfigError("environment_seed must be an integer.")
        if not str(self.forecast_provider_id).strip():
            raise InvalidConfigError("forecast_provider_id must be nonempty.")
        if not str(self.forecast_provider_version).strip():
            raise InvalidConfigError("forecast_provider_version must be nonempty.")
        if self.forecast_provider_id == "builtin":
            effective_id = {
                "native_online": "native_gfs",
                "causal_capacity": "causal_capacity",
            }.get(self.forecast_mode)
            if effective_id is None:
                raise InvalidConfigError(
                    "external_causal requires an explicit forecast_provider_id."
                )
            object.__setattr__(self, "forecast_provider_id", effective_id)
        if self.forecast_mode == "native_online" and self.forecast_provider_id != "native_gfs":
            raise InvalidConfigError("native_online requires forecast_provider_id='native_gfs'.")
        if self.forecast_mode == "causal_capacity" and self.forecast_provider_id != "causal_capacity":
            raise InvalidConfigError("causal_capacity requires forecast_provider_id='causal_capacity'.")
        if isinstance(self.spot_arrival_multiplier, bool) or int(self.spot_arrival_multiplier) != self.spot_arrival_multiplier or self.spot_arrival_multiplier < 1:
            raise InvalidConfigError("spot_arrival_multiplier must be a positive integer.")
        if int(self.spot_arrival_seed) != self.spot_arrival_seed:
            raise InvalidConfigError("spot_arrival_seed must be an integer.")
        if self.spot_stress_start is not None:
            if int(self.spot_stress_start) != self.spot_stress_start:
                raise InvalidConfigError("spot_stress_start must be an integer when set.")
            if self.spot_stress_start < 0 or self.spot_stress_start > self.score_start:
                raise InvalidConfigError(
                    "spot_stress_start must be between warmup_start and score_start."
                )
        if self.B_call < 0:
            raise InvalidConfigError(f"B_call must be non-negative, got {self.B_call}")
        if self.B_call > 0 and self.K < 1:
            raise InvalidConfigError(f"K must be >= 1 when B_call > 0, got {self.K}")
        if self.M < 2:
            raise InvalidConfigError(f"M must be >= 2 to keep fallback and native candidates, got {self.M}")
        if self.control_interval_seconds != 300:
            raise InvalidConfigError(
                "control_interval_seconds must remain 300 to match the GFS quota cadence."
            )
        if self.control_queue_size < 1:
            raise InvalidConfigError("control_queue_size must be positive.")
        if not 0.0 < float(self.max_quota_delta_fraction) <= 1.0:
            raise InvalidConfigError(
                "max_quota_delta_fraction must be within (0, 1]."
            )
        if self.candidate_geometry_version not in {
            "legacy_v1",
            "free_node_extremes_v1",
            "failure_domain_v1",
            "publication_v1",
        }:
            raise InvalidConfigError(
                f"Unknown candidate_geometry_version: {self.candidate_geometry_version}"
            )


# ---------------------------------------------------------------------------
# Phase Cursor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhaseCursor:
    time: int
    phase: Phase
    vc_name: str
    spot_queue_index: int
    offered_jobs: Tuple[int, ...] = field(default_factory=tuple)
    state_version: int = 0


# ---------------------------------------------------------------------------
# Worker and Placement Representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorkerAllocation:
    node_id: int
    card_ids: Tuple[int, ...]

    def __post_init__(self):
        if not isinstance(self.card_ids, tuple):
            object.__setattr__(self, 'card_ids', tuple(self.card_ids))


@dataclass(frozen=True)
class Placement:
    workers: Tuple[WorkerAllocation, ...]
    key: Tuple[Tuple[int, int], ...]
    is_deferral: bool = False

    @classmethod
    def make_deferral(cls) -> "Placement":
        return cls(workers=(), key=(), is_deferral=True)

    @classmethod
    def from_allocations(cls, allocations: List[WorkerAllocation]) -> "Placement":
        # Group by node_id to create canonical key
        counts: Dict[int, int] = {}
        for alloc in allocations:
            counts[alloc.node_id] = counts.get(alloc.node_id, 0) + 1
        sorted_key = tuple(sorted(counts.items()))
        return cls(workers=tuple(allocations), key=sorted_key, is_deferral=False)


@dataclass(frozen=True)
class Candidate:
    slot: int
    source: str
    placement: Placement
    features: Tuple[float, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CandidateSet:
    state_version: int
    job_id: int
    time: int
    candidates: Tuple[Candidate, ...]
    truncation_count: int = 0
    unique_count: int = 0


# ---------------------------------------------------------------------------
# Decision Context
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecisionContext:
    state_version: int
    time: int
    job_id: int
    job_info: Dict[str, Any]
    budget_remaining: int
    cursor: PhaseCursor
    candidates: CandidateSet
    empty_cards_by_node: Dict[Any, Tuple[int, ...]] = field(default_factory=dict)
    node_capacities: Dict[Any, int] = field(default_factory=dict)
    eviction_scores: Dict[Any, float] = field(default_factory=dict)
    spot_occupied_cards: Dict[Any, int] = field(default_factory=dict)


@dataclass(frozen=True)
class AdmissionJobView:
    """Causal queue record exposed to admission controllers."""

    job_id: int
    pool: str
    submit_time: int
    worker_num: int
    gpu_request: float
    total_gpu_demand: float
    guarantee_request: int
    estimated_duration_seconds: int
    features: Tuple[float, ...]


@dataclass(frozen=True)
class ControlDecisionContext:
    """Read-only state at a 300 second quota and admission control tick."""

    state_version: int
    time: int
    base_quota_by_pool: Dict[str, Dict[int, int]]
    capacity_by_pool: Dict[str, int]
    hp_used_by_pool: Dict[str, float]
    spot_used_by_pool: Dict[str, float]
    hp_queue_count: int
    hp_queue_gpu_demand: float
    admission_jobs: Tuple[AdmissionJobView, ...]
    eviction_failure_rate_by_pool: Dict[str, float]
    score_progress: float


# ---------------------------------------------------------------------------
# Event and Ledger Records
# ---------------------------------------------------------------------------

EVENT_SCHEMA_VERSION = "2.0"


@dataclass(frozen=True)
class SimulationEvent:
    schema_version: str
    run_id: str
    episode_id: str
    event_seq: int
    time: int
    phase: str
    event_type: str
    job_id: Optional[Any] = None
    vc_name: Optional[str] = None
    attempt_index: Optional[int] = None
    gpu_count: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": getattr(self, "schema_version", "2.0"),
            "run_id": getattr(self, "run_id", ""),
            "episode_id": getattr(self, "episode_id", ""),
            "event_seq": getattr(self, "event_seq", 0),
            "time": self.time,
            "phase": getattr(self, "phase", "warmup"),
            "event_type": self.event_type,
            "job_id": getattr(self, "job_id", None),
            "vc_name": getattr(self, "vc_name", None),
            "attempt_index": getattr(self, "attempt_index", None),
            "details": getattr(self, "details", {}),
        }


@dataclass(frozen=True)
class TransitionDelta:
    reward: float
    elapsed_sim_seconds: int
    solver_calls: int
    events: Tuple[SimulationEvent, ...]
    terminated: bool
    truncated: bool
    terminal_reason: Optional[str] = None
    initial_reward_gpu_hours: float = 0.0
    initial_solver_calls: int = 0
    initial_deadline_events: int = 0
    initial_evictions: int = 0
    initial_lost_gpu_seconds: float = 0.0
    spot_queue_gpu_hours: float = 0.0
    lost_gpu_hours: float = 0.0
    control_reward: Optional[float] = None


# ---------------------------------------------------------------------------
# Read-Only Verifier State Snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NodeStateSnapshot:
    node_id: int
    gpu_model: str
    total_cards: int
    free_cards: Tuple[int, ...]
    hp_occupied_cards: Tuple[int, ...]
    spot_occupied_cards: Tuple[int, ...]
    eviction_score: float = 0.0


@dataclass(frozen=True)
class PreemptibleSpotUnit:
    job_id: int
    start_time: int              # s_j
    duration_remaining: Optional[int]  # None means causal residual estimate unsupported
    allocations: Tuple[Tuple[int, int], ...] # ((node_id, card_count), ...)
    gpu_model: str
    checkpoint_time: int = 0
    worker_num: Optional[int] = None
    gpu_request: Optional[float] = None
    # Exact occupied cards, needed to reproduce the simulator's victim rule.
    # Empty on legacy fixtures; a card level predictor must then fail closed.
    card_allocations: Tuple[Tuple[int, Tuple[int, ...]], ...] = field(default_factory=tuple)
    spot_duration: Optional[int] = None


@dataclass(frozen=True)
class QueuedHPJob:
    job_id: int
    submit_time: int
    worker_num: int
    gpu_request: int
    gpu_model: str
    duration: int


@dataclass(frozen=True)
class FixedNodeOccupancy:
    node_id: int
    occupied_cards: int
    duration_remaining: Optional[int]  # None means causal residual estimate unsupported
    card_ids: Tuple[int, ...] = field(default_factory=tuple)
    job_id: Optional[int] = None


@dataclass(frozen=True)
class ClusterStateSnapshot:
    time: int
    state_version: int
    nodes: Tuple[NodeStateSnapshot, ...]
    quota: float
    budget_remaining: int
    running_spot_count: int
    queued_spot_count: int
    queued_hp_count: int
    spot_units: Tuple[PreemptibleSpotUnit, ...] = field(default_factory=tuple)
    queued_hp_jobs: Tuple[QueuedHPJob, ...] = field(default_factory=tuple)
    fixed_occupancy: Tuple[FixedNodeOccupancy, ...] = field(default_factory=tuple)
    # (gpu_model, failed_count, succeeded_count, vc_total_cards, time_span).
    # These are causal scheduler state, required by native single worker cost.
    native_preemption_stats: Tuple[Tuple[str, int, int, int, int], ...] = field(default_factory=tuple)
