"""
core.py: Maintained GFS research simulation engine and pausable decision surface.
Maintains exact second-by-second phase ordering:
QUOTA -> RUNNING -> ARRIVALS -> HP_SCAN -> SPOT_SCAN -> RECORD -> TICK_END.
Pauses only at eligible whole-card Spot placement opportunities and resumes
without repeating earlier phases of the second.
"""

import os
import copy
import itertools
import logging
import random
from types import SimpleNamespace
from typing import Dict, Any, Tuple, Optional, List, Set, Mapping
import numpy as np
import pandas as pd
from scipy.stats import norm

from .gfs.cluster import Cluster
from .gfs.job import Job, Trace
from .gfs.policy.spot import SpotScheduler
from .gfs.policy.placer.consolidate import ConsolidatePlacement
from .types import (
    EpisodeSpec,
    Phase,
    PhaseCursor,
    DecisionContext,
    ControlDecisionContext,
    CandidateSet,
    Placement,
    TransitionDelta,
    SimulationEvent,
    ClusterStateSnapshot,
    NodeStateSnapshot,
    PreemptibleSpotUnit,
    QueuedHPJob,
    FixedNodeOccupancy,
    ResearchSimError,
    InvalidActionError,
    StaleStateError,
    ExhaustedReplayError,
    InvalidConfigError,
    NATIVE_GFS_SEQUENCE_LENGTH_HOURS,
)
from .trace import EpisodeTraceView
from .candidates import CandidateGenerator, has_complete_fit
from .verifier import BudgetManager, NoCheckVerifier, VerifierProtocol
from .metrics import EventLedger, EpisodeMetrics
from .observations import build_observation
from .risk import ActionFilter
from .forecasts import CausalForecastContext, validate_quota_update
from .control import (
    UnifiedControlAction,
    WeightedAdmissionRanker,
    apply_quota_delta,
    build_admission_job_view,
    native_admission_key,
)


SOURCE_COMPATIBLE_PROFILES = {
    "yarn_cs_source_compatible",
    "chronus_source_compatible",
    "lyra_source_compatible",
    "fgd_source_compatible",
}
MECHANISM_PROBE_PROFILES = {
    "capacity_open_fcfs_best_fit_probe",
    "staggered_lease_best_fit_probe",
    "whole_card_cost_best_fit_probe",
    "whole_card_fgd_probe",
}
CHRONUS_PROFILES = {
    "chronus_source_compatible",
    "staggered_lease_best_fit_probe",
}
CAPACITY_QUOTA_PROFILES = {
    "chronus_source_compatible",
    "capacity_open_fcfs_best_fit_probe",
    "staggered_lease_best_fit_probe",
}
LYRA_COST_PROFILES = {
    "lyra_source_compatible",
    "whole_card_cost_best_fit_probe",
}
FGD_PROFILES = {
    "fgd_source_compatible",
    "whole_card_fgd_probe",
}
SRGFS_COST_PROFILES = {"srgfs_cost_aware"}


class ResearchSimulator(SpotScheduler):
    """
    Maintained GFS research simulator engine.
    Inherits upstream GFS placement, preemption, and quota mechanisms.
    Adds pausable decision surface, atomic commit, and event ledger.
    """
    def __init__(
        self,
        spec: EpisodeSpec,
        view: EpisodeTraceView,
        policy_type: str = "native_gfs",
        scheduler_profile: str = "native_gfs",
        verifier: Optional[VerifierProtocol] = None,
        duration_estimator: Optional[Any] = None,
        quota_provider: Optional[Any] = None,
        risk_model: Optional[Any] = None,
        action_filter: Optional[Any] = None,
        filter_mode: str = "passthrough",
        log_dir: Optional[str] = None,
        save_warmup_snapshot_to: Optional[str] = None,
        run_id: Optional[str] = None,
        episode_id: Optional[str] = None
    ):
        self.spec = spec
        self.view = view
        self.policy_type = policy_type
        self.scheduler_profile = "native_gfs"
        self.verifier = verifier or NoCheckVerifier()
        self.duration_estimator = duration_estimator
        self.quota_provider = quota_provider
        if spec.forecast_mode == "external_causal":
            if quota_provider is None:
                raise InvalidConfigError(
                    "external_causal requires a quota_provider instance."
                )
            provider_id = str(getattr(quota_provider, "provider_id", ""))
            provider_version = str(getattr(quota_provider, "provider_version", ""))
            if provider_id != spec.forecast_provider_id:
                raise InvalidConfigError(
                    f"quota_provider id {provider_id!r} does not match "
                    f"spec id {spec.forecast_provider_id!r}."
                )
            if provider_version != spec.forecast_provider_version:
                raise InvalidConfigError(
                    f"quota_provider version {provider_version!r} does not match "
                    f"spec version {spec.forecast_provider_version!r}."
                )
        elif quota_provider is not None:
            raise InvalidConfigError(
                "quota_provider may be supplied only for forecast_mode='external_causal'."
            )
        self.risk_model = risk_model
        self.filter_mode = filter_mode
        if action_filter is not None:
            self.action_filter = action_filter
        else:
            self.action_filter = ActionFilter(
                mode=filter_mode,
                risk_model=risk_model,
                verifier=self.verifier
            )
        self.save_warmup_snapshot_to = save_warmup_snapshot_to
        self._warmup_snapshot_hook = None
        stress_tag = (
            f"_spot{int(spec.spot_arrival_multiplier)}_stress{int(spec.spot_arrival_seed)}"
            if int(spec.spot_arrival_multiplier) != 1
            else ""
        )
        if spec.spot_stress_start is not None:
            stress_tag += f"_from{int(spec.spot_stress_start)}"
        self.episode_id = episode_id or (
            f"{spec.node_filter}_{spec.trace_mode}_env{spec.environment_seed}_"
            f"policy{spec.policy_seed}{stress_tag}"
        )

        # Upstream arguments adapter
        args = SimpleNamespace(
            experiment_name="research_sim",
            log_dir=log_dir or "/tmp/research_sim_logs",
            scheduler="spot_scheduler",
            trace_range=(spec.calendar_anchor, "2024-08-31 23:59:59"),
            log_range=(spec.score_start, spec.score_end),
            guarantee_rate=0.9,
            guarantee_hour=[1],
            pred_len=4,
            seq_len=NATIVE_GFS_SEQUENCE_LENGTH_HOURS,
            label_len=24,
            ckpt_interval=3600,
            checkpoints=os.path.join(log_dir or "/tmp/research_sim_logs", "checkpoints"),
            use_gpu=False,
            colocate=False,
            batch_size=32,
            learning_rate=0.00004,
            devices="0",
            use_multi_gpu=False,
            gpu=0,
            embed="timeF",
            freq="h",
            num_covariates=3,
            num_class=3,
            class_input_dim=[6, 1, 84],
            class_output_dim=8,
            attn_hidden_dim=16,
            num_workers=1,
            train_epochs=10,
            patience=3,
            des="test",
            loss="MSE",
            lradj="type1",
            moving_avg=25,
            enc_in=7,
            inverse=True,
            scaling=False
        )

        logger = logging.getLogger(f"research_sim.{self.episode_id}")
        logger.setLevel(logging.WARNING)

        # Seed random generators before constructing Cluster, Trace, or Estimator
        random.seed(spec.environment_seed)
        np.random.seed(spec.environment_seed)
        try:
            import torch
            torch.manual_seed(spec.environment_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(spec.environment_seed)
        except ImportError:
            pass

        # Build Cluster and Trace from EpisodeTraceView
        cluster = Cluster(view.node_df, cluster_name=spec.node_filter)
        trace = Trace()
        for j_dict in view.get_all_jobs_fresh():
            trace.append_job(Job(j_dict))

        os.makedirs(args.log_dir, exist_ok=True)
        super().__init__(trace, cluster, args, args.log_dir, logger)
        self.set_scheduler_profile(scheduler_profile)

        # Core components
        self.candidate_generator = CandidateGenerator(
            M=spec.M,
            geometry_version=spec.candidate_geometry_version,
        )
        self.run_id = run_id or (
            os.path.basename(os.path.dirname(log_dir.rstrip("/\\")))
            if log_dir and os.path.basename(log_dir.rstrip("/\\")) == "logs"
            else os.path.basename(log_dir.rstrip("/\\")) if log_dir
            else f"run_{self.episode_id}"
        )
        self.ledger = EventLedger(self.episode_id, spec, run_id=self.run_id)
        self._spot_retained_service_seconds: Dict[Any, int] = {}
        self._spot_preemption_overhead_seconds: Dict[Any, int] = {}
        self._spot_service_event_cursor = 0
        self.budget = BudgetManager(
            B_call=spec.B_call,
            K=spec.K,
            log_dir=log_dir,
            ledger=self.ledger,
            run_id=self.run_id,
            episode_id=self.episode_id
        )

        # Attempt and sequence tracking for Spot jobs
        self._spot_attempt_indices: Dict[Any, int] = {}
        self._spot_credit_sequences: Dict[Any, int] = {}
        self._spot_credited_gpu_seconds: Dict[Any, float] = {}

        # Reset prefix accounting
        self.prefix_reward_gpu_hours: float = 0.0
        self.prefix_solver_calls: int = 0
        self.prefix_deadline_events: int = 0
        self.prefix_evictions: int = 0
        self.prefix_lost_gpu_seconds: float = 0.0
        self._first_decision_returned: bool = False
        self.step_reward_gpu_hours_sum: float = 0.0

        # Monotonic decision latency tracking
        import time as _sys_time
        self.timer_func = _sys_time.perf_counter_ns
        self.decision_timing_records: List[Dict[str, Any]] = []
        self._last_candidate_start_ns: Optional[int] = None
        self._last_candidate_end_ns: Optional[int] = None
        self._last_candidate_gen_ms: float = 0.0
        self._decision_paused_time_ns: Optional[int] = None

        # Pausable execution state
        self.state_version: int = 0
        self.phase: Phase = Phase.QUOTA
        self.time: int = spec.warmup_start
        self.offered_jobs_this_tick: Set[int] = set()
        self.current_arrivals_idx: int = 0

        # Scan pointer state for SPOT_SCAN pause / resume
        self.paused_context: Optional[DecisionContext] = None
        self.control_paused_context: Optional[ControlDecisionContext] = None
        self.admission_ranker = WeightedAdmissionRanker()
        self.admission_queue_scans: int = 0
        self.admission_reordered_scans: int = 0
        self.admission_duration_nonconstant_scans: int = 0
        self.active_scan_job: Optional[Job] = None
        self.active_scan_key: Optional[str] = None
        self.active_scan_ls: Optional[List[Job]] = None
        self.active_scan_idx: int = 0
        self.active_failed_gpu_num: int = 16

        # Step accounting delta
        self.useful_spot_seconds_delta: float = 0.0
        self.spot_queue_gpu_seconds_delta: float = 0.0
        self.solver_calls_delta: int = 0
        self.events_delta: List[SimulationEvent] = []
        self.last_decision_time: int = self.time
        self.delta_event_cursor: int = 0
        self.eligibility_epoch: int = 0
        self.deferred_epoch: Dict[int, int] = {}
        self.quota_ready: bool = False
        self.progress_hook = None
        self.decisions_count: int = 0
        self.quota_update_count: int = 0
        self.forecast_training_count: int = 0
        self._is_terminated: bool = False
        self._is_truncated: bool = False
        self._terminal_reason: str = ""
        self.forecast_cache: Optional[Dict[int, Any]] = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["progress_hook"] = None
        state["logger"] = None
        state["_warmup_snapshot_hook"] = None
        rng_state = {
            "random": random.getstate(),
            "numpy": np.random.get_state(),
        }
        try:
            import torch
            rng_state["torch"] = torch.get_rng_state()
        except ImportError:
            pass
        state["_rng_state"] = rng_state
        return state

    def __setstate__(self, state):
        rng = state.pop("_rng_state", None)
        self.__dict__.update(state)
        if not hasattr(self, "scheduler_profile"):
            self.scheduler_profile = "native_gfs"
        if not hasattr(self, "_best_fit_placer"):
            self._best_fit_placer = {
                vc_name: ConsolidatePlacement(self.cluster.vc_dict[vc_name])
                for vc_name in self.cluster.vc_dict
            }
        if not hasattr(self, "_spot_retained_service_seconds"):
            self._spot_retained_service_seconds = {}
        if not hasattr(self, "_spot_preemption_overhead_seconds"):
            self._spot_preemption_overhead_seconds = {}
        if not hasattr(self, "_spot_service_event_cursor"):
            self._spot_service_event_cursor = 0
        if not hasattr(self, "_spot_credited_gpu_seconds"):
            self._spot_credited_gpu_seconds = {}
            for event in getattr(self.ledger, "events", []):
                if event.event_type != "credit":
                    continue
                credited = float(event.details.get("credited_gpu_seconds", 0.0))
                self._spot_credited_gpu_seconds[event.job_id] = (
                    self._spot_credited_gpu_seconds.get(event.job_id, 0.0) + credited
                )
        self.logger = logging.getLogger(f"research_sim.{self.episode_id}")
        self.logger.setLevel(logging.WARNING)
        if rng is not None:
            if "random" in rng:
                random.setstate(rng["random"])
            if "numpy" in rng:
                np.random.set_state(rng["numpy"])
            if "torch" in rng:
                try:
                    import torch
                    torch.set_rng_state(rng["torch"])
                except ImportError:
                    pass

    def set_scheduler_profile(self, profile: str) -> None:
        """Select one capability-declared scheduler without changing trace or metrics."""
        allowed = (
            {"native_gfs"}
            | SOURCE_COMPATIBLE_PROFILES
            | MECHANISM_PROBE_PROFILES
            | SRGFS_COST_PROFILES
        )
        profile = str(profile)
        if profile not in allowed:
            raise InvalidConfigError(
                f"Unsupported scheduler profile {profile!r}; expected one of {sorted(allowed)}."
            )
        self.scheduler_profile = profile
        self._best_fit_placer = {
            vc_name: ConsolidatePlacement(self.cluster.vc_dict[vc_name])
            for vc_name in self.cluster.vc_dict
        }

    def _hp_queue_key(self, job: Job) -> Tuple[Any, ...]:
        if self.scheduler_profile in {
            "yarn_cs_source_compatible",
            "fgd_source_compatible",
            "capacity_open_fcfs_best_fit_probe",
            "whole_card_fgd_probe",
        }:
            return (float(job["submit_time"]), int(job["job_name"]))
        if self.scheduler_profile in CHRONUS_PROFILES:
            return (
                -float(job["gpu_request"]),
                int(job["worker_num"]),
                float(job["submit_time"]),
                int(job["job_name"]),
            )
        if self.scheduler_profile in LYRA_COST_PROFILES:
            return (-float(job["gpu_request"]),)
        return (
            -float(job["gpu_request"]),
            float(job["submit_time"]),
            int(job["job_name"]),
        )

    @staticmethod
    def _fgd_frag_value(free_cards: int) -> float:
        typical = {1: 0.35, 2: 0.10, 4: 0.05, 8: 0.50}
        free_cards = int(free_cards)
        return float(sum(
            weight * (free_cards - (free_cards // pod_size))
            for pod_size, weight in typical.items()
        ))

    def _fgd_place(self, job: Job) -> bool:
        """FGD placement for source-compatible and whole-card probe profiles."""
        request = float(job["gpu_request"])
        scheduler_profile = getattr(self, "scheduler_profile", "fgd_source_compatible")
        vc = self.cluster.vc_dict[job["vc_name"]]
        while len(job["nodes"]) < int(job["worker_num"]):
            if request < 1.0 and scheduler_profile == "fgd_source_compatible":
                share_choices = []
                for node in vc.node_list:
                    for card in range(node.num_gpus):
                        allocated = float(node.node_galloc[card])
                        if 0.0 < allocated < 1.0 - request:
                            share_choices.append((1.0 - allocated, node.node_name, card, node))
                if share_choices:
                    _, _, card, node = min(share_choices, key=lambda row: row[:3])
                    if not node.allocate_share_gpu(card, job):
                        raise ResearchSimError("FGD fractional placement failed.")
                    job["nodes"].append({node.node_name: [card]})
                    continue

            if request < 1.0:
                return super().job_placer(job)
            cards = int(request)
            choices = []
            for node in vc.node_list:
                free = int(node.free_gpus)
                if free < cards:
                    continue
                delta = self._fgd_frag_value(free - cards) - self._fgd_frag_value(free)
                choices.append((delta, node.node_name, node))
            if not choices:
                return False
            node = min(choices, key=lambda row: row[:2])[2]
            allocated = node.allocate_gpu(cards, job)
            job["nodes"].append({node.node_name: allocated})
        return True

    def job_placer(self, job: Job) -> bool:
        if self.scheduler_profile in FGD_PROFILES:
            return self._fgd_place(job)
        if self.scheduler_profile in {
            "yarn_cs_source_compatible",
            "chronus_source_compatible",
            "lyra_source_compatible",
            "capacity_open_fcfs_best_fit_probe",
            "staggered_lease_best_fit_probe",
            "whole_card_cost_best_fit_probe",
        }:
            return self._best_fit_placer[job["vc_name"]].place(job)
        return super().job_placer(job)

    def _accounted_gpu_demand(self, job: Job) -> float:
        """Return running demand in the declared quota accounting units."""
        demand = float(job["gpu_request"])
        if self.spec.gpu_accounting_mode == "total_demand":
            demand *= int(job["worker_num"])
        return demand

    def run_guar(self, job: Job) -> None:
        """Start HP work while keeping demand counters unit consistent."""
        super().run_guar(job)
        if self.scheduler_profile in CHRONUS_PROFILES and self.in_score_window(self.time):
            job["chronus_overhead_remaining_seconds"] = int(
                job.pop("chronus_resume_overhead_seconds", 0)
            )
            if self.scheduler_profile == "staggered_lease_best_fit_probe":
                job["lease_end"] = int(self.time) + 1200
        extra = self._accounted_gpu_demand(job) - float(job["gpu_request"])
        if extra:
            vc_name = job["vc_name"]
            self.gpu_request_dict[vc_name][job["type"]] += extra
            self.guar_request_dict[job["org_key"]] += extra

    def run_spot(self, job: Job) -> None:
        """Start Spot work while keeping quota usage in declared units."""
        super().run_spot(job)
        if self.scheduler_profile in CHRONUS_PROFILES and self.in_score_window(self.time):
            # Lease length and application checkpoint interval are independent.
            # Preserve checkpoint progress across lossless lease turnover while
            # retaining the workload's declared one-hour checkpoint interval.
            checkpoint_progress = int(job.get(
                "chronus_checkpoint_progress_seconds",
                max(0, int(self.time) - int(job.get_ckpt_time())),
            ))
            job["chronus_checkpoint_progress_seconds"] = checkpoint_progress
            job["chronus_overhead_remaining_seconds"] = int(
                job.pop("chronus_resume_overhead_seconds", 0)
            )
            if self.scheduler_profile == "staggered_lease_best_fit_probe":
                job["lease_end"] = int(self.time) + 300
        extra = self._accounted_gpu_demand(job) - float(job["gpu_request"])
        if extra:
            self.gpu_request_dict[job["vc_name"]][job["type"]] += extra

    def preempt(self, job: Job, preempt_jobs: List[Job]) -> None:
        """Preempt Spot work and remove the same demand units added at start."""
        running = [preempt_job for preempt_job in preempt_jobs if preempt_job["status"] == "run"]
        super().preempt(job, preempt_jobs)
        for preempt_job in running:
            extra = self._accounted_gpu_demand(preempt_job) - float(preempt_job["gpu_request"])
            if extra:
                self.gpu_request_dict[job["vc_name"]][preempt_job["type"]] -= extra

    @staticmethod
    def _jobs_on_cards(node: Any, cards: List[int]) -> List[Job]:
        victims: Dict[int, Job] = {}
        for card in cards:
            for candidate in node.node_gpu_dict[card]:
                if candidate["type"] == "Spot":
                    victims[int(candidate["job_name"])] = candidate
        return list(victims.values())

    def _rollback_loss_gpu_seconds(self, job: Job) -> float:
        """Return the same rollback loss quantity recorded by the event ledger."""

        if job["status"] != "run":
            return 0.0
        rollback_seconds = (
            int(self.time) - int(job["start_time"])
        ) % int(self.args.ckpt_interval)
        return float(
            rollback_seconds
            * float(job["worker_num"])
            * float(job["gpu_request"])
        )

    def _minimum_loss_card_set(
        self,
        node: Any,
        available: List[int],
        cards_needed: int,
        existing_victims: Optional[Mapping[int, Job]] = None,
    ) -> Tuple[List[int], Dict[int, Job], Tuple[Any, ...]]:
        """Find the exact minimum loss card combination on one eight GPU node."""

        prior = existing_victims or {}
        choices = []
        for combination in itertools.combinations(sorted(available), int(cards_needed)):
            new_victims = {
                int(spot["job_name"]): spot
                for card in combination
                for spot in node.node_gpu_dict[card]
                if spot["type"] == "Spot" and int(spot["job_name"]) not in prior
            }
            lost = sum(
                self._rollback_loss_gpu_seconds(spot)
                for spot in new_victims.values()
            )
            key = (
                float(lost),
                len(new_victims),
                len(available) - int(cards_needed),
                tuple(combination),
            )
            choices.append((key, list(combination), new_victims))
        if not choices:
            raise ResearchSimError("No feasible minimum loss card combination.")
        key, selected, victims = min(choices, key=lambda row: row[0])
        return selected, victims, key

    def _profile_preemption_plan(
        self,
        job: Job,
    ) -> Tuple[bool, Optional[Any], Optional[List[int]], List[Job]]:
        """Return one-worker placement plan for released GFS comparator logic."""
        cards_needed = int(max(float(job["gpu_request"]), 1.0))
        vc = self.cluster.vc_dict[job["vc_name"]]
        candidates = []
        for node in vc.node_list:
            available = [
                card for card in range(node.num_gpus)
                if node.node_type_galloc[card]["HP"] == 0.0
            ]
            if len(available) < cards_needed:
                continue
            if self.scheduler_profile in LYRA_COST_PROFILES:
                def lyra_card_cost(card: int) -> float:
                    return float(sum(
                        1.0 / max(1.0, float(spot["worker_num"]) * float(spot["gpu_request"]))
                        for spot in node.node_gpu_dict[card]
                        if spot["type"] == "Spot"
                    ))
                selected = sorted(
                    available,
                    key=lambda card: (-lyra_card_cost(card), card),
                )[:cards_needed]
                score = sum(lyra_card_cost(card) for card in selected)
                key = (len(available), score, node.node_name)
            elif self.scheduler_profile in SRGFS_COST_PROFILES:
                selected, victims, loss_key = self._minimum_loss_card_set(
                    node, available, cards_needed
                )
                key = (*loss_key[:-1], node.node_name, loss_key[-1])
                candidates.append((key, node, selected, victims))
                continue
            else:
                selected = available[:cards_needed]
                if self.scheduler_profile in FGD_PROFILES:
                    delta = self._fgd_frag_value(len(available) - cards_needed) - self._fgd_frag_value(len(available))
                    key = (delta, node.node_name)
                else:
                    key = (len(available), node.node_name)
            candidates.append((key, node, selected, None))
        if not candidates:
            return False, None, None, []
        _, node, selected, planned_victims = min(candidates, key=lambda row: row[0])
        victims = (
            list(planned_victims.values())
            if planned_victims is not None
            else self._jobs_on_cards(node, selected)
        )
        return True, node, selected, victims

    def place_with_preemption(self, job: Job) -> Tuple[bool, Optional[List[Job]]]:
        if self.scheduler_profile == "native_gfs":
            return super().place_with_preemption(job)
        if self.scheduler_profile in CHRONUS_PROFILES:
            return False, None
        if float(job["gpu_request"]) < 1.0:
            if self.scheduler_profile in MECHANISM_PROBE_PROFILES:
                return super().place_with_preemption(job)
            vc = self.cluster.vc_dict[job["vc_name"]]
            request = float(job["gpu_request"])
            choices = []
            for node in vc.node_list:
                for card in range(node.num_gpus):
                    hp_alloc = float(node.node_type_galloc[card]["HP"])
                    if not 0.0 < hp_alloc < 1.0 - request:
                        continue
                    victims = []
                    available = 1.0 - float(node.node_galloc[card])
                    cost = 0.0
                    for candidate in node.node_gpu_dict[card]:
                        if candidate["type"] != "Spot":
                            continue
                        if candidate not in victims:
                            victims.append(candidate)
                            available += float(candidate["gpu_request"])
                            if self.scheduler_profile in LYRA_COST_PROFILES:
                                cost += 1.0 / max(1, int(candidate["worker_num"]))
                            elif self.scheduler_profile in SRGFS_COST_PROFILES:
                                cost += self._rollback_loss_gpu_seconds(candidate)
                        if available >= request:
                            break
                    if available >= request:
                        source_cost = (
                            cost
                            if self.scheduler_profile
                            in (LYRA_COST_PROFILES | SRGFS_COST_PROFILES)
                            else 0.0
                        )
                        choices.append((1.0 - hp_alloc, source_cost, node.node_name, card, node, victims))
            if choices:
                _, _, _, card, node, victims = min(choices, key=lambda row: row[:4])
                for victim in victims:
                    vc.release_resource(victim)
                if not node.allocate_share_gpu(card, job):
                    raise ResearchSimError("Source-compatible fractional preemption allocation failed.")
                job["nodes"].append({node.node_name: [card]})
                return True, victims
        ok, node, cards, victims = self._profile_preemption_plan(job)
        if not ok or node is None or cards is None:
            return False, None
        vc = self.cluster.vc_dict[job["vc_name"]]
        for victim in victims:
            vc.release_resource(victim)
        if not node.allocate_gpus_with_preemption(cards, job):
            raise ResearchSimError("Adapted comparator preemption allocation failed.")
        job["nodes"].append({node.node_name: cards})
        return True, victims

    def _expire_chronus_lease(self, job: Job, vc_name: str, vc: Any) -> bool:
        """Release unfinished work at a lease boundary with source-model overhead."""
        if self.scheduler_profile not in CHRONUS_PROFILES or not self.in_score_window(self.time):
            return False
        if self.scheduler_profile == "staggered_lease_best_fit_probe":
            lease_end = job.get("lease_end")
            if lease_end is None:
                lease_seconds = 1200 if job["type"] == "HP" else 300
                job["lease_end"] = int(self.time) + lease_seconds
                return False
            if int(self.time) < int(lease_end) or int(job["remain"]) == 0:
                return False
            lease_seconds = 1200 if job["type"] == "HP" else 300
        else:
            if int(self.time) % 1200 != 0 or int(job["remain"]) == 0:
                return False
            lease_seconds = 1200
        overhead_seconds = 120
        remain_before = float(job["remain"])
        checkpoint_progress = None
        if job["type"] == "Spot":
            checkpoint_progress = int(job.get("chronus_checkpoint_progress_seconds", 0))
            if not 0 <= checkpoint_progress < int(job["spot_duration"]):
                raise ResearchSimError(
                    "Chronus checkpoint progress must remain below the checkpoint interval."
                )
            job["chronus_checkpoint_progress_seconds"] = checkpoint_progress
            # SpotScheduler.run_spot accumulates queue time only after a restart.
            job["preempt_times"] = int(job.get("preempt_times", 0)) + 1
        job["chronus_resume_overhead_seconds"] = overhead_seconds
        if not vc.release_resource(job):
            raise ResearchSimError("Chronus lease release failed.")
        self.run_list[vc_name].remove(job)
        self.gpu_request_dict[vc_name][job["type"]] -= self._accounted_gpu_demand(job)
        if job["type"] == "HP":
            self.guar_request_dict[job["org_key"]] -= self._accounted_gpu_demand(job)
        job["nodes"] = []
        job.pop("lease_end", None)
        self.pend_job(job)
        self.ledger.record_event(
            self.time,
            "lease_expiration",
            job_id=job["job_name"],
            vc_name=vc_name,
            attempt_index=(
                self._spot_attempt_indices.get(job["job_name"], 0)
                if job["type"] == "Spot" else None
            ),
            details={
                "job_type": job["type"],
                "lease_seconds": lease_seconds,
                "rollback_seconds": 0,
                "overhead_seconds": overhead_seconds,
                "remain_before": remain_before,
                "remain_after": float(job["remain"]),
                "checkpoint_progress_seconds": checkpoint_progress,
                "attempt_index": (
                    self._spot_attempt_indices.get(job["job_name"], 0)
                    if job["type"] == "Spot" else None
                ),
                "start_time": int(job["start_time"]),
            },
        )
        self.eligibility_epoch += 1
        return True

    def _record_spot_credit_checked(
        self,
        job: Job,
        vc_name: str,
        gpu_seconds: float,
        reason: str,
        credited_run_seconds: float,
    ) -> float:
        """Record useful service without exceeding the job's physical demand."""
        jid = job["job_name"]
        gpu_equivalents = float(job["gpu_request"] * job["worker_num"])
        cap = float(job["duration"]) * gpu_equivalents
        prior = float(self._spot_credited_gpu_seconds.get(jid, 0.0))
        accepted = min(max(0.0, float(gpu_seconds)), max(0.0, cap - prior))
        if accepted <= 0.0:
            return 0.0
        accepted_run_seconds = accepted / gpu_equivalents
        att_idx = self._spot_attempt_indices.get(jid, 0)
        c_seq = self._spot_credit_sequences.get(jid, 0)
        cumulative = prior + accepted
        scored = self.ledger.record_spot_credit(
            self.time,
            jid,
            accepted,
            reason=reason,
            attempt_index=att_idx,
            credit_sequence=c_seq,
            credited_run_seconds=accepted_run_seconds,
            gpu_equivalents=gpu_equivalents,
            vc_name=vc_name,
            job_credit_cap_gpu_seconds=cap,
            cumulative_credited_gpu_seconds=cumulative,
        )
        self._spot_credited_gpu_seconds[jid] = cumulative
        self._spot_credit_sequences[jid] = c_seq + 1
        return scored

    def _complete_running_job(
        self,
        job: Job,
        vc_name: str,
        vc: Any,
        spot_remainder_seconds: Optional[float] = None,
    ) -> None:
        """Complete one running job and emit physically bounded useful service."""
        job["status"] = "end"
        job["end_time"] = self.time
        self.end_job_num += 1
        assert vc.release_resource(job)
        self.run_list[vc_name].remove(job)
        accounted_demand = self._accounted_gpu_demand(job)
        self.gpu_request_dict[vc_name][job["type"]] -= accounted_demand
        if job["type"] == "Spot":
            if spot_remainder_seconds is None:
                spot_remainder_seconds = min(
                    float(job["duration"] % self.args.ckpt_interval),
                    max(0.0, float(self.time - job.get_ckpt_time() - 1)),
                )
            rem_credit = float(
                spot_remainder_seconds * job["gpu_request"] * job["worker_num"]
            )
            self.useful_spot_seconds_delta += self._record_spot_credit_checked(
                job,
                vc_name,
                rem_credit,
                reason="completion",
                credited_run_seconds=float(spot_remainder_seconds),
            )
            self.record_spot_succeed(job)
            self.ledger.completed_jobs["Spot"] += 1
        else:
            self.guar_request_dict[job["org_key"]] -= accounted_demand
            self.ledger.completed_jobs["HP"] += 1
        self.ledger.record_event(
            self.time,
            "completion",
            job_id=job["job_name"],
            vc_name=vc_name,
            details={
                "job_type": job["type"],
                "completion_time": self.time,
                "duration": job["duration"],
            },
        )
        self.eligibility_epoch += 1

    def in_score_window(self, t: int) -> bool:
        """Determines if t falls inside the declared scoring interval."""
        if self.spec.trace_mode == "source_compatible":
            return self.spec.score_start < t <= self.spec.score_end
        return self.spec.score_start <= t < self.spec.score_end

    def in_guar_window(self, t: int) -> bool:
        """Determines if guar_gpu_time updates at t matching upstream log_range."""
        if self.spec.trace_mode == "source_compatible":
            return self.spec.score_start <= t <= self.spec.score_end
        return self.spec.score_start <= t < self.spec.score_end

    def in_quota_window(self, t: int) -> bool:
        """Determines if quota enforcement is active at t."""
        if self.spec.trace_mode == "source_compatible":
            return self.spec.score_start <= t <= self.spec.score_end
        return self.spec.score_start <= t < self.spec.score_end

    def update_spot_quota_cached(self, t: int, now_timestamp: pd.Timestamp) -> None:
        """Applies cached neural forecast outputs while dynamically evaluating eviction rate."""
        pred_mu_dict, pred_sigma_dict = self.forecast_cache[t]
        for vc_name in pred_mu_dict:
            vc = self.cluster.vc_dict[vc_name]
            gpu_capacity = vc.total_gpus
            vc_mu, vc_sigma = pred_mu_dict[vc_name], np.sqrt(pred_sigma_dict[vc_name])
            usage = norm.ppf(self.args.guarantee_rate, loc=vc_mu, scale=vc_sigma)
            free = gpu_capacity - np.clip(np.floor(usage), 0, gpu_capacity)

            # Eviction rate adjustment
            succeed, faied = vc.get_eviction_rate(self.time, self.args.pred_len)
            if succeed + faied > 0:
                eviction_rate = float(faied) / float(succeed + faied)
                if eviction_rate >= 1.5 * (1.0 - self.args.guarantee_rate):
                    self.spot_quota[vc_name]["eta"] = (
                        self.spot_quota[vc_name]["eta"] * (1.0 - self.args.guarantee_rate) / eviction_rate
                    )
                if (
                    eviction_rate <= 0.5 * (1.0 - self.args.guarantee_rate)
                    and len(self.spot_que_list[vc_name]) > 0
                    and self.time - self.spot_que_list[vc_name][0].get_preempt_time() > 3600
                ):
                    self.spot_quota[vc_name]["eta"] = (
                        self.spot_quota[vc_name]["eta"] * (1.5 - (1.0 - self.args.guarantee_rate) / eviction_rate)
                    )

            # Update spot quota
            new_spot_quota = {}
            for h in self.args.guarantee_hour:
                new_spot_quota[h] = np.min(free[:h]) * self.spot_quota[vc_name]["eta"]
                self.spot_quota_record.append(
                    [now_timestamp, vc_name, h, np.min(free[:h]), self.spot_quota[vc_name]["eta"]]
                )
            self.spot_quota[vc_name]["quota"] = new_spot_quota

    def estimate_job_duration(
        self,
        worker_num: int,
        gpu_request: int,
        is_spot: bool = False,
        gpu_model: Optional[str] = None
    ) -> int:
        """Estimates job duration strictly from past fitted estimator."""
        if self.duration_estimator is not None:
            if hasattr(self.duration_estimator, "estimate_duration"):
                return int(self.duration_estimator.estimate_duration(
                    worker_num, gpu_request, is_spot=is_spot, gpu_model=gpu_model or "A100-SXM4-80GB"
                ))
            elif is_spot and hasattr(self.duration_estimator, "estimate_spot_duration"):
                return int(self.duration_estimator.estimate_spot_duration(
                    worker_num, gpu_request, gpu_model=gpu_model or "A100-SXM4-80GB"
                ))
            elif not is_spot and hasattr(self.duration_estimator, "estimate_hp_duration"):
                return int(self.duration_estimator.estimate_hp_duration(
                    worker_num, gpu_request, gpu_model=gpu_model or "A100-SXM4-80GB"
                ))
            elif callable(self.duration_estimator):
                return int(self.duration_estimator(worker_num, gpu_request))
        return 0

    def observe(self) -> Dict[str, np.ndarray]:
        """Returns causal observation tensors and legal action mask."""
        if self.paused_context is None:
            raise ResearchSimError("Cannot call observe() when not paused at a decision.")
        first_vc = self.cluster.vc_list[0]
        vc = self.cluster.vc_dict[first_vc]
        verifier_ready = getattr(self.verifier, "is_ready", True)
        return build_observation(
            context=self.paused_context,
            cluster_view=vc,
            N=self.view.total_nodes,
            M=self.spec.M,
            K=self.spec.K,
            verifier_ready=verifier_ready,
            filter_mode=self.filter_mode
        )

    def _admission_views(self, vc_name: str, jobs: List[Job]) -> Dict[int, Any]:
        """Build sanitized queue views for a controller without future labels."""
        vc = self.cluster.vc_dict[vc_name]
        total_capacity = int(vc.total_gpus)
        max_gpu_per_worker = max(int(node.num_gpus) for node in vc.node_list)
        max_guarantee = max((int(v) for v in self.args.guarantee_hour), default=1)
        views: Dict[int, Any] = {}
        for job in jobs:
            job_id = int(job["job_name"])
            estimated_duration = self.estimate_job_duration(
                int(job["worker_num"]),
                int(job["gpu_request"]),
                is_spot=True,
                gpu_model=job.get("gpu_model", vc_name),
            )
            views[job_id] = build_admission_job_view(
                job,
                time=self.time,
                total_nodes=len(vc.node_list),
                total_capacity=total_capacity,
                max_gpu_per_worker=max_gpu_per_worker,
                max_guarantee_hours=max_guarantee,
                estimated_duration_seconds=estimated_duration,
            )
        return views

    def _order_spot_queue(self, vc_name: str) -> List[Job]:
        """Apply v4 admission weights, with exact native ordering at zero weights."""
        queue = self.spot_que_list[vc_name]
        if self.scheduler_profile in {
            "yarn_cs_source_compatible",
            "capacity_open_fcfs_best_fit_probe",
        }:
            return queue.copy()
        if self.scheduler_profile in FGD_PROFILES:
            queue.sort(key=lambda job: (float(job["submit_time"]), int(job["job_name"])))
            return queue.copy()
        if self.scheduler_profile in CHRONUS_PROFILES:
            queue.sort(key=lambda job: (
                -float(job["gpu_request"]),
                int(job["worker_num"]),
                float(job["submit_time"]),
                int(job["job_name"]),
            ))
            return queue.copy()
        if self.scheduler_profile in LYRA_COST_PROFILES:
            queue.sort(key=lambda job: -float(job["gpu_request"]))
            return queue.copy()
        if self.policy_type != "unified_control":
            queue.sort(key=native_admission_key)
            return queue.copy()
        if not any(abs(weight) > 1e-12 for weight in self.admission_ranker.weights):
            queue.sort(key=native_admission_key)
            return queue.copy()
        views = self._admission_views(vc_name, queue)
        ordered_ids = self.admission_ranker.order(queue, views)
        if len(ordered_ids) != len(queue) or len(set(ordered_ids)) != len(queue):
            raise ResearchSimError("Admission ranker must return one unique ID per queued job.")
        jobs_by_id = {int(job["job_name"]): job for job in queue}
        if set(ordered_ids) != set(jobs_by_id):
            raise ResearchSimError("Admission ranker output is not a queue permutation.")
        native_ids = tuple(int(job["job_name"]) for job in sorted(queue, key=native_admission_key))
        self.admission_queue_scans = int(getattr(self, "admission_queue_scans", 0)) + 1
        if ordered_ids != native_ids:
            self.admission_reordered_scans = int(
                getattr(self, "admission_reordered_scans", 0)
            ) + 1
        duration_values = {
            int(view.estimated_duration_seconds) for view in views.values()
        }
        if len(duration_values) > 1:
            self.admission_duration_nonconstant_scans = int(
                getattr(self, "admission_duration_nonconstant_scans", 0)
            ) + 1
        queue[:] = [jobs_by_id[job_id] for job_id in ordered_ids]
        return queue.copy()

    def _build_control_context(self) -> ControlDecisionContext:
        """Capture the causal state after the base quota update and before scheduling."""
        capacity_by_pool = {
            vc_name: int(self.cluster.vc_dict[vc_name].total_gpus)
            for vc_name in self.cluster.vc_list
        }
        hp_used_by_pool = {
            vc_name: float(sum(
                job["gpu_request"] * job["worker_num"]
                for job in self.run_list[vc_name]
                if job["type"] == "HP"
            ))
            for vc_name in self.cluster.vc_list
        }
        spot_used_by_pool = {
            vc_name: float(self.gpu_request_dict[vc_name]["Spot"])
            for vc_name in self.cluster.vc_list
        }
        base_quota_by_pool = {
            vc_name: {
                int(hour): int(np.clip(np.rint(value), 0, capacity_by_pool[vc_name]))
                for hour, value in self.spot_quota[vc_name]["quota"].items()
            }
            for vc_name in self.cluster.vc_list
        }
        admission_jobs = []
        for vc_name in self.cluster.vc_list:
            native_jobs = sorted(self.spot_que_list[vc_name], key=native_admission_key)
            views = self._admission_views(vc_name, native_jobs)
            admission_jobs.extend(views[int(job["job_name"])] for job in native_jobs)
        hp_jobs = [job for jobs in self.guar_que_list.values() for job in jobs]
        hp_queue_gpu_demand = float(sum(
            job["gpu_request"] * job["worker_num"] for job in hp_jobs
        ))
        eviction_failure_rate_by_pool: Dict[str, float] = {}
        for vc_name in self.cluster.vc_list:
            succeeded, failed = self.cluster.vc_dict[vc_name].get_eviction_rate(
                self.time, self.args.pred_len
            )
            total = int(succeeded) + int(failed)
            eviction_failure_rate_by_pool[vc_name] = (
                float(failed) / total if total > 0 else 0.0
            )
        denominator = max(1, self.spec.score_end - self.spec.score_start)
        progress = (self.time - self.spec.score_start) / denominator
        return ControlDecisionContext(
            state_version=self.state_version,
            time=self.time,
            base_quota_by_pool=base_quota_by_pool,
            capacity_by_pool=capacity_by_pool,
            hp_used_by_pool=hp_used_by_pool,
            spot_used_by_pool=spot_used_by_pool,
            hp_queue_count=len(hp_jobs),
            hp_queue_gpu_demand=hp_queue_gpu_demand,
            admission_jobs=tuple(admission_jobs),
            eviction_failure_rate_by_pool=eviction_failure_rate_by_pool,
            score_progress=float(np.clip(progress, 0.0, 1.0)),
        )

    def _consume_control_delta(self) -> TransitionDelta:
        """Consume transition accounting at a v4 control boundary."""
        delta_events = tuple(self.ledger.events[self.delta_event_cursor:])
        reward_val = self.useful_spot_seconds_delta / 3600.0
        deadlines = sum(1 for event in delta_events if event.event_type == "hp_deadline")
        evictions = sum(1 for event in delta_events if event.event_type == "preemption")
        lost_seconds = sum(
            float(event.details.get("lost_gpu_seconds", 0.0))
            for event in delta_events if event.event_type == "preemption"
        )
        queue_gpu_hours = self.spot_queue_gpu_seconds_delta / 3600.0
        lost_gpu_hours = lost_seconds / 3600.0
        control_reward = reward_val - queue_gpu_hours - lost_gpu_hours
        if not self._first_decision_returned:
            self._first_decision_returned = True
            self.prefix_reward_gpu_hours = reward_val
            self.prefix_solver_calls = self.solver_calls_delta
            self.prefix_deadline_events = deadlines
            self.prefix_evictions = evictions
            self.prefix_lost_gpu_seconds = lost_seconds
            initial_reward = reward_val
            initial_calls = self.solver_calls_delta
            initial_deadlines = deadlines
            initial_evictions = evictions
            initial_lost_seconds = lost_seconds
        else:
            self.step_reward_gpu_hours_sum += reward_val
            initial_reward = 0.0
            initial_calls = 0
            initial_deadlines = 0
            initial_evictions = 0
            initial_lost_seconds = 0.0
        delta = TransitionDelta(
            reward=reward_val,
            elapsed_sim_seconds=self.time - self.last_decision_time,
            solver_calls=self.solver_calls_delta,
            events=delta_events,
            terminated=False,
            truncated=False,
            initial_reward_gpu_hours=initial_reward,
            initial_solver_calls=initial_calls,
            initial_deadline_events=initial_deadlines,
            initial_evictions=initial_evictions,
            initial_lost_gpu_seconds=initial_lost_seconds,
            spot_queue_gpu_hours=queue_gpu_hours,
            lost_gpu_hours=lost_gpu_hours,
            control_reward=control_reward,
        )
        self.useful_spot_seconds_delta = 0.0
        self.spot_queue_gpu_seconds_delta = 0.0
        self.solver_calls_delta = 0
        self.events_delta = []
        self.delta_event_cursor = len(self.ledger.events)
        self.last_decision_time = self.time
        return delta

    def apply_control(self, action: Any) -> Tuple[SimulationEvent, ...]:
        """Apply a bounded quota residual and admission weights for one interval."""
        if self.control_paused_context is None or self.phase != Phase.CONTROL:
            raise ResearchSimError("Simulator is not paused at a v4 control decision.")
        context = self.control_paused_context
        normalized = action if isinstance(action, UnifiedControlAction) else UnifiedControlAction.from_array(action)
        adjusted = apply_quota_delta(
            context,
            normalized,
            max_delta_fraction=self.spec.max_quota_delta_fraction,
        )
        if self.scheduler_profile in CAPACITY_QUOTA_PROFILES:
            adjusted = {
                pool: {int(hour): int(context.capacity_by_pool[pool]) for hour in quota}
                for pool, quota in adjusted.items()
            }
        for vc_name, quota in adjusted.items():
            self.spot_quota[vc_name]["quota"] = dict(quota)
        self.admission_ranker.set_weights(normalized.priority_weights)
        event = self.ledger.record_event(
            self.time,
            "control_action",
            details={
                "control_schema_version": "4.0",
                "state_version": self.state_version,
                "base_quota": context.base_quota_by_pool,
                "executed_quota": adjusted,
                "quota_delta_fraction": normalized.quota_delta_fraction,
                "max_quota_delta_fraction": self.spec.max_quota_delta_fraction,
                "priority_weights": list(normalized.priority_weights),
                "admission_ranker": self.admission_ranker.ranker_id,
            },
        )
        self.state_version += 1
        self.control_paused_context = None
        self.phase = Phase.RUNNING
        return (event,)

    def snapshot(self) -> ClusterStateSnapshot:
        """Alias for snapshot_for_verifier."""
        return self.snapshot_for_verifier()

    def _sync_spot_service_history(self) -> None:
        """Accumulate observed retained work from new preemption ledger events."""
        events = self.ledger.events
        if self._spot_service_event_cursor > len(events):
            self._spot_retained_service_seconds.clear()
            self._spot_preemption_overhead_seconds.clear()
            self._spot_service_event_cursor = 0
        for ev in events[self._spot_service_event_cursor:]:
            if getattr(ev, "event_type", None) != "preemption" or ev.job_id is None:
                continue
            details = ev.details
            start_time = int(details.get("start_time", ev.time))
            rollback = int(round(float(details.get("rollback_runtime_seconds", 0.0))))
            retained = max(0, int(ev.time) - start_time - rollback)
            job_id = ev.job_id
            self._spot_retained_service_seconds[job_id] = (
                self._spot_retained_service_seconds.get(job_id, 0) + retained
            )
            overhead = int(round(float(details.get("overhead", 0.0))))
            self._spot_preemption_overhead_seconds[job_id] = (
                self._spot_preemption_overhead_seconds.get(job_id, 0) + overhead
            )
        self._spot_service_event_cursor = len(events)

    def _estimate_running_job_residual(self, job: Job, gpu_model: str, is_spot: bool) -> Optional[int]:
        """Estimate remaining work from observed service age, never hidden remain."""
        estimator = self.duration_estimator
        if estimator is None or not hasattr(estimator, "estimate_remaining"):
            return None
        start_time = int(job.get("start_time", self.time))
        current_attempt_age = max(0, self.time - start_time)
        additional_work = 0
        service_age = current_attempt_age
        if is_spot:
            self._sync_spot_service_history()
            job_id = job["job_name"]
            service_age += self._spot_retained_service_seconds.get(job_id, 0)
            additional_work = self._spot_preemption_overhead_seconds.get(job_id, 0)
        return estimator.estimate_remaining(
            worker_num=int(job.get("worker_num", 1)),
            gpu_request=int(float(job.get("gpu_request", 1))),
            service_age_seconds=service_age,
            is_spot=is_spot,
            gpu_model=gpu_model,
            additional_work_seconds=additional_work,
        )

    def snapshot_for_verifier(self) -> ClusterStateSnapshot:
        """Returns an immutable snapshot of current cluster state for verifier."""
        nodes_snapshots: List[NodeStateSnapshot] = []
        for vc_name in self.cluster.vc_list:
            vc = self.cluster.vc_dict[vc_name]
            evict = getattr(vc, "eviction_score", None) or {}
            for node in vc.node_list:
                free_cards = tuple(node.check_free_gpu_list())
                hp_occ = tuple(c for c in range(node.num_gpus) if node.node_type_galloc[c]["HP"] > 0)
                spot_occ = tuple(c for c in range(node.num_gpus) if node.node_type_galloc[c]["Spot"] > 0)
                nodes_snapshots.append(
                    NodeStateSnapshot(
                        node_id=node.node_name,
                        gpu_model=vc_name,
                        total_cards=node.num_gpus,
                        free_cards=free_cards,
                        hp_occupied_cards=hp_occ,
                        spot_occupied_cards=spot_occ,
                        eviction_score=float(evict.get(node.node_name, 0.0))
                    )
                )

        first_vc = self.cluster.vc_list[0]
        if self.quota_ready:
            quota_val = self.spot_quota[first_vc]["quota"].get(1, self.cluster.vc_dict[first_vc].total_gpus)
        else:
            quota_val = 0.0

        # Running residuals use causal conditional survival estimates.
        self._sync_spot_service_history()
        spot_units: List[PreemptibleSpotUnit] = []
        fixed_occupancy: List[FixedNodeOccupancy] = []
        for vc_name in self.run_list:
            for job in self.run_list[vc_name]:
                if job["type"] == "Spot":
                    allocs: List[Tuple[int, int]] = []
                    card_allocs: List[Tuple[int, Tuple[int, ...]]] = []
                    for entry in job["nodes"]:
                        for node_name, card_ids in entry.items():
                            allocs.append((node_name, len(card_ids)))
                            card_allocs.append((node_name, tuple(int(card) for card in card_ids)))
                    residual = self._estimate_running_job_residual(job, vc_name, is_spot=True)
                    spot_units.append(
                        PreemptibleSpotUnit(
                            job_id=job["job_name"],
                            start_time=job.get("start_time", self.time),
                            duration_remaining=residual,
                            allocations=tuple(allocs),
                            gpu_model=vc_name,
                            checkpoint_time=job.get_ckpt_time(),
                            worker_num=int(job.get("worker_num", 1)),
                            gpu_request=float(job.get("gpu_request", 1.0)),
                            card_allocations=tuple(card_allocs),
                            spot_duration=int(job.get("spot_duration", self.args.ckpt_interval)),
                        )
                    )
                else:
                    # Nonpreemptible HP job
                    residual = self._estimate_running_job_residual(job, vc_name, is_spot=False)
                    for entry in job["nodes"]:
                        for node_name, card_ids in entry.items():
                            fixed_occupancy.append(
                                FixedNodeOccupancy(
                                    node_id=node_name,
                                    occupied_cards=len(card_ids),
                                    duration_remaining=residual,
                                    card_ids=tuple(int(card) for card in card_ids),
                                    job_id=int(job["job_name"]),
                                )
                            )

        # Queued HP whole-card gang jobs
        queued_hp_jobs: List[QueuedHPJob] = []
        for vc_name in self.guar_que_list:
            for job in self.guar_que_list[vc_name]:
                if job["gpu_request"] >= 1.0 and float(job["gpu_request"]).is_integer():
                    est_dur = self.estimate_job_duration(int(job["worker_num"]), int(job["gpu_request"]), is_spot=False, gpu_model=vc_name)
                    queued_hp_jobs.append(
                        QueuedHPJob(
                            job_id=job["job_name"],
                            submit_time=int(job["submit_time"]),
                            worker_num=int(job["worker_num"]),
                            gpu_request=int(job["gpu_request"]),
                            gpu_model=vc_name,
                            duration=est_dur
                        )
                    )

        return ClusterStateSnapshot(
            time=self.time,
            state_version=self.state_version,
            nodes=tuple(nodes_snapshots),
            quota=float(quota_val),
            budget_remaining=self.budget.remaining_calls,
            running_spot_count=sum(len(self.run_list[vc]) for vc in self.run_list),
            queued_spot_count=sum(len(self.spot_que_list[vc]) for vc in self.spot_que_list),
            queued_hp_count=sum(len(self.guar_que_list[vc]) for vc in self.guar_que_list),
            spot_units=tuple(spot_units),
            queued_hp_jobs=tuple(queued_hp_jobs),
            fixed_occupancy=tuple(fixed_occupancy),
            native_preemption_stats=tuple(
                (str(vc_name), int(self.spot_failed[vc_name]),
                 int(self.spot_succeed[vc_name]),
                 int(self.cluster.vc_dict[vc_name].total_gpus), int(self.timespan))
                for vc_name in self.cluster.vc_list
            ),
        )

    def _execute_preemption_and_record(self, hp_job: Job, preempt_jobs: List[Job]):
        """
        Executes preemption atomically and records preemption events with strict remain assertions.
        """
        if not preempt_jobs:
            return
        captured = []
        for pj in preempt_jobs:
            pjid = pj["job_name"]
            att_idx = self._spot_attempt_indices.get(pjid, 0)
            st_time = pj["start_time"]
            wn = pj["worker_num"]
            req = pj["gpu_request"]
            rem_before = pj["remain"]
            overhead = self.preempt_overhead(pj)
            rollback_sec = (self.time - st_time) % self.args.ckpt_interval
            lost_sec = float(rollback_sec * req * wn)
            captured.append({
                "job": pj,
                "job_id": pjid,
                "attempt_index": att_idx,
                "start_time": st_time,
                "worker_num": wn,
                "gpu_request": req,
                "remain_before": rem_before,
                "overhead": overhead,
                "rollback_sec": rollback_sec,
                "lost_sec": lost_sec,
                "vc_name": pj["vc_name"],
            })

        self.preempt(hp_job, preempt_jobs)

        for c in captured:
            pj = c["job"]
            rem_after = pj["remain"]
            assert rem_after - c["remain_before"] == c["rollback_sec"] + c["overhead"], (
                f"Preemption remain delta mismatch: {rem_after} - {c['remain_before']} != "
                f"{c['rollback_sec']} + {c['overhead']}"
            )
            self.ledger.record_preemption(
                time=self.time,
                job_id=c["job_id"],
                lost_seconds=c["lost_sec"],
                attempt_index=c["attempt_index"],
                start_time=c["start_time"],
                worker_num=c["worker_num"],
                gpu_request=c["gpu_request"],
                rollback_runtime_seconds=float(c["rollback_sec"]),
                remain_before=float(c["remain_before"]),
                remain_after=float(rem_after),
                overhead=float(c["overhead"]),
                preempting_hp_job_id=hp_job["job_name"],
                vc_name=c["vc_name"],
            )

    def apply(self, action: Any) -> Tuple[SimulationEvent, ...]:
        """
        Validates state version, executes atomic placement or deferral,
        and advances the Spot scan cursor.
        """
        if self.paused_context is None:
            raise ResearchSimError("Cannot call apply() when simulator is not paused at a decision.")

        context = self.paused_context
        job = self.active_scan_job
        key = self.active_scan_key
        vc = self.cluster.vc_dict[key]

        # Decode action
        supplied_version = None
        if isinstance(action, dict):
            if set(action) != {"slot", "verify", "state_version"}:
                raise InvalidActionError("Action record needs slot, verify, and state_version.")
            slot, verify_flag = action["slot"], action["verify"]
            supplied_version = action["state_version"]
        elif isinstance(action, (int, np.integer)):
            slot = int(action) // 2
            verify_flag = int(action) % 2
        elif isinstance(action, (tuple, list)) and len(action) in (2, 3):
            slot, verify_flag = action[:2]
            if len(action) == 3:
                supplied_version = action[2]
        else:
            raise InvalidActionError(f"Unsupported action type: {type(action)}")
        if not isinstance(slot, (int, np.integer)) or not isinstance(verify_flag, (int, np.integer)):
            raise InvalidActionError("Action slot and verify flag must be integers.")
        slot, verify_flag = int(slot), int(verify_flag)
        if verify_flag not in (0, 1):
            raise InvalidActionError("Verify flag must be zero or one.")
        if supplied_version is not None and supplied_version != self.state_version:
            raise StaleStateError("Action state version differs from paused simulator state.")
        if context.state_version != self.state_version or context.candidates.state_version != self.state_version:
            raise StaleStateError("Paused decision has a stale candidate set.")
        if context.job_id != job["job_name"] or job["status"] != "pend":
            raise StaleStateError("Paused job is no longer the pending decision job.")

        cands = context.candidates.candidates
        if slot < 0 or slot >= len(cands):
            raise InvalidActionError(f"Action slot {slot} out of range [0, {len(cands)-1}].")

        cand = cands[slot]
        t_apply_start_ns = self.timer_func()

        if cand.slot != slot:
            raise InvalidActionError("Candidate slot mapping is inconsistent.")
        if not cand.placement.is_deferral:
            self._validate_placement(job, vc, cand.placement)
        if verify_flag == 1:
            if not self.budget.can_verify(cand.placement.is_deferral, getattr(self.verifier, "is_ready", True)):
                raise InvalidActionError("Verification requested but budget insufficient, placement is deferral, or verifier not ready.")
        executed_placement = cand.placement
        actual_calls = 0
        result = None

        # Route through ActionFilter
        obs = self.observe()
        snapshot = self.snapshot_for_verifier()

        exec_slot, actual_calls, filter_details = self.action_filter.filter_decision(
            proposed_slot=slot,
            verify_flag=verify_flag,
            context=context,
            observation=obs,
            snapshot=snapshot,
            budget_manager=self.budget,
            K=self.spec.K,
            decision_id=self.decisions_count - 1
        )

        executed_placement = cands[exec_slot].placement if exec_slot < len(cands) else Placement.make_deferral()

        if actual_calls > 0:
            self.ledger.record_solver_calls(self.time, actual_calls, "scenario_check")
            self.solver_calls_delta += actual_calls

        if not executed_placement.is_deferral:
            self._validate_placement(job, vc, executed_placement)

        action_details = {
            "decision_id": self.decisions_count - 1,
            "time": self.time,
            "state_version": self.state_version,
            "candidate_keys": [str(c.placement.key) for c in cands],
            "action_mask": [bool(x) for x in obs["action_mask"]],
            "proposed_slot": slot,
            "verify_requested": (verify_flag == 1),
            "proposed_candidate_key": str(cand.placement.key),
            "executed_slot": exec_slot,
            "executed_candidate_key": str(executed_placement.key),
            "executed_deferral": bool(executed_placement.is_deferral),
            "quota_window": self.time // 300,
            "calls_used": actual_calls,
            "budget_after": self.budget.remaining_calls,
            "filter_mode": self.filter_mode,
            "risk_score": filter_details.get("u_score"),
            "filter_details": filter_details,
            "check_outcome": filter_details.get("check_outcome"),
            "verifier_timeout": bool(filter_details.get("timeout", False)),
        }
        action_event = self.ledger.record_event(
            self.time, "action", job_id=job["job_name"], vc_name=key,
            details=action_details
        )

        immediate_events: List[SimulationEvent] = [action_event]

        if executed_placement.is_deferral:
            # Deferral: job remains queued, marked offered for this second
            self.offered_jobs_this_tick.add(job["job_name"])
            if self.spec.retry_mode == "next_event":
                self.deferred_epoch[job["job_name"]] = self.eligibility_epoch
            ev = self.ledger.record_event(
                self.time, "deferral", job_id=job["job_name"], vc_name=key,
                details={"slot": slot, "source": cand.source}
            )
            immediate_events.append(ev)
            self.events_delta.append(ev)
        else:
            # Atomic commit of complete placement
            staged: List[Tuple[Any, List[int]]] = []
            for alloc in executed_placement.workers:
                staged.append((vc.get_node(alloc.node_id), list(alloc.card_ids)))
            touched_nodes: Dict[int, Tuple[Any, Set[int]]] = {}
            for node, cards in staged:
                if node.node_name not in touched_nodes:
                    touched_nodes[node.node_name] = (node, set())
                touched_nodes[node.node_name][1].update(cards)
            node_snapshots = {
                name: (
                    node.job_num,
                    {c: list(node.node_gpu_dict[c]) for c in cards},
                    {c: node.node_galloc[c] for c in cards},
                    {c: dict(node.node_type_galloc[c]) for c in cards},
                ) for name, (node, cards) in touched_nodes.items()
            }
            job_snapshot = dict(job)
            job_attrs_snapshot = dict(job.__dict__)
            old_job_nodes = copy.deepcopy(job.get("nodes", []))
            queue_snapshot = list(self.spot_que_list[key])
            running_snapshot = list(self.run_list[key])
            usage_snapshot = self.gpu_request_dict[key]["Spot"]
            event_count = len(self.ledger.events)
            event_seq = self.ledger._seq
            try:
                # Commit
                for node, gpu_list in staged:
                    node.job_num += 1
                    for c in gpu_list:
                        node.node_gpu_dict[c].append(job)
                        node.node_galloc[c] = min(job["gpu_request"], 1.0)
                        node.node_type_galloc[c]["Spot"] += min(job["gpu_request"], 1.0)
                    job["nodes"].append({node.node_name: gpu_list})

                self.run_spot(job)
                jid = job["job_name"]
                att_idx = self._spot_attempt_indices.get(jid, -1) + 1
                self._spot_attempt_indices[jid] = att_idx
                self._spot_credit_sequences[jid] = 0
                allocs = copy.deepcopy(job.get("nodes", []))
                self.ledger.record_job_start(
                    self.time, jid, "Spot",
                    nodes=job["nodes"],
                    requested_worker_count=job["worker_num"],
                    actual_worker_count=len(job.get("nodes", [])),
                    complete_start=(len(job.get("nodes", [])) == job["worker_num"]),
                    allocations=allocs,
                    vc_name=key,
                    attempt_index=att_idx
                )

                ev_p = self.ledger.record_event(
                    self.time, "placement", job_id=job["job_name"], vc_name=key,
                    details={"key": executed_placement.key, "slot": slot, "allocations": allocs}
                )
                immediate_events.append(ev_p)
                self.events_delta.append(ev_p)
                self.state_version += 1
                self.deferred_epoch.pop(job["job_name"], None)
            except Exception as e:
                for name, (node, _) in touched_nodes.items():
                    count, gpu_dict, alloc_dict, type_dict = node_snapshots[name]
                    node.job_num = count
                    for card, jobs in gpu_dict.items():
                        node.node_gpu_dict[card] = jobs
                        node.node_galloc[card] = alloc_dict[card]
                        node.node_type_galloc[card] = type_dict[card]
                job.clear()
                job.update(job_snapshot)
                job.__dict__.clear()
                job.__dict__.update(job_attrs_snapshot)
                job["nodes"] = copy.deepcopy(old_job_nodes)
                self.spot_que_list[key][:] = queue_snapshot
                self.run_list[key][:] = running_snapshot
                self.gpu_request_dict[key]["Spot"] = usage_snapshot
                del self.ledger.events[event_count:]
                self.ledger._seq = event_seq
                self.ledger.record_event(
                    self.time, "invalid_action", job_id=job["job_name"], vc_name=key,
                    details={"slot": slot, "error": str(e)}
                )
                raise ResearchSimError(f"Atomic placement commit failed: {e}") from e

        # Latency measurement
        t_apply_end_ns = self.timer_func()
        resp_ms = (t_apply_end_ns - self._decision_paused_time_ns) / 1e6 if self._decision_paused_time_ns is not None else 0.0
        pipe_ms = (t_apply_end_ns - self._last_candidate_start_ns) / 1e6 if self._last_candidate_start_ns is not None else resp_ms
        self.decision_timing_records.append({
            "decision_id": self.decisions_count - 1,
            "replay_time": self.time,
            "state_version": self.state_version,
            "selected_slot": slot,
            "verify_flag": verify_flag,
            "candidate_generation_ms": self._last_candidate_gen_ms,
            "policy_response_ms": resp_ms,
            "decision_pipeline_ms": pipe_ms,
            "candidate_start_ns": self._last_candidate_start_ns,
            "candidate_end_ns": self._last_candidate_end_ns,
            "response_start_ns": self._decision_paused_time_ns,
            "response_end_ns": t_apply_end_ns,
            "calls_used": actual_calls,
            "status": "committed" if not executed_placement.is_deferral else "deferred",
        })

        # Advance scan pointer
        self.active_scan_idx += 1
        self.paused_context = None
        return tuple(immediate_events)

    def _validate_placement(self, job: Job, vc: Any, placement: Placement) -> None:
        if placement.is_deferral or len(placement.workers) != int(job["worker_num"]):
            raise InvalidActionError("Placement must assign every worker exactly once.")
        cards_seen: Set[Tuple[int, int]] = set()
        counts: Dict[int, int] = {}
        for allocation in placement.workers:
            if not vc.check_node_inside_vc(allocation.node_id):
                raise InvalidActionError("Placement uses an incompatible node.")
            if len(allocation.card_ids) != int(job["gpu_request"]):
                raise InvalidActionError("Worker card count differs from demand.")
            node = vc.get_node(allocation.node_id)
            counts[allocation.node_id] = counts.get(allocation.node_id, 0) + 1
            for card in allocation.card_ids:
                token = (allocation.node_id, card)
                if token in cards_seen or card not in node.node_gpu_dict or node.node_gpu_dict[card]:
                    raise InvalidActionError("Placement repeats or occupies a card.")
                cards_seen.add(token)
        if tuple(sorted(counts.items())) != placement.key:
            raise InvalidActionError("Placement key differs from its worker mapping.")
        if self.quota_ready and self.in_quota_window(self.time):
            quota = self.spot_quota[job["vc_name"]]["quota"][job["guarantee_request"]]
            demand = job["gpu_request"] * job["worker_num"]
            if self.gpu_request_dict[job["vc_name"]]["Spot"] + demand > quota:
                raise InvalidActionError("Placement exceeds the current Spot quota.")

    def _place_research_hp_gang(self, job: Job, vc: Any) -> Tuple[bool, List[Job]]:
        """Place every whole card HP worker or leave the cluster unchanged.

        The released preemption path admits one worker only. The full history
        experiment uses this declared complete gang extension for all policies.
        """
        workers = int(job["worker_num"])
        cards_per_worker = int(job["gpu_request"])
        scheduler_profile = getattr(self, "scheduler_profile", "native_gfs")
        free = {node.node_name: len(node.check_free_gpu_list()) for node in vc.node_list}
        direct_source_preemption = scheduler_profile in {
            "yarn_cs_source_compatible",
            "fgd_source_compatible",
        }
        if not direct_source_preemption and has_complete_fit(workers, cards_per_worker, free):
            if not self.job_placer(job) or len(job["nodes"]) != workers:
                raise ResearchSimError("Native HP placement violated the complete fit gate.")
            return True, []

        if scheduler_profile in CHRONUS_PROFILES:
            return False, []

        hp_available = {
            node.node_name: sum(
                node.node_type_galloc[c]["HP"] == 0 for c in range(node.num_gpus)
            ) for node in vc.node_list
        }
        if not has_complete_fit(workers, cards_per_worker, hp_available):
            return False, []

        reserved: Dict[int, Set[int]] = {node.node_name: set() for node in vc.node_list}
        victims: Dict[int, Job] = {}
        plan: List[Tuple[Any, List[int]]] = []
        for _ in range(workers):
            choices = []
            for node in vc.node_list:
                available = [
                    c for c in range(node.num_gpus)
                    if node.node_type_galloc[c]["HP"] == 0 and c not in reserved[node.node_name]
                ]
                if len(available) < cards_per_worker:
                    continue

                def native_card_cost(card):
                    new_jobs = [
                        spot for spot in node.node_gpu_dict[card]
                        if spot["type"] == "Spot" and spot["job_name"] not in victims
                    ]
                    return (len(new_jobs), sum(s.get_preempt_cost(self.time) for s in new_jobs), card)

                if scheduler_profile in LYRA_COST_PROFILES:
                    def lyra_card_cost(card: int) -> float:
                        return float(sum(
                            1.0 / max(1.0, float(spot["worker_num"]) * float(spot["gpu_request"]))
                            for spot in node.node_gpu_dict[card]
                            if spot["type"] == "Spot" and spot["job_name"] not in victims
                        ))
                    selected = sorted(available, key=lambda card: (-lyra_card_cost(card), card))[:cards_per_worker]
                    choice_key = (sum(lyra_card_cost(card) for card in selected), node.node_name)
                elif scheduler_profile in SRGFS_COST_PROFILES:
                    selected, new_victims, loss_key = self._minimum_loss_card_set(
                        node, available, cards_per_worker, victims
                    )
                    choice_key = (*loss_key[:-1], node.node_name, loss_key[-1])
                elif scheduler_profile in FGD_PROFILES:
                    selected = sorted(available)[:cards_per_worker]
                    choice_key = (
                        self._fgd_frag_value(len(available) - cards_per_worker)
                        - self._fgd_frag_value(len(available)),
                        node.node_name,
                    )
                elif scheduler_profile in {
                    "yarn_cs_source_compatible",
                    "capacity_open_fcfs_best_fit_probe",
                }:
                    selected = sorted(available)[:cards_per_worker]
                    choice_key = (len(available), node.node_name)
                else:
                    selected = sorted(available, key=native_card_cost)[:cards_per_worker]
                    choice_key = None
                if scheduler_profile not in SRGFS_COST_PROFILES:
                    new_victims = {
                        spot["job_name"]: spot
                        for card in selected for spot in node.node_gpu_dict[card]
                        if spot["type"] == "Spot" and spot["job_name"] not in victims
                    }
                if choice_key is None:
                    choice_key = (
                        len(new_victims),
                        sum(s.get_preempt_cost(self.time) for s in new_victims.values()),
                        node.node_name,
                    )
                choices.append((choice_key, selected, node, new_victims))
            if not choices:
                return False, []
            _, selected, node, new_victims = min(choices, key=lambda c: c[0])
            reserved[node.node_name].update(selected)
            victims.update(new_victims)
            plan.append((node, selected))

        affected_nodes = {
            node.node_name: node for node in vc.node_list
            if reserved[node.node_name]
        }
        for spot in victims.values():
            for entry in spot["nodes"]:
                for node_name in entry:
                    affected_nodes[node_name] = vc.get_node(node_name)
        node_snapshots = {
            name: (node.job_num,
                   {c: list(jobs) for c, jobs in node.node_gpu_dict.items()},
                   dict(node.node_galloc),
                   {c: dict(types) for c, types in node.node_type_galloc.items()})
            for name, node in affected_nodes.items()
        }
        old_job_nodes = copy.deepcopy(job["nodes"])
        try:
            for spot in victims.values():
                vc.release_resource(spot)
            for node, cards in plan:
                if any(node.node_gpu_dict[c] for c in cards):
                    raise ResearchSimError("Planned HP gang card remained occupied after preemption.")
                node.allocate_gpus_with_preemption(cards, job)
                job["nodes"].append({node.node_name: list(cards)})
        except Exception as exc:
            for name, (count, gpu_jobs, galloc, type_galloc) in node_snapshots.items():
                node = affected_nodes[name]
                node.job_num = count
                node.node_gpu_dict.clear()
                node.node_gpu_dict.update(gpu_jobs)
                node.node_galloc.clear()
                node.node_galloc.update(galloc)
                node.node_type_galloc.clear()
                node.node_type_galloc.update(type_galloc)
            job["nodes"] = old_job_nodes
            raise ResearchSimError("Atomic HP gang preemption failed.") from exc
        return True, list(victims.values())

    def advance_until_decision(
        self, stop_at_time: Optional[int] = None
    ) -> Tuple[Optional[DecisionContext | ControlDecisionContext], TransitionDelta]:
        """
        Advances to the next Spot decision, replay end, or optional diagnostic time limit.

        A diagnostic limit leaves the simulator resumable and does not finalize
        the ledger. The default production path is unchanged.
        """
        if self.paused_context is not None:
            raise ResearchSimError("Simulator is paused. Call apply(action) before advancing.")
        if self.control_paused_context is not None:
            raise ResearchSimError(
                "Simulator is paused. Call apply_control(action) before advancing."
            )

        while (self.time <= self.spec.followup_end
               and (stop_at_time is None or self.time <= stop_at_time)
               and self.end_job_num < self.total_job_num):

            # ---------------------------------------------------------------
            # Phase 0: QUOTA
            # ---------------------------------------------------------------
            if self.phase == Phase.QUOTA:
                self.budget.on_second(self.time)
                if self.time % 300 == 0:
                    if self.in_quota_window(self.time):
                        now_timestamp = pd.Timestamp(self.spec.calendar_anchor) + pd.Timedelta(seconds=self.time)
                        if self.scheduler_profile in CAPACITY_QUOTA_PROFILES:
                            for vc_name in self.cluster.vc_list:
                                capacity = self.cluster.vc_dict[vc_name].total_gpus
                                self.spot_quota[vc_name]["quota"] = {
                                    h: float(capacity) for h in self.args.guarantee_hour
                                }
                            self.quota_update_count += 1
                            forecast_source = {
                                "provider_id": f"{self.scheduler_profile}_capacity",
                                "provider_version": "adapted_v1",
                            }
                        elif self.spec.forecast_mode == "native_online":
                            if not self.recorder.vc_record_df_list:
                                raise ResearchSimError(
                                    f"GFS quota has no recorder history at t={self.time}."
                                )
                            try:
                                if self.forecast_cache is not None and self.time in self.forecast_cache:
                                    self.update_spot_quota_cached(self.time, now_timestamp)
                                else:
                                    org_request_df = self.recorder.agg_seq_recorder()
                                    schema_changed = self.estimator.requires_schema_refresh(
                                        org_request_df
                                    )
                                    if self.time % 86400 == 0 or schema_changed:
                                        self.estimator.train(org_request_df, now_timestamp)
                                        self.forecast_training_count += 1
                                    self.update_spot_quota(org_request_df, now_timestamp)
                                self.quota_update_count += 1
                            except Exception as exc:
                                raise ResearchSimError(
                                    f"Native GFS forecast or quota failed at t={self.time}."
                                ) from exc
                            forecast_source = {
                                "provider_id": self.spec.forecast_provider_id,
                                "provider_version": self.spec.forecast_provider_version,
                            }
                        elif self.spec.forecast_mode == "causal_capacity":
                            for vc_name in self.cluster.vc_list:
                                capacity = self.cluster.vc_dict[vc_name].total_gpus
                                hp_used = sum(
                                    job["gpu_request"] * job["worker_num"]
                                    for job in self.run_list[vc_name] if job["type"] == "HP"
                                )
                                quota = max(0.0, capacity - hp_used)
                                self.spot_quota[vc_name]["quota"] = {
                                    h: quota for h in self.args.guarantee_hour
                                }
                            self.quota_update_count += 1
                            forecast_source = {
                                "provider_id": self.spec.forecast_provider_id,
                                "provider_version": self.spec.forecast_provider_version,
                            }
                        elif self.spec.forecast_mode == "external_causal":
                            if not self.recorder.vc_record_df_list:
                                raise ResearchSimError(
                                    f"External quota provider has no recorder history at t={self.time}."
                                )
                            try:
                                org_request_df = self.recorder.agg_seq_recorder()
                                capacity_by_pool = {
                                    vc_name: int(self.cluster.vc_dict[vc_name].total_gpus)
                                    for vc_name in self.cluster.vc_list
                                }
                                hp_used_by_pool = {
                                    vc_name: float(sum(
                                        job["gpu_request"] * job["worker_num"]
                                        for job in self.run_list[vc_name]
                                        if job["type"] == "HP"
                                    ))
                                    for vc_name in self.cluster.vc_list
                                }
                                spot_used_by_pool = {
                                    vc_name: float(self.gpu_request_dict[vc_name]["Spot"])
                                    for vc_name in self.cluster.vc_list
                                }
                                eviction_outcomes = {
                                    vc_name: tuple(int(v) for v in self.cluster.vc_dict[vc_name].get_eviction_rate(
                                        self.time, self.args.pred_len
                                    ))
                                    for vc_name in self.cluster.vc_list
                                }
                                forecast_context = CausalForecastContext.build(
                                    time=self.time,
                                    calendar_timestamp=str(now_timestamp),
                                    org_request_history=org_request_df,
                                    capacity_by_pool=capacity_by_pool,
                                    hp_used_by_pool=hp_used_by_pool,
                                    spot_used_by_pool=spot_used_by_pool,
                                    current_quota_by_pool=copy.deepcopy(self.spot_quota),
                                    eviction_outcomes_by_pool=eviction_outcomes,
                                    guarantee_rate=self.args.guarantee_rate,
                                    guarantee_hours=self.args.guarantee_hour,
                                )
                                raw_quota = self.quota_provider.update_quota(forecast_context)
                                self.spot_quota = validate_quota_update(
                                    raw_quota,
                                    capacity_by_pool=capacity_by_pool,
                                    guarantee_hours=self.args.guarantee_hour,
                                )
                                self.quota_update_count += 1
                            except Exception as exc:
                                raise ResearchSimError(
                                    f"External causal quota provider failed at t={self.time}."
                                ) from exc
                            forecast_source = {
                                "provider_id": self.spec.forecast_provider_id,
                                "provider_version": self.spec.forecast_provider_version,
                            }
                        else:
                            raise InvalidConfigError(f"Unsupported forecast mode: {self.spec.forecast_mode}")
                        self.quota_ready = True
                        self.eligibility_epoch += 1
                        self.ledger.record_event(
                            self.time, "quota_update",
                            details={
                                "spot_quota": {k: v["quota"] for k, v in self.spot_quota.items()},
                                **forecast_source,
                            }
                        )
                    for vc_name in self.guar_que_list:
                        vc = self.cluster.vc_dict[vc_name]
                        vc.update_eviction_score(self.time)
                    if (
                        self.policy_type == "unified_control"
                        and self.in_score_window(self.time)
                    ):
                        self.phase = Phase.CONTROL
                        self.control_paused_context = self._build_control_context()
                        self.decisions_count += 1
                        return self.control_paused_context, self._consume_control_delta()
                self.phase = Phase.RUNNING
                if self.time == self.spec.score_start:
                    if getattr(self, "save_warmup_snapshot_to", None) is not None:
                        from .warmup_snapshot import save_warmup_snapshot
                        save_warmup_snapshot(self, self.save_warmup_snapshot_to)
                    if getattr(self, "_warmup_snapshot_hook", None) is not None:
                        self._warmup_snapshot_hook(self)

            if self.phase == Phase.CONTROL:
                raise ResearchSimError(
                    "CONTROL phase requires apply_control(action) before replay can continue."
                )


            # ---------------------------------------------------------------
            # Phase 1: RUNNING
            # ---------------------------------------------------------------
            if self.phase == Phase.RUNNING:
                for vc_name in self.run_list:
                    run_ls = self.run_list[vc_name].copy()
                    vc = self.cluster.vc_dict[vc_name]
                    for job in run_ls:
                        if (
                            self.scheduler_profile in CHRONUS_PROFILES
                            and self.in_score_window(self.time)
                        ):
                            if (
                                self.scheduler_profile == "chronus_source_compatible"
                                and self._expire_chronus_lease(job, vc_name, vc)
                            ):
                                continue
                            overhead_remaining = int(
                                job.get("chronus_overhead_remaining_seconds", 0)
                            )
                            if overhead_remaining > 0:
                                job["chronus_overhead_remaining_seconds"] = overhead_remaining - 1
                            else:
                                job["remain"] = max(0.0, float(job["remain"]) - 1.0)
                                if job["type"] == "Spot":
                                    progress = int(
                                        job.get("chronus_checkpoint_progress_seconds", 0)
                                    ) + 1
                                    if progress >= int(job["spot_duration"]):
                                        credit = float(
                                            job["spot_duration"]
                                            * job["gpu_request"]
                                            * job["worker_num"]
                                        )
                                        self.useful_spot_seconds_delta += (
                                            self._record_spot_credit_checked(
                                                job,
                                                vc_name,
                                                credit,
                                                reason="checkpoint",
                                                credited_run_seconds=float(job["spot_duration"]),
                                            )
                                        )
                                        self.record_spot_succeed(job)
                                        progress = 0
                                    job["chronus_checkpoint_progress_seconds"] = progress
                            if job["remain"] == 0:
                                self._complete_running_job(
                                    job,
                                    vc_name,
                                    vc,
                                    spot_remainder_seconds=(
                                        float(job.get("chronus_checkpoint_progress_seconds", 0))
                                        if job["type"] == "Spot" else None
                                    ),
                                )
                                continue
                            if (
                                self.scheduler_profile == "staggered_lease_best_fit_probe"
                                and self._expire_chronus_lease(job, vc_name, vc)
                            ):
                                continue
                            continue

                        # Checkpoint milestone
                        if job["type"] == "Spot" and self.time - job.get_ckpt_time() >= job["spot_duration"]:
                            credit = float(job["spot_duration"] * job["gpu_request"] * job["worker_num"])
                            self.useful_spot_seconds_delta += self._record_spot_credit_checked(
                                job,
                                vc_name,
                                credit,
                                reason="checkpoint",
                                credited_run_seconds=float(job["spot_duration"]),
                            )
                            self.record_spot_succeed(job)
                            job.set_ckpt_time(self.time)

                        if self._expire_chronus_lease(job, vc_name, vc):
                            continue

                        if job["remain"] == 0:
                            self._complete_running_job(job, vc_name, vc)
                        else:
                            job["remain"] -= 1
                self.phase = Phase.ARRIVALS

            # ---------------------------------------------------------------
            # Phase 2: ARRIVALS
            # ---------------------------------------------------------------
            if self.phase == Phase.ARRIVALS:
                while self.current_arrivals_idx < self.total_job_num:
                    job = self.trace.job_list[self.current_arrivals_idx]
                    if job["submit_time"] <= self.time:
                        self.pend_job(job)
                        self.current_arrivals_idx += 1
                        if job["type"] == "HP":
                            self.ledger.record_hp_arrival(
                                self.time, job["job_name"], job["worker_num"], job["gpu_request"],
                                submit_time=job["submit_time"], vc_name=job["vc_name"]
                            )
                        else:
                            self.ledger.record_event(
                                self.time, "arrival", job_id=job["job_name"], vc_name=job["vc_name"],
                                details={
                                    "job_type": "Spot",
                                    "worker_num": job["worker_num"],
                                    "gpu_request": job["gpu_request"],
                                    "submit_time": job["submit_time"],
                                    "executed_arrival_time": self.time,
                                    "vc_name": job["vc_name"],
                                    "hp_cohort_eligible": False,
                                    "exclusion_reason": "spot_job"
                                }
                            )
                        self.eligibility_epoch += 1
                    else:
                        break
                self.phase = Phase.HP_SCAN

            # ---------------------------------------------------------------
            # Phase 3: HP_SCAN
            # ---------------------------------------------------------------
            if self.phase == Phase.HP_SCAN:
                if (
                    self.scheduler_profile in CHRONUS_PROFILES
                    and self.in_score_window(self.time)
                    and self.time % 300 != 0
                ):
                    self.phase = Phase.RECORD
                    continue
                for vc in self.guar_que_list:
                    self.guar_que_list[vc].sort(key=self._hp_queue_key)
                    que_ls = self.guar_que_list[vc].copy()
                    failed_gpu_num = 16

                    for job in que_ls:
                        if job["gpu_request"] >= failed_gpu_num:
                            continue
                        research_gang = (
                            self.spec.trace_mode == "research_full_history"
                            and job["worker_num"] > 1
                            and job["gpu_request"] >= 1
                            and float(job["gpu_request"]).is_integer()
                        )
                        if research_gang:
                            placed, preempt_jobs = self._place_research_hp_gang(
                                job, self.cluster.vc_dict[vc]
                            )
                            if placed:
                                self.run_guar(job)
                                allocs = copy.deepcopy(job.get("nodes", []))
                                act_workers = len(job.get("nodes", []))
                                req_workers = job["worker_num"]
                                self.ledger.record_job_start(
                                    self.time, job["job_name"], "HP",
                                    nodes=job["nodes"],
                                    requested_worker_count=req_workers,
                                    actual_worker_count=act_workers,
                                    complete_start=(act_workers == req_workers),
                                    allocations=allocs,
                                    vc_name=job["vc_name"]
                                )
                                self.eligibility_epoch += 1
                                if preempt_jobs:
                                    self._execute_preemption_and_record(job, preempt_jobs)
                                    self.eligibility_epoch += 1
                            else:
                                failed_gpu_num = job["gpu_request"]
                        elif self.scheduler_profile not in {
                            "yarn_cs_source_compatible",
                            "fgd_source_compatible",
                        } and self.job_placer(job):
                            self.run_guar(job)
                            allocs = copy.deepcopy(job.get("nodes", []))
                            act_workers = len(job.get("nodes", []))
                            req_workers = job["worker_num"]
                            self.ledger.record_job_start(
                                self.time, job["job_name"], "HP",
                                nodes=job["nodes"],
                                requested_worker_count=req_workers,
                                actual_worker_count=act_workers,
                                complete_start=(act_workers == req_workers),
                                allocations=allocs,
                                vc_name=job["vc_name"]
                            )
                            self.eligibility_epoch += 1
                        elif (
                            self.scheduler_profile not in CHRONUS_PROFILES
                            and (
                                self.time >= self.args.log_range[0]
                                or self.spec.trace_mode == "research_full_history"
                            )
                        ):
                            place_flag, preempt_jobs = self.place_with_preemption(job)
                            if place_flag:
                                self.run_guar(job)
                                allocs = copy.deepcopy(job.get("nodes", []))
                                act_workers = len(job.get("nodes", []))
                                req_workers = job["worker_num"]
                                self.ledger.record_job_start(
                                    self.time, job["job_name"], "HP",
                                    nodes=job["nodes"],
                                    requested_worker_count=req_workers,
                                    actual_worker_count=act_workers,
                                    complete_start=(act_workers == req_workers),
                                    allocations=allocs,
                                    vc_name=job["vc_name"]
                                )
                                self.eligibility_epoch += 1
                                if preempt_jobs:
                                    self._execute_preemption_and_record(job, preempt_jobs)
                                    self.eligibility_epoch += 1
                            else:
                                failed_gpu_num = job["gpu_request"]
                        else:
                            failed_gpu_num = job["gpu_request"]

                # Check eligible gang deadlines after HP scheduling at simulator second t
                self.ledger.check_hp_deadlines(self.time)

                self.phase = Phase.SPOT_SCAN
                self.active_scan_key = None
                self.active_scan_ls = None
                self.active_scan_idx = 0
                self.active_failed_gpu_num = 16

            # ---------------------------------------------------------------
            # Phase 4: SPOT_SCAN
            # ---------------------------------------------------------------
            if self.phase == Phase.SPOT_SCAN:
                vc_keys = list(self.spot_que_list.keys())

                # If starting new scan
                if self.active_scan_key is None:
                    if vc_keys:
                        self.active_scan_key = vc_keys[0]
                        vc = self.cluster.vc_dict[self.active_scan_key]
                        self.active_scan_ls = self._order_spot_queue(self.active_scan_key)
                        self.active_scan_idx = 0
                        self.active_failed_gpu_num = 16

                while self.active_scan_key is not None:
                    key = self.active_scan_key
                    vc = self.cluster.vc_dict[key]
                    spot_que_ls = self.active_scan_ls

                    while self.active_scan_idx < len(spot_que_ls):
                        job = spot_que_ls[self.active_scan_idx]

                        # Skip if already offered / deferred in this second
                        if job["job_name"] in self.offered_jobs_this_tick:
                            self.active_scan_idx += 1
                            continue
                        if (self.policy_type != "native_gfs" and self.spec.retry_mode == "next_event"
                                and self.deferred_epoch.get(job["job_name"], -1) >= self.eligibility_epoch):
                            self.active_scan_idx += 1
                            continue

                        # Skip threshold
                        if job["gpu_request"] >= self.active_failed_gpu_num:
                            self.active_scan_idx += 1
                            continue

                        # Quota check
                        job_demand = job["gpu_request"] * job["worker_num"]
                        current_spot_usage = self.gpu_request_dict[key]["Spot"]
                        if self.quota_ready:
                            quota_val = self.spot_quota[key]["quota"].get(job.get("guarantee_request", 1), vc.total_gpus)
                        else:
                            quota_val = vc.total_gpus if self.time < self.spec.score_start else 0.0
                        if self.spec.trace_mode == "source_compatible":
                            quota_ok = (
                                (self.time > self.args.log_range[1]) or
                                ((self.args.log_range[0] <= self.time <= self.args.log_range[1]) and
                                 self.quota_ready and current_spot_usage + job_demand <= quota_val)
                            )
                        else:
                            quota_ok = (
                                (self.time >= self.spec.score_end) or
                                (self.time < self.spec.score_start) or
                                (self.in_quota_window(self.time) and
                                 self.quota_ready and current_spot_usage + job_demand <= quota_val)
                            )

                        if quota_ok:
                            is_whole_card = (job["gpu_request"] >= 1.0 and (job["gpu_request"] % 1.0 == 0.0))
                            empty_cards_by_node = {n.node_name: tuple(n.check_free_gpu_list()) for n in vc.node_list}
                            free_counts = {n: len(cards) for n, cards in empty_cards_by_node.items()}
                            fit_ok = has_complete_fit(job["worker_num"], int(job["gpu_request"]), free_counts)

                            if is_whole_card and fit_ok:
                                if self.policy_type in {"native_gfs", "unified_control"} or not self.in_score_window(self.time):
                                    # Native GFS places directly
                                    if self.job_placer(job):
                                        self.run_spot(job)
                                        jid = job["job_name"]
                                        att_idx = self._spot_attempt_indices.get(jid, -1) + 1
                                        self._spot_attempt_indices[jid] = att_idx
                                        self._spot_credit_sequences[jid] = 0
                                        allocs = copy.deepcopy(job.get("nodes", []))
                                        self.ledger.record_job_start(
                                            self.time, jid, "Spot",
                                            nodes=job["nodes"],
                                            requested_worker_count=job["worker_num"],
                                            actual_worker_count=len(job.get("nodes", [])),
                                            complete_start=(len(job.get("nodes", [])) == job["worker_num"]),
                                            allocations=allocs,
                                            vc_name=key,
                                            attempt_index=att_idx
                                        )
                                        self.eligibility_epoch += 1
                                        self.ledger.record_event(
                                            self.time, "placement", job_id=job["job_name"], vc_name=key,
                                            details={"source": "native_gfs", "slot": 0, "key": None, "allocations": allocs}
                                        )
                                    else:
                                        vc.release_resource(job)
                                        job.update({"nodes": []})
                                        self.active_failed_gpu_num = job["gpu_request"]
                                else:
                                    # Pausable RL Decision Boundary
                                    evict_dict = getattr(vc, "eviction_score", None) or {}
                                    capacities = {n.node_name: n.num_gpus for n in vc.node_list}
                                    occupied_spot = {
                                        n.node_name: sum(
                                            1 for c in range(n.num_gpus)
                                            if n.node_type_galloc[c]["Spot"] > 0
                                        ) for n in vc.node_list
                                    }
                                    self._last_candidate_start_ns = self.timer_func()
                                    cand_set = self.candidate_generator.generate(
                                        job_id=job["job_name"],
                                        worker_num=job["worker_num"],
                                        gpu_request=int(job["gpu_request"]),
                                        empty_cards_by_node=empty_cards_by_node,
                                        eviction_scores=evict_dict,
                                        node_capacities=capacities,
                                        spot_occupied_cards=occupied_spot,
                                        state_version=self.state_version,
                                        time=self.time
                                    )
                                    self._last_candidate_end_ns = self.timer_func()
                                    self._last_candidate_gen_ms = (self._last_candidate_end_ns - self._last_candidate_start_ns) / 1e6

                                    queued_hp_jobs = [
                                        queued_job
                                        for queued_vc in self.guar_que_list.values()
                                        for queued_job in queued_vc
                                    ]
                                    queued_hp_demands = [
                                        float(queued_job["worker_num"])
                                        * float(queued_job["gpu_request"])
                                        for queued_job in queued_hp_jobs
                                    ]
                                    queued_hp_waits = [
                                        max(0.0, float(self.time - queued_job["submit_time"]))
                                        for queued_job in queued_hp_jobs
                                    ]
                                    total_gpu_capacity = float(sum(capacities.values()))
                                    fully_free_node_count = sum(
                                        1
                                        for node_id, capacity in capacities.items()
                                        if len(empty_cards_by_node.get(node_id, ())) == capacity
                                    )
                                    estimated_duration = self.estimate_job_duration(
                                        int(job["worker_num"]),
                                        int(job["gpu_request"]),
                                        is_spot=True,
                                        gpu_model=job.get("gpu_model", key),
                                    )

                                    context = DecisionContext(
                                        state_version=self.state_version,
                                        time=self.time,
                                        job_id=job["job_name"],
                                        job_info={
                                            "job_name": job["job_name"],
                                            "organization": job.get("organization", 0),
                                            "gpu_model": job.get("gpu_model", key),
                                            "worker_num": job["worker_num"],
                                            "gpu_request": job["gpu_request"],
                                            "cpu_request": job.get("cpu_request", 0.0),
                                            "submit_time": job["submit_time"],
                                            "elapsed_queue_time": self.time - job["submit_time"],
                                            "guarantee_request": job.get("guarantee_request", 1),
                                            "current_quota": float(quota_val),
                                            "spot_in_use": float(current_spot_usage),
                                            "queued_hp_count": sum(len(self.guar_que_list[v]) for v in self.guar_que_list),
                                            "queued_spot_count": sum(len(self.spot_que_list[v]) for v in self.spot_que_list),
                                            "offer_ordinal": self.active_scan_idx,
                                            "split_id": getattr(self.spec, "split_id", "train"),
                                            "proposed_duration": estimated_duration,
                                            "estimated_duration_seconds": estimated_duration,
                                            "service_horizon": self.spec.L_H,
                                            "score_start": self.spec.score_start,
                                            "score_end": self.spec.score_end,
                                            "followup_end": self.spec.followup_end,
                                            "total_gpu_capacity": total_gpu_capacity,
                                            "verification_budget_capacity": self.spec.B_call,
                                            "queued_hp_gpu_demand": sum(queued_hp_demands),
                                            "largest_queued_hp_gang": max(queued_hp_demands, default=0.0),
                                            "oldest_queued_hp_wait": max(queued_hp_waits, default=0.0),
                                            "fully_free_node_count": fully_free_node_count,
                                        },
                                        budget_remaining=self.budget.remaining_calls,
                                        cursor=PhaseCursor(
                                            time=self.time,
                                            phase=Phase.SPOT_SCAN,
                                            vc_name=key,
                                            spot_queue_index=self.active_scan_idx,
                                            offered_jobs=tuple(self.offered_jobs_this_tick),
                                            state_version=self.state_version
                                        ),
                                        candidates=cand_set,
                                        empty_cards_by_node=empty_cards_by_node,
                                        node_capacities=capacities,
                                        eviction_scores=dict(evict_dict),
                                        spot_occupied_cards=occupied_spot,
                                    )

                                    self.active_scan_job = job
                                    self.paused_context = context

                                    # Package delta
                                    delta_events = tuple(self.ledger.events[self.delta_event_cursor:])
                                    reward_val = self.useful_spot_seconds_delta / 3600.0

                                    if not self._first_decision_returned:
                                        self._first_decision_returned = True
                                        init_reward = reward_val
                                        init_calls = self.solver_calls_delta
                                        init_deadlines = sum(1 for e in delta_events if e.event_type == "hp_deadline")
                                        init_evictions = sum(1 for e in delta_events if e.event_type == "preemption")
                                        init_lost_sec = sum(float(e.details.get("lost_gpu_seconds", 0.0)) for e in delta_events if e.event_type == "preemption")
                                        self.prefix_reward_gpu_hours = init_reward
                                        self.prefix_solver_calls = init_calls
                                        self.prefix_deadline_events = init_deadlines
                                        self.prefix_evictions = init_evictions
                                        self.prefix_lost_gpu_seconds = init_lost_sec
                                    else:
                                        init_reward = 0.0
                                        init_calls = 0
                                        init_deadlines = 0
                                        init_evictions = 0
                                        init_lost_sec = 0.0
                                        self.step_reward_gpu_hours_sum += reward_val

                                    delta = TransitionDelta(
                                        reward=reward_val,
                                        elapsed_sim_seconds=self.time - self.last_decision_time,
                                        solver_calls=self.solver_calls_delta,
                                        events=delta_events,
                                        terminated=False,
                                        truncated=False,
                                        initial_reward_gpu_hours=init_reward,
                                        initial_solver_calls=init_calls,
                                        initial_deadline_events=init_deadlines,
                                        initial_evictions=init_evictions,
                                        initial_lost_gpu_seconds=init_lost_sec,
                                    )
                                    self.useful_spot_seconds_delta = 0.0
                                    self.solver_calls_delta = 0
                                    self.events_delta = []
                                    self.delta_event_cursor = len(self.ledger.events)
                                    self.last_decision_time = self.time

                                    # PAUSE and return decision context!
                                    self._decision_paused_time_ns = self.timer_func()
                                    self.decisions_count += 1
                                    return context, delta
                            else:
                                # Fractional or native path
                                if self.job_placer(job):
                                    self.run_spot(job)
                                    jid = job["job_name"]
                                    att_idx = self._spot_attempt_indices.get(jid, -1) + 1
                                    self._spot_attempt_indices[jid] = att_idx
                                    self._spot_credit_sequences[jid] = 0
                                    allocs = copy.deepcopy(job.get("nodes", []))
                                    self.ledger.record_job_start(
                                        self.time, jid, "Spot",
                                        nodes=job["nodes"],
                                        requested_worker_count=job["worker_num"],
                                        actual_worker_count=len(job.get("nodes", [])),
                                        complete_start=(len(job.get("nodes", [])) == job["worker_num"]),
                                        allocations=allocs,
                                        vc_name=key,
                                        attempt_index=att_idx
                                    )
                                    self.eligibility_epoch += 1
                                else:
                                    vc.release_resource(job)
                                    job.update({"nodes": []})
                                    self.active_failed_gpu_num = job["gpu_request"]

                        self.active_scan_idx += 1

                    # Finished scanning this VC
                    curr_idx = vc_keys.index(self.active_scan_key)
                    if curr_idx + 1 < len(vc_keys):
                        self.active_scan_key = vc_keys[curr_idx + 1]
                        self.active_scan_ls = self._order_spot_queue(self.active_scan_key)
                        self.active_scan_idx = 0
                        self.active_failed_gpu_num = 16
                    else:
                        self.active_scan_key = None
                        break

                self.phase = Phase.RECORD

            # ---------------------------------------------------------------
            # Phase 5: RECORD
            # ---------------------------------------------------------------
            if self.phase == Phase.RECORD:
                if self.in_score_window(self.time):
                    self.spot_queue_gpu_seconds_delta += float(sum(
                        job["gpu_request"] * job["worker_num"]
                        for jobs in self.spot_que_list.values()
                        for job in jobs
                    ))
                if self.time % 60 == 0 and not (self.forecast_cache is not None and self.time >= self.spec.score_start):
                    self.recorder.update_seq_recorder()
                if self.in_guar_window(self.time):
                    self.update_guar_gpu_time()
                self.phase = Phase.TICK_END

            # ---------------------------------------------------------------
            # Phase 6: TICK_END
            # ---------------------------------------------------------------
            if self.phase == Phase.TICK_END:
                if self.progress_hook is not None and (self.time % 86400 == 0 or (self.time >= self.spec.score_start and self.time % 3600 == 0)):
                    self.progress_hook(self.time, self.end_job_num, self.total_job_num)
                self.time += 1
                self.offered_jobs_this_tick.clear()
                self.phase = Phase.QUOTA

        if stop_at_time is not None and self.time > stop_at_time:
            return None, TransitionDelta(
                reward=0.0,
                elapsed_sim_seconds=self.time - self.last_decision_time,
                solver_calls=0,
                events=tuple(self.ledger.events[self.delta_event_cursor:]),
                terminated=False,
                truncated=True,
                terminal_reason="diagnostic_time_limit",
            )

        # -------------------------------------------------------------------
        # Terminal / Replay Finished
        # -------------------------------------------------------------------
        is_terminal = self.end_job_num == self.total_job_num or self.time > self.spec.followup_end
        is_truncated = False
        term_reason = "all_jobs_completed" if self.end_job_num == self.total_job_num else "planned_horizon_reached"
        term_status = "completed"

        self._is_terminated = True
        self._is_truncated = is_truncated
        self._terminal_reason = term_reason

        self.ledger.finalize_hp_deadlines(self.time, externally_truncated=is_truncated)

        remaining_queued = sum(len(self.guar_que_list[v]) for v in self.guar_que_list) + sum(len(self.spot_que_list[v]) for v in self.spot_que_list)
        remaining_running = sum(len(self.run_list[v]) for v in self.run_list)

        self.ledger.record_event(
            self.time, "terminal",
            details={
                "time": self.time,
                "terminal_reason": term_reason,
                "remaining_queued": remaining_queued,
                "remaining_running": remaining_running,
                "last_event_seq": self.ledger._seq,
                "status": term_status,
            }
        )

        delta_events = tuple(self.ledger.events[self.delta_event_cursor:])
        reward_val = self.useful_spot_seconds_delta / 3600.0
        terminal_lost_seconds = sum(
            float(event.details.get("lost_gpu_seconds", 0.0))
            for event in delta_events if event.event_type == "preemption"
        )
        terminal_queue_gpu_hours = self.spot_queue_gpu_seconds_delta / 3600.0
        terminal_lost_gpu_hours = terminal_lost_seconds / 3600.0

        if not self._first_decision_returned:
            self._first_decision_returned = True
            init_reward = reward_val
            init_calls = self.solver_calls_delta
            init_deadlines = sum(1 for e in delta_events if e.event_type == "hp_deadline")
            init_evictions = sum(1 for e in delta_events if e.event_type == "preemption")
            init_lost_sec = sum(float(e.details.get("lost_gpu_seconds", 0.0)) for e in delta_events if e.event_type == "preemption")
            self.prefix_reward_gpu_hours = init_reward
            self.prefix_solver_calls = init_calls
            self.prefix_deadline_events = init_deadlines
            self.prefix_evictions = init_evictions
            self.prefix_lost_gpu_seconds = init_lost_sec
        else:
            init_reward = 0.0
            init_calls = 0
            init_deadlines = 0
            init_evictions = 0
            init_lost_sec = 0.0
            self.step_reward_gpu_hours_sum += reward_val

        delta = TransitionDelta(
            reward=reward_val,
            elapsed_sim_seconds=self.time - self.last_decision_time,
            solver_calls=self.solver_calls_delta,
            events=delta_events,
            terminated=is_terminal,
            truncated=is_truncated,
            terminal_reason=term_reason,
            initial_reward_gpu_hours=init_reward,
            initial_solver_calls=init_calls,
            initial_deadline_events=init_deadlines,
            initial_evictions=init_evictions,
            initial_lost_gpu_seconds=init_lost_sec,
            spot_queue_gpu_hours=terminal_queue_gpu_hours,
            lost_gpu_hours=terminal_lost_gpu_hours,
            control_reward=(
                reward_val - terminal_queue_gpu_hours - terminal_lost_gpu_hours
            ),
        )
        self.useful_spot_seconds_delta = 0.0
        self.spot_queue_gpu_seconds_delta = 0.0
        self.solver_calls_delta = 0
        self.events_delta = []
        self.delta_event_cursor = len(self.ledger.events)
        self.last_decision_time = self.time

        return None, delta

    def get_metrics(self) -> EpisodeMetrics:
        """Returns reconciled summary metrics from event ledger."""
        remaining_queued = sum(len(self.guar_que_list[v]) for v in self.guar_que_list) + sum(len(self.spot_que_list[v]) for v in self.spot_que_list)
        remaining_running = sum(len(self.run_list[v]) for v in self.run_list)
        is_terminal = self._is_terminated or (self.end_job_num == self.total_job_num or self.time > self.spec.followup_end)
        is_truncated = self._is_truncated
        term_reason = self._terminal_reason or ("all_jobs_completed" if self.end_job_num == self.total_job_num else "planned_horizon_reached")

        return self.ledger.compute_summary_metrics(
            current_time=self.time,
            remaining_queued=remaining_queued,
            remaining_running=remaining_running,
            terminated=is_terminal,
            truncated=is_truncated,
            terminal_reason=term_reason,
            initial_reward_gpu_hours=self.prefix_reward_gpu_hours,
            step_reward_gpu_hours_sum=self.step_reward_gpu_hours_sum,
        )

    def terminate(self, reason: str = "user_halt", externally_truncated: bool = True):
        """Forces graceful termination of the simulation ledger and records terminal event."""
        if any(e.event_type == "terminal" for e in self.ledger.events):
            return
        self._is_terminated = True
        self._is_truncated = externally_truncated
        self._terminal_reason = reason
        self.ledger.finalize_hp_deadlines(self.time, externally_truncated=externally_truncated)
        remaining_queued = sum(len(self.guar_que_list[v]) for v in self.guar_que_list) + sum(len(self.spot_que_list[v]) for v in self.spot_que_list)
        remaining_running = sum(len(self.run_list[v]) for v in self.run_list)
        self.ledger.record_event(
            self.time, "terminal",
            details={
                "time": self.time,
                "terminal_reason": reason,
                "remaining_queued": remaining_queued,
                "remaining_running": remaining_running,
                "last_event_seq": self.ledger._seq,
                "status": "externally_truncated" if externally_truncated else "completed",
            }
        )
