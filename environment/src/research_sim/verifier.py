import json
import math
import dataclasses
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Protocol, Tuple
from dataclasses import dataclass
from .types import Placement, ClusterStateSnapshot


def _json_safe(obj: Any) -> Any:
    """Recursively converts objects to JSON-serializable structures, handling dataclasses, Enums, inf/nan."""
    if obj is None:
        return None
    if isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        if math.isnan(obj):
            return "nan"
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if dataclasses.is_dataclass(obj):
        return _json_safe(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "item"):
        return _json_safe(obj.item())
    return str(obj)



@dataclass(frozen=True)
class VerifierResult:
    approved: bool
    calls_used: int
    timeout: bool = False
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            object.__setattr__(self, 'details', {})


class VerifierProtocol(Protocol):
    """Protocol for scenario verifiers."""
    def verify(
        self,
        snapshot: ClusterStateSnapshot,
        candidate_placement: Placement,
        K: int
    ) -> VerifierResult:
        ...


class NoCheckVerifier:
    """Runnable baseline verifier that performs no checks and incurs zero cost."""
    def __init__(self):
        self.is_ready = False

    def verify(
        self,
        snapshot: ClusterStateSnapshot,
        candidate_placement: Placement,
        K: int
    ) -> VerifierResult:
        raise RuntimeError("NoCheckVerifier cannot execute a scenario check.")


class BudgetManager:
    """Manages the 300-second window solver call budget B_call and solver invocation logs."""
    def __init__(
        self,
        B_call: int = 2000,
        K: int = 4,
        log_dir: Optional[str] = None,
        ledger: Optional[Any] = None,
        run_id: str = "",
        episode_id: str = ""
    ):
        self.B_call = B_call
        self.K = K
        self.log_dir = log_dir
        self.ledger = ledger
        self.run_id = run_id
        self.episode_id = episode_id
        self.current_window_index: int = -1
        self.current_time: int = 0
        self.remaining_calls: int = B_call
        self.window_invocations: int = 0
        self.total_invocations: int = 0
        self.decision_counter: int = 0
        self.checks_by_decision_id: Dict[int, Dict[str, Any]] = {}
        self.check_requests_log: List[Dict[str, Any]] = []

        self.invocations_file: Optional[Path] = None
        self.started_file: Optional[Path] = None
        self.checks_file: Optional[Path] = None
        if log_dir is not None:
            norm_dir = Path(log_dir)
            if norm_dir.name == "logs" and norm_dir.parent.name == "logs":
                norm_dir = norm_dir.parent
            self.log_dir = str(norm_dir)
            p = norm_dir
            self.invocations_file = p / "solver_invocations.jsonl"
            self.started_file = p / "solver_invocations.started.jsonl"
            self.checks_file = p / "scenario_checks.jsonl"
            p.mkdir(parents=True, exist_ok=True)
            for log_file in (self.invocations_file, self.started_file, self.checks_file):
                with open(log_file, "a", encoding="utf-8"):
                    pass

        self._active_invocations: Dict[int, Dict[str, Any]] = {}
        self._inv_seq = 0
        self._current_check_inv_ids: List[int] = []

    def on_second(self, current_time: int):
        """Called every second; resets budget at 300-second boundaries."""
        self.current_time = current_time
        window_idx = current_time // 300
        if window_idx != self.current_window_index:
            self.current_window_index = window_idx
            self.remaining_calls = self.B_call
            self.window_invocations = 0

    def can_verify(self, is_deferral: bool, verifier_ready: bool) -> bool:
        """A check is legal only if placement is not deferral, verifier is ready, and remaining calls >= 2*K."""
        if is_deferral or not verifier_ready:
            return False
        return self.remaining_calls >= (2 * self.K)

    def get_check_record(self, decision_id: int) -> Optional[Dict[str, Any]]:
        return self.checks_by_decision_id.get(decision_id)

    def record_invocation_before_solve(
        self,
        decision_id: int,
        quota_window: int,
        scenario_id: int,
        scenario_seed: int,
        side: str,
        T_max: float,
        num_workers: int,
        time: Optional[int] = None
    ) -> int:
        """Records an invocation immediately before each solver call, including calls that time out or raise."""
        inv_id = self._inv_seq
        self._inv_seq += 1
        self.total_invocations += 1
        self.window_invocations += 1
        self._current_check_inv_ids.append(inv_id)

        t = time if time is not None else self.current_time

        rec = {
            "schema_version": "2.0",
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "call_id": inv_id,
            "invocation_id": inv_id,
            "decision_id": decision_id,
            "quota_window": quota_window,
            "scenario_id": scenario_id,
            "scenario_seed": scenario_seed,
            "side": side,
            "T_max": T_max,
            "worker_count": num_workers,
            "time": t,
            "status": "STARTED",
            "objective_gpu_seconds": None,
            "objective_gpu_hours": None,
            "wall_time": None,
            "exception": None
        }
        self._active_invocations[inv_id] = rec

        if self.started_file is not None:
            with open(self.started_file, "a") as f:
                f.write(json.dumps(_json_safe(rec)) + "\n")
                f.flush()

        if self.ledger is not None:
            self.ledger.record_solver_invocation(
                time=t,
                call_id=inv_id,
                decision_id=decision_id,
                quota_window=quota_window,
                scenario_id=scenario_id,
                scenario_seed=scenario_seed,
                side=side,
                T_max=T_max,
                worker_count=num_workers
            )

        return inv_id

    def record_invocation_after_solve(
        self,
        inv_id: int,
        status: str,
        objective_gpu_seconds: Optional[int],
        objective_gpu_hours: Optional[float],
        wall_time: float,
        model_build_time: float = 0.0,
        solver_search_time: float = 0.0,
        solver_diagnostics: Optional[Dict[str, Any]] = None,
        objective_bound_gpu_seconds: Optional[float] = None,
        objective_bound_gpu_hours: Optional[float] = None,
        exception: Optional[str] = None
    ):
        rec = self._active_invocations.pop(inv_id, {})
        rec["status"] = status
        rec["objective_gpu_seconds"] = objective_gpu_seconds
        rec["objective_gpu_hours"] = objective_gpu_hours
        rec["wall_time"] = wall_time
        rec["wall_time_seconds"] = wall_time
        rec["model_build_time_seconds"] = round(model_build_time, 6)
        rec["solver_search_time_seconds"] = round(solver_search_time, 6)
        rec["solver_diagnostics"] = solver_diagnostics or {}
        rec["objective_bound_gpu_seconds"] = objective_bound_gpu_seconds
        rec["objective_bound_gpu_hours"] = objective_bound_gpu_hours
        rec["exception"] = exception

        if self.invocations_file is not None:
            with open(self.invocations_file, "a") as f:
                f.write(json.dumps(_json_safe(rec)) + "\n")
                f.flush()

    def record_scenario_check(self, check_record: Dict[str, Any]):
        if self.checks_file is not None:
            with open(self.checks_file, "a") as f:
                f.write(json.dumps(_json_safe(check_record)) + "\n")
                f.flush()

    def execute_check(
        self,
        verifier: VerifierProtocol,
        snapshot: ClusterStateSnapshot,
        placement: Placement,
        job_info: Optional[Dict[str, Any]] = None,
        decision_id: Optional[int] = None
    ) -> Tuple[Placement, int, VerifierResult]:
        reservation = 2 * self.K
        if self.remaining_calls < reservation:
            return Placement.make_deferral(), 0, VerifierResult(approved=False, calls_used=0, timeout=False)

        if decision_id is None:
            decision_id = self.decision_counter
        self.decision_counter += 1

        # Track starting invocations
        start_inv_count = self.total_invocations
        self._current_check_inv_ids.clear()

        # Hold reservation
        self.remaining_calls -= reservation

        try:
            import inspect
            sig = inspect.signature(verifier.verify)
            kwargs = {}
            if "job_info" in sig.parameters:
                kwargs["job_info"] = job_info
            if "budget_manager" in sig.parameters:
                kwargs["budget_manager"] = self
            if "decision_id" in sig.parameters:
                kwargs["decision_id"] = decision_id
            if "quota_window" in sig.parameters:
                kwargs["quota_window"] = self.current_window_index
            if "offered_job_id" in sig.parameters:
                kwargs["offered_job_id"] = (job_info or {}).get("job_name")
            if "offer_ordinal" in sig.parameters:
                kwargs["offer_ordinal"] = int((job_info or {}).get("offer_ordinal", 0))
            if "split_id" in sig.parameters:
                kwargs["split_id"] = str((job_info or {}).get("split_id", "train"))

            result = verifier.verify(snapshot, placement, self.K, **kwargs)
        except Exception as exc:
            # Even on exception, actual invocations made before failure must be charged
            recorded_invocations = self.total_invocations - start_inv_count
            unused = reservation - recorded_invocations
            self.remaining_calls += max(0, unused)
            self._current_check_inv_ids.clear()
            raise exc

        recorded_invocations = self.total_invocations - start_inv_count
        if recorded_invocations > 0:
            actual_calls = recorded_invocations
        else:
            actual_calls = result.calls_used
            self.total_invocations += actual_calls
            self.window_invocations += actual_calls

        # Assertions per Point 6
        if actual_calls > reservation:
            raise ValueError(f"Actual solver invocations ({actual_calls}) exceeded reservation ({reservation}).")
        assert actual_calls <= reservation, f"Actual solver invocations ({actual_calls}) exceeded reservation ({reservation})."
        assert self.window_invocations <= self.B_call, f"Window solver invocations ({self.window_invocations}) exceeded B_call ({self.B_call})."

        unused = reservation - actual_calls
        self.remaining_calls += unused

        executed = placement if result.approved else Placement.make_deferral()

        # Persist and index scenario_checks.jsonl keyed by decision_id
        details = result.details or {}
        check_row = {
            "decision_id": decision_id,
            "quota_window": self.current_window_index,
            "time": snapshot.time,
            "candidate_key": list(placement.key) if hasattr(placement, "key") else None,
            "total_calls": actual_calls,
            "reservation": reservation,
            "released_calls": unused,
            "remaining_calls": self.remaining_calls,
            "invocation_ids": list(self._current_check_inv_ids),
            "all_resolved": details.get("all_resolved", not result.timeout),
            "mean_risk": details.get("mean_risk"),
            "Y_any": details.get("Y_any"),
            "approved": result.approved,
            "latency_seconds": details.get("latency_seconds", 0.0),
            "cache_hit_count": int(details.get("cache_hit_count", 0)),
            "pair_results": [
                {
                    "scenario_id": p.scenario_id,
                    "resolved": p.resolved,
                    "Y": p.Y,
                    "extra_waste_gpu_hours": p.extra_waste_gpu_hours,
                    "solver_calls": p.solver_calls_used,
                    "deferral_cache_hit": bool((p.deferral_result.details or {}).get("cache_hit", False)),
                    "deferral_status": p.deferral_result.status.value if hasattr(p.deferral_result, "status") else None,
                    "deferral_objective_gpu_seconds": p.deferral_result.objective_gpu_seconds if hasattr(p.deferral_result, "objective_gpu_seconds") else None,
                    "placement_status": p.placement_result.status.value if (p.placement_result and hasattr(p.placement_result, "status")) else None,
                    "placement_objective_gpu_seconds": p.placement_result.objective_gpu_seconds if (p.placement_result and hasattr(p.placement_result, "objective_gpu_seconds")) else None,
                }
                if hasattr(p, "scenario_id") else p
                for p in details.get("pair_results", [])
            ]
        }
        self.checks_by_decision_id[decision_id] = check_row
        self.record_scenario_check(check_row)

        request_log_entry = {
            "decision_id": decision_id,
            "time": snapshot.time,
            "quota_window": self.current_window_index,
            "reservation": reservation,
            "actual_calls": actual_calls,
            "released_calls": unused,
            "remaining_calls": self.remaining_calls,
            "invocation_ids": list(self._current_check_inv_ids),
            "window_invocations": self.window_invocations,
            "B_call": self.B_call
        }
        self.check_requests_log.append(request_log_entry)
        self._current_check_inv_ids.clear()

        # Update details in result to provide structured decision_id result
        result_details = dict(details)
        result_details["decision_id"] = decision_id
        result_details["check_outcome"] = check_row
        result_details["Y_any"] = check_row["Y_any"]
        result = VerifierResult(
            approved=result.approved,
            calls_used=actual_calls,
            timeout=result.timeout,
            details=result_details
        )

        return executed, actual_calls, result


