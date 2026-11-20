"""Deterministic process parallelism helpers for independent simulator tasks."""

from __future__ import annotations

import concurrent.futures
import multiprocessing as mp
import os
import sys
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Sequence, TypeVar


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class ParallelismPlan:
    """Resource bounded process plan with no nested CPU oversubscription."""

    requested_workers: int
    worker_count: int
    available_cpus: int
    threads_per_worker: int
    memory_limit_gib: float | None
    estimated_worker_gib: float | None
    limiting_factors: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def plan_parallelism(
    requested_workers: int | str,
    *,
    available_cpus: int | None = None,
    threads_per_worker: int = 1,
    memory_limit_gib: float | None = None,
    estimated_worker_gib: float | None = None,
) -> ParallelismPlan:
    """Choose a safe worker count from CPU and optional memory limits."""
    cpus = int(os.cpu_count() or 1) if available_cpus is None else int(available_cpus)
    if cpus < 1:
        raise ValueError("available_cpus must be positive.")
    threads = int(threads_per_worker)
    if threads < 1:
        raise ValueError("threads_per_worker must be positive.")
    cpu_cap = max(1, cpus // threads)
    if isinstance(requested_workers, str):
        if requested_workers.lower() != "auto":
            raise ValueError("requested_workers must be a positive integer or 'auto'.")
        requested = cpu_cap
    else:
        requested = int(requested_workers)
        if requested < 1:
            raise ValueError("requested_workers must be positive.")

    caps = [(requested, "requested"), (cpu_cap, "cpu")]
    if memory_limit_gib is not None or estimated_worker_gib is not None:
        if memory_limit_gib is None or estimated_worker_gib is None:
            raise ValueError(
                "memory_limit_gib and estimated_worker_gib must be set together."
            )
        if memory_limit_gib <= 0 or estimated_worker_gib <= 0:
            raise ValueError("Memory limits must be positive.")
        memory_cap = int(memory_limit_gib // estimated_worker_gib)
        if memory_cap < 1:
            raise ValueError("memory_limit_gib cannot fit one estimated worker.")
        caps.append((memory_cap, "memory"))

    workers = min(value for value, _ in caps)
    limiting = tuple(name for value, name in caps if value == workers)
    return ParallelismPlan(
        requested_workers=requested,
        worker_count=workers,
        available_cpus=cpus,
        threads_per_worker=threads,
        memory_limit_gib=memory_limit_gib,
        estimated_worker_gib=estimated_worker_gib,
        limiting_factors=limiting,
    )


def configure_worker_threads(threads_per_worker: int = 1) -> None:
    """Limit numerical libraries inside one simulator worker process."""
    threads = int(threads_per_worker)
    if threads < 1:
        raise ValueError("threads_per_worker must be positive.")
    value = str(threads)
    for name in _THREAD_ENV_VARS:
        os.environ[name] = value
    torch_module = sys.modules.get("torch")
    if torch_module is not None:
        torch_module.set_num_threads(threads)


def ordered_process_map(
    function: Callable[[InputT], OutputT],
    items: Iterable[InputT],
    *,
    requested_workers: int | str = "auto",
    available_cpus: int | None = None,
    threads_per_worker: int = 1,
    start_method: str = "spawn",
    chunksize: int = 1,
) -> list[OutputT]:
    """Map independent tasks in input order and propagate worker failures."""
    if start_method not in {"spawn", "forkserver"}:
        raise ValueError("start_method must be 'spawn' or 'forkserver'.")
    if chunksize < 1:
        raise ValueError("chunksize must be positive.")
    task_items: Sequence[InputT] = tuple(items)
    if not task_items:
        return []
    plan = plan_parallelism(
        requested_workers,
        available_cpus=available_cpus,
        threads_per_worker=threads_per_worker,
    )
    if plan.worker_count == 1:
        configure_worker_threads(threads_per_worker)
        return [function(item) for item in task_items]
    context = mp.get_context(start_method)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=plan.worker_count,
        mp_context=context,
        initializer=configure_worker_threads,
        initargs=(threads_per_worker,),
    ) as executor:
        return list(executor.map(function, task_items, chunksize=chunksize))
