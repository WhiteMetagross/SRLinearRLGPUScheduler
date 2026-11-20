"""Process vectorization and central tensor batching for version 4 control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import numpy as np

from .parallel import configure_worker_threads, plan_parallelism


@dataclass(frozen=True)
class ThreadLimitedEnvFactory:
    """Pickle-safe wrapper that limits numerical threads before env creation."""

    env_fn: Callable[[], gym.Env]
    threads_per_worker: int = 1

    def __call__(self) -> gym.Env:
        configure_worker_threads(self.threads_per_worker)
        return self.env_fn()


def make_async_control_vector_env(
    env_fns: Sequence[Callable[[], gym.Env]],
    *,
    threads_per_worker: int = 1,
    start_method: str = "spawn",
    shared_memory: bool = True,
) -> AsyncVectorEnv:
    """Create one sequential simulator per CPU process.

    The caller owns seed and window selection. Policy inference stays in the
    parent process so one batched tensor call can use a GPU.
    """
    factories = tuple(env_fns)
    if not factories:
        raise ValueError("env_fns must contain at least one environment factory.")
    if start_method not in {"spawn", "forkserver"}:
        raise ValueError("start_method must be 'spawn' or 'forkserver'.")
    plan = plan_parallelism(
        len(factories),
        threads_per_worker=threads_per_worker,
    )
    if plan.worker_count != len(factories):
        raise ValueError(
            f"Requested {len(factories)} vector workers, but only "
            f"{plan.worker_count} fit the CPU thread limit."
        )
    wrapped = tuple(
        ThreadLimitedEnvFactory(factory, threads_per_worker)
        for factory in factories
    )
    return AsyncVectorEnv(
        wrapped,
        shared_memory=shared_memory,
        context=start_method,
        daemon=True,
    )


def resolve_policy_device(requested: str = "auto") -> Any:
    """Resolve the central policy device without silently accepting no CUDA."""
    import torch

    value = str(requested).lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but this PyTorch build cannot use CUDA.")
        return torch.device("cuda")
    raise ValueError("requested must be one of: auto, cpu, cuda.")


def batch_observations_to_device(
    observations: Mapping[str, np.ndarray],
    device: Any,
) -> dict[str, Any]:
    """Transfer a complete vector observation batch once per policy step."""
    import torch

    target = torch.device(device)
    tensors: dict[str, Any] = {}
    for key, value in observations.items():
        array = np.asarray(value)
        tensor = torch.as_tensor(array)
        if array.dtype != np.bool_:
            tensor = tensor.to(dtype=torch.float32)
        tensors[key] = tensor.to(
            device=target,
            non_blocking=target.type == "cuda",
        )
    return tensors
