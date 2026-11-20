"""Deterministic seed derivation for paired benchmark runs."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


DETERMINISTIC_SEED_DERIVATION = "master_seed_identity_v1"
STOCHASTIC_SEED_DERIVATION = "numpy_seedsequence_spawn_uint32_v1"


def derive_run_seeds(master_seed: int, count: int) -> tuple[int, ...]:
    """Derive independent uint32 seeds without storing the generated values."""
    if isinstance(master_seed, bool) or int(master_seed) != master_seed or master_seed < 0:
        raise ValueError("master_seed must be a nonnegative integer.")
    if isinstance(count, bool) or int(count) != count or count < 1:
        raise ValueError("seed count must be a positive integer.")
    children = np.random.SeedSequence(int(master_seed)).spawn(int(count))
    return tuple(
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in children
    )


def deterministic_seed_from_suite(suite: Mapping[str, Any]) -> int:
    """Return the single provenance seed used by deterministic baselines."""
    seed_cfg = suite.get("randomness")
    if not isinstance(seed_cfg, Mapping):
        raise ValueError("Benchmark suite requires a randomness mapping.")
    if seed_cfg.get("deterministic_seed_derivation") != DETERMINISTIC_SEED_DERIVATION:
        raise ValueError(
            "Unsupported deterministic seed derivation "
            f"{seed_cfg.get('deterministic_seed_derivation')!r}; "
            f"expected {DETERMINISTIC_SEED_DERIVATION!r}."
        )
    return int(seed_cfg["master_seed"])


def stochastic_seeds_from_suite(suite: Mapping[str, Any]) -> tuple[int, ...]:
    """Generate seeds reserved for learned or otherwise stochastic methods."""
    seed_cfg = suite.get("randomness")
    if not isinstance(seed_cfg, Mapping):
        raise ValueError("Benchmark suite requires a randomness mapping.")
    if seed_cfg.get("stochastic_seed_derivation") != STOCHASTIC_SEED_DERIVATION:
        raise ValueError(
            "Unsupported stochastic seed derivation "
            f"{seed_cfg.get('stochastic_seed_derivation')!r}; "
            f"expected {STOCHASTIC_SEED_DERIVATION!r}."
        )
    return derive_run_seeds(
        int(seed_cfg["master_seed"]),
        int(seed_cfg["stochastic_seed_count"]),
    )
