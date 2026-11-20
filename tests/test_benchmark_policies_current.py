from types import SimpleNamespace

import numpy as np

from benchmark.policies import FixedActionPolicy, LinearResidualPolicy


def _context() -> SimpleNamespace:
    return SimpleNamespace(
        time=0,
        capacity_by_pool={"A100": 100.0},
        hp_used_by_pool={"A100": 40.0},
        spot_used_by_pool={"A100": 20.0},
        base_quota_by_pool={"A100": {1: 50.0}},
        hp_queue_gpu_demand=0.0,
        admission_jobs=(),
    )


def test_fixed_action_policy_returns_an_independent_copy():
    expected = np.asarray([0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float32)
    policy = FixedActionPolicy(expected)
    first = policy.action(_context())
    first[0] = 1.0
    np.testing.assert_array_equal(policy.action(_context()), expected)


def test_contextual_linear_changes_only_quota_component():
    policy = LinearResidualPolicy(
        coefficients=(0.0,) * 9,
        intercept=0.04,
        feature_mean=(0.0,) * 9,
        feature_scale=(1.0,) * 9,
        safety_offset=-0.01,
    )
    action = policy.action(_context())
    assert action[0] == np.float32(0.03)
    np.testing.assert_array_equal(action[1:], np.zeros(6, dtype=np.float32))
