import json
from pathlib import Path

import numpy as np
import torch

from benchmark.measure_forecast_efficiency import (
    count_parameters,
    patchtst_flops,
    profile_flops,
    temporal_mamba_flops,
)
from methods.forecasters.advanced import _NLinear, _PatchTST
from methods.forecasters.models import _DLinear, _OrgLinearR, orglinear_args
from methods.forecasters.srlinear import _AdaptiveSeasonalResidualNet
from research_sim.gfs.estimator.model.OrgLinear import Model as ReleasedOrgLinear


def test_fixed_protocol_parameter_and_flop_counts(tmp_path: Path) -> None:
    organizations = 28
    lookback = 672
    horizon = 4
    shared_input = torch.zeros(organizations, lookback)

    nlinear = _NLinear(lookback, horizon)
    assert count_parameters(nlinear) == 2_692
    assert profile_flops(nlinear, (shared_input,)) == 150_528

    dlinear = _DLinear(lookback, horizon, 25)
    assert count_parameters(dlinear) == 5_384
    assert profile_flops(dlinear, (shared_input,)) == 301_056

    patchtst = _PatchTST(
        lookback=lookback,
        horizon=horizon,
        patch_length=16,
        patch_stride=8,
        model_width=64,
        heads=4,
        layers=2,
        feedforward_width=128,
        dropout=0.1,
    )
    assert count_parameters(patchtst) == 94_724
    assert patchtst_flops(
        organizations=organizations,
        lookback=lookback,
        horizon=horizon,
        patch_length=16,
        patch_stride=8,
        model_width=64,
        layers=2,
        feedforward_width=128,
    ) == 409_321_472

    args = orglinear_args(
        checkpoint_dir=tmp_path,
        use_gpu=False,
        gpu=0,
        lookback=lookback,
        horizon=horizon,
        epochs=10,
        batch_size=32,
        learning_rate=0.00004,
        num_workers=0,
    )
    orglinear = ReleasedOrgLinear(args)
    orglinear_inputs = (
        torch.zeros(1, lookback, organizations),
        torch.zeros(1, organizations, 3, dtype=torch.long),
        torch.zeros(1, 3, organizations, dtype=torch.long),
    )
    assert count_parameters(orglinear) == 10_004
    assert profile_flops(orglinear, orglinear_inputs) == 549_696
    orglinear_r = _OrgLinearR(args)
    assert count_parameters(orglinear_r) == 10_004
    assert profile_flops(orglinear_r, orglinear_inputs) == 549_696

    feature_count = 72
    period_count = 5
    srlinear = _AdaptiveSeasonalResidualNet(
        horizon=horizon,
        feature_count=feature_count,
        period_count=period_count,
        base_period_index=2,
        base_gate_logit=8.0,
        organization_coefficients=np.zeros((horizon, feature_count)),
        organization_intercepts=np.zeros(horizon),
        pool_coefficients=np.zeros((horizon, feature_count)),
        pool_intercepts=np.zeros(horizon),
        blend_weights=np.full(horizon, 0.5),
        point_bias_offsets=np.zeros(horizon),
        means=np.zeros(organizations),
        scales=np.ones(organizations),
        pool_mean=0.0,
        pool_scale=1.0,
        fallback_shares=np.ones(organizations),
        seed=17,
    )
    srlinear_inputs = (
        torch.zeros(1, horizon, organizations, feature_count),
        torch.zeros(1, horizon, feature_count),
        torch.zeros(1, horizon, organizations, period_count),
        torch.zeros(1, horizon, period_count),
    )
    assert count_parameters(srlinear) == 2_052
    assert profile_flops(srlinear, srlinear_inputs) == 20_744

    assert temporal_mamba_flops(
        organizations=organizations,
        lookback=lookback,
        horizon=horizon,
        patch_length=24,
        patch_stride=24,
        model_width=32,
        state_width=8,
        convolution_width=4,
        expansion=2,
    ) == 17_110_016

