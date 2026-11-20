from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from benchmark.forecast_data import SnapshotHistory, align_prediction, build_origin_data
from methods.forecasters.advanced import _NLinear, _PatchTST, _TemporalMamba
from methods.forecasters.models import DLinearForecaster, SeasonalNaiveForecaster
from methods.forecasters.srlinear import (
    AdaptiveSeasonalResidualLinearForecaster,
    _AdaptiveSeasonalResidualNet,
    _peak_tail_composite_loss,
    _seasonal_anchors,
)
from methods.forecasters.srlinear_components import _build_inference_design


def _minute_frame(days: int = 20) -> pd.DataFrame:
    index = pd.date_range("2024-03-01 20:00:00", periods=days * 24 * 60, freq="1min")
    hour = np.arange(len(index)) // 60
    return pd.DataFrame(
        {
            "A100&&org_a": (hour % 24).astype(float),
            "A100&&org_b": ((hour + 3) % 12).astype(float),
        },
        index=index,
    )


def test_origin_data_is_causal_and_alignment_handles_new_organizations():
    frame = _minute_frame()
    frame["A100&&future_org"] = 0.0
    frame.loc[frame.index[-120:], "A100&&future_org"] = 4.0
    origin = frame.index[-180]
    source = SnapshotHistory(frame, frame.index[0], 128.0, {}, "hash")
    data = build_origin_data(source, origin=origin, horizon=2, minimum_history_hours=24)
    assert "A100&&future_org" not in data.causal_history.columns
    assert data.causal_history.index.max() < origin
    aligned = align_prediction(
        prediction=np.asarray([[1.0], [2.0]]),
        prediction_columns=("known",),
        target_columns=("known", "new"),
    )
    np.testing.assert_array_equal(aligned[:, 1], np.zeros(2))


def test_seasonal_naive_uses_exact_weekly_lag(tmp_path: Path):
    frame = _minute_frame()
    origin = frame.index[-240]
    history = frame.loc[frame.index < origin]
    output = SeasonalNaiveForecaster(horizon=4, seasonal_lag=168).fit_predict(
        history, origin, tmp_path
    )
    expected = history.resample("1h").max().iloc[-168:-164].to_numpy(dtype=np.float32)
    np.testing.assert_array_equal(output.point, expected)


def test_dlinear_smoke_is_deterministic(tmp_path: Path):
    frame = _minute_frame(days=8)
    origin = frame.index[-1] + pd.Timedelta(minutes=1)

    def run(path: Path) -> np.ndarray:
        model = DLinearForecaster(
            lookback=24,
            horizon=2,
            moving_average=5,
            epochs=1,
            batch_size=64,
            learning_rate=0.001,
            weight_decay=0.0,
            validation_hours=24,
            sample_stride_hours=2,
            patience=1,
            device=torch.device("cpu"),
            seed=17,
            num_workers=0,
        )
        return model.fit_predict(frame, origin, path).point

    np.testing.assert_array_equal(run(tmp_path / "a"), run(tmp_path / "b"))


def test_nlinear_matches_official_last_value_normalization():
    model = _NLinear(lookback=4, horizon=2)
    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias.zero_()
    values = torch.tensor([[1.0, 3.0, 2.0, 7.0]])
    torch.testing.assert_close(model(values), torch.tensor([[7.0, 7.0]]))


def test_srlinear_features_and_anchors_are_causal():
    values = np.arange(800 * 2, dtype=np.float64).reshape(800, 2)
    features, base = _build_inference_design(
        values,
        origin=pd.Timestamp("2024-05-01 20:00:00"),
        calendar_timestamp=pd.Timestamp("2024-05-02 00:00:00"),
        horizon_step=3,
        periods=(24, 48, 168, 336, 672),
        base_period=168,
        recent_windows=(4, 12, 24, 72, 168),
        residual_lookback=48,
    )
    repeated, repeated_base = _build_inference_design(
        np.vstack((values, np.full((4, 2), 1e9)))[:800],
        origin=pd.Timestamp("2024-05-01 20:00:00"),
        calendar_timestamp=pd.Timestamp("2024-05-02 00:00:00"),
        horizon_step=3,
        periods=(24, 48, 168, 336, 672),
        base_period=168,
        recent_windows=(4, 12, 24, 72, 168),
        residual_lookback=48,
    )
    np.testing.assert_array_equal(features, repeated)
    np.testing.assert_array_equal(base, repeated_base)
    anchors = _seasonal_anchors(
        values[:30], np.asarray([20, 21]), horizon_step=1, periods=(4, 8)
    )
    assert anchors.shape == (2, 2, 2)


def test_srlinear_gate_is_convex_and_reconciliation_is_exact():
    model = _AdaptiveSeasonalResidualNet(
        horizon=2,
        feature_count=3,
        period_count=2,
        base_period_index=1,
        base_gate_logit=4.0,
        organization_coefficients=np.zeros((2, 3)),
        organization_intercepts=np.zeros(2),
        pool_coefficients=np.zeros((2, 3)),
        pool_intercepts=np.zeros(2),
        blend_weights=np.asarray([0.5, 0.5]),
        point_bias_offsets=np.zeros(2),
        means=np.asarray([2.0, 3.0]),
        scales=np.ones(2),
        pool_mean=5.0,
        pool_scale=1.0,
        fallback_shares=np.asarray([1.0, 2.0]),
        seed=17,
    )
    organization, pool, gate = model(
        torch.zeros((4, 2, 2, 3)),
        torch.zeros((4, 2, 3)),
        torch.zeros((4, 2, 2, 2)),
        torch.zeros((4, 2, 2)),
    )
    torch.testing.assert_close(gate.sum(dim=-1), torch.ones((4, 2)))
    torch.testing.assert_close(organization.sum(dim=-1), pool)


def test_srlinear_loss_penalizes_peak_underprediction():
    target = torch.tensor([[2.0, 2.0], [10.0, 10.0]])
    thresholds = torch.tensor([8.0, 8.0])
    loss, components = _peak_tail_composite_loss(
        torch.tensor([[1.0, -1.0], [-4.0, -4.0]]),
        target,
        thresholds,
        rmse_weight=0.25,
        tail_weight=0.10,
        tail_fraction=0.25,
        peak_weight=0.10,
        underprediction_weight=0.10,
    )
    assert torch.isfinite(loss)
    assert components["peak_underprediction"] > 0.0


def test_srlinear_smoke_is_deterministic_and_coherent(tmp_path: Path):
    frame = _minute_frame(days=40)
    origin = frame.index[-1] + pd.Timedelta(minutes=1)

    def run(path: Path) -> np.ndarray:
        model = AdaptiveSeasonalResidualLinearForecaster(
            horizon=2,
            periods=(24, 48, 168, 336, 672),
            base_period=168,
            recent_windows=(4, 12, 24, 72, 168),
            residual_lookback=48,
            validation_hours=48,
            validation_cadence_hours=24,
            tuning_validation_cadence_hours=6,
            sample_stride_hours=2,
            ridge_alphas=(0.001, 0.01),
            reconciliation_grid=(0.0, 0.5, 1.0),
            epochs=2,
            learning_rate=0.001,
            weight_decay=0.0001,
            patience=1,
            organization_loss_weight=0.15,
            gate_stability_weight=0.0001,
            loss_rmse_weight=0.25,
            loss_tail_weight=0.10,
            loss_tail_fraction=0.10,
            loss_peak_weight=0.10,
            loss_underprediction_weight=0.10,
            loss_peak_quantile=0.90,
            validation_rmse_weight=0.50,
            base_gate_logit=8.0,
            gradient_clip=1.0,
            device=torch.device("cpu"),
            seed=17,
        )
        return model.fit_predict(frame, origin, path).point

    first = run(tmp_path / "a")
    second = run(tmp_path / "b")
    np.testing.assert_array_equal(first, second)
    assert np.all(first >= 0.0)


def test_patchtst_forward_is_finite():
    model = _PatchTST(
        lookback=24,
        horizon=3,
        patch_length=8,
        patch_stride=4,
        model_width=16,
        heads=4,
        layers=1,
        feedforward_width=32,
        dropout=0.0,
    )
    output = model(torch.arange(48, dtype=torch.float32).reshape(2, 24))
    assert output.shape == (2, 3)
    assert torch.isfinite(output).all()


def test_temporal_mamba_uses_official_package():
    pytest.importorskip("mamba_ssm")
    model = _TemporalMamba(
        lookback=24,
        horizon=3,
        patch_length=4,
        patch_stride=4,
        model_width=16,
        state_width=4,
        convolution_width=2,
        expansion=1,
        layers=1,
        dropout=0.0,
    )
    inputs = torch.randn(2, 24, requires_grad=True)
    output = model(inputs)
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    assert inputs.grad is not None
