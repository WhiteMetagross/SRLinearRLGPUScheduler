"""Learnable adaptive seasonal residual linear forecaster.

SRLinear preserves the causal feature construction and ridge initialization
from its deterministic component model. It adds one small horizon specific convex gate over seasonal
anchors, then jointly fine tunes the gate, organization residual head, direct
aggregate residual head, and coherent aggregate blend.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

from .models import ForecastOutput, hourly_organization_maximum, set_forecast_seed
from .srlinear_components import (
    _build_design,
    _build_inference_design,
    _positions,
    _select_head,
    _validate_structure,
)


def _peak_tail_composite_loss(
    error: torch.Tensor,
    target: torch.Tensor,
    peak_thresholds: torch.Tensor,
    *,
    rmse_weight: float,
    tail_weight: float,
    tail_fraction: float,
    peak_weight: float,
    underprediction_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return scale-free point loss and auditable component values.

    ``error`` must be prediction minus target after causal train-only scaling.
    Peak thresholds remain in target units and are estimated from train targets.
    """

    if error.shape != target.shape:
        raise ValueError("SRLinear loss error and target shapes must match.")
    if error.ndim != 2:
        raise ValueError("SRLinear pool loss requires [sample, horizon] tensors.")
    if peak_thresholds.shape != (error.shape[1],):
        raise ValueError("SRLinear peak thresholds require one value per horizon.")
    if not 0.0 < float(tail_fraction) <= 1.0:
        raise ValueError("SRLinear tail fraction must be in (0, 1].")
    weights = (
        float(rmse_weight),
        float(tail_weight),
        float(peak_weight),
        float(underprediction_weight),
    )
    if any(weight < 0.0 for weight in weights):
        raise ValueError("SRLinear composite loss weights must be nonnegative.")

    absolute = error.abs()
    mae = absolute.mean()
    rmse = torch.sqrt(error.square().mean() + 1e-12) - 1e-6
    tail_count = max(1, int(np.ceil(absolute.numel() * float(tail_fraction))))
    tail = torch.topk(absolute.reshape(-1), k=tail_count, sorted=False).values.mean()
    peak_mask = (target >= peak_thresholds[None, :]).to(error.dtype)
    peak_denominator = peak_mask.sum().clamp_min(1.0)
    peak_absolute = (peak_mask * absolute).sum() / peak_denominator
    peak_underprediction = (
        peak_mask * torch.relu(-error)
    ).sum() / peak_denominator
    components = {
        "mae": mae,
        "rmse": rmse,
        "tail": tail,
        "peak_absolute": peak_absolute,
        "peak_underprediction": peak_underprediction,
    }
    loss = (
        mae
        + float(rmse_weight) * rmse
        + float(tail_weight) * tail
        + float(peak_weight) * peak_absolute
        + float(underprediction_weight) * peak_underprediction
    )
    return loss, components


def _seasonal_anchors(
    values: np.ndarray,
    positions: np.ndarray,
    *,
    horizon_step: int,
    periods: tuple[int, ...],
) -> np.ndarray:
    """Return causal anchors with shape [position, channel, period]."""

    values = np.asarray(values, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError("SRLinear values require shape [time, channel].")
    indices = np.stack(
        [positions + int(horizon_step) - int(period) for period in periods],
        axis=-1,
    )
    if indices.min() < 0 or indices.max() >= len(values):
        raise ValueError("SRLinear seasonal anchor crosses causal history.")
    channel_indices = np.arange(values.shape[1], dtype=np.int64)[None, :, None]
    return values[indices[:, None, :], channel_indices]


class _AdaptiveSeasonalResidualNet(nn.Module):
    """Small coherent mixture initialized from fitted ridge heads."""

    def __init__(
        self,
        *,
        horizon: int,
        feature_count: int,
        period_count: int,
        base_period_index: int,
        base_gate_logit: float,
        organization_coefficients: np.ndarray,
        organization_intercepts: np.ndarray,
        pool_coefficients: np.ndarray,
        pool_intercepts: np.ndarray,
        blend_weights: np.ndarray,
        point_bias_offsets: np.ndarray,
        means: np.ndarray,
        scales: np.ndarray,
        pool_mean: float,
        pool_scale: float,
        fallback_shares: np.ndarray,
        seed: int,
    ):
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        gate_weight = torch.randn(
            int(horizon), int(period_count), int(feature_count), generator=generator
        ) * 1e-3
        gate_bias = torch.zeros(int(horizon), int(period_count))
        gate_bias[:, int(base_period_index)] = float(base_gate_logit)
        self.gate_weight = nn.Parameter(gate_weight)
        self.gate_bias = nn.Parameter(gate_bias)
        self.organization_weight = nn.Parameter(
            torch.as_tensor(organization_coefficients, dtype=torch.float32).clone()
        )
        self.organization_bias = nn.Parameter(
            torch.as_tensor(organization_intercepts, dtype=torch.float32).clone()
        )
        self.pool_weight = nn.Parameter(
            torch.as_tensor(pool_coefficients, dtype=torch.float32).clone()
        )
        self.pool_bias = nn.Parameter(
            torch.as_tensor(pool_intercepts, dtype=torch.float32).clone()
        )
        clipped_blend = np.clip(np.asarray(blend_weights), 1e-4, 1.0 - 1e-4)
        self.blend_logit = nn.Parameter(
            torch.as_tensor(
                np.log(clipped_blend / (1.0 - clipped_blend)),
                dtype=torch.float32,
            )
        )
        self.point_bias = nn.Parameter(
            torch.as_tensor(point_bias_offsets, dtype=torch.float32).clone()
        )
        self.register_buffer("means", torch.as_tensor(means, dtype=torch.float32))
        self.register_buffer("scales", torch.as_tensor(scales, dtype=torch.float32))
        self.register_buffer(
            "pool_mean", torch.as_tensor(float(pool_mean), dtype=torch.float32)
        )
        self.register_buffer(
            "pool_scale", torch.as_tensor(float(pool_scale), dtype=torch.float32)
        )
        shares = np.clip(np.asarray(fallback_shares, dtype=np.float64), 0.0, None)
        if float(shares.sum()) <= 1e-12:
            shares = np.full(len(shares), 1.0 / len(shares), dtype=np.float64)
        else:
            shares = shares / shares.sum()
        self.register_buffer(
            "fallback_shares", torch.as_tensor(shares, dtype=torch.float32)
        )

    def forward(
        self,
        organization_features: torch.Tensor,
        pool_features: torch.Tensor,
        organization_anchors: torch.Tensor,
        pool_anchors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return reconciled organizations, aggregate point, and gate weights."""

        gate_logits = torch.einsum(
            "phf,hkf->phk", pool_features, self.gate_weight
        ) + self.gate_bias[None, :, :]
        gate = torch.softmax(gate_logits, dim=-1)
        organization_base = torch.einsum("phk,phck->phc", gate, organization_anchors)
        pool_base = torch.einsum("phk,phk->ph", gate, pool_anchors)
        organization_residual = torch.einsum(
            "phcf,hf->phc", organization_features, self.organization_weight
        ) + self.organization_bias[None, :, None]
        pool_residual = torch.einsum(
            "phf,hf->ph", pool_features, self.pool_weight
        ) + self.pool_bias[None, :]
        organization_raw = torch.relu(
            (organization_base + organization_residual)
            * self.scales[None, None, :]
            + self.means[None, None, :]
        )
        direct_pool = torch.relu(
            (pool_base + pool_residual) * self.pool_scale + self.pool_mean
        )
        organization_pool = organization_raw.sum(dim=-1)
        blend = torch.sigmoid(self.blend_logit)[None, :]
        final_pool = torch.relu(
            (1.0 - blend) * organization_pool
            + blend * direct_pool
            + self.point_bias[None, :]
        )
        denominator = organization_pool.unsqueeze(-1)
        proportional = organization_raw * final_pool.unsqueeze(-1) / denominator.clamp_min(1e-8)
        fallback = final_pool.unsqueeze(-1) * self.fallback_shares[None, None, :]
        reconciled = torch.where(denominator > 1e-8, proportional, fallback)
        return reconciled, final_pool, gate


class AdaptiveSeasonalResidualLinearForecaster:
    """Learnable multiseasonal linear model with exact aggregate coherence."""

    model_id = "srlinear_learnable_reconciled_v1"
    stochastic = True

    def __init__(
        self,
        *,
        horizon: int,
        periods: Iterable[int],
        base_period: int,
        recent_windows: Iterable[int],
        residual_lookback: int,
        validation_hours: int,
        validation_cadence_hours: int,
        tuning_validation_cadence_hours: int,
        sample_stride_hours: int,
        ridge_alphas: Iterable[float],
        reconciliation_grid: Iterable[float],
        epochs: int,
        learning_rate: float,
        weight_decay: float,
        patience: int,
        organization_loss_weight: float,
        gate_stability_weight: float,
        loss_rmse_weight: float,
        loss_tail_weight: float,
        loss_tail_fraction: float,
        loss_peak_weight: float,
        loss_underprediction_weight: float,
        loss_peak_quantile: float,
        validation_rmse_weight: float,
        base_gate_logit: float,
        gradient_clip: float,
        device: torch.device,
        seed: int,
    ):
        self.horizon = int(horizon)
        self.periods = tuple(sorted({int(value) for value in periods}))
        self.base_period = int(base_period)
        self.recent_windows = tuple(sorted({int(value) for value in recent_windows}))
        self.residual_lookback = int(residual_lookback)
        self.validation_hours = int(validation_hours)
        self.validation_cadence_hours = int(validation_cadence_hours)
        self.tuning_validation_cadence_hours = int(tuning_validation_cadence_hours)
        self.sample_stride_hours = int(sample_stride_hours)
        self.ridge_alphas = tuple(sorted({float(value) for value in ridge_alphas}))
        self.reconciliation_grid = tuple(sorted({float(value) for value in reconciliation_grid}))
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.patience = int(patience)
        self.organization_loss_weight = float(organization_loss_weight)
        self.gate_stability_weight = float(gate_stability_weight)
        self.loss_rmse_weight = float(loss_rmse_weight)
        self.loss_tail_weight = float(loss_tail_weight)
        self.loss_tail_fraction = float(loss_tail_fraction)
        self.loss_peak_weight = float(loss_peak_weight)
        self.loss_underprediction_weight = float(loss_underprediction_weight)
        self.loss_peak_quantile = float(loss_peak_quantile)
        self.validation_rmse_weight = float(validation_rmse_weight)
        self.base_gate_logit = float(base_gate_logit)
        self.gradient_clip = float(gradient_clip)
        self.device = torch.device(device)
        self.seed = int(seed)
        _validate_structure(
            horizon=self.horizon,
            periods=self.periods,
            base_period=self.base_period,
            recent_windows=self.recent_windows,
        )
        if self.epochs <= 0 or self.patience <= 0:
            raise ValueError("SRLinear epochs and patience must be positive.")
        if self.validation_cadence_hours <= 0 or self.tuning_validation_cadence_hours <= 0:
            raise ValueError("SRLinear validation cadences must be positive.")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("SRLinear optimizer settings are invalid.")
        if self.organization_loss_weight < 0.0 or self.gate_stability_weight < 0.0:
            raise ValueError("SRLinear loss weights must be nonnegative.")
        if any(
            weight < 0.0
            for weight in (
                self.loss_rmse_weight,
                self.loss_tail_weight,
                self.loss_peak_weight,
                self.loss_underprediction_weight,
            )
        ):
            raise ValueError("SRLinear composite loss weights must be nonnegative.")
        if not 0.0 < self.loss_tail_fraction <= 1.0:
            raise ValueError("SRLinear tail fraction must be in (0, 1].")
        if not 0.0 < self.loss_peak_quantile < 1.0:
            raise ValueError("SRLinear peak quantile must be in (0, 1).")
        if not 0.0 <= self.validation_rmse_weight <= 1.0:
            raise ValueError("SRLinear validation RMSE weight must be in [0, 1].")
        if self.gradient_clip <= 0.0:
            raise ValueError("SRLinear gradient clip must be positive.")

    @staticmethod
    def _tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=device)

    def fit_predict(
        self,
        history: pd.DataFrame,
        origin: pd.Timestamp,
        run_dir: Path,
        calendar_timestamp: pd.Timestamp | None = None,
    ) -> ForecastOutput:
        set_forecast_seed(self.seed)
        hourly = hourly_organization_maximum(history, origin=origin)
        raw = hourly.to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(raw)):
            raise ValueError("SRLinear history contains nonfinite values.")
        calendar_timestamp = pd.Timestamp(calendar_timestamp or origin)
        calendar_offset = calendar_timestamp - pd.Timestamp(origin)
        history_required = max(max(self.periods), max(self.recent_windows), self.residual_lookback)
        fit_end = len(hourly) - self.validation_hours
        if fit_end <= history_required + self.horizon:
            raise ValueError("Insufficient history for SRLinear train and validation windows.")

        means = raw[:fit_end].mean(axis=0)
        scales = raw[:fit_end].std(axis=0)
        scales = np.where(scales > 1e-6, scales, 1.0)
        standardized = (raw - means) / scales
        raw_pool = raw.sum(axis=1, keepdims=True)
        pool_mean = raw_pool[:fit_end].mean(axis=0)
        pool_scale = raw_pool[:fit_end].std(axis=0)
        pool_scale = np.where(pool_scale > 1e-6, pool_scale, 1.0)
        standardized_pool = (raw_pool - pool_mean) / pool_scale
        train_positions = _positions(
            length=fit_end,
            history_required=history_required,
            horizon=self.horizon,
            stride=self.sample_stride_hours,
        )
        initialization_positions = _positions(
            length=len(hourly),
            history_required=history_required,
            horizon=self.horizon,
            stride=self.validation_cadence_hours,
            start=fit_end,
        )
        validation_positions = _positions(
            length=len(hourly),
            history_required=history_required,
            horizon=self.horizon,
            stride=self.tuning_validation_cadence_hours,
            start=fit_end,
        )
        if (
            train_positions.size == 0
            or initialization_positions.size == 0
            or validation_positions.size == 0
        ):
            raise ValueError("SRLinear train or validation design is empty.")

        train_organization_features = []
        validation_organization_features = []
        train_pool_features = []
        validation_pool_features = []
        train_organization_anchors = []
        validation_organization_anchors = []
        train_pool_anchors = []
        validation_pool_anchors = []
        organization_coefficients = []
        organization_intercepts = []
        organization_feature_means = []
        organization_feature_scales = []
        pool_coefficients = []
        pool_intercepts = []
        pool_feature_means = []
        pool_feature_scales = []
        blend_weights = []
        point_bias_offsets = []
        initialization_validation_mae = []

        for step in range(self.horizon):
            train_design = _build_design(
                standardized,
                hourly.index,
                positions=train_positions,
                horizon_step=step,
                periods=self.periods,
                base_period=self.base_period,
                recent_windows=self.recent_windows,
                residual_lookback=self.residual_lookback,
                calendar_offset=calendar_offset,
                include_target=True,
            )
            validation_design = _build_design(
                standardized,
                hourly.index,
                positions=initialization_positions,
                horizon_step=step,
                periods=self.periods,
                base_period=self.base_period,
                recent_windows=self.recent_windows,
                residual_lookback=self.residual_lookback,
                calendar_offset=calendar_offset,
                include_target=True,
            )
            organization_head, organization_prediction, _ = _select_head(
                train_design,
                validation_design,
                alphas=self.ridge_alphas,
                means=means,
                scales=scales,
            )
            tuning_validation_design = _build_design(
                standardized,
                hourly.index,
                positions=validation_positions,
                horizon_step=step,
                periods=self.periods,
                base_period=self.base_period,
                recent_windows=self.recent_windows,
                residual_lookback=self.residual_lookback,
                calendar_offset=calendar_offset,
                include_target=True,
            )
            train_pool_design = _build_design(
                standardized_pool,
                hourly.index,
                positions=train_positions,
                horizon_step=step,
                periods=self.periods,
                base_period=self.base_period,
                recent_windows=self.recent_windows,
                residual_lookback=self.residual_lookback,
                calendar_offset=calendar_offset,
                include_target=True,
            )
            validation_pool_design = _build_design(
                standardized_pool,
                hourly.index,
                positions=initialization_positions,
                horizon_step=step,
                periods=self.periods,
                base_period=self.base_period,
                recent_windows=self.recent_windows,
                residual_lookback=self.residual_lookback,
                calendar_offset=calendar_offset,
                include_target=True,
            )
            pool_head, pool_prediction_2d, _ = _select_head(
                train_pool_design,
                validation_pool_design,
                alphas=self.ridge_alphas,
                means=pool_mean,
                scales=pool_scale,
            )
            tuning_validation_pool_design = _build_design(
                standardized_pool,
                hourly.index,
                positions=validation_positions,
                horizon_step=step,
                periods=self.periods,
                base_period=self.base_period,
                recent_windows=self.recent_windows,
                residual_lookback=self.residual_lookback,
                calendar_offset=calendar_offset,
                include_target=True,
            )
            direct_pool = pool_prediction_2d[:, 0]
            organization_pool = organization_prediction.sum(axis=1)
            truth_pool = raw_pool[initialization_positions + step, 0]
            best_blend = min(
                self.reconciliation_grid,
                key=lambda weight: (
                    float(np.abs((1.0 - weight) * organization_pool + weight * direct_pool - truth_pool).mean()),
                    float(weight),
                ),
            )
            blended = (1.0 - best_blend) * organization_pool + best_blend * direct_pool
            point_bias = float(np.median(truth_pool - blended))

            feature_count = int(organization_head.coefficients.size)
            channels = raw.shape[1]
            train_organization_features.append(
                ((train_design.features - organization_head.feature_mean) / organization_head.feature_scale).reshape(
                    len(train_positions), channels, feature_count
                )
            )
            validation_organization_features.append(
                ((tuning_validation_design.features - organization_head.feature_mean) / organization_head.feature_scale).reshape(
                    len(validation_positions), channels, feature_count
                )
            )
            train_pool_features.append(
                (train_pool_design.features - pool_head.feature_mean) / pool_head.feature_scale
            )
            validation_pool_features.append(
                (tuning_validation_pool_design.features - pool_head.feature_mean)
                / pool_head.feature_scale
            )
            train_organization_anchors.append(
                _seasonal_anchors(
                    standardized,
                    train_positions,
                    horizon_step=step,
                    periods=self.periods,
                )
            )
            validation_organization_anchors.append(
                _seasonal_anchors(
                    standardized,
                    validation_positions,
                    horizon_step=step,
                    periods=self.periods,
                )
            )
            train_pool_anchors.append(
                _seasonal_anchors(
                    standardized_pool,
                    train_positions,
                    horizon_step=step,
                    periods=self.periods,
                )[:, 0, :]
            )
            validation_pool_anchors.append(
                _seasonal_anchors(
                    standardized_pool,
                    validation_positions,
                    horizon_step=step,
                    periods=self.periods,
                )[:, 0, :]
            )
            organization_coefficients.append(organization_head.coefficients)
            organization_intercepts.append(organization_head.intercept)
            organization_feature_means.append(organization_head.feature_mean)
            organization_feature_scales.append(organization_head.feature_scale)
            pool_coefficients.append(pool_head.coefficients)
            pool_intercepts.append(pool_head.intercept)
            pool_feature_means.append(pool_head.feature_mean)
            pool_feature_scales.append(pool_head.feature_scale)
            blend_weights.append(best_blend)
            point_bias_offsets.append(point_bias)
            initialization_validation_mae.append(
                float(np.abs(blended + point_bias - truth_pool).mean())
            )

        def stack_horizon(items: list[np.ndarray]) -> np.ndarray:
            return np.stack(items, axis=1).astype(np.float32)

        train_org_x = stack_horizon(train_organization_features)
        validation_org_x = stack_horizon(validation_organization_features)
        train_pool_x = stack_horizon(train_pool_features)
        validation_pool_x = stack_horizon(validation_pool_features)
        train_org_anchors = stack_horizon(train_organization_anchors)
        validation_org_anchors = stack_horizon(validation_organization_anchors)
        train_pool_anchors_array = stack_horizon(train_pool_anchors)
        validation_pool_anchors_array = stack_horizon(validation_pool_anchors)
        train_org_target = np.stack(
            [raw[train_positions + step] for step in range(self.horizon)], axis=1
        ).astype(np.float32)
        validation_org_target = np.stack(
            [raw[validation_positions + step] for step in range(self.horizon)], axis=1
        ).astype(np.float32)
        train_pool_target = train_org_target.sum(axis=-1)
        validation_pool_target = validation_org_target.sum(axis=-1)
        train_peak_thresholds = np.quantile(
            train_pool_target,
            self.loss_peak_quantile,
            axis=0,
        ).astype(np.float32)
        fallback_shares = raw[-min(168, len(raw)) :].mean(axis=0)

        model = _AdaptiveSeasonalResidualNet(
            horizon=self.horizon,
            feature_count=feature_count,
            period_count=len(self.periods),
            base_period_index=self.periods.index(self.base_period),
            base_gate_logit=self.base_gate_logit,
            organization_coefficients=np.stack(organization_coefficients),
            organization_intercepts=np.asarray(organization_intercepts),
            pool_coefficients=np.stack(pool_coefficients),
            pool_intercepts=np.asarray(pool_intercepts),
            blend_weights=np.asarray(blend_weights),
            point_bias_offsets=np.asarray(point_bias_offsets),
            means=means,
            scales=scales,
            pool_mean=float(pool_mean[0]),
            pool_scale=float(pool_scale[0]),
            fallback_shares=fallback_shares,
            seed=self.seed,
        ).to(self.device)
        train_tensors = tuple(
            self._tensor(value, self.device)
            for value in (
                train_org_x,
                train_pool_x,
                train_org_anchors,
                train_pool_anchors_array,
                train_org_target,
                train_pool_target,
            )
        )
        validation_tensors = tuple(
            self._tensor(value, self.device)
            for value in (
                validation_org_x,
                validation_pool_x,
                validation_org_anchors,
                validation_pool_anchors_array,
                validation_org_target,
                validation_pool_target,
            )
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            fused=self.device.type == "cuda",
        )
        peak_threshold_tensor = self._tensor(train_peak_thresholds, self.device)

        @torch.no_grad()
        def validation_score() -> tuple[float, float, float]:
            model.eval()
            predicted_org, predicted_pool, _ = model(*validation_tensors[:4])
            pool_error = predicted_pool - validation_tensors[5]
            pool_mae = float(torch.mean(torch.abs(pool_error)).item())
            pool_rmse = float(torch.sqrt(torch.mean(pool_error.square())).item())
            organization_mae = float(torch.mean(torch.abs(predicted_org - validation_tensors[4])).item())
            return pool_mae, pool_rmse, organization_mae

        best_pool_mae, best_pool_rmse, best_organization_mae = validation_score()
        initial_pool_mae = best_pool_mae
        initial_pool_rmse = best_pool_rmse
        best_selection_score = 1.0
        best_state = deepcopy(model.state_dict())
        best_epoch = 0
        epochs_without_improvement = 0
        training_history = [
            {
                "epoch": 0,
                "validation_pool_mae": best_pool_mae,
                "validation_pool_rmse": best_pool_rmse,
                "validation_selection_score": best_selection_score,
                "validation_organization_mae": best_organization_mae,
            }
        ]
        for epoch in range(1, self.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            predicted_org, predicted_pool, _ = model(*train_tensors[:4])
            pool_error = (predicted_pool - train_tensors[5]) / model.pool_scale.clamp_min(1e-6)
            organization_error = (predicted_org - train_tensors[4]) / model.scales[None, None, :].clamp_min(1e-6)
            pool_loss, pool_loss_components = _peak_tail_composite_loss(
                pool_error,
                train_tensors[5],
                peak_threshold_tensor,
                rmse_weight=self.loss_rmse_weight,
                tail_weight=self.loss_tail_weight,
                tail_fraction=self.loss_tail_fraction,
                peak_weight=self.loss_peak_weight,
                underprediction_weight=self.loss_underprediction_weight,
            )
            organization_loss = torch.mean(torch.abs(organization_error))
            stability_loss = model.gate_weight.square().mean()
            loss = (
                pool_loss
                + self.organization_loss_weight * organization_loss
                + self.gate_stability_weight * stability_loss
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
            optimizer.step()
            pool_mae, pool_rmse, organization_mae = validation_score()
            selection_score = (
                (1.0 - self.validation_rmse_weight)
                * pool_mae
                / max(initial_pool_mae, 1e-6)
                + self.validation_rmse_weight
                * pool_rmse
                / max(initial_pool_rmse, 1e-6)
            )
            training_history.append(
                {
                    "epoch": epoch,
                    "training_loss": float(loss.detach().cpu()),
                    **{
                        f"training_pool_{name}": float(value.detach().cpu())
                        for name, value in pool_loss_components.items()
                    },
                    "validation_pool_mae": pool_mae,
                    "validation_pool_rmse": pool_rmse,
                    "validation_selection_score": selection_score,
                    "validation_organization_mae": organization_mae,
                }
            )
            candidate = (selection_score, pool_mae, pool_rmse, organization_mae)
            incumbent = (
                best_selection_score,
                best_pool_mae,
                best_pool_rmse,
                best_organization_mae,
            )
            if candidate < incumbent:
                (
                    best_selection_score,
                    best_pool_mae,
                    best_pool_rmse,
                    best_organization_mae,
                ) = candidate
                best_state = deepcopy(model.state_dict())
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break
        model.load_state_dict(best_state)
        model.eval()

        inference_org_features = []
        inference_pool_features = []
        inference_org_anchors = []
        inference_pool_anchors = []
        position = np.asarray([len(standardized)], dtype=np.int64)
        for step in range(self.horizon):
            org_features, _ = _build_inference_design(
                standardized,
                origin=origin,
                calendar_timestamp=calendar_timestamp,
                horizon_step=step,
                periods=self.periods,
                base_period=self.base_period,
                recent_windows=self.recent_windows,
                residual_lookback=self.residual_lookback,
            )
            pool_features, _ = _build_inference_design(
                standardized_pool,
                origin=origin,
                calendar_timestamp=calendar_timestamp,
                horizon_step=step,
                periods=self.periods,
                base_period=self.base_period,
                recent_windows=self.recent_windows,
                residual_lookback=self.residual_lookback,
            )
            inference_org_features.append(
                (org_features - organization_feature_means[step])
                / organization_feature_scales[step]
            )
            inference_pool_features.append(
                (
                    (pool_features - pool_feature_means[step])
                    / pool_feature_scales[step]
                )[0]
            )
            inference_org_anchors.append(
                _seasonal_anchors(
                    standardized,
                    position,
                    horizon_step=step,
                    periods=self.periods,
                )[0]
            )
            inference_pool_anchors.append(
                _seasonal_anchors(
                    standardized_pool,
                    position,
                    horizon_step=step,
                    periods=self.periods,
                )[0, 0]
            )
        with torch.no_grad():
            predicted_organization, predicted_pool, gate = model(
                self._tensor(np.stack(inference_org_features, axis=0)[None, ...], self.device),
                self._tensor(np.stack(inference_pool_features, axis=0)[None, ...], self.device),
                self._tensor(np.stack(inference_org_anchors, axis=0)[None, ...], self.device),
                self._tensor(np.stack(inference_pool_anchors, axis=0)[None, ...], self.device),
            )
        point = predicted_organization[0].detach().cpu().numpy().astype(np.float32)
        pool_point = predicted_pool[0].detach().cpu().numpy()
        if not np.allclose(point.sum(axis=1), pool_point, rtol=1e-5, atol=1e-4):
            raise AssertionError("SRLinear reconciliation lost aggregate coherence.")

        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        learned_parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
        joblib.dump(
            {
                "model_id": self.model_id,
                "seed": self.seed,
                "origin": pd.Timestamp(origin).isoformat(),
                "columns": tuple(hourly.columns),
                "periods": self.periods,
                "base_period": self.base_period,
                "recent_windows": self.recent_windows,
                "residual_lookback": self.residual_lookback,
                "validation_hours": self.validation_hours,
                "validation_cadence_hours": self.validation_cadence_hours,
                "tuning_validation_cadence_hours": self.tuning_validation_cadence_hours,
                "sample_stride_hours": self.sample_stride_hours,
                "ridge_alphas": self.ridge_alphas,
                "reconciliation_grid": self.reconciliation_grid,
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "patience": self.patience,
                "organization_loss_weight": self.organization_loss_weight,
                "gate_stability_weight": self.gate_stability_weight,
                "loss_rmse_weight": self.loss_rmse_weight,
                "loss_tail_weight": self.loss_tail_weight,
                "loss_tail_fraction": self.loss_tail_fraction,
                "loss_peak_weight": self.loss_peak_weight,
                "loss_underprediction_weight": self.loss_underprediction_weight,
                "loss_peak_quantile": self.loss_peak_quantile,
                "validation_rmse_weight": self.validation_rmse_weight,
                "base_gate_logit": self.base_gate_logit,
                "gradient_clip": self.gradient_clip,
                "best_epoch": best_epoch,
                "best_validation_pool_mae": best_pool_mae,
                "best_validation_pool_rmse": best_pool_rmse,
                "best_validation_selection_score": best_selection_score,
                "best_validation_organization_mae": best_organization_mae,
                "train_peak_thresholds": train_peak_thresholds,
                "initialization_validation_mae_by_horizon": tuple(initialization_validation_mae),
                "training_history": training_history,
                "learned_parameter_count": learned_parameter_count,
                "final_gate_weights": gate[0].detach().cpu().numpy(),
                "state_dict": {
                    name: value.detach().cpu().numpy()
                    for name, value in model.state_dict().items()
                },
                "organization_feature_means": np.stack(organization_feature_means),
                "organization_feature_scales": np.stack(organization_feature_scales),
                "pool_feature_means": np.stack(pool_feature_means),
                "pool_feature_scales": np.stack(pool_feature_scales),
            },
            checkpoint_dir / f"{pd.Timestamp(origin).strftime('%Y%m%dT%H%M%S')}.joblib",
            compress=3,
        )
        return ForecastOutput(tuple(hourly.columns), point, {})
