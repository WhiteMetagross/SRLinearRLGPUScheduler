"""Forecast model adapters used by the causal rolling benchmark.

Released OrgLinear is imported without modification. OrgLinear-R is an
explicitly separate research baseline with scaled attention, an all-target
Gaussian likelihood, and a numerical scale floor. It is not a replacement for
the released implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from research_sim.gfs.estimator.gpu_request_estimator import GPURequestEstimator
from research_sim.gfs.estimator.model import OrgLinear as released_orglinear


@dataclass(frozen=True)
class ForecastOutput:
    """One forecast at one causal origin."""

    columns: tuple[str, ...]
    point: np.ndarray
    intrinsic_pool_quantiles: Mapping[float, np.ndarray]

    def __post_init__(self) -> None:
        point = np.asarray(self.point)
        if point.ndim != 2:
            raise ValueError("Forecast point must have shape [horizon, organization].")
        if point.shape[1] != len(self.columns):
            raise ValueError("Forecast columns do not match point width.")
        if not np.all(np.isfinite(point)):
            raise ValueError("Forecast point contains nonfinite values.")
        for quantile, values in self.intrinsic_pool_quantiles.items():
            if not 0.0 < float(quantile) < 1.0:
                raise ValueError("Forecast quantiles must lie inside (0, 1).")
            array = np.asarray(values)
            if array.shape != (point.shape[0],) or not np.all(np.isfinite(array)):
                raise ValueError("Pool quantile shape or values are invalid.")


def set_forecast_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for one isolated forecast run."""

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)


def hourly_organization_maximum(
    history: pd.DataFrame,
    *,
    origin: pd.Timestamp,
) -> pd.DataFrame:
    """Aggregate complete causal minute bins into origin-aligned hourly peaks."""

    if history.empty:
        raise ValueError("Forecast history is empty.")
    if not isinstance(history.index, pd.DatetimeIndex):
        raise TypeError("Forecast history requires a DatetimeIndex.")
    causal = history.loc[history.index < origin]
    if causal.empty or causal.index.max() >= origin:
        raise ValueError("Forecast history is not strictly causal.")
    offset = pd.Timedelta(
        minutes=int(origin.minute), seconds=int(origin.second)
    )
    return causal.resample("1h", origin="start_day", offset=offset).max()


class SeasonalNaiveForecaster:
    """Per organization seasonal persistence at a fixed hourly lag."""

    model_id = "seasonal_naive_168h"
    stochastic = False

    def __init__(self, *, horizon: int, seasonal_lag: int = 168):
        self.horizon = int(horizon)
        self.seasonal_lag = int(seasonal_lag)

    def fit_predict(
        self,
        history: pd.DataFrame,
        origin: pd.Timestamp,
        run_dir: Path,
        calendar_timestamp: pd.Timestamp | None = None,
    ) -> ForecastOutput:
        del run_dir, calendar_timestamp
        hourly = hourly_organization_maximum(history, origin=origin)
        required = self.seasonal_lag
        if len(hourly) < required:
            raise ValueError(
                f"Seasonal naive requires {required} complete history hours, got {len(hourly)}."
            )
        start = len(hourly) - self.seasonal_lag
        values = hourly.iloc[start : start + self.horizon].to_numpy(dtype=np.float32)
        if values.shape[0] != self.horizon:
            raise ValueError("Seasonal lag does not cover the requested horizon.")
        return ForecastOutput(tuple(hourly.columns), values, {})


class _MovingAverage(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.pool = nn.AvgPool1d(self.kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel_size - 1) // 2
        front = x[:, :1].repeat(1, pad)
        end = x[:, -1:].repeat(1, pad)
        return self.pool(torch.cat((front, x, end), dim=1).unsqueeze(1)).squeeze(1)


class _DLinear(nn.Module):
    """Shared univariate DLinear used across organization series."""

    def __init__(self, lookback: int, horizon: int, moving_average: int):
        super().__init__()
        self.moving_average = _MovingAverage(moving_average)
        self.seasonal = nn.Linear(lookback, horizon)
        self.trend = nn.Linear(lookback, horizon)
        nn.init.constant_(self.seasonal.weight, 1.0 / lookback)
        nn.init.constant_(self.trend.weight, 1.0 / lookback)
        nn.init.zeros_(self.seasonal.bias)
        nn.init.zeros_(self.trend.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self.moving_average(x)
        return self.seasonal(x - trend) + self.trend(trend)


def _windowed_series(
    values: np.ndarray,
    *,
    lookback: int,
    horizon: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create shared-model samples without crossing the causal cutoff."""

    if values.ndim != 2:
        raise ValueError("Hourly values must have shape [time, organization].")
    last_start = values.shape[0] - lookback - horizon
    if last_start < 0:
        raise ValueError("Insufficient hourly history for requested lookback and horizon.")
    starts = np.arange(0, last_start + 1, int(stride), dtype=np.int64)
    x = np.stack([values[s : s + lookback].T for s in starts], axis=0)
    y = np.stack(
        [values[s + lookback : s + lookback + horizon].T for s in starts],
        axis=0,
    )
    return x.reshape(-1, lookback), y.reshape(-1, horizon)


class DLinearForecaster:
    """Causal shared DLinear baseline with per organization train-only scaling."""

    model_id = "dlinear_shared"
    stochastic = True

    def __init__(
        self,
        *,
        lookback: int,
        horizon: int,
        moving_average: int,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        validation_hours: int,
        sample_stride_hours: int,
        patience: int,
        device: torch.device,
        seed: int,
        num_workers: int,
    ):
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.moving_average = int(moving_average)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.validation_hours = int(validation_hours)
        self.sample_stride_hours = int(sample_stride_hours)
        self.patience = int(patience)
        self.device = device
        self.seed = int(seed)
        self.num_workers = int(num_workers)

    def fit_predict(
        self,
        history: pd.DataFrame,
        origin: pd.Timestamp,
        run_dir: Path,
        calendar_timestamp: pd.Timestamp | None = None,
    ) -> ForecastOutput:
        del calendar_timestamp
        set_forecast_seed(self.seed)
        hourly = hourly_organization_maximum(history, origin=origin)
        values = hourly.to_numpy(dtype=np.float32)
        validation_span = self.validation_hours + self.lookback + self.horizon
        if len(hourly) <= validation_span:
            raise ValueError("Insufficient history for DLinear train and validation windows.")

        scale_fit_end = len(hourly) - self.validation_hours
        scale_values = values[:scale_fit_end]
        means = scale_values.mean(axis=0)
        scales = scale_values.std(axis=0)
        scales = np.where(scales > 1e-6, scales, 1.0)
        standardized = (values - means) / scales

        train_values = standardized[:scale_fit_end]
        validation_values = standardized[
            scale_fit_end - self.lookback :
        ]
        train_x, train_y = _windowed_series(
            train_values,
            lookback=self.lookback,
            horizon=self.horizon,
            stride=self.sample_stride_hours,
        )
        val_x, val_y = _windowed_series(
            validation_values,
            lookback=self.lookback,
            horizon=self.horizon,
            stride=self.sample_stride_hours,
        )

        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
        model = _DLinear(self.lookback, self.horizon, self.moving_average).to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        criterion = nn.MSELoss()
        val_x_tensor = torch.from_numpy(val_x).to(self.device)
        val_y_tensor = torch.from_numpy(val_y).to(self.device)
        best_loss = math.inf
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0
        for _ in range(self.epochs):
            model.train()
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss = float(criterion(model(val_x_tensor), val_y_tensor).item())
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break
        if best_state is None:
            raise RuntimeError("DLinear training did not produce a finite checkpoint.")
        model.load_state_dict(best_state)
        model.eval()
        x = torch.from_numpy(standardized[-self.lookback :].T.copy()).to(self.device)
        with torch.no_grad():
            prediction = model(x).cpu().numpy().T
        prediction = prediction * scales[None, :] + means[None, :]
        prediction = np.clip(prediction, 0.0, None).astype(np.float32)

        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_id": self.model_id,
                "state_dict": best_state,
                "means": means,
                "scales": scales,
                "columns": tuple(hourly.columns),
                "origin": origin.isoformat(),
                "seed": self.seed,
            },
            checkpoint_dir / f"{origin.strftime('%Y%m%dT%H%M%S')}.pth",
        )
        return ForecastOutput(tuple(hourly.columns), prediction, {})


class _OrgLinearR(released_orglinear.Model):
    """Separate OrgLinear revision with conventional scaled attention."""

    def encoder(self, x: torch.Tensor, x_dec: torch.Tensor, idx: torch.Tensor):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        attributes = torch.stack(
            [self.embedding[i](idx[:, i, :]) for i in range(self.num_class)],
            dim=2,
        )
        q = self.q(attributes)
        k = self.k(attributes)
        v = self.v(attributes)
        attention = torch.softmax(
            torch.matmul(q, k.transpose(-1, -2)) * self._norm_fact,
            dim=-1,
        )
        attention_output = torch.matmul(attention, v).flatten(start_dim=2)

        time_features = torch.stack(
            [
                self.time_embedding[i](x_dec[:, :, i].to(torch.long))
                for i in range(self.num_covariates)
            ],
            dim=2,
        ).flatten(start_dim=2)

        seasonal_input = torch.cat((seasonal_init, time_features, attention_output), dim=2)
        trend_input = torch.cat((trend_init, time_features, attention_output), dim=2)
        sigma_input = torch.cat((x.permute(0, 2, 1), time_features, attention_output), dim=2)
        seasonal_output = self.Linear_Seasonal(seasonal_input)
        trend_output = self.Linear_Trend(trend_input)
        sigma_output = self.distribution_sigma(self.Linear_Sigma(sigma_input)) + 1e-4
        mean = seasonal_output + trend_output
        return mean.permute(0, 2, 1), sigma_output.permute(0, 2, 1)


class _AllTargetGaussianNLL(nn.Module):
    def forward(
        self,
        mean: torch.Tensor,
        sigma: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return -torch.distributions.Normal(mean, sigma).log_prob(labels).mean()


class _OrgLinearREstimator(GPURequestEstimator):
    def _build_model(self):
        model = _OrgLinearR(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _select_criterion(self):
        return _AllTargetGaussianNLL()


def orglinear_args(
    *,
    checkpoint_dir: Path,
    use_gpu: bool,
    gpu: int,
    lookback: int,
    horizon: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    num_workers: int,
) -> SimpleNamespace:
    """Build explicit released-compatible OrgLinear arguments."""

    return SimpleNamespace(
        seq_len=int(lookback),
        label_len=24,
        pred_len=int(horizon),
        checkpoints=str(checkpoint_dir),
        use_gpu=bool(use_gpu),
        colocate=False,
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        devices=str(gpu),
        use_multi_gpu=False,
        gpu=int(gpu),
        device_ids=[int(gpu)],
        embed="timeF",
        freq="h",
        num_covariates=3,
        num_class=3,
        class_input_dim=[6, 1, 84],
        class_output_dim=8,
        attn_hidden_dim=16,
        num_workers=int(num_workers),
        train_epochs=int(epochs),
        patience=3,
        des="forecast_benchmark",
        loss="MLE",
        lradj="type1",
        moving_avg=25,
        enc_in=7,
        inverse=True,
        scaling=False,
    )


class _OrgLinearAdapter:
    stochastic = True
    estimator_class = GPURequestEstimator

    def __init__(
        self,
        *,
        model_id: str,
        lookback: int,
        horizon: int,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        device: torch.device,
        seed: int,
        num_workers: int,
        run_dir: Path,
        quantiles: tuple[float, ...],
    ):
        self.model_id = model_id
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.seed = int(seed)
        self.quantiles = tuple(float(q) for q in quantiles)
        set_forecast_seed(self.seed)
        args = orglinear_args(
            checkpoint_dir=run_dir / "checkpoints",
            use_gpu=device.type == "cuda",
            gpu=device.index or 0,
            lookback=lookback,
            horizon=horizon,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_workers=num_workers,
        )
        self.estimator = self.estimator_class(args)

    def fit_predict(
        self,
        history: pd.DataFrame,
        origin: pd.Timestamp,
        run_dir: Path,
        calendar_timestamp: pd.Timestamp | None = None,
    ) -> ForecastOutput:
        del run_dir
        set_forecast_seed(self.seed)
        model_time = pd.Timestamp(calendar_timestamp) if calendar_timestamp is not None else origin
        self.estimator.train(history, model_time)
        mean, sigma = self.estimator.test(history, model_time)
        point = np.clip(mean[0], 0.0, None).astype(np.float32)
        sigma_array = sigma[0].astype(np.float64)
        pool_mean = mean[0].sum(axis=1, dtype=np.float64)
        pool_sigma = np.sqrt(np.square(sigma_array).sum(axis=1))
        intrinsic = {
            quantile: np.clip(
                norm.ppf(quantile, loc=pool_mean, scale=pool_sigma),
                0.0,
                None,
            ).astype(np.float64)
            for quantile in self.quantiles
        }
        return ForecastOutput(tuple(history.columns), point, intrinsic)


class OrgLinearReleasedForecaster(_OrgLinearAdapter):
    """Unmodified released OrgLinear behind the common benchmark interface."""

    model_id = "orglinear_released_v1"
    estimator_class = GPURequestEstimator

    def __init__(self, **kwargs):
        super().__init__(model_id=self.model_id, **kwargs)


class OrgLinearRForecaster(_OrgLinearAdapter):
    """OrgLinear-R: separate minimal revision baseline."""

    model_id = "orglinear_r_v1"
    estimator_class = _OrgLinearREstimator

    def __init__(self, **kwargs):
        super().__init__(model_id=self.model_id, **kwargs)
