"""Additional forecasting controls for the causal GFS benchmark.

NLinear preserves the official shared-weight forward semantics. PatchTST and
Temporal Mamba preserve their defining architecture semantics while using the
benchmark's shared causal training and evaluation contract. Temporal Mamba
loads the official ``mamba_ssm.Mamba`` implementation from the ``mambahar``
Conda environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .models import (
    ForecastOutput,
    _windowed_series,
    hourly_organization_maximum,
    set_forecast_seed,
)


class _NLinear(nn.Module):
    """Official NLinear shared-weight semantics for pooled univariate series."""

    def __init__(self, lookback: int, horizon: int):
        super().__init__()
        self.linear = nn.Linear(int(lookback), int(horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_last = x[:, -1:].detach()
        return self.linear(x - sequence_last) + sequence_last


class _PatchTST(nn.Module):
    """Compact channel-independent PatchTST with RevIN and a flatten head."""

    def __init__(
        self,
        *,
        lookback: int,
        horizon: int,
        patch_length: int,
        patch_stride: int,
        model_width: int,
        heads: int,
        layers: int,
        feedforward_width: int,
        dropout: float,
    ):
        super().__init__()
        if model_width % heads:
            raise ValueError("PatchTST model width must be divisible by head count.")
        if patch_length > lookback:
            raise ValueError("PatchTST patch length exceeds lookback.")
        self.lookback = int(lookback)
        self.patch_length = int(patch_length)
        self.patch_stride = int(patch_stride)
        self.patch_count = math.ceil(
            (self.lookback - self.patch_length) / self.patch_stride
        ) + 1
        covered = (self.patch_count - 1) * self.patch_stride + self.patch_length
        self.end_padding = covered - self.lookback
        self.patch_projection = nn.Linear(self.patch_length, model_width)
        self.position = nn.Parameter(torch.zeros(1, self.patch_count, model_width))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_width,
            nhead=heads,
            dim_feedforward=feedforward_width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            norm=nn.LayerNorm(model_width),
            enable_nested_tensor=False,
        )
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(dropout),
            nn.Linear(self.patch_count * model_width, int(horizon)),
        )
        nn.init.normal_(self.position, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True).detach()
        scale = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        normalized = (x - mean) / scale
        if self.end_padding:
            normalized = F.pad(normalized, (0, self.end_padding), mode="replicate")
        patches = normalized.unfold(
            dimension=-1,
            size=self.patch_length,
            step=self.patch_stride,
        )
        if patches.shape[1] != self.patch_count:
            raise RuntimeError("PatchTST produced an unexpected patch count.")
        tokens = self.patch_projection(patches) + self.position
        forecast = self.head(self.encoder(tokens))
        return forecast * scale + mean


class _TemporalMamba(nn.Module):
    """Temporal patch tokenizer followed by official Mamba mixers."""

    def __init__(
        self,
        *,
        lookback: int,
        horizon: int,
        patch_length: int,
        patch_stride: int,
        model_width: int,
        state_width: int,
        convolution_width: int,
        expansion: int,
        layers: int,
        dropout: float,
    ):
        super().__init__()
        if patch_length > lookback:
            raise ValueError("Temporal Mamba patch length exceeds lookback.")
        self.lookback = int(lookback)
        self.patch_length = int(patch_length)
        self.patch_stride = int(patch_stride)
        self.patch_count = math.ceil(
            (self.lookback - self.patch_length) / self.patch_stride
        ) + 1
        covered = (self.patch_count - 1) * self.patch_stride + self.patch_length
        self.end_padding = covered - self.lookback
        self.patch_projection = nn.Linear(self.patch_length, model_width)
        try:
            from mamba_ssm import Mamba
        except ImportError as exc:
            raise RuntimeError(
                "Temporal Mamba requires mamba_ssm from Conda environment "
                "'mambahar'."
            ) from exc
        self.pre_norms = nn.ModuleList(
            [nn.LayerNorm(model_width) for _ in range(int(layers))]
        )
        self.blocks = nn.ModuleList(
            [
                Mamba(
                    d_model=int(model_width),
                    d_state=int(state_width),
                    d_conv=int(convolution_width),
                    expand=int(expansion),
                )
                for _ in range(int(layers))
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_width)
        self.head = nn.Linear(self.patch_count * model_width, int(horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True).detach()
        scale = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        normalized = (x - mean) / scale
        if self.end_padding:
            normalized = F.pad(normalized, (0, self.end_padding), mode="replicate")
        patches = normalized.unfold(
            dimension=-1,
            size=self.patch_length,
            step=self.patch_stride,
        )
        tokens = self.patch_projection(patches)
        for pre_norm, block in zip(self.pre_norms, self.blocks):
            tokens = tokens + self.dropout(block(pre_norm(tokens)))
        forecast = self.head(self.norm(tokens).flatten(start_dim=1))
        return forecast * scale + mean


@dataclass(frozen=True)
class _TorchTrainingConfig:
    lookback: int
    horizon: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    validation_hours: int
    sample_stride_hours: int
    patience: int
    device: torch.device
    seed: int
    num_workers: int
    precision: str
    fused_optimizer: bool
    gpu_resident: bool
    validation_batch_size: int


class _SharedTorchForecaster:
    stochastic = True

    def __init__(
        self,
        *,
        model_id: str,
        model_factory: Callable[[], nn.Module],
        model_config: Mapping[str, object],
        lookback: int,
        horizon: int,
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
        precision: str = "fp32",
        fused_optimizer: bool = False,
        gpu_resident: bool = False,
        validation_batch_size: int | None = None,
    ):
        precision = str(precision).lower()
        if precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError(f"Unsupported training precision: {precision}.")
        self.model_id = str(model_id)
        self.model_factory = model_factory
        self.model_config = dict(model_config)
        self.config = _TorchTrainingConfig(
            lookback=int(lookback),
            horizon=int(horizon),
            epochs=int(epochs),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            validation_hours=int(validation_hours),
            sample_stride_hours=int(sample_stride_hours),
            patience=int(patience),
            device=device,
            seed=int(seed),
            num_workers=int(num_workers),
            precision=precision,
            fused_optimizer=bool(fused_optimizer),
            gpu_resident=bool(gpu_resident),
            validation_batch_size=int(validation_batch_size or batch_size),
        )

    def fit_predict(
        self,
        history: pd.DataFrame,
        origin: pd.Timestamp,
        run_dir: Path,
        calendar_timestamp: pd.Timestamp | None = None,
    ) -> ForecastOutput:
        del calendar_timestamp
        config = self.config
        set_forecast_seed(config.seed)
        hourly = hourly_organization_maximum(history, origin=origin)
        values = hourly.to_numpy(dtype=np.float32)
        scale_fit_end = len(hourly) - config.validation_hours
        if scale_fit_end <= config.lookback + config.horizon:
            raise ValueError(f"Insufficient history for {self.model_id} training.")
        scale_values = values[:scale_fit_end]
        means = scale_values.mean(axis=0)
        scales = scale_values.std(axis=0)
        scales = np.where(scales > 1e-6, scales, 1.0)
        standardized = (values - means) / scales
        train_x, train_y = _windowed_series(
            standardized[:scale_fit_end],
            lookback=config.lookback,
            horizon=config.horizon,
            stride=config.sample_stride_hours,
        )
        validation_values = standardized[scale_fit_end - config.lookback :]
        validation_x, validation_y = _windowed_series(
            validation_values,
            lookback=config.lookback,
            horizon=config.horizon,
            stride=config.sample_stride_hours,
        )
        generator = torch.Generator().manual_seed(config.seed)
        pin_memory = config.device.type == "cuda"
        use_gpu_resident = config.gpu_resident and config.device.type == "cuda"
        if use_gpu_resident:
            train_tensors = (
                torch.from_numpy(train_x).to(config.device),
                torch.from_numpy(train_y).to(config.device),
            )
            validation_tensors = (
                torch.from_numpy(validation_x).to(config.device),
                torch.from_numpy(validation_y).to(config.device),
            )
            train_loader = None
            validation_loader = None
        else:
            train_tensors = None
            validation_tensors = None
            train_loader = DataLoader(
                TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
                batch_size=config.batch_size,
                shuffle=True,
                generator=generator,
                num_workers=config.num_workers,
                pin_memory=pin_memory,
                persistent_workers=config.num_workers > 0,
            )
            validation_loader = DataLoader(
                TensorDataset(
                    torch.from_numpy(validation_x), torch.from_numpy(validation_y)
                ),
                batch_size=config.validation_batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=pin_memory,
            )
        model = self.model_factory().to(config.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            fused=config.fused_optimizer and config.device.type == "cuda",
        )
        criterion = nn.MSELoss()
        amp_enabled = config.device.type == "cuda" and config.precision != "fp32"
        amp_dtype = torch.float16 if config.precision == "fp16" else torch.bfloat16
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=amp_enabled and amp_dtype == torch.float16,
        )

        def training_batches():
            if train_tensors is None:
                assert train_loader is not None
                yield from train_loader
                return
            features, targets = train_tensors
            permutation = torch.randperm(len(features), generator=generator)
            for start in range(0, len(permutation), config.batch_size):
                indices = permutation[start : start + config.batch_size].to(config.device)
                yield features.index_select(0, indices), targets.index_select(0, indices)

        def evaluation_batches():
            if validation_tensors is None:
                assert validation_loader is not None
                yield from validation_loader
                return
            features, targets = validation_tensors
            for start in range(0, len(features), config.validation_batch_size):
                end = start + config.validation_batch_size
                yield features[start:end], targets[start:end]

        best_loss = math.inf
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0
        trained_epochs = 0
        for epoch in range(config.epochs):
            trained_epochs = epoch + 1
            model.train()
            for batch_x, batch_y in training_batches():
                batch_x = batch_x.to(config.device, non_blocking=pin_memory)
                batch_y = batch_y.to(config.device, non_blocking=pin_memory)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=config.device.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    loss = criterion(model(batch_x), batch_y)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"{self.model_id} produced nonfinite training loss.")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            model.eval()
            validation_sum = 0.0
            validation_count = 0
            with torch.no_grad():
                for batch_x, batch_y in evaluation_batches():
                    batch_x = batch_x.to(config.device, non_blocking=pin_memory)
                    batch_y = batch_y.to(config.device, non_blocking=pin_memory)
                    with torch.autocast(
                        device_type=config.device.type,
                        dtype=amp_dtype,
                        enabled=amp_enabled,
                    ):
                        batch_loss = criterion(model(batch_x), batch_y)
                    validation_sum += float(batch_loss.item()) * len(batch_x)
                    validation_count += len(batch_x)
            validation_loss = validation_sum / validation_count
            if not math.isfinite(validation_loss):
                raise RuntimeError(f"{self.model_id} produced nonfinite validation loss.")
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= config.patience:
                    break
        if best_state is None:
            raise RuntimeError(f"{self.model_id} did not produce a checkpoint.")
        model.load_state_dict(best_state)
        model.eval()
        inference = torch.from_numpy(
            standardized[-config.lookback :].T.copy()
        ).to(config.device)
        with torch.no_grad():
            prediction = model(inference).cpu().numpy().T
        prediction = prediction * scales[None, :] + means[None, :]
        prediction = np.clip(prediction, 0.0, None).astype(np.float32)
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_id": self.model_id,
                "model_config": self.model_config,
                "state_dict": best_state,
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "means": means,
                "scales": scales,
                "columns": tuple(hourly.columns),
                "origin": origin.isoformat(),
                "seed": config.seed,
                "trained_epochs": trained_epochs,
                "best_validation_loss": best_loss,
                "training_runtime": {
                    "precision": config.precision if amp_enabled else "fp32",
                    "fused_optimizer": config.fused_optimizer and config.device.type == "cuda",
                    "gpu_resident": use_gpu_resident,
                    "validation_batch_size": config.validation_batch_size,
                },
            },
            checkpoint_dir / f"{origin.strftime('%Y%m%dT%H%M%S')}.pth",
        )
        return ForecastOutput(tuple(hourly.columns), prediction, {})


class NLinearForecaster(_SharedTorchForecaster):
    """Official shared NLinear semantics under the common causal trainer."""

    model_id = "nlinear_shared_v1"

    def __init__(self, **kwargs):
        lookback = int(kwargs["lookback"])
        horizon = int(kwargs["horizon"])
        super().__init__(
            model_id=self.model_id,
            model_factory=lambda: _NLinear(lookback, horizon),
            model_config={
                "implementation": "official_shared_forward_semantics",
                "lookback": lookback,
                "horizon": horizon,
            },
            **kwargs,
        )


class PatchTSTForecaster(_SharedTorchForecaster):
    """Channel-independent PatchTST adaptation for changing organizations."""

    model_id = "patchtst_channel_independent_v1"

    def __init__(
        self,
        *,
        patch_length: int,
        patch_stride: int,
        model_width: int,
        heads: int,
        layers: int,
        feedforward_width: int,
        dropout: float,
        **kwargs,
    ):
        lookback = int(kwargs["lookback"])
        horizon = int(kwargs["horizon"])
        model_config = {
            "patch_length": int(patch_length),
            "patch_stride": int(patch_stride),
            "model_width": int(model_width),
            "heads": int(heads),
            "layers": int(layers),
            "feedforward_width": int(feedforward_width),
            "dropout": float(dropout),
            "revin": True,
            "channel_independent": True,
        }
        super().__init__(
            model_id=self.model_id,
            model_factory=lambda: _PatchTST(
                lookback=lookback,
                horizon=horizon,
                patch_length=int(patch_length),
                patch_stride=int(patch_stride),
                model_width=int(model_width),
                heads=int(heads),
                layers=int(layers),
                feedforward_width=int(feedforward_width),
                dropout=float(dropout),
            ),
            model_config=model_config,
            **kwargs,
        )


class TemporalMambaForecaster(_SharedTorchForecaster):
    """Temporal Mamba baseline using official mamba_ssm.Mamba mixers."""

    model_id = "temporal_mamba_official_ssm_v1"

    def __init__(
        self,
        *,
        patch_length: int,
        patch_stride: int,
        model_width: int,
        state_width: int,
        convolution_width: int,
        expansion: int,
        layers: int,
        dropout: float,
        **kwargs,
    ):
        lookback = int(kwargs["lookback"])
        horizon = int(kwargs["horizon"])
        model_config = {
            "implementation": "mamba_ssm.Mamba_2.3.2.post1",
            "patch_length": int(patch_length),
            "patch_stride": int(patch_stride),
            "model_width": int(model_width),
            "state_width": int(state_width),
            "convolution_width": int(convolution_width),
            "expansion": int(expansion),
            "layers": int(layers),
            "dropout": float(dropout),
        }
        super().__init__(
            model_id=self.model_id,
            model_factory=lambda: _TemporalMamba(
                lookback=lookback,
                horizon=horizon,
                patch_length=int(patch_length),
                patch_stride=int(patch_stride),
                model_width=int(model_width),
                state_width=int(state_width),
                convolution_width=int(convolution_width),
                expansion=int(expansion),
                layers=int(layers),
                dropout=float(dropout),
            ),
            model_config=model_config,
            **kwargs,
        )
