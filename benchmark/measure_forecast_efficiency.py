"""Measure parameter and inference operation counts for forecast models.

The protocol profiles one complete cluster forecast at a fixed causal origin.
PyTorch ``FlopCounterMode`` counts supported matrix multiplication and
convolution operators. One multiply accumulate is reported as two FLOPs.
Data preparation, model fitting, indexing, normalization, reductions,
activations, and unsupported custom kernel operations are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
from typing import Any

import joblib
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from methods.forecasters.advanced import _NLinear, _PatchTST, _TemporalMamba
from methods.forecasters.models import _DLinear, _OrgLinearR, orglinear_args
from methods.forecasters.srlinear import _AdaptiveSeasonalResidualNet
from research_sim.gfs.estimator.model.OrgLinear import Model as ReleasedOrgLinear


EXPECTED_PARAMETERS = {
    "seasonal_naive": 0,
    "orglinear_released": 10_004,
    "orglinear_r": 10_004,
    "nlinear": 2_692,
    "dlinear": 5_384,
    "temporal_mamba": 12_900,
    "patchtst": 94_724,
    "srlinear": 2_052,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def count_parameters(model: nn.Module) -> int:
    """Return the exact number of trainable scalar parameters."""

    return int(sum(parameter.numel() for parameter in model.parameters()))


def profile_flops(
    model: nn.Module,
    args: tuple[torch.Tensor, ...],
) -> int:
    """Return supported operator FLOPs for one inference forward pass."""

    model.eval()
    # Keep autograd enabled during the forward call. PyTorch otherwise dispatches
    # TransformerEncoder through a fused inference operator whose internal
    # attention matrix products are invisible to FlopCounterMode. No backward
    # pass is executed and gradients are never retained after this function.
    with FlopCounterMode(display=False) as counter:
        model(*args)
    if any(argument.is_cuda for argument in args):
        torch.cuda.synchronize()
    return int(counter.get_total_flops())


def _record(
    model: nn.Module,
    args: tuple[torch.Tensor, ...],
    *,
    device: str,
    notes: str,
    analytical_flops: int | None = None,
) -> dict[str, Any]:
    parameters = count_parameters(model)
    profiler_flops = profile_flops(model, args)
    flops = int(analytical_flops) if analytical_flops is not None else profiler_flops
    if flops % 2:
        raise AssertionError(f"FLOP count must be even, got {flops}.")
    return {
        "parameters": parameters,
        "macs_per_cluster_forecast": flops // 2,
        "flops_per_cluster_forecast": flops,
        "profiler_supported_flops": profiler_flops,
        "analytical_correction_flops": flops - profiler_flops,
        "profile_device": device,
        "notes": notes,
    }


def patchtst_flops(
    *,
    organizations: int,
    lookback: int,
    horizon: int,
    patch_length: int,
    patch_stride: int,
    model_width: int,
    layers: int,
    feedforward_width: int,
) -> int:
    """Return dense and attention FLOPs for the fixed PatchTST forward pass."""

    patch_count = math.ceil((lookback - patch_length) / patch_stride) + 1
    projection_macs = patch_count * patch_length * model_width
    layer_macs = (
        4 * patch_count * model_width * model_width
        + 2 * patch_count * patch_count * model_width
        + 2 * patch_count * model_width * feedforward_width
    )
    head_macs = patch_count * model_width * horizon
    return int(
        2
        * organizations
        * (projection_macs + layers * layer_macs + head_macs)
    )


def temporal_mamba_flops(
    *,
    organizations: int,
    lookback: int,
    horizon: int,
    patch_length: int,
    patch_stride: int,
    model_width: int,
    state_width: int,
    convolution_width: int,
    expansion: int,
) -> int:
    """Return Mamba FLOPs using its standard selective scan estimate.

    Dense projections and depthwise convolution use two FLOPs per multiply
    accumulate. Selective scan uses ``9 L D N + L D`` FLOPs per sequence,
    where L is token count, D is expanded width, and N is state width.
    """

    patch_count = math.ceil((lookback - patch_length) / patch_stride) + 1
    inner_width = model_width * expansion
    delta_rank = math.ceil(model_width / 16)
    dense_macs = (
        patch_count * patch_length * model_width
        + patch_count * model_width * (2 * inner_width)
        + patch_count * inner_width * convolution_width
        + patch_count * inner_width * (delta_rank + 2 * state_width)
        + patch_count * delta_rank * inner_width
        + patch_count * inner_width * model_width
        + patch_count * model_width * horizon
    )
    selective_scan_flops = (
        9 * patch_count * inner_width * state_width
        + patch_count * inner_width
    )
    return int(
        organizations * (2 * dense_macs + selective_scan_flops)
    )


def _build_srlinear(
    checkpoint_path: Path,
) -> tuple[_AdaptiveSeasonalResidualNet, tuple[torch.Tensor, ...], dict[str, int]]:
    payload = joblib.load(checkpoint_path)
    state = payload["state_dict"]
    gate_bias = np.asarray(state["gate_bias"])
    organization_weight = np.asarray(state["organization_weight"])
    horizon, period_count = gate_bias.shape
    feature_count = int(organization_weight.shape[1])
    channels = int(np.asarray(state["means"]).shape[0])
    model = _AdaptiveSeasonalResidualNet(
        horizon=horizon,
        feature_count=feature_count,
        period_count=period_count,
        base_period_index=0,
        base_gate_logit=0.0,
        organization_coefficients=np.zeros_like(organization_weight),
        organization_intercepts=np.zeros(horizon, dtype=np.float32),
        pool_coefficients=np.zeros_like(np.asarray(state["pool_weight"])),
        pool_intercepts=np.zeros(horizon, dtype=np.float32),
        blend_weights=np.full(horizon, 0.5, dtype=np.float32),
        point_bias_offsets=np.zeros(horizon, dtype=np.float32),
        means=np.asarray(state["means"]),
        scales=np.asarray(state["scales"]),
        pool_mean=float(np.asarray(state["pool_mean"])),
        pool_scale=float(np.asarray(state["pool_scale"])),
        fallback_shares=np.asarray(state["fallback_shares"]),
        seed=int(payload["seed"]),
    )
    tensor_state = {
        name: torch.as_tensor(value)
        for name, value in state.items()
    }
    model.load_state_dict(tensor_state, strict=True)
    inputs = (
        torch.zeros(1, horizon, channels, feature_count),
        torch.zeros(1, horizon, feature_count),
        torch.zeros(1, horizon, channels, period_count),
        torch.zeros(1, horizon, period_count),
    )
    geometry = {
        "organizations": channels,
        "horizon_hours": horizon,
        "feature_count": feature_count,
        "seasonal_anchor_count": period_count,
    }
    stored_count = int(payload["learned_parameter_count"])
    if count_parameters(model) != stored_count:
        raise AssertionError(
            "SRLinear checkpoint parameter count does not match reconstructed model."
        )
    return model, inputs, geometry


def measure(
    *,
    config_path: Path,
    srlinear_checkpoint: Path,
    repository_root: Path,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rolling = config["rolling"]
    models = config["models"]
    lookback = int(rolling["lookback_hours"])
    horizon = int(rolling["horizon_hours"])

    srlinear_model, srlinear_inputs, geometry = _build_srlinear(srlinear_checkpoint)
    organizations = geometry["organizations"]
    if geometry["horizon_hours"] != horizon:
        raise AssertionError("SRLinear checkpoint horizon does not match suite config.")

    shared_input = torch.zeros(organizations, lookback)
    measured: dict[str, dict[str, Any]] = {
        "seasonal_naive": {
            "parameters": 0,
            "macs_per_cluster_forecast": 0,
            "flops_per_cluster_forecast": 0,
            "profile_device": "CPU",
            "notes": "No learned arithmetic core. Array indexing and copying are excluded.",
        }
    }

    nlinear = _NLinear(lookback, horizon)
    measured["nlinear"] = _record(
        nlinear,
        (shared_input,),
        device="CPU",
        notes="Shared channel independent linear model evaluated for all organizations.",
    )

    dlinear_config = models["dlinear"]
    dlinear = _DLinear(
        lookback,
        horizon,
        int(dlinear_config["moving_average"]),
    )
    measured["dlinear"] = _record(
        dlinear,
        (shared_input,),
        device="CPU",
        notes="Seasonal and trend linear heads counted. Average pooling is not counted by the operator profiler.",
    )

    patch_config = models["patchtst"]
    patchtst = _PatchTST(
        lookback=lookback,
        horizon=horizon,
        patch_length=int(patch_config["patch_length"]),
        patch_stride=int(patch_config["patch_stride"]),
        model_width=int(patch_config["model_width"]),
        heads=int(patch_config["heads"]),
        layers=int(patch_config["layers"]),
        feedforward_width=int(patch_config["feedforward_width"]),
        dropout=float(patch_config["dropout"]),
    )
    measured["patchtst"] = _record(
        patchtst,
        (shared_input,),
        device="CPU",
        notes="Patch projection, attention matrix products, feedforward layers, and output head are counted. Analytical attention terms correct fused operator undercounting.",
        analytical_flops=patchtst_flops(
            organizations=organizations,
            lookback=lookback,
            horizon=horizon,
            patch_length=int(patch_config["patch_length"]),
            patch_stride=int(patch_config["patch_stride"]),
            model_width=int(patch_config["model_width"]),
            layers=int(patch_config["layers"]),
            feedforward_width=int(patch_config["feedforward_width"]),
        ),
    )

    args = orglinear_args(
        checkpoint_dir=repository_root / "tmp" / "forecast_efficiency",
        use_gpu=False,
        gpu=0,
        lookback=lookback,
        horizon=horizon,
        epochs=int(models["orglinear_released"]["epochs"]),
        batch_size=int(models["orglinear_released"]["batch_size"]),
        learning_rate=float(models["orglinear_released"]["learning_rate"]),
        num_workers=0,
    )
    orglinear = ReleasedOrgLinear(args)
    orglinear_inputs = (
        torch.zeros(1, lookback, organizations),
        torch.zeros(1, organizations, 3, dtype=torch.long),
        torch.zeros(1, 3, organizations, dtype=torch.long),
    )
    measured["orglinear_released"] = _record(
        orglinear,
        orglinear_inputs,
        device="CPU",
        notes="Attribute projections, class attention products, and three forecast heads are counted.",
    )
    orglinear_r = _OrgLinearR(args)
    measured["orglinear_r"] = _record(
        orglinear_r,
        orglinear_inputs,
        device="CPU",
        notes="Scaled attention revision retains the Released OrgLinear parameter and operation geometry.",
    )

    mamba_config = models["temporal_mamba"]
    if not torch.cuda.is_available():
        raise RuntimeError("Temporal Mamba efficiency profiling requires CUDA.")
    temporal_mamba = _TemporalMamba(
        lookback=lookback,
        horizon=horizon,
        patch_length=int(mamba_config["patch_length"]),
        patch_stride=int(mamba_config["patch_stride"]),
        model_width=int(mamba_config["model_width"]),
        state_width=int(mamba_config["state_width"]),
        convolution_width=int(mamba_config["convolution_width"]),
        expansion=int(mamba_config["expansion"]),
        layers=int(mamba_config["layers"]),
        dropout=float(mamba_config["dropout"]),
    ).cuda()
    measured["temporal_mamba"] = _record(
        temporal_mamba,
        (shared_input.cuda(),),
        device="CUDA",
        notes="Dense and convolution terms are profiler cross checked. Fused selective scan uses the declared 9 L D N plus L D FLOP estimate.",
        analytical_flops=temporal_mamba_flops(
            organizations=organizations,
            lookback=lookback,
            horizon=horizon,
            patch_length=int(mamba_config["patch_length"]),
            patch_stride=int(mamba_config["patch_stride"]),
            model_width=int(mamba_config["model_width"]),
            state_width=int(mamba_config["state_width"]),
            convolution_width=int(mamba_config["convolution_width"]),
            expansion=int(mamba_config["expansion"]),
        ),
    )

    measured["srlinear"] = _record(
        srlinear_model,
        srlinear_inputs,
        device="CPU",
        notes="Seasonal gate, organization residual, aggregate residual, and reconciliation matrix products are counted.",
    )

    mismatches = {
        model_id: {
            "expected": EXPECTED_PARAMETERS[model_id],
            "measured": values["parameters"],
        }
        for model_id, values in measured.items()
        if values["parameters"] != EXPECTED_PARAMETERS[model_id]
    }
    if mismatches:
        raise AssertionError(f"Parameter count mismatch: {mismatches}")

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unavailable"

    try:
        import mamba_ssm

        mamba_version = str(mamba_ssm.__version__)
    except ImportError:
        mamba_version = "unavailable"

    return {
        "schema_version": "1.0",
        "protocol": {
            "scope": "one_complete_cluster_inference_forward_pass",
            "lookback_hours": lookback,
            "horizon_hours": horizon,
            "organizations": organizations,
            "organization_basis": "active prediction columns at the first scored SRLinear origin",
            "batch_size": 1,
            "mac_definition": "one multiply accumulate equals two supported operator FLOPs",
            "flop_counter": "torch.utils.flop_counter.FlopCounterMode with disclosed analytical corrections",
            "included": "matrix multiplication, convolution, attention products, and selective scan estimate",
            "excluded": [
                "data preparation",
                "model fitting and optimizer work",
                "indexing and data movement",
                "normalization and reductions",
                "activations and softmax",
                "unsupported custom kernel operations outside the stated selective scan estimate",
            ],
            "temporal_mamba_estimator": "9 L D N plus L D FLOPs per sequence for selective scan",
        },
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "mamba_ssm": mamba_version,
            "git_commit": commit,
        },
        "sources": {
            "forecast_suite": _relative(config_path, repository_root),
            "forecast_suite_sha256": _sha256(config_path),
            "srlinear_checkpoint": _relative(srlinear_checkpoint, repository_root),
            "srlinear_checkpoint_sha256": _sha256(srlinear_checkpoint),
        },
        "srlinear_geometry": geometry,
        "models": measured,
        "validation": {
            "parameter_counts_match_reported_values": True,
            "forward_pass_completed_for_every_learned_model": True,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("../ResearchWork/benchmark/forecast_suite.yaml"),
    )
    parser.add_argument(
        "--srlinear-checkpoint",
        type=Path,
        default=Path(
            "../ResearchWork/runs/forecast_benchmark/"
            "a100_native_hp_forecast_train_v1/run/20261014T094110439430Z/"
            "srlinear_v3/seed3302413169/checkpoints/20240501T200000.joblib"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/forecast_efficiency.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    result = measure(
        config_path=args.config.resolve(),
        srlinear_checkpoint=args.srlinear_checkpoint.resolve(),
        repository_root=repository_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["models"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
