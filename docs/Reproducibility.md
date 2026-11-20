# Reproducibility:

## System requirements:

- Windows with WSL 2.
- Conda installed inside WSL.
- Sufficient local disk for the Alibaba trace and generated snapshots.
- An NVIDIA GPU for CUDA forecasting and future RL training.
- CPU execution for simulator event transitions.

## Environment setup:

Use the existing `trainenv` environment for this workspace. A fresh machine can create the pinned CPU environment from `environment.yml`. GPU users should use `environment-gpu.yml` and verify compatibility with their installed driver.

```bash
conda env create -f environment.yml
conda run --no-capture-output -n iml-research-rmproject python -m pip install -e .
```

No package should be installed into Conda `base`.

Temporal Mamba uses the separate exact specification in `benchmark/forecast_environment.lock.yml`. Existing Mamba runs used the project environment named `mambahar`. Create an equivalent environment from that lock when the official Mamba control must be reproduced.

## Data setup:

Follow `data/README.md`. Raw data must match both SHA256 values. Do not commit raw data.

## Integrity checks:

```bash
conda run --no-capture-output -n trainenv python environment/tools/verify_freeze.py
conda run --no-capture-output -n trainenv python -m pytest -q
```

The freeze verifier checks the immutable simulator bundle. Tests check current wrappers and methods outside that bundle.

## Generated snapshots:

```bash
conda run --no-capture-output -n trainenv python environment/tools/generate_snapshots.py environment/configs/natural_train.yaml environment/configs/natural_val.yaml
```

Snapshots live under `runs/` and are not tracked by Git. Every reported run must record the snapshot hash.

## Scheduling commands:

```bash
conda run --no-capture-output -n trainenv python benchmark/experiments.py --plan benchmark/experiment_plan.yaml prepare
conda run --no-capture-output -n trainenv python benchmark/experiments.py --plan benchmark/experiment_plan.yaml launch
conda run --no-capture-output -n trainenv python benchmark/experiments.py status --batch latest
```

Use `benchmark/method_plan.yaml` for SRGFS and the final deterministic quota methods.

## Forecast commands:

```bash
conda run --no-capture-output -n trainenv python benchmark/forecast_run.py --models seasonal_naive,dlinear,nlinear,srlinear,patchtst,orglinear_released run
conda run --no-capture-output -n trainenv python benchmark/forecast_run.py --models temporal_mamba run
```

Temporal Mamba requires the official `mamba_ssm` package in the selected environment.

## Seed policy:

- Master seed is `17`.
- Deterministic methods run once.
- Stochastic seeds are generated at run time with NumPy `SeedSequence.spawn`.
- All stochastic comparison methods must use the same paired seed set.

Seeds are generated at run time. They are recorded in every run manifest and are not copied into model code.

## SRAC commands:

```bash
conda run --no-capture-output -n trainenv python benchmark/rl_run.py --suite benchmark/srac_rl_suite.yaml smoke --train-steps 16 --eval-steps 16
conda run --no-capture-output -n trainenv python benchmark/rl_experiments.py --suite benchmark/srac_rl_suite.yaml prepare
conda run --no-capture-output -n trainenv python benchmark/rl_experiments.py --suite benchmark/srac_rl_suite.yaml launch
```

## Required run metadata:

Every reported run must store:

- Git commit and dirty state.
- Suite and configuration hashes.
- Trace and snapshot hashes.
- Split name and exact boundaries.
- Model or policy identity.
- Seed.
- Python, framework, CUDA, and driver versions.
- CPU thread and process worker counts.
- Wall time and peak CUDA memory where relevant.
- Raw metrics and audit status.

## Final test rule:

The final test is intentionally unavailable without `test_access_grant.json`. Do not create this file until the method, hyperparameters, seeds, checkpoint selection, and evaluation commands are frozen.
