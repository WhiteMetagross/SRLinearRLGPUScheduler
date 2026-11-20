# Code Base Index:

## Final experiment entry points:

| Path | Role |
| --- | --- |
| `benchmark/forecast_run.py` | Runs the forecasting benchmark and SRLinear evaluation. |
| `benchmark/run.py` | Runs deterministic scheduler and quota methods. |
| `benchmark/experiments.py` | Launches and monitors deterministic replay batches. |
| `benchmark/rl_run.py` | Trains or evaluates one quota RL method. |
| `benchmark/rl_experiments.py` | Launches and monitors multi seed RL batches. |
| `benchmark/measure_forecast_efficiency.py` | Measures parameter count, operations, and inference cost. |

## Final configurations:

| Path | Role |
| --- | --- |
| `benchmark/forecast_suite.yaml` | Shared forecasting protocol and model parameters. |
| `benchmark/suite.yaml` | Scheduler baseline definitions. |
| `benchmark/paper_scheduler_plan.yaml` | Main deterministic baseline run plan. |
| `benchmark/non_rl_suite.yaml` | Native GFS, Contextual Linear Quota, SRGFS, and deterministic SRLinear controls. |
| `benchmark/method_plan.yaml` | Final deterministic method comparison. |
| `benchmark/rl_suite.yaml` | CLQ PPO and SRLinear quota PPO controls. |
| `benchmark/srlinear_rl_suite.yaml` | SRLinear utility and risk objective controls. |
| `benchmark/srac_rl_suite.yaml` | Final SRAC configuration. |

## Methods:

| Path | Role |
| --- | --- |
| `methods/forecasters/srlinear.py` | SRLinear training, prediction, and uncertainty interface. |
| `methods/forecasters/srlinear_components.py` | Seasonal anchors, residual features, loss, and reconciliation. |
| `methods/forecasters/models.py` | DLinear, NLinear, PatchTST, and OrgLinear interfaces. |
| `methods/forecasters/advanced.py` | Temporal Mamba interface. |
| `methods/quota_rl/env.py` | Quota RL observation, action, reward, cost, and shield logic. |
| `methods/quota_rl/model.py` | Recurrent actor critic model. |
| `methods/quota_rl/ppo.py` | Constrained recurrent PPO and generalized advantage estimation. |
| `methods/quota_rl/factory.py` | Common simulator and policy construction. |

## Simulator:

| Path | Role |
| --- | --- |
| `environment/protocol.yaml` | Frozen data, split, metric, and execution contract. |
| `environment/configs/` | Natural train, validation, and locked test configs. |
| `environment/src/research_sim/core.py` | Shared event simulator. |
| `environment/src/research_sim/control_env.py` | Five minute quota control interface. |
| `environment/src/research_sim/measurement.py` | Common event reduction and metric accounting. |
| `environment/src/research_sim/forecasts.py` | Causal forecast provider contract. |
| `environment/src/research_sim/gfs/` | GFS forecast and scheduling implementation. |
| `environment/tools/generate_snapshots.py` | Generates reusable causal warmup snapshots. |
| `environment/tools/verify_freeze.py` | Verifies the released environment checksums. |

## Training only fit tools:

| Path | Role |
| --- | --- |
| `benchmark/fit_srlinear_quota_calibration.py` | Fits SRLinear quota uncertainty from training forecasts. |
| `benchmark/fit_contextual_control.py` | Fits Contextual Linear Quota from a training replay. |
| `benchmark/fit_srgfs.py` | Builds deterministic SRGFS controller settings. |
| `benchmark/protocol_helpers.py` | Applies split locks and causal duration fitting. |

## Paper and documentation:

| Path | Role |
| --- | --- |
| `paper/main.tex` | IEEE paper source. |
| `paper/main.pdf` | Built paper. |
| `paper/figures/` | Figures required by the LaTeX source and README. |
| `docs/DatasetAndSplits.md` | Data provenance, chronology, and leakage controls. |
| `docs/ForecastingMethods.md` | Forecast model definitions. |
| `docs/SchedulingMethods.md` | Scheduler and quota method definitions. |
| `docs/Reproducibility.md` | Environment, data, seed, and run requirements. |
| `docs/ThirdPartyCode.md` | GFS, trace, and baseline code provenance. |

## Excluded material:

- Raw trace files.
- Warmup snapshots.
- Fitted JSON controls.
- Run logs and progress files.
- Model checkpoints.
- Cached forecasts.
- Failed or discarded placement RL code.
- Development only viability probes.
- Obsolete forecasting candidates.
- Paper build intermediates.
