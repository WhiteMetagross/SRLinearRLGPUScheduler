# Dataset And Splits:

## Source:

The simulator uses the Alibaba Cluster Trace Program GPU Spot trace. The frozen protocol records source commit `7e77a86ac139298e8f19da2be84f07bd6d23f477` from `cluster-trace-v2026-spot-gpu`.

Only nodes with GPU type `A100-SXM4-80GB` enter the primary benchmark. The filtered cluster contains 3,456 GPUs in the frozen snapshot.

The required source files are:

- `node_info_df.csv` with SHA256 `1aba161961a5a4a1a61aa581383c5e5abe3400b59f8597ba8c4eef7597bc9d18`.
- `job_info_df.csv` with SHA256 `113ccee4c28f5c3bbaaca974cd164b9280b7d4c39e53b745443b28eea05e03dd`.

## Chronological protocol:

| Role | Interval | Duration |
| --- | --- | ---: |
| Warmup | 1 March to 30 April 2024. | 61 days. |
| Train | 1 May to 30 June 2024. | 61 days. |
| Validation | 2 July to 31 July 2024. | 30 days. |
| Test | 2 August to 31 August 2024. | 30 days. |

One complete day is embargoed between train and validation. One complete day is embargoed between validation and test. The embargo prevents state and target overlap at adjacent boundaries.

## Workload roles:

- Natural one times arrivals are the primary publication workload.
- Two times and four times Spot arrival stress are optional robustness conditions.
- Stress conditions must never replace the natural headline result.
- The current public validation results use natural one times arrivals only.

## Forecast target:

The forecast benchmark predicts four hourly values. Each value is the sum of each observed organization's maximum GPU demand within that hour. This is the quota target used by GFS. It can exceed the maximum simultaneous total because organization maxima may occur at different minutes.

Forecast models receive 672 hours of causal history. Development uses 61 daily origins in May and June. The first 14 origins calibrate forecast uncertainty for quantile evaluation.

## Scheduling target:

The primary scheduling metric is useful Spot GPU hours credited within the score interval. Credited service cannot exceed the requested job duration times declared GPU demand.

The primary service constraint is the fraction of eligible high priority gang arrivals that miss the one hour service condition. Feasible methods require a rate at or below two percent.

## Leakage controls:

- Trace filtering occurs before split evaluation under fixed hashes.
- Training artifacts state `test_outcomes_accessed: false`.
- Forecast features use rows strictly earlier than each origin.
- Forecast thresholds and uncertainty use training history only.
- Validation selects methods and parameters.
- Test execution requires an external access grant that is absent from this repository.
