# Scheduling Methods:

## Scheduling baseline suite:

The public suite contains twelve methods.

| Method | Main capability |
| --- | --- |
| Native GFS | Released forecast, quota, queue order, placement, and preemption. |
| FIFO admission | Native GFS with waiting time priority. |
| Shortest causal estimate first | Native GFS with causal duration priority. |
| Fixed headroom quota | Current free capacity with a train fitted high priority margin. |
| YARN Capacity Scheduler | GFS quota, first come queue, best fit placement, and source style preemption. |
| Chronus | Five minute batching, global twenty minute lease flush, and no live preemption. |
| Lyra | Native queue and quota with preemption cost aware placement. |
| FGD | Native quota with whole card and fractional FGD placement. |
| Capacity open FCFS best fit probe | Full capacity quota with FCFS, best fit, and preemption. |
| Staggered lease best fit probe | Per job leases with turnover cost. |
| Whole card cost best fit probe | Whole card rollback cost placement. |
| Whole card FGD probe | Whole card FGD scoring with native fractional fallback. |

The named scheduler methods are controlled adaptations inside the common event simulator. Mechanism probes are diagnostic controls.

## Current quota methods:

Contextual Linear predicts a bounded quota residual every five minutes from nine causal features. Features include current high priority use, base quota, Spot use, queue pressure, and calendar phase. Coefficients and the safety offset are fitted on the training split.

SRGFS replaces the OrgLinear quota forecast with SRLinear and training calibrated residual Gaussian uncertainty. It keeps native scheduling.

SRGFS Contextual applies a contextual residual fitted on the SRLinear training quota series.

SRGFS Contextual Mix combines a native safe quota and the SRLinear quota. It projects the contextual target onto the interval between those endpoints. It returns to the safe endpoint after observed rollback loss, high priority queueing, a sharp high priority rise, or an eviction failure signal. Its scheduler uses minimum rollback loss victim selection.

## Current interpretation:

Contextual Linear Quota is the strongest observed deterministic quota ruler. SRGFS Contextual Mix provides an SRLinear based risk and throughput tradeoff. SRAC is evaluated against Contextual Linear Quota under the same service constraint.
