# Forecasting Methods:

## Shared protocol:

Every model receives the same 672 hour causal history and predicts four hourly organization demand values. Point metrics use all 61 development origins. Stochastic models use three seeds generated from master seed 17. Deterministic models run once.

## Supported baselines:

- Seasonal Naive copies the demand from the same hour one week earlier.
- DLinear separates moving trend and residual components, then applies linear projections.
- NLinear subtracts the last value, applies a shared linear map, then restores the last value.
- PatchTST applies a channel independent Transformer to temporal patches.
- Temporal Mamba applies the official Mamba state space block to daily patches.
- OrgLinear uses the model bundled with GFS.

## SRLinear:

SRLinear is the proposed lightweight forecaster. It contains 2,052 trainable parameters in the evaluated configuration.

Its input paths are:

1. Seasonal anchors at 24, 48, 168, 336, and 672 hours.
2. Recent means and slopes at 4, 12, 24, 72, and 168 hours.
3. A 48 hour residual history.
4. Causal hour and weekday features.

Horizon specific convex gates mix the seasonal anchors. Linear residual heads predict organization corrections. A separate aggregate head predicts total demand. Exact proportional reconciliation makes the final organization forecasts sum to the aggregate forecast.

The normalized aggregate loss is:

\[
L = L_{MAE} + 0.75L_{RMSE} + 0.25L_{tail} + 0.20L_{peak} + 0.25L_{peak\ under}.
\]

The model also uses organization MAE with weight `0.15` and gate regularization with weight `0.0001`. Peak thresholds are estimated from the causal training window only.

## Removed development models:

OrgMamba, pooled GBDT, deterministic SRLinear, shrinkage SRLinear, and earlier loss variants were development experiments. Their executable code is excluded from this release.
