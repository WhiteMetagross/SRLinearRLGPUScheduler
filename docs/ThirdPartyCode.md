# Third Party Code:

## GFS:

The frozen simulator contains code derived from the released GFS artifact under `environment/src/research_sim/gfs/`. Its GNU General Public License version 3 is preserved at the repository root and at `environment/src/research_sim/gfs/LICENSE`.

Local changes provide explicit research interfaces, total GPU demand accounting, common metrics, snapshot checks, and source compatible scheduler profiles. These changes do not imply endorsement by the original authors.

## Alibaba trace:

The trace is acquired from the Alibaba Cluster Trace Program. Raw files are not redistributed by this repository. Source revision and file hashes are recorded in `environment/protocol.yaml` and `data/raw/spot_gpu_v2026/SHA256SUMS`.

## Forecast baselines:

- OrgLinear comes from the GFS artifact.
- DLinear and NLinear retain their defining linear semantics through local PyTorch implementations.
- PatchTST is a compact local adaptation using the PyTorch Transformer encoder.
- Temporal Mamba imports the official `mamba_ssm.Mamba` implementation.

External baseline names describe method families. Numerical reproduction claims require the original data and protocol, which differ from this benchmark.
