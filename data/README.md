# Data Setup:

## Source:

Download the Alibaba GPU Spot trace from the Alibaba Cluster Trace Program repository:

`https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2026-spot-gpu`.

This repository does not redistribute raw trace files.

## Required layout:

```text
data/
└── raw/
    └── spot_gpu_v2026/
        ├── node_info_df.csv
        ├── job_info_df.csv
        └── SHA256SUMS
```

## Verification:

Run this command inside WSL from the repository root:

```bash
cd data/raw/spot_gpu_v2026
sha256sum --check SHA256SUMS
```

Both files must report `OK`. Do not continue if a hash differs.

## Frozen use:

- Primary experiments filter `A100-SXM4-80GB` nodes only.
- Trace commit and hashes are fixed in `environment/protocol.yaml`.
- Raw files remain ignored by Git.
- Generated snapshots remain under `runs/` and are also ignored.
