"""Generate frozen paper warmup snapshots serially and record provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


ENVIRONMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ENVIRONMENT_ROOT.parent
sys.path.insert(0, str(ENVIRONMENT_ROOT / "src"))

from research_sim.rl_cli import (
    _paper_manifest_for,
    _validate_trace_hashes,
    build_spec_from_dict,
    load_yaml,
)
from research_sim.trace import TraceStore
from research_sim.warmup_snapshot import generate_warmup_snapshot, get_snapshot_path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", type=Path)
    args = parser.parse_args()

    reports = []
    for config_arg in args.configs:
        config_path = config_arg.resolve()
        cfg = load_yaml(str(config_path))
        spec = build_spec_from_dict(cfg)
        manifest = _paper_manifest_for(str(config_path), cfg, spec)
        if manifest is None:
            raise SystemExit(f"Snapshot generation requires a frozen paper config: {config_path}")
        if spec.forecast_mode != "native_online":
            raise SystemExit(
                "This command supports native_online snapshots only. External causal "
                "providers must be passed explicitly by their method runner."
            )
        trace_dir = Path(cfg.get("trace_dir", "data/raw/spot_gpu_v2026"))
        if not trace_dir.is_absolute():
            trace_dir = REPOSITORY_ROOT / trace_dir
        store = TraceStore.get_instance(str(trace_dir))
        view = store.create_episode_view(spec)
        _validate_trace_hashes(manifest, view)
        snapshot_dir = Path(cfg.get("snapshot_dir", "runs/warmup_snapshots"))
        if not snapshot_dir.is_absolute():
            snapshot_dir = REPOSITORY_ROOT / snapshot_dir
        target = get_snapshot_path(spec, str(snapshot_dir))
        existed = target.is_file()
        started = time.perf_counter()
        generated = generate_warmup_snapshot(
            spec=spec,
            view=view,
            snapshot_dir=str(snapshot_dir),
        )
        elapsed = time.perf_counter() - started
        reports.append({
            "config": str(config_path),
            "split": spec.split_id,
            "snapshot": str(generated),
            "status": "REUSED" if existed else "GENERATED",
            "elapsed_seconds": elapsed,
            "bytes": generated.stat().st_size,
            "sha256": sha256(generated),
            "protocol_id": manifest["protocol_id"],
        })

    print(json.dumps(reports, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
