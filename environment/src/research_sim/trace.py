"""
trace.py: Immutable trace loader, integrity verification, and time mapping.
Loads official Spot GPU CSV traces once, verifies SHA256 checksums, and produces
reproducible, isolated trace views for episodes without modifying raw artifacts.
"""

import os
import hashlib
import math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from .types import EpisodeSpec, InvalidConfigError

EXPECTED_NODE_SHA256 = "1aba161961a5a4a1a61aa581383c5e5abe3400b59f8597ba8c4eef7597bc9d18"
EXPECTED_JOB_SHA256 = "113ccee4c28f5c3bbaaca974cd164b9280b7d4c39e53b745443b28eea05e03dd"


def compute_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class EpisodeTraceView:
    """Immutable view of trace data tailored to an EpisodeSpec."""
    def __init__(
        self,
        node_df: pd.DataFrame,
        jobs_df: pd.DataFrame,
        spec: EpisodeSpec,
        hashes: Dict[str, str],
        workload_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.spec = spec
        self.hashes = hashes
        self.workload_metadata = dict(workload_metadata or {})
        self._node_df = node_df.copy()
        self._jobs_df = jobs_df.copy()

        # Standardize job columns for GFS compatibility
        if "type" not in self._jobs_df.columns and "job_type" in self._jobs_df.columns:
            self._jobs_df["type"] = self._jobs_df["job_type"]
        elif "job_type" not in self._jobs_df.columns and "type" in self._jobs_df.columns:
            self._jobs_df["job_type"] = self._jobs_df["type"]

        if "vc_name" not in self._jobs_df.columns:
            self._jobs_df["vc_name"] = self._jobs_df["gpu_model"]
        if "org_key" not in self._jobs_df.columns:
            org_series = self._jobs_df["organization"] if "organization" in self._jobs_df.columns else 0
            self._jobs_df["org_key"] = self._jobs_df["gpu_model"].astype(str) + "&&cluster&&" + org_series.astype(str)
        if "remain" not in self._jobs_df.columns:
            self._jobs_df["remain"] = self._jobs_df["duration"]
        if "guarantee_request" not in self._jobs_df.columns:
            self._jobs_df["guarantee_request"] = 1
        if "spot_duration" not in self._jobs_df.columns:
            self._jobs_df["spot_duration"] = 3600
        if "job_index" not in self._jobs_df.columns:
            self._jobs_df["job_index"] = self._jobs_df["job_name"]
        if "status" not in self._jobs_df.columns:
            self._jobs_df["status"] = None

        # Build index of arrivals by integer second
        self.node_ids: Tuple[int, ...] = tuple(sorted(self._node_df["node_name"].astype(int).unique()))
        self.total_nodes = len(self.node_ids)
        self.total_gpus = int(self._node_df["gpu_capacity_num"].sum())

        # The released trace parser preserves CSV row order for equal timestamps.
        self._jobs_df.sort_values(by="submit_time", kind="stable", inplace=True)
        self._jobs_df.reset_index(drop=True, inplace=True)

        self.job_counts_by_class: Dict[str, int] = {
            str(k): int(v) for k, v in self._jobs_df["job_type"].value_counts().items()
        }
        self.total_jobs = len(self._jobs_df)
        hash_columns = [
            "job_name", "submit_time", "job_type", "gpu_model",
            "gpu_request", "worker_num", "duration",
        ]
        canonical_jobs = self._jobs_df[hash_columns].to_csv(index=False, lineterminator="\n")
        self.workload_sha256 = hashlib.sha256(canonical_jobs.encode("utf-8")).hexdigest()
        prefix_jobs = self._jobs_df[self._jobs_df["submit_time"] < spec.score_start][hash_columns]
        canonical_prefix = prefix_jobs.to_csv(index=False, lineterminator="\n")
        self.prefix_workload_sha256 = hashlib.sha256(canonical_prefix.encode("utf-8")).hexdigest()

        # Preindex jobs by integer submit_time second
        self._arrivals_by_sec: Dict[int, List[Dict[str, Any]]] = {}
        for row in self._jobs_df.to_dict(orient="records"):
            t = int(round(row["submit_time"]))
            if t not in self._arrivals_by_sec:
                self._arrivals_by_sec[t] = []
            self._arrivals_by_sec[t].append(row)

    @property
    def node_df(self) -> pd.DataFrame:
        return self._node_df.copy()

    @property
    def jobs_df(self) -> pd.DataFrame:
        return self._jobs_df.copy()

    @property
    def job_df(self) -> pd.DataFrame:
        return self._jobs_df.copy()

    def get_arrivals_at(self, second: int) -> List[Dict[str, Any]]:
        """Returns fresh mutable job dictionaries arriving at integer second."""
        if second not in self._arrivals_by_sec:
            return []
        import sys
        return [{
            **job_dict,
            "nodes": [],
            "priority": -1,
            "random": 0,
            "status": None,
            "start_time": sys.maxsize,
            "end_time": sys.maxsize,
            "remain": job_dict.get("remain", job_dict.get("duration", 0.0)),
            "ckpt_times": 0,
            "preempt_times": 0
        } for job_dict in self._arrivals_by_sec[second]]

    def get_all_jobs_fresh(self) -> List[Dict[str, Any]]:
        """Returns fresh mutable job dictionaries for all jobs in view."""
        import sys
        return [{
            **job_dict,
            "nodes": [],
            "priority": -1,
            "random": 0,
            "status": None,
            "start_time": sys.maxsize,
            "end_time": sys.maxsize,
            "remain": job_dict.get("remain", job_dict.get("duration", 0.0)),
            "ckpt_times": 0,
            "preempt_times": 0
        } for job_dict in self._jobs_df.to_dict(orient="records")]


class TraceStore:
    """Singleton-like loader for raw trace CSV files."""
    _instance: Optional["TraceStore"] = None

    def __init__(self, trace_dir: str):
        self.trace_dir = trace_dir
        self.node_path = os.path.join(trace_dir, "node_info_df.csv")
        self.job_path = os.path.join(trace_dir, "job_info_df.csv")

        if not os.path.isfile(self.node_path):
            raise FileNotFoundError(f"node_info_df.csv not found at {self.node_path}")
        if not os.path.isfile(self.job_path):
            raise FileNotFoundError(f"job_info_df.csv not found at {self.job_path}")

        node_sha = compute_sha256(self.node_path)
        job_sha = compute_sha256(self.job_path)

        if node_sha != EXPECTED_NODE_SHA256:
            raise ValueError(
                f"node_info_df.csv SHA256 mismatch: expected {EXPECTED_NODE_SHA256}, got {node_sha}"
            )
        if job_sha != EXPECTED_JOB_SHA256:
            raise ValueError(
                f"job_info_df.csv SHA256 mismatch: expected {EXPECTED_JOB_SHA256}, got {job_sha}"
            )

        self.hashes = {
            "node_sha256": node_sha,
            "job_sha256": job_sha
        }

        # Load once
        self._raw_nodes = pd.read_csv(self.node_path)
        self._raw_jobs = pd.read_csv(self.job_path)

        # Standardize node columns
        if "vc_name" not in self._raw_nodes.columns:
            self._raw_nodes["vc_name"] = self._raw_nodes["gpu_model"]

        # Ensure submit_time is float
        self._raw_jobs["submit_time"] = self._raw_jobs["submit_time"].astype(float)
        self._raw_jobs["duration"] = self._raw_jobs["duration"].astype(float)
        self._raw_jobs["cpu_request"] = self._raw_jobs["cpu_request"].astype(float)
        self._raw_jobs["gpu_request"] = self._raw_jobs["gpu_request"].astype(float)
        self._raw_jobs["worker_num"] = self._raw_jobs["worker_num"].astype(int)

    @classmethod
    def get_instance(cls, trace_dir: str = "data/raw/spot_gpu_v2026") -> "TraceStore":
        if cls._instance is None or cls._instance.trace_dir != trace_dir:
            cls._instance = cls(trace_dir)
        return cls._instance

    def create_episode_view(self, spec: EpisodeSpec) -> EpisodeTraceView:
        """Filters nodes and jobs based on EpisodeSpec."""
        # 1. Filter Nodes
        if spec.node_filter == "all":
            node_df = self._raw_nodes.copy()
        else:
            node_df = self._raw_nodes[self._raw_nodes["gpu_model"] == spec.node_filter].copy()

        if len(node_df) == 0:
            raise InvalidConfigError(f"No nodes match filter: {spec.node_filter}")

        # 2. Filter Jobs
        if spec.node_filter == "all":
            job_df = self._raw_jobs.copy()
        else:
            job_df = self._raw_jobs[self._raw_jobs["gpu_model"] == spec.node_filter].copy()

        # Apply trace inclusion mode
        if spec.trace_mode == "source_compatible":
            # The released preprocessing keeps all HP arrivals in the replay horizon.
            cond_hp = (job_df["job_type"] == "HP") & (job_df["submit_time"] >= 0) & (job_df["submit_time"] <= spec.followup_end)
            cond_spot = (job_df["job_type"] == "Spot") & (job_df["submit_time"] >= spec.score_start) & (job_df["submit_time"] <= spec.score_end)
            job_df = job_df[cond_hp | cond_spot].copy()
        elif spec.trace_mode == "research_full_history":
            # Include both classes from time zero so earlier occupancy survives.
            cond = (job_df["submit_time"] >= 0) & (job_df["submit_time"] <= spec.followup_end)
            job_df = job_df[cond].copy()
        else:
            raise InvalidConfigError(f"Unknown trace_mode: {spec.trace_mode}")

        if len(job_df) == 0:
            raise InvalidConfigError(f"No jobs retained after applying spec: {spec}")

        stress_start = (
            int(spec.spot_stress_start)
            if spec.spot_stress_start is not None
            else int(spec.score_start)
        )
        surge_spots = job_df[
            (job_df["job_type"] == "Spot")
            & (job_df["submit_time"] >= stress_start)
            & (job_df["submit_time"] <= spec.followup_end)
        ].sort_values(["submit_time", "job_name"], kind="stable")
        original_count = int(len(surge_spots))
        prefix_original_count = int(
            ((surge_spots["submit_time"] < spec.score_start)).sum()
        )
        workload_metadata = {
            "workload_mode": (
                "natural"
                if int(spec.spot_arrival_multiplier) == 1
                else "synthetic_spot_arrival_stress"
            ),
            "synthetic_stress_enabled": int(spec.spot_arrival_multiplier) > 1,
            "spot_arrival_multiplier": int(spec.spot_arrival_multiplier),
            "spot_arrival_seed": int(spec.spot_arrival_seed),
            "spot_stress_start": stress_start if spec.spot_stress_start is not None else None,
            "stress_effective_start": stress_start,
            "stress_transform": "none" if int(spec.spot_arrival_multiplier) == 1 else "same_job_shape_copies_after_source_within_300_second_window",
            "stress_original_spot_arrivals": original_count,
            "stress_synthetic_spot_arrivals": 0,
            "stress_prefix_original_spot_arrivals": prefix_original_count,
            "stress_prefix_synthetic_spot_arrivals": 0,
        }
        multiplier = int(spec.spot_arrival_multiplier)
        if multiplier > 1:
            if spec.trace_mode != "research_full_history":
                raise InvalidConfigError(
                    "Spot arrival stress currently requires research_full_history trace mode."
                )
            clones: List[Dict[str, Any]] = []
            if original_count:
                next_job_id = int(self._raw_jobs["job_name"].max()) + 1
                rng = np.random.default_rng(int(spec.spot_arrival_seed))
                for source in surge_spots.to_dict(orient="records"):
                    source_time = float(source["submit_time"])
                    earliest = int(math.ceil(source_time))
                    quota_window_end = (int(source_time // 300) + 1) * 300
                    latest_exclusive = min(quota_window_end, int(spec.followup_end) + 1)
                    for _ in range(multiplier - 1):
                        clone = dict(source)
                        clone["job_name"] = next_job_id
                        next_job_id += 1
                        if latest_exclusive > earliest:
                            clone["submit_time"] = int(rng.integers(earliest, latest_exclusive))
                        else:
                            clone["submit_time"] = int(round(source_time))
                        clones.append(clone)
                if clones:
                    job_df = pd.concat(
                        [job_df, pd.DataFrame.from_records(clones)],
                        ignore_index=True,
                    )
            workload_metadata.update({
                "stress_transform": "same_job_shape_copies_after_source_within_300_second_window",
                "stress_original_spot_arrivals": original_count,
                "stress_synthetic_spot_arrivals": len(clones),
                "stress_prefix_original_spot_arrivals": prefix_original_count,
                "stress_prefix_synthetic_spot_arrivals": sum(
                    1 for clone in clones if float(clone["submit_time"]) < spec.score_start
                ),
            })

        return EpisodeTraceView(node_df, job_df, spec, self.hashes, workload_metadata)
