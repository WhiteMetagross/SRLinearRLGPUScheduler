"""
risk.py: Preemption waste risk model and action filtering.
Predicts P(Y=1 | observation, candidate) using a lightweight PyTorch neural ensemble.
Computes conservative upper-confidence risk score U = mean_risk + kappa * uncertainty.
Enforces explicit action filter modes: 'passthrough', 'risk_model', 'paired_check'.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .types import Placement, ClusterStateSnapshot


from dataclasses import dataclass

RISK_FEATURE_NAMES: Tuple[str, ...] = (
    "cand_valid",
    "cand_source_code",
    "cand_nodes_used",
    "cand_worker_num",
    "cand_gpu_request",
    "job_worker_num",
    "job_gpu_request",
    "job_cpu_request",
    "job_elapsed_queue_time",
    "job_guarantee_request",
    "global_time",
    "global_quota",
    "global_spot_in_use",
    "global_budget_remaining",
    "global_time_since_quota_update",
    "global_queued_hp",
    "global_queued_spot",
    "global_causal_forecast",
)


@dataclass
class FeatureNormalizer:
    means: np.ndarray
    scales: np.ndarray
    feature_names: Tuple[str, ...] = RISK_FEATURE_NAMES
    schema_version: str = "2.0"

    def transform(self, x: np.ndarray) -> np.ndarray:
        if x.shape[-1] != len(self.means):
            raise ValueError(f"Feature dimension mismatch: expected {len(self.means)}, got {x.shape[-1]}")
        return ((x - self.means) / self.scales).astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "feature_names": list(self.feature_names),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureNormalizer":
        return cls(
            means=np.array(d["means"], dtype=np.float32),
            scales=np.array(d["scales"], dtype=np.float32),
            feature_names=tuple(d.get("feature_names", RISK_FEATURE_NAMES)),
            schema_version=d.get("schema_version", "2.0")
        )

    @classmethod
    def fit(cls, X: np.ndarray, feature_names: Tuple[str, ...] = RISK_FEATURE_NAMES) -> "FeatureNormalizer":
        if len(X) == 0:
            raise ValueError("Cannot fit FeatureNormalizer on empty dataset")
        if X.shape[1] != len(feature_names):
            raise ValueError(f"Feature count mismatch: expected {len(feature_names)}, got {X.shape[1]}")
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        scales = np.where(stds < 1e-6, 1.0, stds)
        return cls(means=means.astype(np.float32), scales=scales.astype(np.float32), feature_names=feature_names)


class RiskEnsembleMember(nn.Module):
    def __init__(self, in_dim: int = 18, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RiskEnsembleModel:
    """
    Small neural ensemble predicting preemption waste risk P(Y=1 | o, p).
    Uncertainty quantified via ensemble standard deviation.
    """
    def __init__(
        self,
        in_dim: int = 18,
        ensemble_size: int = 3,
        kappa: float = 1.0,
        tau_safe: float = 0.2,
        seed: int = 42,
        normalizer: Optional[FeatureNormalizer] = None,
        feature_names: Tuple[str, ...] = RISK_FEATURE_NAMES
    ):
        self.in_dim = in_dim
        self.ensemble_size = ensemble_size
        self.kappa = kappa
        self.tau_safe = tau_safe
        self.seed = seed
        self.normalizer = normalizer
        self.feature_names = feature_names
        self.is_fitted = False

        torch.manual_seed(seed)
        self.members = [RiskEnsembleMember(in_dim=in_dim) for _ in range(ensemble_size)]

    def extract_features(
        self,
        candidate_feature_row: np.ndarray,
        job_features: np.ndarray,
        global_features: np.ndarray,
        apply_norm: bool = True
    ) -> np.ndarray:
        """
        Concatenates candidate (5), job (5), and global (8) features into exact 18-dim vector.
        Raises ValueError if dimensions differ or if any value is NaN/Inf.
        Applies feature normalization if fitted and apply_norm=True.
        """
        if len(candidate_feature_row) != 5:
            raise ValueError(f"Expected 5 candidate features, got {len(candidate_feature_row)}")
        if len(job_features) != 5:
            raise ValueError(f"Expected 5 job features, got {len(job_features)}")
        if len(global_features) != 8:
            raise ValueError(f"Expected 8 global features, got {len(global_features)}")

        feat = np.concatenate([candidate_feature_row, job_features, global_features]).astype(np.float32)
        if len(feat) != self.in_dim:
            raise ValueError(f"Extracted feature dimension {len(feat)} does not match model in_dim {self.in_dim}")

        if not np.all(np.isfinite(feat)):
            raise ValueError(f"Extracted features contain NaN or Inf values: {feat}")

        if apply_norm and self.normalizer is not None:
            feat = self.normalizer.transform(feat)

        return feat

    def predict(self, x: np.ndarray) -> Tuple[float, float, float]:
        """
        Returns (mean_risk, uncertainty, U) for a single feature vector x.
        U = mean_risk + kappa * uncertainty.
        """
        if not self.is_fitted:
            # If not yet fitted, return conservative high risk
            return 1.0, 0.0, 1.0

        if not np.all(np.isfinite(x)):
            raise ValueError(f"Input feature vector contains NaN or Inf values: {x}")

        t_x = torch.from_numpy(x).float().unsqueeze(0)
        preds = []
        with torch.no_grad():
            for member in self.members:
                member.eval()
                p = float(member(t_x).item())
                preds.append(p)

        mean_risk = float(np.mean(preds))
        uncertainty = float(np.std(preds))
        u_score = min(1.0, mean_risk + self.kappa * uncertainty)
        return mean_risk, uncertainty, u_score

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int = 25,
        lr: float = 0.005,
        batch_size: int = 16
    ) -> Dict[str, Any]:
        """
        Trains ensemble strictly on resolved labels Y in {0, 1}.
        """
        if len(X) == 0 or len(Y) == 0:
            return {"loss": 0.0, "samples": 0}

        if not np.all(np.isfinite(X)):
            raise ValueError("Training features X contain NaN or Inf values")
        if not np.all(np.isfinite(Y)):
            raise ValueError("Training labels Y contain NaN or Inf values")

        t_X = torch.from_numpy(X).float()
        t_Y = torch.from_numpy(Y).float().unsqueeze(1)
        loss_fn = nn.BCELoss()

        total_loss = 0.0
        n_samples = len(X)

        for m_idx, member in enumerate(self.members):
            member.train()
            optimizer = optim.Adam(member.parameters(), lr=lr)
            # Bootstrap sample for this member with independent deterministic seed
            rng = np.random.default_rng(self.seed + m_idx * 100)
            indices = rng.integers(0, n_samples, size=n_samples)
            m_X = t_X[indices]
            m_Y = t_Y[indices]

            for epoch in range(epochs):
                permutation = torch.randperm(n_samples)
                for i in range(0, n_samples, batch_size):
                    batch_idx = permutation[i:i + batch_size]
                    b_x, b_y = m_X[batch_idx], m_Y[batch_idx]
                    optimizer.zero_grad()
                    out = member(b_x)
                    loss = loss_fn(out, b_y)
                    loss.backward()
                    optimizer.step()
            total_loss += float(loss.item())

        self.is_fitted = True
        return {
            "loss": total_loss / self.ensemble_size,
            "samples": n_samples,
            "positive_count": int(np.sum(Y > 0.5)),
            "negative_count": int(np.sum(Y <= 0.5)),
            "class_balance": float(np.mean(Y))
        }

    def evaluate_validation(self, X: np.ndarray, Y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates held-out calibration on validation labels with model weights fixed.
        Reports:
        - Brier score for mean probability
        - Reliability bins with counts
        - False safe numerator and denominator among true positives (P(Y=1 | U <= tau_safe))
        - Accepted fraction (U <= tau_safe)
        - Confidence interval for false safe rate (Wilson score)
        """
        if len(X) == 0:
            return {"status": "inconclusive", "reason": "empty_validation_set"}

        means, stdevs, u_scores = [], [], []
        for x in X:
            m, s, u = self.predict(x)
            means.append(m)
            stdevs.append(s)
            u_scores.append(u)

        means = np.array(means, dtype=np.float32)
        u_scores = np.array(u_scores, dtype=np.float32)
        Y_arr = np.array(Y, dtype=np.float32)

        n_total = len(Y_arr)
        brier = float(np.mean((means - Y_arr) ** 2))
        pos_mask = (Y_arr > 0.5)
        pos_count = int(np.sum(pos_mask))
        neg_count = n_total - pos_count

        # False safe means an accepted placement (U <= tau) whose observed label is
        # unsafe. The denominator is accepted placements, matching the deployment
        # safety claim P(Y=1 | accepted), not all positive labels.
        accepted_mask = u_scores <= self.tau_safe
        accepted_count = int(np.sum(accepted_mask))
        false_safe_num = int(np.sum(accepted_mask & pos_mask))
        if accepted_count > 0 and pos_count > 0:
            false_safe_rate = float(false_safe_num / accepted_count)
            # Wilson score interval for false safe rate
            z = 1.96
            p_hat = false_safe_rate
            denom = 1 + z**2 / accepted_count
            centre = (p_hat + z**2 / (2 * accepted_count)) / denom
            spread = z * np.sqrt(p_hat * (1 - p_hat) / accepted_count + z**2 / (4 * accepted_count**2)) / denom
            ci_lower = max(0.0, float(centre - spread))
            ci_upper = min(1.0, float(centre + spread))
        else:
            false_safe_num = 0
            false_safe_rate = None
            ci_lower = None
            ci_upper = None

        # Accepted fraction
        accepted_fraction = float(accepted_count / n_total)

        # Reliability bins (10 equal bins in [0, 1])
        bins = []
        bin_edges = np.linspace(0.0, 1.0, 11)
        for b_idx in range(10):
            low, high = bin_edges[b_idx], bin_edges[b_idx + 1]
            in_bin = (means >= low) & (means < high if b_idx < 9 else means <= high)
            b_count = int(np.sum(in_bin))
            b_pred = float(np.mean(means[in_bin])) if b_count > 0 else None
            b_obs = float(np.mean(Y_arr[in_bin])) if b_count > 0 else None
            bins.append({
                "bin_index": b_idx,
                "range": [round(float(low), 2), round(float(high), 2)],
                "count": b_count,
                "mean_pred": round(b_pred, 4) if b_pred is not None else None,
                "empirical_rate": round(b_obs, 4) if b_obs is not None else None
            })

        return {
            "brier_score": round(brier, 4),
            "sample_count": n_total,
            "positive_count": pos_count,
            "negative_count": neg_count,
            "false_safe_numerator": false_safe_num,
            "false_safe_denominator": accepted_count,
            "false_safe_accepted_denominator": accepted_count,
            "false_safe_positive_denominator_legacy": pos_count,
            "false_safe_rate": round(false_safe_rate, 4) if false_safe_rate is not None else None,
            "false_safe_ci_95": [round(ci_lower, 4), round(ci_upper, 4)] if ci_lower is not None else None,
            "accepted_fraction": round(accepted_fraction, 4),
            "reliability_bins": bins,
            "inconclusive": (accepted_count == 0 or pos_count == 0)
        }

    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> Dict[str, Any]:
        """Legacy compatibility wrapper around evaluate_validation."""
        return self.evaluate_validation(X, Y)


import hashlib
from pathlib import Path


class ActionFilter:
    """
    Implements four explicit filter modes:
    1. 'passthrough': Executes legal proposed placement without checks.
    2. 'risk_model': Uses conservative risk score U <= tau_safe for v=0.
    3. 'paired_check': Uses solver check for v=1, and risk model for v=0.
    4. 'label_collection': Dedicated training data collection mode routing v=1 checks via BudgetManager.
    """
    VALID_MODES = ("passthrough", "risk_model", "paired_check", "label_collection")

    def __init__(
        self,
        mode: str = "passthrough",
        risk_model: Optional[RiskEnsembleModel] = None,
        verifier: Optional[Any] = None
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown filter mode: {mode}. Must be one of {self.VALID_MODES}")
        self.mode = mode
        self.risk_model = risk_model
        self.verifier = verifier

        if self.mode in ("risk_model", "paired_check"):
            if self.risk_model is None or not self.risk_model.is_fitted:
                raise ValueError(f"Filter mode '{self.mode}' requires a fitted risk model.")
        if self.mode in ("paired_check", "label_collection"):
            if self.verifier is None or not getattr(self.verifier, "is_ready", True):
                raise ValueError(f"Filter mode '{self.mode}' requires an active, ready verifier.")

    def filter_decision(
        self,
        proposed_slot: int,
        verify_flag: int,
        context: Any,
        observation: Dict[str, np.ndarray],
        snapshot: ClusterStateSnapshot,
        budget_manager: Optional[Any] = None,
        K: int = 4,
        decision_id: Optional[int] = None
    ) -> Tuple[int, int, Dict[str, Any]]:
        """
        Returns (executed_slot, actual_calls, filter_details).
        Slot 0 is always deferral.
        """
        if proposed_slot == 0:
            # Proposed deferral is always executed as deferral with 0 calls
            return 0, 0, {"mode": self.mode, "reason": "proposed_deferral"}

        cand_set = context.candidates.candidates
        if proposed_slot >= len(cand_set):
            return 0, 0, {"mode": self.mode, "reason": "out_of_range_slot"}
        candidate = cand_set[proposed_slot]

        # Mode 1: Passthrough
        if self.mode == "passthrough":
            return proposed_slot, 0, {"mode": "passthrough", "approved": True}

        # Mode 2: Risk Model only (no solver calls)
        if self.mode == "risk_model":
            if self.risk_model is None or not self.risk_model.is_fitted:
                raise RuntimeError("Filter mode 'risk_model' requires a fitted risk model.")
            cand_feat = observation["candidate_features"][proposed_slot]
            job_feat = observation["job_features"]
            glob_feat = observation["global_features"]
            x = self.risk_model.extract_features(cand_feat, job_feat, glob_feat)
            m, s, u = self.risk_model.predict(x)
            approved = bool(u <= self.risk_model.tau_safe)
            exec_slot = proposed_slot if approved else 0
            return exec_slot, 0, {
                "mode": "risk_model",
                "approved": approved,
                "mean_risk": m,
                "uncertainty": s,
                "u_score": u
            }

        # Mode 3: Paired Check
        if self.mode == "paired_check":
            if self.risk_model is None or not self.risk_model.is_fitted:
                raise RuntimeError("Filter mode 'paired_check' requires a fitted risk model before rollout.")

            if verify_flag == 1:
                # Requested solver check - must route through BudgetManager
                if budget_manager is None:
                    raise RuntimeError("BudgetManager is required to authorize and execute scenario check.")
                executed_p, actual_calls, result = budget_manager.execute_check(
                    verifier=self.verifier,
                    snapshot=snapshot,
                    placement=candidate.placement,
                    job_info=context.job_info,
                    decision_id=decision_id
                )
                exec_slot = proposed_slot if result.approved else 0
                return exec_slot, actual_calls, {
                    "mode": "paired_check",
                    "path": "solver_check",
                    "decision_id": decision_id,
                    "approved": result.approved,
                    "calls_used": actual_calls,
                    "timeout": result.timeout,
                    "check_outcome": result.details.get("check_outcome", result.details),
                    "details": result.details
                }
            else:
                # v=0: check not requested -> use risk model
                cand_feat = observation["candidate_features"][proposed_slot]
                job_feat = observation["job_features"]
                glob_feat = observation["global_features"]
                x = self.risk_model.extract_features(cand_feat, job_feat, glob_feat)
                m, s, u = self.risk_model.predict(x)
                approved = bool(u <= self.risk_model.tau_safe)
                exec_slot = proposed_slot if approved else 0
                return exec_slot, 0, {
                    "mode": "paired_check",
                    "path": "risk_model",
                    "decision_id": decision_id,
                    "approved": approved,
                    "mean_risk": m,
                    "uncertainty": s,
                    "u_score": u
                }

        # Mode 4: Label collection (dedicated training data collection)
        if self.mode == "label_collection":
            if verify_flag == 1:
                if budget_manager is None:
                    raise RuntimeError("BudgetManager is required to authorize and execute scenario check.")
                if not budget_manager.can_verify(is_deferral=candidate.placement.is_deferral, verifier_ready=getattr(self.verifier, "is_ready", False)):
                    return 0, 0, {"mode": "label_collection", "decision_id": decision_id, "reason": "budget_or_readiness_blocked", "approved": False}
                executed_p, actual_calls, result = budget_manager.execute_check(
                    verifier=self.verifier,
                    snapshot=snapshot,
                    placement=candidate.placement,
                    job_info=context.job_info,
                    decision_id=decision_id
                )
                exec_slot = proposed_slot if result.approved else 0
                return exec_slot, actual_calls, {
                    "mode": "label_collection",
                    "path": "solver_check",
                    "decision_id": decision_id,
                    "approved": result.approved,
                    "calls_used": actual_calls,
                    "timeout": result.timeout,
                    "check_outcome": result.details.get("check_outcome", result.details),
                    "details": result.details
                }
            else:
                # v=0 in label collection: unverified actions defer
                return 0, 0, {"mode": "label_collection", "reason": "unverified_defer", "approved": False}

        return 0, 0, {"mode": self.mode, "reason": "fallback_defer"}


def save_risk_checkpoint(
    path: str,
    model: RiskEnsembleModel,
    extra_meta: Optional[Dict[str, Any]] = None
) -> str:
    """
    Saves fitted RiskEnsembleModel weights, normalizer, and provenance metadata to disk.
    Asserts model is fitted. Returns SHA256 hex digest of the saved file.
    """
    if not model.is_fitted:
        raise ValueError("Cannot save unfitted RiskEnsembleModel checkpoint.")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "feature_dim": model.in_dim,
        "ensemble_size": model.ensemble_size,
        "kappa": model.kappa,
        "tau_safe": model.tau_safe,
        "seed": model.seed,
        "feature_names": list(model.feature_names),
        "feature_order": ["candidate_features", "job_features", "global_features"]
    }
    if extra_meta:
        meta.update(extra_meta)

    # A model trained directly on raw features has no learned transform. Store
    # an explicit identity transform so every persisted checkpoint still has a
    # complete, auditable feature pipeline. Production fitting stores its real
    # training normalizer here.
    if model.normalizer is None:
        checkpoint_normalizer = FeatureNormalizer(
            means=np.zeros(model.in_dim, dtype=np.float32),
            scales=np.ones(model.in_dim, dtype=np.float32),
            feature_names=tuple(model.feature_names),
        )
    else:
        checkpoint_normalizer = model.normalizer

    payload = {
        "schema_version": "2.0",
        "in_dim": model.in_dim,
        "ensemble_size": model.ensemble_size,
        "kappa": model.kappa,
        "tau_safe": model.tau_safe,
        "seed": model.seed,
        "is_fitted": True,
        "feature_names": list(model.feature_names),
        "normalizer": checkpoint_normalizer.to_dict(),
        "members": [m.state_dict() for m in model.members],
        "metadata": meta
    }

    torch.save(payload, p)
    with open(p, "rb") as f:
        ckpt_hash = hashlib.sha256(f.read()).hexdigest()
    return ckpt_hash


def load_risk_checkpoint(
    path: str,
    expected_in_dim: int = 18,
    expected_config_hash: Optional[str] = None,
    expected_K: Optional[int] = None,
) -> RiskEnsembleModel:
    """
    Validates and loads a fitted RiskEnsembleModel from checkpoint.
    Validates schema version ('2.0'), is_fitted, input dimension, feature names, and config hash if supplied.
    Fails loudly if checkpoint is missing, unfitted, or incompatible.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Risk model checkpoint not found: {path}")

    try:
        payload = torch.load(p, weights_only=False)
    except Exception as exc:
        raise ValueError(f"Failed to load risk model checkpoint from {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid risk model checkpoint at {path}: expected dict, got {type(payload)}")

    schema = payload.get("schema_version")
    if schema != "2.0":
        raise ValueError(f"Incompatible risk model schema version '{schema}' in {path}; expected '2.0'")

    if not payload.get("is_fitted", False):
        raise ValueError(f"Risk model checkpoint at {path} is not fitted (is_fitted=False)")

    in_dim = payload.get("in_dim")
    if in_dim != expected_in_dim:
        raise ValueError(f"Risk model checkpoint feature dimension mismatch: got {in_dim}, expected {expected_in_dim}")

    stored_features = payload.get("feature_names")
    if stored_features is None:
        raise ValueError(f"Risk model checkpoint at {path} has no feature schema metadata")
    if tuple(stored_features) != RISK_FEATURE_NAMES:
        raise ValueError(f"Risk model feature schema mismatch in {path}: expected {RISK_FEATURE_NAMES}, got {stored_features}")

    meta = payload.get("metadata", {})
    if expected_K is not None:
        stored_k = meta.get("K")
        if stored_k is None:
            raise ValueError(f"Risk model checkpoint at {path} has no scenario count K metadata")
        if int(stored_k) != int(expected_K):
            raise ValueError(
                f"Risk model checkpoint K mismatch: got {stored_k}, expected {expected_K}"
            )
    if expected_config_hash is not None:
        cfg_h = meta.get("config_hash")
        if not cfg_h:
            raise ValueError(f"Risk model checkpoint at {path} has no config hash metadata")
        if cfg_h != expected_config_hash:
            raise ValueError(f"Risk model checkpoint config hash mismatch: {cfg_h} vs expected {expected_config_hash}")

    ens_size = payload.get("ensemble_size", len(payload.get("members", [])))
    kappa = float(payload.get("kappa", 1.0))
    tau_safe = float(payload.get("tau_safe", 0.2))
    seed = int(payload.get("seed", 42))

    norm_data = payload.get("normalizer")
    if norm_data is None:
        raise ValueError(f"Risk model checkpoint at {path} has no fitted feature normalizer")
    normalizer = FeatureNormalizer.from_dict(norm_data)
    if normalizer.feature_names != RISK_FEATURE_NAMES or len(normalizer.means) != expected_in_dim:
        raise ValueError(f"Risk model normalizer schema mismatch in {path}")
    feat_names = tuple(stored_features)

    model = RiskEnsembleModel(
        in_dim=in_dim,
        ensemble_size=ens_size,
        kappa=kappa,
        tau_safe=tau_safe,
        seed=seed,
        normalizer=normalizer,
        feature_names=feat_names
    )

    members_data = payload.get("members", [])
    if len(members_data) != len(model.members):
        raise ValueError(f"Ensemble size mismatch: weights have {len(members_data)}, model expects {len(model.members)}")

    for member, state in zip(model.members, members_data):
        member.load_state_dict(state)
        member.eval()

    model.is_fitted = True
    return model
