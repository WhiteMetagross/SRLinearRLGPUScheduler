from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from research_sim.forecasts import CausalForecastContext, validate_quota_update
from research_sim.gfs.estimator.data_loader import Dataset_GPU_Org
from research_sim.gfs.estimator.gpu_request_estimator import GPURequestEstimator
from research_sim.types import EpisodeSpec, InvalidConfigError


def test_causal_forecast_context_copies_history_and_wraps_mappings():
    history = pd.DataFrame({"pool&&org": [1.0, 2.0]})
    context = CausalForecastContext.build(
        time=300,
        calendar_timestamp="2024-03-01 00:05:00",
        org_request_history=history,
        capacity_by_pool={"pool": 8},
        hp_used_by_pool={"pool": 2},
        spot_used_by_pool={"pool": 1},
        current_quota_by_pool={"pool": {"quota": {1: 4}, "eta": 1.0}},
        eviction_outcomes_by_pool={"pool": (3, 1)},
        guarantee_rate=0.9,
        guarantee_hours=[1],
    )

    history.iloc[0, 0] = 999.0
    assert context.org_request_history.iloc[0, 0] == 1.0
    assert context.history_end_exclusive == 300
    assert isinstance(context.capacity_by_pool, MappingProxyType)
    with pytest.raises(TypeError):
        context.capacity_by_pool["pool"] = 4


def test_external_quota_output_validation_is_strict():
    valid = validate_quota_update(
        {"pool": {"quota": {1: 6}, "eta": 0.9}},
        capacity_by_pool={"pool": 8},
        guarantee_hours=[1],
    )
    assert valid == {"pool": {"quota": {1: 6}, "eta": 0.9}}

    with pytest.raises(ValueError, match="within"):
        validate_quota_update(
            {"pool": {"quota": {1: 9}, "eta": 1.0}},
            capacity_by_pool={"pool": 8},
            guarantee_hours=[1],
        )
    with pytest.raises(ValueError, match="pools"):
        validate_quota_update(
            {}, capacity_by_pool={"pool": 8}, guarantee_hours=[1]
        )


def test_external_forecast_requires_explicit_identity():
    with pytest.raises(InvalidConfigError, match="explicit forecast_provider_id"):
        EpisodeSpec(forecast_mode="external_causal")

    spec = EpisodeSpec(
        forecast_mode="external_causal",
        forecast_provider_id="proposed_v1",
        forecast_provider_version="0.1.0",
    )
    assert spec.forecast_provider_id == "proposed_v1"


def test_native_estimator_detects_new_organization_columns():
    estimator = GPURequestEstimator.__new__(GPURequestEstimator)
    estimator.class_columns = ("pool&&class&&org-a",)

    same = pd.DataFrame(columns=["pool&&class&&org-a"])
    expanded = pd.DataFrame(
        columns=["pool&&class&&org-a", "pool&&class&&org-b"]
    )

    assert estimator.requires_schema_refresh(same) is False
    assert estimator.requires_schema_refresh(expanded) is True


def test_native_estimator_recovers_schema_from_legacy_snapshot_width():
    estimator = GPURequestEstimator.__new__(GPURequestEstimator)
    estimator.class_data = np.zeros((3, 2), dtype=np.int64)
    observed = pd.DataFrame(
        columns=["pool&&class&&org-a", "pool&&class&&org-b"]
    )

    assert estimator.requires_schema_refresh(observed) is False
    assert estimator.class_columns == tuple(observed.columns)


def test_orglinear_hourly_cache_matches_reference_slicing():
    args = type("Args", (), {"num_class": 3, "num_covariates": 3})()
    index = pd.date_range("2024-03-01", periods=45 * 24 * 60, freq="min")
    values = np.arange(len(index) * 2, dtype=np.float64).reshape(len(index), 2)
    frame = pd.DataFrame(
        values,
        index=index,
        columns=["pool&&class&&org-a", "pool&&class&&org-b"],
    )
    dataset = Dataset_GPU_Org(
        args=args,
        flag="train",
        data=frame,
        timestamp=pd.Timestamp("2024-04-10"),
        size=[4, 2, 3],
        scale=False,
    )

    for item_index in (0, 1, 11, 12, len(dataset) - 1):
        seq_x, _, _, seq_y = dataset[item_index]
        s_begin = item_index * dataset.stride_size
        s_end = s_begin + dataset.seq_len * 60
        reference_x = dataset.data_x[s_begin:s_end]
        reference_x = np.max(
            reference_x.reshape(dataset.seq_len, 60, reference_x.shape[1]),
            axis=1,
        )
        r_begin = s_end - dataset.label_len * 60
        r_end = r_begin + (dataset.label_len + dataset.pred_len) * 60
        reference_y = dataset.data_y[r_begin:r_end]
        reference_y = np.max(
            reference_y.reshape(
                dataset.label_len + dataset.pred_len,
                60,
                reference_y.shape[1],
            ),
            axis=1,
        )
        np.testing.assert_array_equal(seq_x, reference_x)
        np.testing.assert_array_equal(seq_y, reference_y)
