from types import SimpleNamespace

import pandas as pd
from pandas.testing import assert_frame_equal

from research_sim.gfs.policy.recorder.recorder import Recorder


def _frame(rows):
    return pd.DataFrame(rows, columns=["time", "org_key", "gpu_request"])


def _recorder():
    return Recorder(SimpleNamespace(logger=None), log_dir=None)


def test_incremental_org_pivot_matches_full_history_rebuild():
    recorder = _recorder()
    first = _frame([
        (0, "A100&&org_a", 2.0),
        (0, "A100&&org_b", 1.0),
        (60, "A100&&org_a", 3.0),
        (60, "A100&&org_b", 0.0),
    ])
    second = _frame([
        (120, "A100&&org_a", 4.0),
        (120, "A100&&org_b", 2.0),
        (120, "A100&&org_c", 5.0),
        (180, "A100&&org_a", 0.0),
        (180, "A100&&org_b", 1.0),
        (180, "A100&&org_c", 6.0),
    ])
    recorder.vc_record_df_list = [pd.DataFrame({"time": [0]})]
    recorder.record_df_list = [pd.DataFrame({"time": [0]})]
    recorder.org_request_df_list = [first]
    assert_frame_equal(
        recorder.agg_seq_recorder(),
        recorder.process_seq_data(first),
    )

    recorder.vc_record_df_list.append(pd.DataFrame({"time": [60]}))
    recorder.record_df_list.append(pd.DataFrame({"time": [60]}))
    recorder.org_request_df_list.append(second)
    incremental = recorder.agg_seq_recorder()
    rebuilt = recorder.process_seq_data(pd.concat((first, second), ignore_index=True))
    assert_frame_equal(incremental, rebuilt)


def test_incremental_org_pivot_returns_cache_without_new_rows():
    recorder = _recorder()
    frame = _frame([(0, "A100&&org_a", 2.0)])
    recorder.vc_record_df_list = [pd.DataFrame({"time": [0]})]
    recorder.record_df_list = [pd.DataFrame({"time": [0]})]
    recorder.org_request_df_list = [frame]
    expected = recorder.agg_seq_recorder()
    assert recorder.agg_seq_recorder() is expected


def test_legacy_snapshot_without_cache_rebuilds_once():
    recorder = _recorder()
    del recorder._org_request_wide_cache
    frame = _frame([
        (0, "A100&&org_a", 2.0),
        (60, "A100&&org_a", 3.0),
    ])
    recorder.vc_record_df_list = [pd.DataFrame({"time": [0]})]
    recorder.record_df_list = [pd.DataFrame({"time": [0]})]
    recorder.org_request_df_list = [frame]
    assert_frame_equal(
        recorder.agg_seq_recorder(),
        recorder.process_seq_data(frame),
    )
    assert recorder.org_request_df_list == []
