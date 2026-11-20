import os

import torch

from benchmark.run import _configure_numeric_threads


def test_numeric_thread_configuration_is_enforced():
    previous = int(torch.get_num_threads())
    try:
        report = _configure_numeric_threads({"resources": {"threads_per_worker": 4}})
        assert report["declared_threads_per_worker"] == 4
        assert report["torch_num_threads"] == 4
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            assert report["environment"][name] == "4"
            assert os.environ[name] == "4"
    finally:
        torch.set_num_threads(previous)
