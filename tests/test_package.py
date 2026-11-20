import pytest

def test_package_import():
    import research_sim
    assert research_sim.__version__ == "0.1.0"

def test_gfs_import():
    import research_sim.gfs
    from research_sim.gfs import Cluster, Job, Trace, SpotScheduler, FifoWithSpot
    assert Cluster is not None
    assert Job is not None
    assert Trace is not None
    assert SpotScheduler is not None
    assert FifoWithSpot is not None
