from types import SimpleNamespace

from research_sim.core import ResearchSimulator


def _spot(job_id, start_time):
    return {
        "job_name": job_id,
        "type": "Spot",
        "status": "run",
        "start_time": start_time,
        "worker_num": 1,
        "gpu_request": 1,
    }


def test_minimum_loss_cards_use_ledger_rollback_cost_and_unique_victims():
    simulator = object.__new__(ResearchSimulator)
    simulator.time = 1000
    simulator.args = SimpleNamespace(ckpt_interval=3600)
    expensive = _spot(1, 0)
    cheap = _spot(2, 900)
    node = SimpleNamespace(
        node_gpu_dict={
            0: [expensive],
            1: [cheap],
            2: [cheap],
        }
    )
    selected, victims, key = simulator._minimum_loss_card_set(
        node, [0, 1, 2], 2
    )
    assert selected == [1, 2]
    assert list(victims) == [2]
    assert key[0] == 100.0
    assert key[1] == 1


def test_srgfs_scheduler_profile_is_opt_in():
    simulator = object.__new__(ResearchSimulator)
    simulator.cluster = SimpleNamespace(vc_dict={})
    simulator.set_scheduler_profile("srgfs_cost_aware")
    assert simulator.scheduler_profile == "srgfs_cost_aware"
