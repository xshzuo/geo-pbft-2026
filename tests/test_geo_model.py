import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from geo_model import GeoConfig, example_config, stage_times, geo_service_time, queue_metrics


def single_region(n=4, delay=0.0):
    return GeoConfig(
        nodes_per_region=(n,),
        bandwidth_bytes_s=((500e6,),),
        delay_s=((delay,),),
        processing_s_byte=(1e-8,),
        block_capacity=10,
    )


def test_quorum_and_stage_output():
    cfg = example_config()
    stages = stage_times(cfg, 0)
    assert cfg.quorum == 67
    assert set(stages) == {"pre_prepare", "prepare", "commit", "reply", "total"}
    assert stages["total"] == sum(stages[x] for x in ("pre_prepare", "prepare", "commit", "reply"))
    assert all(v >= 0 for v in stages.values())


def test_zero_delay_and_identical_regions_are_symmetric():
    cfg = GeoConfig(
        nodes_per_region=(2, 2),
        bandwidth_bytes_s=((500e6, 500e6), (500e6, 500e6)),
        delay_s=((0.0, 0.0), (0.0, 0.0)),
        processing_s_byte=(1e-8, 1e-8),
        block_capacity=10,
    )
    a, b = stage_times(cfg, 0), stage_times(cfg, 1)
    for key in a:
        assert math.isclose(a[key], b[key], rel_tol=0, abs_tol=1e-15)


def test_queue_metrics_are_stable():
    cfg = single_region()
    metrics = queue_metrics(geo_service_time(cfg), lambda_rate=1.0, k=cfg.block_capacity)
    assert 0 < metrics["alpha"] < 1
    assert metrics["throughput"] > 1
    assert metrics["E_sojourn"] > metrics["T"]


def test_single_region_pre_prepare_excludes_only_primary_node():
    cfg = single_region()
    stages = stage_times(cfg, 0)
    assert stages["pre_prepare"] > 0


def test_quorum_does_not_double_count_local_vote():
    cfg = GeoConfig(
        nodes_per_region=(1, 3),
        bandwidth_bytes_s=((500e6, 500e6), (500e6, 500e6)),
        delay_s=((0.001, 0.200), (0.200, 0.001)),
        processing_s_byte=(0.0, 0.0),
        block_capacity=10,
    )
    # N=4, q=3. Receiver in the one-node region has only one local vote,
    # therefore it needs two 200ms remote votes. Double-counting local would
    # incorrectly return ~1ms.
    from geo_model import node_regions, quorum_completion
    value = quorum_completion(cfg, node_regions(cfg), receiver_region=0,
                              payload_bytes=cfg.message_bytes)
    assert value >= 0.2


def test_unstable_queue_is_rejected():
    cfg = single_region()
    try:
        queue_metrics(geo_service_time(cfg), lambda_rate=1e9, k=cfg.block_capacity)
    except ValueError:
        pass
    else:
        raise AssertionError("unstable queue was accepted")


def test_client_visible_mode_is_not_slower():
    cfg = example_config()
    conservative = stage_times(cfg, 0, "all_receiver_quorum")
    visible = stage_times(cfg, 0, "client_visible_quorum")
    assert visible["prepare"] <= conservative["prepare"]
    assert visible["commit"] <= conservative["commit"]
    assert visible["total"] <= conservative["total"]


def test_unknown_completion_mode_is_rejected():
    cfg = example_config()
    try:
        stage_times(cfg, 0, "not-a-mode")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown completion mode was accepted")
