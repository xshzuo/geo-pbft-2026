"""Client-visible completion semantics for PBFT.

The conservative all_receiver_quorum mode requires every receiver to obtain
its quorum of messages.  A more realistic completion model uses the PBFT
client-observable commit certificate: once the client has received 2f+1
matching reply messages, the block is considered final.

This module provides:
  * `client_visible_round_time` — time for the slowest necessary client
    response path to deliver f+1 commit messages and 2f+1 reply messages.
  * `reply_quorum_at_client` — q-th arrival at the client (analogous to
    `quorum_completion` but for client-observable completion).
"""
from __future__ import annotations

from typing import Sequence

from geo_model import GeoConfig, node_regions, link_cost


def client_visible_round_time(cfg: GeoConfig) -> dict[str, float]:
    """Compute the time for client-visible commit completion.

    Stages:
      1. pre-prepare  : max over backups of the full block delivery time
      2. commit       : q-th commit message arrival time (conservative)
      3. reply        : q-th reply message arrival time
    """
    cfg.validate()
    # Pre-prepare: every backup must receive the full block.  The
    # primary region has the primary node; for pre-prepare we choose a
    # round-robin primary.  We use region 0 as a representative here.
    primary = 0
    regions = node_regions(cfg)
    backups = list(regions)
    backups.remove(primary)
    pp = max(
        link_cost(cfg, primary, dst, cfg.block_bytes + cfg.message_bytes)
        for dst in backups
    )
    # Commit: each receiver needs q messages; we use the q-th fastest
    # receiver-side quorum completion (client-visible).  This is
    # analogous to client_visible_quorum in stage_times but only the
    # q-th fastest receiver.
    receiver_quorum_times = [
        _receiver_quorum_at(cfg, regions, dst, cfg.message_bytes)
        for dst in regions
    ]
    receiver_quorum_times.sort()
    q = cfg.quorum
    commit_visible = receiver_quorum_times[q - 1]
    # Reply: client needs q reply messages
    reply_costs = sorted(
        link_cost(cfg, src, cfg.client_region, cfg.message_bytes)
        for src in regions
    )
    reply = reply_costs[q - 1]
    return {
        "pre_prepare": pp,
        "commit_visible": commit_visible,
        "reply": reply,
        "total": pp + commit_visible + reply,
    }


def _receiver_quorum_at(cfg: GeoConfig, source_regions: Sequence[int],
                        receiver_region: int, payload_bytes: float) -> float:
    senders = list(source_regions)
    senders.remove(receiver_region)
    costs = [0.0]  # self-vote
    costs.extend(link_cost(cfg, src, receiver_region, payload_bytes) for src in senders)
    return sorted(costs)[cfg.quorum - 1]


def client_visible_vs_conservative(cfg: GeoConfig) -> dict[str, float]:
    """Compare client_visible_round_time against all_receiver_quorum.

    Note: this is a structural comparison; the difference indicates how
    much overestimation the conservative mode introduces.
    """
    from geo_model import stage_times
    conservative = max(stage_times(cfg, p)["total"] for p in range(cfg.regions))
    visible = client_visible_round_time(cfg)
    return {
        "conservative_T_geo": conservative,
        "client_visible_total": visible["total"],
        "ratio": visible["total"] / conservative if conservative > 0 else float("nan"),
    }


if __name__ == "__main__":
    from geo_model import example_config
    cfg = example_config()
    print(client_visible_round_time(cfg))
    print(client_visible_vs_conservative(cfg))
