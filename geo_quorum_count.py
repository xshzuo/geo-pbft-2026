"""Quorum formation analysis: how many cross-region messages does a single
PBFT round actually require?

For a global PBFT instance with regions ``(N_1, N_2, ..., N_R)`` and quorum
``q = 2f+1``, each receiver needs to collect ``q`` messages.  Some of those
messages come from within the receiver's own region (zero-cost via
``quorum_completion``'s self-vote) and the rest from other regions.  This
module breaks down the expected number of cross-region messages per round
per region.
"""
from __future__ import annotations

import math
from typing import Sequence


def cross_region_messages_per_round(cfg_nodes_per_region: Sequence[int], q: int) -> dict[str, list[int]]:
    """For each region, compute:
       * self_votes (always 1)
       * intra_region_votes (same-region replicas that must send explicitly)
       * cross_region_votes (replicas from other regions that must send)
    """
    regions = cfg_nodes_per_region
    R = len(regions)
    out = {"region": list(range(R)), "self_votes": [], "intra": [], "cross": []}
    for i, N_i in enumerate(regions):
        # The receiver has 1 self-vote (cost 0).  The other q-1 messages
        # must come from outside.  In the conservative model, every other
        # replica must send its own message.  We assume the protocol does
        # not require explicit intra-region unicast (a single broadcast can
        # satisfy all same-region receivers).
        out["self_votes"].append(1)
        out["intra"].append(N_i - 1)  # same region replicas
        out["cross"].append(sum(regions[j] for j in range(R) if j != i))
    return out


def total_messages_per_round(cfg_nodes_per_region: Sequence[int], q: int) -> int:
    """Total number of prepare/commit messages exchanged per round under
    the conservative model.
    """
    R = len(cfg_nodes_per_region)
    total = 0
    for i in range(R):
        for src_region in range(R):
            total += cfg_nodes_per_region[src_region]
    return total


def messages_breakdown_table(cfg_nodes_per_region: Sequence[int], q: int) -> list[dict[str, int | str]]:
    breakdown = cross_region_messages_per_round(cfg_nodes_per_region, q)
    table = []
    for i, region in enumerate(breakdown["region"]):
        table.append({
            "region": region,
            "self_vote": breakdown["self_votes"][i],
            "intra_region_sources": breakdown["intra"][i],
            "cross_region_sources": breakdown["cross"][i],
            "q": q,
        })
    return table


if __name__ == "__main__":
    for npr in [(30, 25, 25, 20), (10, 10, 10, 10), (1, 1, 1, 97)]:
        N = sum(npr)
        f = (N - 1) // 3
        q = 2 * f + 1
        print(f"NPR={npr} N={N} f={f} q={q}")
        for row in messages_breakdown_table(npr, q):
            print(" ", row)
        print(f"  total messages per round = {total_messages_per_round(npr, q)}")
