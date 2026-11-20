"""
candidates.py: Deterministic, pure complete placement candidate generator for Spot jobs.
Generates complete gang worker placements without mutating live cluster state.
Canonicalizes placements by (node_id, worker_count) key, assigns lowest available
card IDs deterministically, and reserves slot 0 for deferral.
"""

from typing import List, Dict, Tuple, Optional, Any, Set
import numpy as np
from .types import (
    WorkerAllocation,
    Placement,
    Candidate,
    CandidateSet,
    InvalidActionError
)


def has_complete_fit(worker_num: int, gpu_request: int, free_cards_by_node: Dict[int, int]) -> bool:
    """
    Feasibility gate: returns True if and only if sum_n floor(R_n / g_i) >= w_i.
    """
    if gpu_request < 1 or worker_num < 1:
        return False
    total_slots = sum(count // gpu_request for count in free_cards_by_node.values())
    return total_slots >= worker_num


def assign_card_ids(
    allocations_by_node: Dict[int, int],
    gpu_request: int,
    empty_cards_by_node: Dict[int, Tuple[int, ...]]
) -> Tuple[WorkerAllocation, ...]:
    """
    Converts node worker counts to exact WorkerAllocations by taking
    the lowest available card IDs on each chosen node in stable order.
    """
    worker_allocs: List[WorkerAllocation] = []
    for node_id in sorted(allocations_by_node.keys()):
        count = allocations_by_node[node_id]
        available_cards = list(sorted(empty_cards_by_node[node_id]))
        for w in range(count):
            start_idx = w * gpu_request
            end_idx = start_idx + gpu_request
            cards = tuple(available_cards[start_idx:end_idx])
            worker_allocs.append(WorkerAllocation(node_id=node_id, card_ids=cards))
    return tuple(worker_allocs)


def build_candidate_placement(
    allocations_by_node: Dict[int, int],
    gpu_request: int,
    empty_cards_by_node: Dict[int, Tuple[int, ...]]
) -> Placement:
    """Builds a Placement object with canonical key and exact card allocations."""
    # Filter out nodes with 0 workers
    clean_counts = {n: c for n, c in allocations_by_node.items() if c > 0}
    worker_allocs = assign_card_ids(clean_counts, gpu_request, empty_cards_by_node)
    return Placement.from_allocations(list(worker_allocs))


class CandidateGenerator:
    """Pure candidate generator producing complete placements."""

    def __init__(
        self,
        M: int = 8,
        max_local_moves: int = 16,
        geometry_version: str = "legacy_v1",
    ):
        if geometry_version not in {
            "legacy_v1",
            "free_node_extremes_v1",
            "failure_domain_v1",
            "publication_v1",
        }:
            raise ValueError(f"Unknown candidate geometry version: {geometry_version}")
        self.M = M
        self.max_local_moves = max_local_moves
        self.geometry_version = geometry_version

    def generate(
        self,
        job_id: int,
        worker_num: int,
        gpu_request: int,
        empty_cards_by_node: Dict[int, Tuple[int, ...]],
        eviction_scores: Optional[Dict[int, float]] = None,
        node_capacities: Optional[Dict[int, int]] = None,
        spot_occupied_cards: Optional[Dict[int, int]] = None,
        state_version: int = 0,
        time: int = 0
    ) -> CandidateSet:
        free_counts = {n: len(cards) for n, cards in empty_cards_by_node.items()}

        # 0. Check feasibility gate
        if not has_complete_fit(worker_num, gpu_request, free_counts):
            # Feasibility gate failed: only deferral is valid
            defer_cand = Candidate(
                slot=0,
                source="defer",
                placement=Placement.make_deferral(),
                features=(0.0, 0.0, 0.0, 0.0)
            )
            return CandidateSet(
                state_version=state_version,
                job_id=job_id,
                time=time,
                candidates=(defer_cand,),
                truncation_count=0,
                unique_count=0
            )

        unique_placements: Dict[Tuple[Tuple[int, int], ...], Tuple[str, Placement]] = {}

        # 1. Fallback: Reserve a complete deterministic fallback first. Provenance is always 'fallback'.
        fallback_allocs = self._generate_fallback(worker_num, gpu_request, free_counts)
        if fallback_allocs:
            p_fallback = build_candidate_placement(fallback_allocs, gpu_request, empty_cards_by_node)
            unique_placements[p_fallback.key] = ("fallback", p_fallback)

        # 2. Native GFS: Add only when distinct from fallback so duplicate native does not overwrite fallback provenance.
        native_allocs = self._generate_native(
            worker_num, gpu_request, free_counts, eviction_scores,
            node_capacities, spot_occupied_cards
        )
        if native_allocs:
            p_native = build_candidate_placement(native_allocs, gpu_request, empty_cards_by_node)
            if p_native.key not in unique_placements:
                unique_placements[p_native.key] = ("native_gfs", p_native)

        # Failure-domain geometry adds the complete placement that uses the
        # fewest possible nodes. It is opt-in so legacy candidate sets remain
        # bit-for-bit stable for existing runs and snapshots.
        if self.geometry_version in ("failure_domain_v1", "publication_v1"):
            min_node_allocs = self._generate_min_nodes(
                worker_num=worker_num,
                gpu_request=gpu_request,
                free_counts=free_counts,
            )
            if min_node_allocs:
                p_min_nodes = build_candidate_placement(
                    min_node_allocs, gpu_request, empty_cards_by_node
                )
                if p_min_nodes.key not in unique_placements:
                    unique_placements[p_min_nodes.key] = (
                        "failure_domain_min_nodes",
                        p_min_nodes,
                    )

        # Explicit geometry variants make the candidate set span placements
        # that preserve or consume currently empty full nodes. They use only
        # current free cards and node capacities, never future HP arrivals.
        if self.geometry_version in ("free_node_extremes_v1", "publication_v1"):
            geometry_candidates = (
                ("preserve_full_nodes", True),
                ("consume_full_nodes", False),
            )
            for source, preserve_full_nodes in geometry_candidates:
                allocations = self._generate_full_node_extreme(
                    worker_num=worker_num,
                    gpu_request=gpu_request,
                    free_counts=free_counts,
                    node_capacities=node_capacities or {},
                    preserve_full_nodes=preserve_full_nodes,
                )
                if allocations:
                    placement = build_candidate_placement(
                        allocations, gpu_request, empty_cards_by_node
                    )
                    if placement.key not in unique_placements:
                        unique_placements[placement.key] = (source, placement)

        # 3. Deterministic round-robin among remaining families (tight_pack, spread)
        remaining_families = []
        pack_allocs = self._generate_tight_pack(worker_num, gpu_request, free_counts)
        if pack_allocs:
            remaining_families.append(("tight_pack", build_candidate_placement(pack_allocs, gpu_request, empty_cards_by_node)))

        spread_allocs = self._generate_spread(worker_num, gpu_request, free_counts)
        if spread_allocs:
            remaining_families.append(("spread", build_candidate_placement(spread_allocs, gpu_request, empty_cards_by_node)))

        for source, p in remaining_families:
            if p.key not in unique_placements:
                unique_placements[p.key] = (source, p)

        # 4. Bounded local moves from existing unique placements
        seed_keys = list(unique_placements.keys())
        moves_tried = 0
        for seed_key in seed_keys:
            if moves_tried >= self.max_local_moves:
                break
            seed_source, seed_p = unique_placements[seed_key]
            local_allocs_list = self._generate_local_moves(
                seed_p, gpu_request, free_counts, max_moves=self.max_local_moves - moves_tried
            )
            for l_allocs in local_allocs_list:
                moves_tried += 1
                p_loc = build_candidate_placement(l_allocs, gpu_request, empty_cards_by_node)
                if p_loc.key not in unique_placements:
                    unique_placements[p_loc.key] = ("local_move", p_loc)
                if moves_tried >= self.max_local_moves:
                    break

        total_unique = len(unique_placements)
        truncation_count = max(0, total_unique - self.M)

        # Assemble Candidate list: Slot 0 = Deferral
        candidates: List[Candidate] = [
            Candidate(
                slot=0,
                source="defer",
                placement=Placement.make_deferral(),
                features=(0.0, 0.0, 0.0, 0.0)
            )
        ]

        # Slots 1..M
        for slot, (key, (source, placement)) in enumerate(list(unique_placements.items())[:self.M], start=1):
            nodes_used = len(placement.key)
            candidates.append(
                Candidate(
                    slot=slot,
                    source=source,
                    placement=placement,
                    features=(float(nodes_used), float(worker_num), float(gpu_request))
                )
            )

        return CandidateSet(
            state_version=state_version,
            job_id=job_id,
            time=time,
            candidates=tuple(candidates),
            truncation_count=truncation_count,
            unique_count=total_unique
        )

    def _generate_full_node_extreme(
        self,
        worker_num: int,
        gpu_request: int,
        free_counts: Dict[int, int],
        node_capacities: Dict[int, int],
        preserve_full_nodes: bool,
    ) -> Optional[Dict[int, int]]:
        scratch = dict(free_counts)

        def order_key(node_id: int):
            capacity = int(node_capacities.get(node_id, free_counts[node_id]))
            is_full = free_counts[node_id] >= capacity
            if preserve_full_nodes:
                return (is_full, free_counts[node_id], node_id)
            return (not is_full, -free_counts[node_id], node_id)

        allocations: Dict[int, int] = {}
        workers_left = int(worker_num)
        for node_id in sorted(scratch, key=order_key):
            slots = scratch[node_id] // int(gpu_request)
            if slots <= 0:
                continue
            assigned = min(slots, workers_left)
            allocations[node_id] = assigned
            workers_left -= assigned
            if workers_left == 0:
                return allocations
        return None

    def _generate_fallback(
        self,
        worker_num: int,
        gpu_request: int,
        free_counts: Dict[int, int]
    ) -> Optional[Dict[int, int]]:
        scratch = free_counts.copy()
        allocs: Dict[int, int] = {}
        rem = worker_num

        for node_id in sorted(scratch.keys()):
            slots = scratch[node_id] // gpu_request
            if slots > 0:
                take = min(slots, rem)
                allocs[node_id] = take
                scratch[node_id] -= take * gpu_request
                rem -= take
                if rem == 0:
                    break

        return allocs if rem == 0 else None

    def _generate_tight_pack(
        self,
        worker_num: int,
        gpu_request: int,
        free_counts: Dict[int, int]
    ) -> Optional[Dict[int, int]]:
        scratch = free_counts.copy()
        allocs: Dict[int, int] = {}

        for _ in range(worker_num):
            candidates = [n for n, f in scratch.items() if f >= gpu_request]
            if not candidates:
                return None
            # Smallest remainder: free[n] - gpu_request
            best_node = min(candidates, key=lambda n: (scratch[n] - gpu_request, n))
            allocs[best_node] = allocs.get(best_node, 0) + 1
            scratch[best_node] -= gpu_request

        return allocs

    def _generate_min_nodes(
        self,
        worker_num: int,
        gpu_request: int,
        free_counts: Dict[int, int],
    ) -> Optional[Dict[int, int]]:
        """Return a complete placement using the minimum number of nodes."""
        slots_by_node = {
            node_id: free_cards // gpu_request
            for node_id, free_cards in free_counts.items()
            if free_cards >= gpu_request
        }
        allocations: Dict[int, int] = {}
        workers_left = int(worker_num)
        for node_id in sorted(slots_by_node, key=lambda n: (-slots_by_node[n], n)):
            assigned = min(slots_by_node[node_id], workers_left)
            if assigned <= 0:
                continue
            allocations[node_id] = assigned
            workers_left -= assigned
            if workers_left == 0:
                return allocations
        return None

    def _generate_spread(
        self,
        worker_num: int,
        gpu_request: int,
        free_counts: Dict[int, int]
    ) -> Optional[Dict[int, int]]:
        scratch = free_counts.copy()
        allocs: Dict[int, int] = {}

        for _ in range(worker_num):
            candidates = [n for n, f in scratch.items() if f >= gpu_request]
            if not candidates:
                return None
            # Prefer 0 workers, then lowest worker count, then largest free cards, then node_id
            best_node = min(
                candidates,
                key=lambda n: (allocs.get(n, 0), -scratch[n], n)
            )
            allocs[best_node] = allocs.get(best_node, 0) + 1
            scratch[best_node] -= gpu_request

        return allocs

    def _generate_native(
        self,
        worker_num: int,
        gpu_request: int,
        free_counts: Dict[int, int],
        eviction_scores: Optional[Dict[int, float]],
        node_capacities: Optional[Dict[int, int]],
        spot_occupied_cards: Optional[Dict[int, int]]
    ) -> Optional[Dict[int, int]]:
        scratch = free_counts.copy()
        allocs: Dict[int, int] = {}
        occupied = dict(spot_occupied_cards or {})

        for _ in range(worker_num):
            candidates = [n for n, f in scratch.items() if f >= gpu_request]
            if not candidates:
                return None

            # Replay Node.get_scheduler_score and stable source node order.
            def score_fn(n):
                sched_score = (scratch[n] + 1) * 100 - occupied.get(n, 0)
                evict = eviction_scores.get(n, 0.0) if eviction_scores else 0.0
                return (sched_score, -evict)

            best_node = min(candidates, key=score_fn)
            allocs[best_node] = allocs.get(best_node, 0) + 1
            scratch[best_node] -= gpu_request
            occupied[best_node] = occupied.get(best_node, 0) + gpu_request

        return allocs

    def _generate_local_moves(
        self,
        seed_placement: Placement,
        gpu_request: int,
        free_counts: Dict[int, int],
        max_moves: int = 4
    ) -> List[Dict[int, int]]:
        # Current node counts
        current_counts: Dict[int, int] = dict(seed_placement.key)
        # Residual free cards after seed placement
        residual_free = free_counts.copy()
        for n, count in current_counts.items():
            residual_free[n] -= count * gpu_request

        new_allocs_list: List[Dict[int, int]] = []

        # Try moving 1 worker from source node to destination node
        src_nodes = sorted(current_counts.keys())
        for src in src_nodes:
            if current_counts[src] <= 0:
                continue
            for dst in sorted(free_counts.keys()):
                if dst == src:
                    continue
                if residual_free[dst] >= gpu_request:
                    candidate_counts = current_counts.copy()
                    candidate_counts[src] -= 1
                    if candidate_counts[src] == 0:
                        del candidate_counts[src]
                    candidate_counts[dst] = candidate_counts.get(dst, 0) + 1
                    new_allocs_list.append(candidate_counts)
                    if len(new_allocs_list) >= max_moves:
                        return new_allocs_list

        return new_allocs_list
