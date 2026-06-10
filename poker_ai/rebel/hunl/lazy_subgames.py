"""Lazy subgame construction + population queueing (p1_design.json STEP 7).

Two pieces, and the place where LAZY meets FUSED:

  * ``SubgameFactory`` -- constructs :class:`SubgameSpec` instances lazily from
    (street, board, pot, stacks, actor, ranges) play/training states, and
    expands a turn spec's river-deal CUT leaf into its 48 runout river specs
    (``expand_to_river``). The card-removal range masking is EXACTLY the
    mapping ``poker_ai/rebel/turn_river.py::turn_leaf_river_cfv`` uses and
    ``terminal_eval.exact_river_cfv_population`` consumes: ranges live on the
    canonical global 1326 index, so hands containing the runout card simply do
    not exist in that river board's local hand set (the per-board gather IS the
    mask). The shared expansion lives here as :func:`river_runout_specs`;
    ``terminal_eval`` delegates to it rather than duplicating the loop.

  * ``PopulationQueue`` -- accumulates specs across boards/beliefs, groups
    them by exact topology bucket (``SubgameSpec.bucket_key``), and flushes a
    bucket to ``population_solver.solve_population`` the moment it reaches its
    memory-derived B_max. :func:`b_max_for` exposes the census memory-model
    arithmetic (autoresearch-session/rebel/hunl_topology_census.json):

        dense_per_elem    = (2*nn*A*H + 4*nn*H + nn_dec*A*H) * value_bytes
        terminal_per_elem = 8 * H*H * terminal_value_bytes   (W/L/T/valid +
                                                              transposes, f32)
        B_max             = floor(budget / (dense_per_elem + terminal_per_elem))

NOTHING GLOBAL IS EVER ENUMERATED: trees are built lazily per (pot, stacks,
first-to-act) entry config through ``tree_builder``'s LRU cache, specs are
built only on request, and the queue only ever sees what callers enqueue
(guarded by test_hunl_lazy_subgames.py's cache-stats spy).
"""
from __future__ import annotations

import numpy as np
import torch

from poker_ai.rebel.hunl import population_solver as pop
from poker_ai.rebel.hunl import subgame_spec as sgs
from poker_ai.rebel.hunl.tree_builder import fast_cfr

__all__ = [
    "DEFAULT_BUDGET_GB",
    "STREET_H",
    "b_max_for",
    "b_max_for_tree",
    "river_runout_specs",
    "SubgameFactory",
    "PopulationQueue",
]

DEFAULT_BUDGET_GB = 6.5  # census protocol: 6.5GB usable of 8GB (3070 Ti)
N_ACTIONS = 9            # solver.py abstraction: fold/call + 6 fracs + all-in

# Local kernel widths per street (the batch runs at local width, not 1326).
STREET_H = {"turn": sgs.TURN_H, "river": sgs.RIVER_H}


# ---------------------------------------------------------------------------
# Memory-derived B_max (the census/memory-model arithmetic)
# ---------------------------------------------------------------------------

def b_max_for(
    street,
    n_nodes,
    budget_gb,
    n_decision=None,
    n_actions=N_ACTIONS,
    value_bytes=8,
    terminal_value_bytes=4,
):
    """Largest fused batch size B for one bucket under a memory budget.

    Exactly the census arithmetic (hunl_topology_census.json ``memory_model``
    + ``b_max_table``): float64 dense kernel tensors per element
    (regret_sum + strategy_sum [nn,A,H] x2, hr/vr/hvals/vvals [nn,H] x4,
    retained per-level strategies ~[nn_dec,A,H]) plus 8 float32 [H,H] terminal
    matrices (W/L/T/valid + contiguous transposes).

    Args:
        street: "turn" (H=1128) or "river" (H=1081) -- the local kernel width.
        n_nodes: betting-tree node count of the bucket's shared topology.
        budget_gb: usable device memory in GB (decimal, 1 GB = 1e9 bytes;
            census protocol budget is 6.5).
        n_decision: decision-node count; ``None`` uses the conservative upper
            bound ``n_nodes`` (exact counts come free from a built tree via
            :func:`b_max_for_tree`).
        value_bytes: bytes per kernel scalar (8 = float64, the mandated
            production default).
        terminal_value_bytes: bytes per terminal-matrix scalar (4 = float32,
            the census convention; the current CPU port casts terminals to the
            kernel dtype, so pass 8 to be strict about that path).

    Returns:
        int >= 1 (a bucket must always be flushable at B=1).
    """
    try:
        H = STREET_H[str(street).lower()]
    except KeyError:
        raise ValueError(
            f"street must be one of {sorted(STREET_H)}, got {street!r}") from None
    nn = int(n_nodes)
    nd = nn if n_decision is None else int(n_decision)
    if nn < 1 or nd < 0 or nd > nn:
        raise ValueError(f"bad node counts: n_nodes={nn} n_decision={nd}")
    budget = float(budget_gb) * 1e9
    if budget <= 0:
        raise ValueError(f"budget_gb must be positive, got {budget_gb!r}")
    a = int(n_actions)
    dense = (2 * nn * a * H + 4 * nn * H + nd * a * H) * int(value_bytes)
    terminal = 8 * H * H * int(terminal_value_bytes)
    return max(1, int(budget // (dense + terminal)))


def b_max_for_tree(tree, street, budget_gb, **kwargs):
    """:func:`b_max_for` with exact node counts read off a built tree."""
    n_nodes = int(tree["n_nodes"])
    n_decision = int(np.sum(np.asarray(tree["terminal_type"]) == fast_cfr.T_DECISION))
    return b_max_for(street, n_nodes, budget_gb, n_decision=n_decision, **kwargs)


# ---------------------------------------------------------------------------
# Turn-leaf -> river-runout expansion (the shared card-removal mapping)
# ---------------------------------------------------------------------------

def river_runout_specs(turn_board, pot, stack0, stack1, first_to_act,
                       r0_global, r1_global):
    """The 48 runout river specs of one turn river-deal cut.

    One spec per river card not on the turn board, ascending-card order --
    exactly the runout loop of ``turn_river.turn_leaf_river_cfv``. Card-removal
    range masking is implicit in the global 1326 representation: gathering
    ``r0_global``/``r1_global`` at a river board's local hand set drops every
    hand containing the runout card (parity-tested against the reference
    mapping in test_hunl_lazy_subgames.py).

    This is the single shared expansion: ``SubgameFactory.expand_to_river``
    and ``terminal_eval.exact_river_cfv_population`` both consume it.
    """
    key = sgs.board_key(turn_board)
    if len(key) != 4:
        raise ValueError(f"turn board must have 4 cards, got {len(key)}")
    return [
        sgs.SubgameSpec(
            street="river",
            board=key + (river,),
            pot=pot,
            stack0=stack0,
            stack1=stack1,
            first_to_act=first_to_act,
            r0=r0_global,
            r1=r1_global,
        )
        for river in range(sgs.N_CARDS)
        if river not in key
    ]


class SubgameFactory:
    """Lazy SubgameSpec construction from play/training states.

    Builds exactly what is requested (``n_specs_built`` counts every spec) and
    triggers NO tree construction by itself -- trees materialize lazily through
    ``tree_builder``'s LRU cache only when a spec's ``tree()``/``bucket_key()``
    is first needed (``expand_to_river`` reads the turn tree once to locate the
    cut's pot/stacks).
    """

    def __init__(self):
        self.n_specs_built = 0

    def from_state(self, street, board, pot, stack0, stack1, first_to_act,
                   r0, r1) -> "sgs.SubgameSpec":
        """Spec from a public state + both players' GLOBAL [1326] beliefs."""
        spec = sgs.SubgameSpec(
            street=street, board=tuple(board), pot=pot, stack0=stack0,
            stack1=stack1, first_to_act=first_to_act, r0=r0, r1=r1)
        self.n_specs_built += 1
        return spec

    def expand_to_river(self, turn_spec, leaf_node, r0=None, r1=None):
        """River specs for one river-deal CUT leaf of a turn spec.

        Args:
            turn_spec: the turn :class:`SubgameSpec`.
            leaf_node: node index of a ``showdown`` terminal in the turn tree
                (turn showdowns ARE river-deal cuts); the river subgames start
                at that node's (pot, stacks).
            r0, r1: optional GLOBAL [1326] reaches at the cut (range x strategy
                reach). Default: the turn spec's entry beliefs. Turn-local
                reaches map through ``subgame_spec.scatter_global`` first.

        Returns:
            list of 48 river SubgameSpecs (card-removal masking implicit).
        """
        if turn_spec.street != "turn":
            raise ValueError(
                f"expand_to_river needs a turn spec, got {turn_spec.street!r}")
        tree = turn_spec.tree()
        leaf = int(leaf_node)
        if not (0 <= leaf < int(tree["n_nodes"])):
            raise ValueError(
                f"leaf_node {leaf} out of range for tree with "
                f"{int(tree['n_nodes'])} nodes")
        if int(tree["terminal_type"][leaf]) != fast_cfr.T_SHOWDOWN:
            raise ValueError(
                f"leaf_node {leaf} is not a river-deal cut (showdown terminal)")
        specs = river_runout_specs(
            turn_spec.board,
            int(tree["pot"][leaf]),
            int(tree["stacks_h"][leaf]),
            int(tree["stacks_v"][leaf]),
            turn_spec.first_to_act,
            turn_spec.r0 if r0 is None else r0,
            turn_spec.r1 if r1 is None else r1,
        )
        self.n_specs_built += len(specs)
        return specs


# ---------------------------------------------------------------------------
# PopulationQueue: accumulate across boards/beliefs, flush per bucket at B_max
# ---------------------------------------------------------------------------

class PopulationQueue:
    """Topology-bucketed accumulation with memory-derived flushing.

    ``add(spec)`` files the spec under its exact topology bucket
    (``SubgameSpec.bucket_key``: blake2b tree digest + street + local width --
    members may differ in board, pot, stacks and ranges) and, when the bucket
    reaches its B_max, flushes it through ``solve_population`` as ONE fused
    same-topology batch, returning that bucket's results (empty list
    otherwise). ``flush_all()`` drains every remaining partial bucket.

    B_max per bucket: the explicit ``b_max`` override, else
    :func:`b_max_for_tree` on the bucket's own tree under ``budget_gb`` (the
    census arithmetic). Chunk-exactness is the P-B property: queue-flushed
    solves are bit-identical on CPU to one direct ``solve_population`` call
    over the same specs.
    """

    def __init__(
        self,
        n_iterations,
        leaf=None,
        dtype=torch.float64,
        device="cpu",
        b_max=None,
        budget_gb=DEFAULT_BUDGET_GB,
        compute_value_pass=True,
    ):
        if b_max is not None and int(b_max) < 1:
            raise ValueError(f"b_max must be >= 1, got {b_max!r}")
        self.n_iterations = int(n_iterations)
        self.leaf = leaf
        self.dtype = dtype
        self.device = device
        self.b_max = None if b_max is None else int(b_max)
        self.budget_gb = float(budget_gb)
        self.compute_value_pass = bool(compute_value_pass)
        self._buckets = {}        # bucket_key -> {"specs": [...], "b_max": int}
        self.n_flushes = 0
        self.n_solved = 0

    # -- introspection -------------------------------------------------------

    @property
    def pending(self) -> int:
        return sum(len(b["specs"]) for b in self._buckets.values())

    def b_max_for_spec(self, spec) -> int:
        """The flush threshold this queue applies to ``spec``'s bucket."""
        if self.b_max is not None:
            return self.b_max
        return b_max_for_tree(spec.tree(), spec.street, self.budget_gb)

    # -- queueing ------------------------------------------------------------

    def add(self, spec):
        """Enqueue one spec; returns the flushed bucket's results, or []."""
        if spec.street == "turn" and self.leaf is None:
            raise ValueError(
                "turn specs need a leaf hook (river-deal cut leaves); "
                "construct the queue with leaf=terminal_eval.make_exact_river_leaf(...)")
        key = spec.bucket_key()
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = {"specs": [], "b_max": self.b_max_for_spec(spec)}
            self._buckets[key] = bucket
        bucket["specs"].append(spec)
        if len(bucket["specs"]) >= bucket["b_max"]:
            del self._buckets[key]
            return self._solve(bucket["specs"], bucket["b_max"])
        return []

    def extend(self, specs):
        """Enqueue many specs; returns all results flushed along the way."""
        results = []
        for spec in specs:
            results.extend(self.add(spec))
        return results

    def flush_all(self):
        """Drain every partial bucket (insertion order); returns results."""
        results = []
        buckets, self._buckets = self._buckets, {}
        for bucket in buckets.values():
            results.extend(self._solve(bucket["specs"], bucket["b_max"]))
        return results

    def _solve(self, specs, b_max):
        out = pop.solve_population(
            specs,
            n_iterations=self.n_iterations,
            leaf=self.leaf,
            dtype=self.dtype,
            device=self.device,
            b_max=b_max,
            compute_value_pass=self.compute_value_pass,
        )
        self.n_flushes += 1
        self.n_solved += len(out)
        return out
