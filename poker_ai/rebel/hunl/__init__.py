"""HUNL instance of the lazy batched-subgame substrate (p1_design.json).

Street-local, board-free betting-tree construction + topology bucketing built
ON TOP of the protected/trusted surfaces ``scripts/solver.py`` and
``scripts/fast_cfr.py`` (imported, never edited). Nothing global is ever
enumerated: trees are built lazily per (pot, stacks, first-to-act) entry
config and grouped by exact ``_same_topology_signature`` digest so that
populations of same-shape subgames can be fused into one batched GPU solve.
"""
