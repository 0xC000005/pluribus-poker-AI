"""ReBeL-style search-in-learning substrate (de-risk first on small games).

This package is the foundation for the search-in-learning main line (see
``docs/research_protocols/net_only_ceiling_and_rebel_decision.md``). It starts small-game-first:
the Leduc public tree + an exact leaf oracle, used to pin the two correctness must-fixes
(CFV normalization convention, uniform averaging/sampling) BEFORE any value-net training.
"""
