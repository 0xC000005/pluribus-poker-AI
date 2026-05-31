"""Optimized iterative CFR+ solver with batched matrix operations.

Key optimizations over the recursive solver:
1. Float32 throughout (~1.5x faster matrix ops)
2. Batched GEMM for terminal evaluation (~10-12x faster)
   - Collects reach probabilities at all terminals via forward pass
   - Evaluates all terminals with a few large matrix multiplies (BLAS level-3)
   - Instead of 85 separate (n,n)@(n,) GEMV calls
3. Hand pruning: exclude near-zero probability hands to reduce n
4. Iterative forward/backward passes (no Python recursion overhead)

The solver takes precomputed terminal evaluation matrices (win/lose/tie
or equity versions) and a game tree, then runs CFR+ iterations.
"""
import itertools
from collections import OrderedDict, deque

import numpy as np
import torch

# Terminal types
T_DECISION = 0
T_HERO_FOLD = 1
T_VILLAIN_FOLD = 2
T_SHOWDOWN = 3
SOLVER_UPDATES = ("cfr_plus", "dcfr_plus", "pdcfr_plus")
_DCFR_ALPHA = 1.5
_DCFR_GAMMA = 2.0
_PDCFR_ALPHA = 2.3
_PDCFR_GAMMA = 5.0
_MAX_TORCH_MATRIX_TENSOR_CACHE = 2
_TORCH_MATRIX_TENSOR_CACHE = OrderedDict()
_TORCH_MATRIX_TENSOR_CACHE_HITS = 0
_TORCH_MATRIX_TENSOR_CACHE_MISSES = 0


def clear_torch_matrix_tensor_cache():
    """Clear cached torch terminal-matrix tensor bundles."""
    global _TORCH_MATRIX_TENSOR_CACHE_HITS, _TORCH_MATRIX_TENSOR_CACHE_MISSES
    _TORCH_MATRIX_TENSOR_CACHE.clear()
    _TORCH_MATRIX_TENSOR_CACHE_HITS = 0
    _TORCH_MATRIX_TENSOR_CACHE_MISSES = 0


def torch_matrix_tensor_cache_info():
    """Return diagnostics for repeated terminal-matrix tensor reuse."""
    return {
        "entries": len(_TORCH_MATRIX_TENSOR_CACHE),
        "max_entries": _MAX_TORCH_MATRIX_TENSOR_CACHE,
        "hits": _TORCH_MATRIX_TENSOR_CACHE_HITS,
        "misses": _TORCH_MATRIX_TENSOR_CACHE_MISSES,
    }


def _validate_solver_update(solver_update):
    if solver_update not in SOLVER_UPDATES:
        allowed = ", ".join(SOLVER_UPDATES)
        raise ValueError(f"Unknown solver_update: {solver_update}. Expected one of: {allowed}")
    return solver_update


def _dcfr_discount_factors(iteration_index):
    """Fixed DCFR-style discounts for the opt-in low-budget resolver probe."""
    t = np.float32(max(int(iteration_index), 1))
    positive_regret = np.float32((t ** _DCFR_ALPHA) / ((t ** _DCFR_ALPHA) + 1.0))
    average_strategy = np.float32((t / (t + 1.0)) ** _DCFR_GAMMA)
    return positive_regret, average_strategy


def _pdcfr_discount_factor(iteration_index):
    """PDCFR+ optimistic-regret discount using the published default alpha."""
    t = np.float32(max(int(iteration_index), 1))
    base = np.power(max(t - np.float32(1.0), np.float32(0.0)), _PDCFR_ALPHA)
    return np.float32(base / (base + np.float32(1.0))) if base > 0 else np.float32(0.0)


def _pdcfr_strategy_discount(iteration_index):
    """Published PDCFR+ average-strategy discount with gamma=5."""
    t = np.float32(max(int(iteration_index), 1))
    if t <= 1:
        return np.float32(0.0)
    return np.float32(((t - np.float32(1.0)) / t) ** _PDCFR_GAMMA)


def build_tree_arrays(root):
    """Flatten a Node tree into arrays for iterative CFR.

    Returns dict with all tree data in numpy arrays.
    """
    # BFS to number all nodes
    all_nodes = []
    node_id_to_idx = {}
    queue = deque([root])
    while queue:
        node = queue.popleft()
        idx = len(all_nodes)
        all_nodes.append(node)
        node_id_to_idx[id(node)] = idx
        if not node.is_terminal:
            for a in sorted(node.children.keys()):
                queue.append(node.children[a])

    n_nodes = len(all_nodes)

    # Build arrays
    player = np.zeros(n_nodes, dtype=np.int32)
    pot = np.zeros(n_nodes, dtype=np.int32)
    stacks_h = np.zeros(n_nodes, dtype=np.int32)
    stacks_v = np.zeros(n_nodes, dtype=np.int32)
    to_call = np.zeros(n_nodes, dtype=np.int32)
    terminal_type = np.zeros(n_nodes, dtype=np.int32)
    parent_idx = np.full(n_nodes, -1, dtype=np.int32)
    parent_action = np.full(n_nodes, -1, dtype=np.int32)
    # Determine max action index from the tree.
    max_action = 4  # minimum
    for node in all_nodes:
        if not node.is_terminal:
            for a in node.children:
                if a > max_action:
                    max_action = a
    n_actions = max_action + 1
    children = np.full((n_nodes, n_actions), -1, dtype=np.int32)

    # Maps for terminal type strings
    tt_map = {'': T_DECISION, 'hero_fold': T_HERO_FOLD,
              'villain_fold': T_VILLAIN_FOLD, 'showdown': T_SHOWDOWN}

    for i, node in enumerate(all_nodes):
        player[i] = node.player
        pot[i] = node.pot
        stacks_h[i] = node.stacks[0]
        stacks_v[i] = node.stacks[1]
        to_call[i] = node.to_call
        terminal_type[i] = tt_map.get(node.terminal_type, T_DECISION)
        if not node.is_terminal:
            for a, child in node.children.items():
                ci = node_id_to_idx[id(child)]
                children[i, a] = ci
                parent_idx[ci] = i
                parent_action[ci] = a

    # Find actions available at each decision node
    # decision_actions[i] = sorted list of valid action indices
    decision_actions = []
    for i in range(n_nodes):
        if player[i] != -1:
            acts = sorted([a for a in range(n_actions) if children[i, a] >= 0])
            decision_actions.append(acts)
        else:
            decision_actions.append([])

    return {
        'n_nodes': n_nodes,
        'n_actions': n_actions,
        'player': player,
        'pot': pot,
        'stacks_h': stacks_h,
        'stacks_v': stacks_v,
        'to_call': to_call,
        'terminal_type': terminal_type,
        'parent_idx': parent_idx,
        'parent_action': parent_action,
        'children': children,
        'decision_actions': decision_actions,
        'all_nodes': all_nodes,
        # Pre-classified indices
        'showdown_idx': np.where(terminal_type == T_SHOWDOWN)[0],
        'hero_fold_idx': np.where(terminal_type == T_HERO_FOLD)[0],
        'villain_fold_idx': np.where(terminal_type == T_VILLAIN_FOLD)[0],
        'decision_idx': np.where(player >= 0)[0],
    }


def _frontier_cut_nodes(tree, cut_node_indices):
    if cut_node_indices is None:
        return np.zeros(0, dtype=np.int32), np.zeros(tree['n_nodes'], dtype=bool)
    raw = [int(idx) for idx in cut_node_indices]
    if not raw:
        return np.zeros(0, dtype=np.int32), np.zeros(tree['n_nodes'], dtype=bool)
    n_nodes = int(tree['n_nodes'])
    player = tree['player']
    cut_set = set()
    for idx in raw:
        if idx < 0 or idx >= n_nodes:
            raise ValueError(f"cut node index out of range: {idx}")
        if int(player[idx]) == -1:
            raise ValueError(f"cut node must be a decision node, got terminal {idx}")
        cut_set.add(idx)

    frontier = []
    for idx in sorted(cut_set):
        parent = int(tree['parent_idx'][idx])
        has_cut_ancestor = False
        while parent >= 0:
            if parent in cut_set:
                has_cut_ancestor = True
                break
            parent = int(tree['parent_idx'][parent])
        if not has_cut_ancestor:
            frontier.append(idx)

    cut_idx = np.asarray(frontier, dtype=np.int32)
    cut_mask = np.zeros(n_nodes, dtype=bool)
    cut_mask[cut_idx] = True
    return cut_idx, cut_mask


def _validated_trace_nodes(tree, trace_node_indices):
    if trace_node_indices is None:
        return np.zeros(0, dtype=np.int32)
    raw = [int(idx) for idx in trace_node_indices]
    if not raw:
        return np.zeros(0, dtype=np.int32)
    n_nodes = int(tree['n_nodes'])
    unique = sorted(set(raw))
    bad = [idx for idx in unique if idx < 0 or idx >= n_nodes]
    if bad:
        raise ValueError(f"trace_node_indices out of range: {bad[:5]}")
    return np.asarray(unique, dtype=np.int32)


def _descendants_of_cut_nodes(tree, cut_mask):
    inactive = np.zeros(tree['n_nodes'], dtype=bool)
    if not np.any(cut_mask):
        return inactive
    for idx in range(int(tree['n_nodes'])):
        parent = int(tree['parent_idx'][idx])
        while parent >= 0:
            if cut_mask[parent]:
                inactive[idx] = True
                break
            parent = int(tree['parent_idx'][parent])
    return inactive


def _validated_initial_array(initial, expected_shape, name):
    if initial is None:
        return None
    arr = np.asarray(initial, dtype=np.float32)
    if arr.shape != expected_shape:
        raise ValueError(f"{name} shape {arr.shape} != {expected_shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    if (arr < 0).any():
        raise ValueError(f"{name} must be non-negative for CFR+ warm starts")
    return arr.copy()


def _validated_updated_array(updated, expected_shape, name):
    arr = np.asarray(updated, dtype=np.float32)
    if arr.shape != expected_shape:
        raise ValueError(f"{name} shape {arr.shape} != {expected_shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    if (arr < 0).any():
        raise ValueError(f"{name} must be non-negative for CFR+ update hooks")
    return arr.copy()


def _regret_matching_strategy(regret_rows):
    """Return regret-matching strategy rows for one decision node.

    ``regret_rows`` has shape ``(n_actions, n_hands)``. Columns with no
    positive regret use the standard uniform fallback.
    """
    regret_rows = np.asarray(regret_rows, dtype=np.float32)
    if regret_rows.ndim != 2 or regret_rows.shape[0] == 0:
        raise ValueError("regret_rows must have shape (n_actions, n_hands)")
    positive = np.maximum(regret_rows, np.float32(0.0))
    totals = positive.sum(axis=0, dtype=np.float32)
    safe = np.where(totals > 0.0, totals, np.float32(1.0))
    uniform = np.float32(1.0 / regret_rows.shape[0])
    return np.where(totals.reshape(1, -1) > 0.0, positive / safe.reshape(1, -1), uniform)


def _tree_depths(parent_idx):
    parent_idx = np.asarray(parent_idx, dtype=np.int32)
    depths = np.zeros(parent_idx.shape[0], dtype=np.int32)
    for idx in range(parent_idx.shape[0]):
        parent = int(parent_idx[idx])
        if parent >= 0:
            depths[idx] = depths[parent] + 1
    return depths


def _level_edge_groups(tree):
    """Build parent/action/child edge groups ordered by tree depth."""
    player = tree['player']
    children = tree['children']
    depths = _tree_depths(tree['parent_idx'])
    groups = []
    max_depth = int(depths.max()) if depths.size else 0
    for depth in range(max_depth):
        parents = [
            int(idx)
            for idx in np.nonzero((depths == depth) & (player >= 0))[0]
            if np.any(children[idx] >= 0)
        ]
        if not parents:
            groups.append(None)
            continue
        edge_parent = []
        edge_action = []
        edge_child = []
        edge_parent_pos = []
        for parent_pos, parent in enumerate(parents):
            actions = np.nonzero(children[parent] >= 0)[0]
            for action in actions:
                edge_parent.append(parent)
                edge_action.append(int(action))
                edge_child.append(int(children[parent, action]))
                edge_parent_pos.append(parent_pos)
        groups.append(
            {
                'parents': np.asarray(parents, dtype=np.intp),
                'edge_parent': np.asarray(edge_parent, dtype=np.intp),
                'edge_action': np.asarray(edge_action, dtype=np.intp),
                'edge_child': np.asarray(edge_child, dtype=np.intp),
                'edge_parent_pos': np.asarray(edge_parent_pos, dtype=np.intp),
            }
        )
        groups[-1]['legal_mask'] = children[groups[-1]['parents']] >= 0
        groups[-1]['edge_player'] = player[groups[-1]['edge_parent']]
        groups[-1]['starts'] = np.r_[
            0,
            np.nonzero(np.diff(groups[-1]['edge_parent_pos']))[0] + 1,
        ]
    return groups


def _level_regret_matching(regret_rows, legal_mask):
    positive = np.maximum(regret_rows, np.float32(0.0)) * legal_mask[:, :, None]
    totals = positive.sum(axis=1, dtype=np.float32)
    safe = np.where(totals > 0.0, totals, np.float32(1.0))
    legal_counts = legal_mask.sum(axis=1, dtype=np.float32)
    uniform = np.divide(
        np.float32(1.0),
        legal_counts,
        out=np.zeros_like(legal_counts, dtype=np.float32),
        where=legal_counts > 0.0,
    )
    return np.where(
        totals[:, None, :] > 0.0,
        positive / safe[:, None, :],
        legal_mask[:, :, None] * uniform[:, None, None],
    ).astype(np.float32, copy=False)


def solve_cfr_levelsync(tree, n_hands, win_m, lose_m, tie_m, valid_m,
                        pot_start, hero_stack_start, villain_stack_start,
                        n_iterations=100, hero_range=None, villain_range=None,
                        initial_regret_sum=None, initial_strategy_sum=None):
    """Run CFR+ with level-synchronous vectorized tree updates.

    This is an opt-in prototype for the fused/matrix CFR boundary. It preserves
    the vanilla CFR+ recurrence but intentionally excludes diagnostic callback
    hooks and alternative solver updates until exactness and speed are proven.
    """
    n = n_hands
    nn = tree['n_nodes']
    n_actions = tree['n_actions']
    player = tree['player']
    children = tree['children']
    groups = tree.setdefault('_level_edge_groups', _level_edge_groups(tree))

    show_idx = tree['showdown_idx']
    hfold_idx = tree['hero_fold_idx']
    vfold_idx = tree['villain_fold_idx']

    hi_all = hero_stack_start - tree['stacks_h']
    vi_all = villain_stack_start - tree['stacks_v']

    if len(show_idx) > 0:
        hi_s = hi_all[show_idx].astype(np.float32)
        vi_s = vi_all[show_idx].astype(np.float32)
        hw_s = (pot_start + vi_s).reshape(-1, 1)
        hl_s = (-hi_s).reshape(-1, 1)
        ht_s = ((pot_start + vi_s - hi_s) / 2).reshape(-1, 1)
        vw_s = (pot_start + hi_s).reshape(-1, 1)
        vl_s = (-vi_s).reshape(-1, 1)
        vt_s = ((pot_start + hi_s - vi_s) / 2).reshape(-1, 1)

    if len(hfold_idx) > 0:
        hi_hf = hi_all[hfold_idx].astype(np.float32)
        hf_hero_coeff = (-hi_hf).reshape(-1, 1)
        hf_vill_coeff = (pot_start + hi_hf).reshape(-1, 1)

    if len(vfold_idx) > 0:
        vi_vf = vi_all[vfold_idx].astype(np.float32)
        vf_hero_coeff = (pot_start + vi_vf).reshape(-1, 1)
        vf_vill_coeff = (-vi_vf).reshape(-1, 1)

    expected_shape = (nn, n_actions, n)
    regret_sum = _validated_initial_array(
        initial_regret_sum,
        expected_shape,
        "initial_regret_sum",
    )
    if regret_sum is None:
        regret_sum = np.zeros(expected_shape, dtype=np.float32)
    strategy_sum = _validated_initial_array(
        initial_strategy_sum,
        expected_shape,
        "initial_strategy_sum",
    )
    if strategy_sum is None:
        strategy_sum = np.zeros(expected_shape, dtype=np.float32)

    hr_at = np.zeros((nn, n), dtype=np.float32)
    vr_at = np.zeros((nn, n), dtype=np.float32)
    hvals = np.zeros((nn, n), dtype=np.float32)
    vvals = np.zeros((nn, n), dtype=np.float32)
    hr_init = hero_range if hero_range is not None else np.ones(n, dtype=np.float32)
    vr_init = villain_range if villain_range is not None else np.ones(n, dtype=np.float32)

    win_mT = win_m.T.copy()
    lose_mT = lose_m.T.copy()
    tie_mT = tie_m.T.copy()
    valid_mT = valid_m.T.copy()

    for _iter in range(n_iterations):
        hr_at[0] = hr_init
        vr_at[0] = vr_init
        level_strategies = []

        for group in groups:
            if group is None:
                level_strategies.append(None)
                continue
            parents = group['parents']
            strategies = _level_regret_matching(regret_sum[parents], group['legal_mask'])
            level_strategies.append(strategies)
            edge_parent = group['edge_parent']
            edge_action = group['edge_action']
            edge_child = group['edge_child']
            edge_strategy = strategies[group['edge_parent_pos'], edge_action]
            hero_edges = group['edge_player'] == 0
            villain_edges = ~hero_edges
            if np.any(hero_edges):
                hp = edge_parent[hero_edges]
                hc = edge_child[hero_edges]
                hr_at[hc] = hr_at[hp] * edge_strategy[hero_edges]
                vr_at[hc] = vr_at[hp]
            if np.any(villain_edges):
                vp = edge_parent[villain_edges]
                vc = edge_child[villain_edges]
                hr_at[vc] = hr_at[vp]
                vr_at[vc] = vr_at[vp] * edge_strategy[villain_edges]

        if len(show_idx) > 0:
            VR_s = vr_at[show_idx]
            HR_s = hr_at[show_idx]
            h_win = VR_s @ win_mT
            h_lose = VR_s @ lose_mT
            h_tie = VR_s @ tie_mT
            v_win = HR_s @ win_m
            v_lose = HR_s @ lose_m
            v_tie = HR_s @ tie_m
            hvals[show_idx] = hw_s * h_win + hl_s * h_lose + ht_s * h_tie
            vvals[show_idx] = vw_s * v_lose + vl_s * v_win + vt_s * v_tie

        if len(hfold_idx) > 0:
            VR_hf = vr_at[hfold_idx]
            HR_hf = hr_at[hfold_idx]
            hvals[hfold_idx] = hf_hero_coeff * (VR_hf @ valid_mT)
            vvals[hfold_idx] = hf_vill_coeff * (HR_hf @ valid_m)

        if len(vfold_idx) > 0:
            VR_vf = vr_at[vfold_idx]
            HR_vf = hr_at[vfold_idx]
            hvals[vfold_idx] = vf_hero_coeff * (VR_vf @ valid_mT)
            vvals[vfold_idx] = vf_vill_coeff * (HR_vf @ valid_m)

        for group, strategies in zip(reversed(groups), reversed(level_strategies), strict=False):
            if group is None or strategies is None:
                continue
            parents = group['parents']
            edge_parent = group['edge_parent']
            edge_action = group['edge_action']
            edge_child = group['edge_child']
            edge_parent_pos = group['edge_parent_pos']
            edge_strategy = strategies[edge_parent_pos, edge_action]
            product_h = edge_strategy * hvals[edge_child]
            product_v = edge_strategy * vvals[edge_child]
            parent_hvals = np.add.reduceat(product_h, group['starts'], axis=0)
            parent_vvals = np.add.reduceat(product_v, group['starts'], axis=0)
            hvals[parents] = parent_hvals
            vvals[parents] = parent_vvals

            hero_edges = group['edge_player'] == 0
            villain_edges = ~hero_edges
            if np.any(hero_edges):
                hp = edge_parent[hero_edges]
                ha = edge_action[hero_edges]
                hpos = edge_parent_pos[hero_edges]
                instant = hvals[edge_child[hero_edges]] - parent_hvals[hpos]
                regret_sum[hp, ha, :] = np.maximum(
                    regret_sum[hp, ha, :] + instant,
                    0,
                )
                strategy_sum[hp, ha, :] = (
                    strategy_sum[hp, ha, :]
                    + hr_at[hp] * edge_strategy[hero_edges]
                )
            if np.any(villain_edges):
                vp = edge_parent[villain_edges]
                va = edge_action[villain_edges]
                vpos = edge_parent_pos[villain_edges]
                instant = vvals[edge_child[villain_edges]] - parent_vvals[vpos]
                regret_sum[vp, va, :] = np.maximum(
                    regret_sum[vp, va, :] + instant,
                    0,
                )
                strategy_sum[vp, va, :] = (
                    strategy_sum[vp, va, :]
                    + vr_at[vp] * edge_strategy[villain_edges]
                )

    return regret_sum, strategy_sum


def _torch_level_regret_matching(regret_rows, legal_mask):
    legal = legal_mask.to(dtype=regret_rows.dtype).unsqueeze(-1)
    positive = torch.clamp(regret_rows, min=0.0) * legal
    totals = positive.sum(dim=1)
    safe = totals.clamp(min=1.0)
    legal_counts = legal_mask.sum(dim=1).to(dtype=regret_rows.dtype).clamp(min=1.0)
    uniform = (1.0 / legal_counts).reshape(-1, 1, 1)
    return torch.where(
        totals.unsqueeze(1) > 0.0,
        positive / safe.unsqueeze(1),
        legal * uniform,
    )


def _torch_level_regret_matching_batched(regret_rows, legal_mask):
    legal = legal_mask.to(dtype=regret_rows.dtype).unsqueeze(0).unsqueeze(-1)
    positive = torch.clamp(regret_rows, min=0.0) * legal
    totals = positive.sum(dim=2)
    safe = totals.clamp(min=1.0)
    legal_counts = legal_mask.sum(dim=1).to(dtype=regret_rows.dtype).clamp(min=1.0)
    uniform = (1.0 / legal_counts).reshape(1, -1, 1, 1)
    return torch.where(
        totals.unsqueeze(2) > 0.0,
        positive / safe.unsqueeze(2),
        legal * uniform,
    )


def _torch_static_cache_key(torch_device, n_hands, win_m, lose_m, tie_m, valid_m):
    matrices = (win_m, lose_m, tie_m, valid_m)
    matrix_sig = tuple(
        (
            id(matrix),
            tuple(np.shape(matrix)),
            str(np.asarray(matrix).dtype),
        )
        for matrix in matrices
    )
    return (str(torch_device), int(n_hands), matrix_sig)


def _torch_matrix_tensor_bundle(torch_device, win_m, lose_m, tie_m, valid_m):
    global _TORCH_MATRIX_TENSOR_CACHE_HITS, _TORCH_MATRIX_TENSOR_CACHE_MISSES
    cache_key = _torch_static_cache_key(
        torch_device,
        np.shape(win_m)[0],
        win_m,
        lose_m,
        tie_m,
        valid_m,
    )
    cached = _TORCH_MATRIX_TENSOR_CACHE.get(cache_key)
    if cached is not None:
        _TORCH_MATRIX_TENSOR_CACHE_HITS += 1
        _TORCH_MATRIX_TENSOR_CACHE.move_to_end(cache_key)
        return cached

    _TORCH_MATRIX_TENSOR_CACHE_MISSES += 1
    win_t = torch.as_tensor(win_m, dtype=torch.float32, device=torch_device)
    lose_t = torch.as_tensor(lose_m, dtype=torch.float32, device=torch_device)
    tie_t = torch.as_tensor(tie_m, dtype=torch.float32, device=torch_device)
    valid_t = torch.as_tensor(valid_m, dtype=torch.float32, device=torch_device)
    cached = {
        'win': win_t,
        'lose': lose_t,
        'tie': tie_t,
        'valid': valid_t,
        'win_T': win_t.T.contiguous(),
        'lose_T': lose_t.T.contiguous(),
        'tie_T': tie_t.T.contiguous(),
        'valid_T': valid_t.T.contiguous(),
    }
    _TORCH_MATRIX_TENSOR_CACHE[cache_key] = cached
    _TORCH_MATRIX_TENSOR_CACHE.move_to_end(cache_key)
    while len(_TORCH_MATRIX_TENSOR_CACHE) > _MAX_TORCH_MATRIX_TENSOR_CACHE:
        _TORCH_MATRIX_TENSOR_CACHE.popitem(last=False)
    return cached


def _torch_levelsync_workspace(static, expected_shape, nn, n, n_actions, torch_device):
    workspace = static.setdefault("_workspace", {})
    specs = {
        "regret_sum": expected_shape,
        "strategy_sum": expected_shape,
        "hr_at": (nn, n),
        "vr_at": (nn, n),
        "hvals": (nn, n),
        "vvals": (nn, n),
    }
    for name, shape in specs.items():
        tensor = workspace.get(name)
        if (
            tensor is None
            or tuple(tensor.shape) != tuple(shape)
            or tensor.device != torch_device
            or tensor.dtype != torch.float32
        ):
            tensor = torch.empty(shape, dtype=torch.float32, device=torch_device)
            workspace[name] = tensor
    return workspace


def _torch_levelsync_static_cache(tree, n_hands, win_m, lose_m, tie_m, valid_m, torch_device):
    cache = tree.setdefault("_torch_levelsync_cache", {})
    cache_key = _torch_static_cache_key(torch_device, n_hands, win_m, lose_m, tie_m, valid_m)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    groups_np = tree.setdefault('_level_edge_groups', _level_edge_groups(tree))
    groups = []
    for group in groups_np:
        if group is None:
            groups.append(None)
            continue
        entry = {
            'parents': torch.as_tensor(group['parents'], dtype=torch.long, device=torch_device),
            'edge_parent': torch.as_tensor(group['edge_parent'], dtype=torch.long, device=torch_device),
            'edge_action': torch.as_tensor(group['edge_action'], dtype=torch.long, device=torch_device),
            'edge_child': torch.as_tensor(group['edge_child'], dtype=torch.long, device=torch_device),
            'edge_parent_pos': torch.as_tensor(group['edge_parent_pos'], dtype=torch.long, device=torch_device),
            'legal_mask': torch.as_tensor(group['legal_mask'], dtype=torch.bool, device=torch_device),
        }
        edge_player = torch.as_tensor(group['edge_player'], dtype=torch.long, device=torch_device)
        entry['hero_edges'] = edge_player == 0
        entry['villain_edges'] = ~entry['hero_edges']
        groups.append(entry)

    matrix_bundle = _torch_matrix_tensor_bundle(torch_device, win_m, lose_m, tie_m, valid_m)
    cached = {
        'groups': groups,
        **matrix_bundle,
        'stacks_h': torch.as_tensor(tree['stacks_h'], dtype=torch.float32, device=torch_device),
        'stacks_v': torch.as_tensor(tree['stacks_v'], dtype=torch.float32, device=torch_device),
        'show_idx': torch.as_tensor(tree['showdown_idx'], dtype=torch.long, device=torch_device),
        'hero_fold_idx': torch.as_tensor(tree['hero_fold_idx'], dtype=torch.long, device=torch_device),
        'villain_fold_idx': torch.as_tensor(tree['villain_fold_idx'], dtype=torch.long, device=torch_device),
    }
    cache[cache_key] = cached
    return cached


def solve_cfr_levelsync_torch(tree, n_hands, win_m, lose_m, tie_m, valid_m,
                              pot_start, hero_stack_start, villain_stack_start,
                              n_iterations=100, hero_range=None,
                              villain_range=None, device="cuda",
                              initial_regret_sum=None,
                              initial_strategy_sum=None):
    """Run the level-synchronous CFR+ prototype with torch tensors."""
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA level-sync backend requested but torch.cuda is unavailable.")

    n = int(n_hands)
    nn = int(tree['n_nodes'])
    n_actions = int(tree['n_actions'])
    static = _torch_levelsync_static_cache(
        tree,
        n,
        win_m,
        lose_m,
        tie_m,
        valid_m,
        torch_device,
    )
    groups = static['groups']
    win_t = static['win']
    lose_t = static['lose']
    tie_t = static['tie']
    valid_t = static['valid']
    win_tT = static['win_T']
    lose_tT = static['lose_T']
    tie_tT = static['tie_T']
    valid_tT = static['valid_T']

    stacks_h = static['stacks_h']
    stacks_v = static['stacks_v']
    hi_all = float(hero_stack_start) - stacks_h
    vi_all = float(villain_stack_start) - stacks_v
    show_idx = static['show_idx']
    hfold_idx = static['hero_fold_idx']
    vfold_idx = static['villain_fold_idx']

    if show_idx.numel() > 0:
        hi_s = hi_all.index_select(0, show_idx).reshape(-1, 1)
        vi_s = vi_all.index_select(0, show_idx).reshape(-1, 1)
        hw_s = float(pot_start) + vi_s
        hl_s = -hi_s
        ht_s = (float(pot_start) + vi_s - hi_s) / 2.0
        vw_s = float(pot_start) + hi_s
        vl_s = -vi_s
        vt_s = (float(pot_start) + hi_s - vi_s) / 2.0

    if hfold_idx.numel() > 0:
        hi_hf = hi_all.index_select(0, hfold_idx).reshape(-1, 1)
        hf_hero_coeff = -hi_hf
        hf_vill_coeff = float(pot_start) + hi_hf

    if vfold_idx.numel() > 0:
        vi_vf = vi_all.index_select(0, vfold_idx).reshape(-1, 1)
        vf_hero_coeff = float(pot_start) + vi_vf
        vf_vill_coeff = -vi_vf

    expected_shape = (nn, n_actions, n)
    workspace = _torch_levelsync_workspace(
        static,
        expected_shape,
        nn,
        n,
        n_actions,
        torch_device,
    )
    initial_regret_np = _validated_initial_array(
        initial_regret_sum,
        expected_shape,
        "initial_regret_sum",
    )
    initial_strategy_np = _validated_initial_array(
        initial_strategy_sum,
        expected_shape,
        "initial_strategy_sum",
    )
    if initial_regret_np is None:
        regret_sum = workspace["regret_sum"]
        regret_sum.zero_()
    else:
        regret_sum = torch.as_tensor(initial_regret_np, dtype=torch.float32, device=torch_device)
    if initial_strategy_np is None:
        strategy_sum = workspace["strategy_sum"]
        strategy_sum.zero_()
    else:
        strategy_sum = torch.as_tensor(initial_strategy_np, dtype=torch.float32, device=torch_device)

    hr_at = workspace["hr_at"]
    vr_at = workspace["vr_at"]
    hvals = workspace["hvals"]
    vvals = workspace["vvals"]
    hr_at.zero_()
    vr_at.zero_()
    hvals.zero_()
    vvals.zero_()
    hr_init = (
        torch.as_tensor(hero_range, dtype=torch.float32, device=torch_device)
        if hero_range is not None else torch.ones(n, dtype=torch.float32, device=torch_device)
    )
    vr_init = (
        torch.as_tensor(villain_range, dtype=torch.float32, device=torch_device)
        if villain_range is not None else torch.ones(n, dtype=torch.float32, device=torch_device)
    )

    for _iter in range(n_iterations):
        hr_at[0].copy_(hr_init)
        vr_at[0].copy_(vr_init)
        level_strategies = []

        for group in groups:
            if group is None:
                level_strategies.append(None)
                continue
            parents = group['parents']
            strategies = _torch_level_regret_matching(
                regret_sum.index_select(0, parents),
                group['legal_mask'],
            )
            level_strategies.append(strategies)
            edge_parent = group['edge_parent']
            edge_action = group['edge_action']
            edge_child = group['edge_child']
            edge_parent_pos = group['edge_parent_pos']
            edge_strategy = strategies[edge_parent_pos, edge_action]
            hero_edges = group['hero_edges']
            villain_edges = group['villain_edges']
            if bool(hero_edges.any()):
                hp = edge_parent[hero_edges]
                hc = edge_child[hero_edges]
                hr_at[hc] = hr_at[hp] * edge_strategy[hero_edges]
                vr_at[hc] = vr_at[hp]
            if bool(villain_edges.any()):
                vp = edge_parent[villain_edges]
                vc = edge_child[villain_edges]
                hr_at[vc] = hr_at[vp]
                vr_at[vc] = vr_at[vp] * edge_strategy[villain_edges]

        if show_idx.numel() > 0:
            VR_s = vr_at.index_select(0, show_idx)
            HR_s = hr_at.index_select(0, show_idx)
            h_win = VR_s @ win_tT
            h_lose = VR_s @ lose_tT
            h_tie = VR_s @ tie_tT
            v_win = HR_s @ win_t
            v_lose = HR_s @ lose_t
            v_tie = HR_s @ tie_t
            hvals[show_idx] = hw_s * h_win + hl_s * h_lose + ht_s * h_tie
            vvals[show_idx] = vw_s * v_lose + vl_s * v_win + vt_s * v_tie

        if hfold_idx.numel() > 0:
            VR_hf = vr_at.index_select(0, hfold_idx)
            HR_hf = hr_at.index_select(0, hfold_idx)
            hvals[hfold_idx] = hf_hero_coeff * (VR_hf @ valid_tT)
            vvals[hfold_idx] = hf_vill_coeff * (HR_hf @ valid_t)

        if vfold_idx.numel() > 0:
            VR_vf = vr_at.index_select(0, vfold_idx)
            HR_vf = hr_at.index_select(0, vfold_idx)
            hvals[vfold_idx] = vf_hero_coeff * (VR_vf @ valid_tT)
            vvals[vfold_idx] = vf_vill_coeff * (HR_vf @ valid_t)

        for group, strategies in zip(reversed(groups), reversed(level_strategies), strict=False):
            if group is None or strategies is None:
                continue
            parents = group['parents']
            edge_parent = group['edge_parent']
            edge_action = group['edge_action']
            edge_child = group['edge_child']
            edge_parent_pos = group['edge_parent_pos']
            edge_strategy = strategies[edge_parent_pos, edge_action]

            parent_hvals = torch.zeros(
                (parents.numel(), n),
                dtype=torch.float32,
                device=torch_device,
            )
            parent_vvals = torch.zeros_like(parent_hvals)
            parent_hvals.index_add_(0, edge_parent_pos, edge_strategy * hvals[edge_child])
            parent_vvals.index_add_(0, edge_parent_pos, edge_strategy * vvals[edge_child])
            hvals[parents] = parent_hvals
            vvals[parents] = parent_vvals

            hero_edges = group['hero_edges']
            villain_edges = group['villain_edges']
            if bool(hero_edges.any()):
                hp = edge_parent[hero_edges]
                ha = edge_action[hero_edges]
                hpos = edge_parent_pos[hero_edges]
                instant = hvals[edge_child[hero_edges]] - parent_hvals[hpos]
                regret_sum[hp, ha, :] = torch.clamp(
                    regret_sum[hp, ha, :] + instant,
                    min=0.0,
                )
                strategy_sum[hp, ha, :] = (
                    strategy_sum[hp, ha, :]
                    + hr_at[hp] * edge_strategy[hero_edges]
                )
            if bool(villain_edges.any()):
                vp = edge_parent[villain_edges]
                va = edge_action[villain_edges]
                vpos = edge_parent_pos[villain_edges]
                instant = vvals[edge_child[villain_edges]] - parent_vvals[vpos]
                regret_sum[vp, va, :] = torch.clamp(
                    regret_sum[vp, va, :] + instant,
                    min=0.0,
                )
                strategy_sum[vp, va, :] = (
                    strategy_sum[vp, va, :]
                    + vr_at[vp] * edge_strategy[villain_edges]
                )

    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    return regret_sum.detach().cpu().numpy().copy(), strategy_sum.detach().cpu().numpy().copy()


def _terminal_type_array(tree):
    n_nodes = int(tree["n_nodes"])
    return np.asarray(tree.get("terminal_type", np.zeros(n_nodes, dtype=np.int32)), dtype=np.int32)


def _same_topology_signature(tree):
    return (
        int(tree["n_nodes"]),
        int(tree["n_actions"]),
        np.asarray(tree["player"], dtype=np.int32),
        np.asarray(tree["parent_idx"], dtype=np.int32),
        np.asarray(tree["children"], dtype=np.int32),
        _terminal_type_array(tree),
    )


def _validate_same_topology_batch(trees, n_hands):
    if not trees:
        raise ValueError("trees must contain at least one tree")
    base = _same_topology_signature(trees[0])
    for index, tree in enumerate(trees[1:], start=1):
        candidate = _same_topology_signature(tree)
        if candidate[0] != base[0] or candidate[1] != base[1]:
            raise ValueError(f"all trees must have the same topology; tree {index} shape differs")
        for lhs, rhs in zip(base[2:], candidate[2:], strict=True):
            if lhs.shape != rhs.shape or not np.array_equal(lhs, rhs):
                raise ValueError(f"all trees must have the same topology; tree {index} differs")
    n = int(n_hands)
    for index, tree in enumerate(trees):
        if int(tree["n_actions"]) != int(base[1]):
            raise ValueError(f"tree {index} action count differs")
        for key in ("stacks_h", "stacks_v"):
            if np.asarray(tree[key]).shape != (int(base[0]),):
                raise ValueError(f"tree {index} {key} shape is incompatible with topology")
    return int(base[0]), int(base[1]), n


def _stack_matrix_batch(matrices, n_hands, name, torch_device):
    stacked = np.stack([np.asarray(matrix, dtype=np.float32) for matrix in matrices], axis=0)
    expected = (len(matrices), int(n_hands), int(n_hands))
    if stacked.shape != expected:
        raise ValueError(f"{name} batch shape {stacked.shape} != {expected}")
    return torch.as_tensor(stacked, dtype=torch.float32, device=torch_device)


def _stack_vector(values, name, torch_device):
    return torch.as_tensor(np.asarray(values, dtype=np.float32), dtype=torch.float32, device=torch_device)


def solve_cfr_levelsync_torch_batched_same_topology(
    trees,
    n_hands,
    win_ms,
    lose_ms,
    tie_ms,
    valid_ms,
    pot_starts,
    hero_stack_starts,
    villain_stack_starts,
    n_iterations=100,
    hero_ranges=None,
    villain_ranges=None,
    terminal_eval_mode="batched",
    device="cuda",
):
    """Run level-synchronous CFR+ over a batch of exact-same-topology trees.

    This is a research-only primitive for testing whether same-shape public
    roots can amortize the resolver recurrence. It intentionally handles only
    the exact CFR+ path and rejects topology mismatches instead of padding.
    """
    trees = list(trees)
    batch_size = len(trees)
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA level-sync backend requested but torch.cuda is unavailable.")
    if terminal_eval_mode not in {"batched", "loop", "loop_showdown", "loop_folds"}:
        raise ValueError(
            "terminal_eval_mode must be 'batched', 'loop', 'loop_showdown', or 'loop_folds'"
        )
    loop_showdown = terminal_eval_mode in {"loop", "loop_showdown"}
    loop_folds = terminal_eval_mode in {"loop", "loop_folds"}
    if not (
        len(win_ms)
        == len(lose_ms)
        == len(tie_ms)
        == len(valid_ms)
        == len(pot_starts)
        == len(hero_stack_starts)
        == len(villain_stack_starts)
        == batch_size
    ):
        raise ValueError("all batched inputs must have one entry per tree")

    nn, n_actions, n = _validate_same_topology_batch(trees, n_hands)
    base_tree = trees[0]
    groups_np = base_tree.setdefault("_level_edge_groups", _level_edge_groups(base_tree))
    groups = []
    for group in groups_np:
        if group is None:
            groups.append(None)
            continue
        edge_player = torch.as_tensor(group["edge_player"], dtype=torch.long, device=torch_device)
        groups.append(
            {
                "parents": torch.as_tensor(group["parents"], dtype=torch.long, device=torch_device),
                "edge_parent": torch.as_tensor(group["edge_parent"], dtype=torch.long, device=torch_device),
                "edge_action": torch.as_tensor(group["edge_action"], dtype=torch.long, device=torch_device),
                "edge_child": torch.as_tensor(group["edge_child"], dtype=torch.long, device=torch_device),
                "edge_parent_pos": torch.as_tensor(group["edge_parent_pos"], dtype=torch.long, device=torch_device),
                "legal_mask": torch.as_tensor(group["legal_mask"], dtype=torch.bool, device=torch_device),
                "hero_edges": edge_player == 0,
                "villain_edges": edge_player != 0,
            }
        )

    win_t = _stack_matrix_batch(win_ms, n, "win_ms", torch_device)
    lose_t = _stack_matrix_batch(lose_ms, n, "lose_ms", torch_device)
    tie_t = _stack_matrix_batch(tie_ms, n, "tie_ms", torch_device)
    valid_t = _stack_matrix_batch(valid_ms, n, "valid_ms", torch_device)
    win_tT = win_t.transpose(1, 2).contiguous()
    lose_tT = lose_t.transpose(1, 2).contiguous()
    tie_tT = tie_t.transpose(1, 2).contiguous()
    valid_tT = valid_t.transpose(1, 2).contiguous()

    stacks_h = torch.as_tensor(
        np.stack([np.asarray(tree["stacks_h"], dtype=np.float32) for tree in trees], axis=0),
        dtype=torch.float32,
        device=torch_device,
    )
    stacks_v = torch.as_tensor(
        np.stack([np.asarray(tree["stacks_v"], dtype=np.float32) for tree in trees], axis=0),
        dtype=torch.float32,
        device=torch_device,
    )
    pot_t = _stack_vector(pot_starts, "pot_starts", torch_device).reshape(batch_size, 1, 1)
    hero_stack_t = _stack_vector(hero_stack_starts, "hero_stack_starts", torch_device).reshape(batch_size, 1)
    villain_stack_t = _stack_vector(villain_stack_starts, "villain_stack_starts", torch_device).reshape(batch_size, 1)

    hi_all = hero_stack_t - stacks_h
    vi_all = villain_stack_t - stacks_v
    show_idx = torch.as_tensor(base_tree["showdown_idx"], dtype=torch.long, device=torch_device)
    hfold_idx = torch.as_tensor(base_tree["hero_fold_idx"], dtype=torch.long, device=torch_device)
    vfold_idx = torch.as_tensor(base_tree["villain_fold_idx"], dtype=torch.long, device=torch_device)

    if show_idx.numel() > 0:
        hi_s = hi_all.index_select(1, show_idx).unsqueeze(-1)
        vi_s = vi_all.index_select(1, show_idx).unsqueeze(-1)
        hw_s = pot_t + vi_s
        hl_s = -hi_s
        ht_s = (pot_t + vi_s - hi_s) / 2.0
        vw_s = pot_t + hi_s
        vl_s = -vi_s
        vt_s = (pot_t + hi_s - vi_s) / 2.0

    if hfold_idx.numel() > 0:
        hi_hf = hi_all.index_select(1, hfold_idx).unsqueeze(-1)
        hf_hero_coeff = -hi_hf
        hf_vill_coeff = pot_t + hi_hf

    if vfold_idx.numel() > 0:
        vi_vf = vi_all.index_select(1, vfold_idx).unsqueeze(-1)
        vf_hero_coeff = pot_t + vi_vf
        vf_vill_coeff = -vi_vf

    regret_sum = torch.zeros((batch_size, nn, n_actions, n), dtype=torch.float32, device=torch_device)
    strategy_sum = torch.zeros_like(regret_sum)
    hr_at = torch.zeros((batch_size, nn, n), dtype=torch.float32, device=torch_device)
    vr_at = torch.zeros_like(hr_at)
    hvals = torch.zeros_like(hr_at)
    vvals = torch.zeros_like(hr_at)
    if hero_ranges is None:
        hr_init = torch.ones((batch_size, n), dtype=torch.float32, device=torch_device)
    else:
        hr_init = torch.as_tensor(
            np.stack([np.asarray(value, dtype=np.float32) for value in hero_ranges], axis=0),
            dtype=torch.float32,
            device=torch_device,
        )
    if villain_ranges is None:
        vr_init = torch.ones((batch_size, n), dtype=torch.float32, device=torch_device)
    else:
        vr_init = torch.as_tensor(
            np.stack([np.asarray(value, dtype=np.float32) for value in villain_ranges], axis=0),
            dtype=torch.float32,
            device=torch_device,
        )
    if hr_init.shape != (batch_size, n) or vr_init.shape != (batch_size, n):
        raise ValueError("hero_ranges and villain_ranges must have shape (batch, n_hands)")

    for _iter in range(int(n_iterations)):
        hr_at.zero_()
        vr_at.zero_()
        hvals.zero_()
        vvals.zero_()
        hr_at[:, 0, :].copy_(hr_init)
        vr_at[:, 0, :].copy_(vr_init)
        level_strategies = []

        for group in groups:
            if group is None:
                level_strategies.append(None)
                continue
            parents = group["parents"]
            strategies = _torch_level_regret_matching_batched(
                regret_sum.index_select(1, parents),
                group["legal_mask"],
            )
            level_strategies.append(strategies)
            edge_parent = group["edge_parent"]
            edge_action = group["edge_action"]
            edge_child = group["edge_child"]
            edge_parent_pos = group["edge_parent_pos"]
            edge_strategy = strategies[:, edge_parent_pos, edge_action, :]
            hero_edges = group["hero_edges"]
            villain_edges = group["villain_edges"]
            if bool(hero_edges.any()):
                hp = edge_parent[hero_edges]
                hc = edge_child[hero_edges]
                hr_at[:, hc, :] = hr_at[:, hp, :] * edge_strategy[:, hero_edges, :]
                vr_at[:, hc, :] = vr_at[:, hp, :]
            if bool(villain_edges.any()):
                vp = edge_parent[villain_edges]
                vc = edge_child[villain_edges]
                hr_at[:, vc, :] = hr_at[:, vp, :]
                vr_at[:, vc, :] = vr_at[:, vp, :] * edge_strategy[:, villain_edges, :]

        if show_idx.numel() > 0:
            VR_s = hr_at.new_empty((batch_size, show_idx.numel(), n))
            HR_s = hr_at.new_empty((batch_size, show_idx.numel(), n))
            VR_s.copy_(vr_at.index_select(1, show_idx))
            HR_s.copy_(hr_at.index_select(1, show_idx))
            if loop_showdown:
                for batch_idx in range(batch_size):
                    h_win = VR_s[batch_idx] @ win_tT[batch_idx]
                    h_lose = VR_s[batch_idx] @ lose_tT[batch_idx]
                    h_tie = VR_s[batch_idx] @ tie_tT[batch_idx]
                    v_win = HR_s[batch_idx] @ win_t[batch_idx]
                    v_lose = HR_s[batch_idx] @ lose_t[batch_idx]
                    v_tie = HR_s[batch_idx] @ tie_t[batch_idx]
                    hvals[batch_idx, show_idx, :] = (
                        hw_s[batch_idx] * h_win
                        + hl_s[batch_idx] * h_lose
                        + ht_s[batch_idx] * h_tie
                    )
                    vvals[batch_idx, show_idx, :] = (
                        vw_s[batch_idx] * v_lose
                        + vl_s[batch_idx] * v_win
                        + vt_s[batch_idx] * v_tie
                    )
            else:
                h_win = torch.bmm(VR_s, win_tT)
                h_lose = torch.bmm(VR_s, lose_tT)
                h_tie = torch.bmm(VR_s, tie_tT)
                v_win = torch.bmm(HR_s, win_t)
                v_lose = torch.bmm(HR_s, lose_t)
                v_tie = torch.bmm(HR_s, tie_t)
                hvals[:, show_idx, :] = hw_s * h_win + hl_s * h_lose + ht_s * h_tie
                vvals[:, show_idx, :] = vw_s * v_lose + vl_s * v_win + vt_s * v_tie

        if hfold_idx.numel() > 0:
            VR_hf = vr_at.index_select(1, hfold_idx)
            HR_hf = hr_at.index_select(1, hfold_idx)
            if loop_folds:
                for batch_idx in range(batch_size):
                    hvals[batch_idx, hfold_idx, :] = (
                        hf_hero_coeff[batch_idx] * (VR_hf[batch_idx] @ valid_tT[batch_idx])
                    )
                    vvals[batch_idx, hfold_idx, :] = (
                        hf_vill_coeff[batch_idx] * (HR_hf[batch_idx] @ valid_t[batch_idx])
                    )
            else:
                hvals[:, hfold_idx, :] = hf_hero_coeff * torch.bmm(VR_hf, valid_tT)
                vvals[:, hfold_idx, :] = hf_vill_coeff * torch.bmm(HR_hf, valid_t)

        if vfold_idx.numel() > 0:
            VR_vf = vr_at.index_select(1, vfold_idx)
            HR_vf = hr_at.index_select(1, vfold_idx)
            if loop_folds:
                for batch_idx in range(batch_size):
                    hvals[batch_idx, vfold_idx, :] = (
                        vf_hero_coeff[batch_idx] * (VR_vf[batch_idx] @ valid_tT[batch_idx])
                    )
                    vvals[batch_idx, vfold_idx, :] = (
                        vf_vill_coeff[batch_idx] * (HR_vf[batch_idx] @ valid_t[batch_idx])
                    )
            else:
                hvals[:, vfold_idx, :] = vf_hero_coeff * torch.bmm(VR_vf, valid_tT)
                vvals[:, vfold_idx, :] = vf_vill_coeff * torch.bmm(HR_vf, valid_t)

        for group, strategies in zip(reversed(groups), reversed(level_strategies), strict=False):
            if group is None or strategies is None:
                continue
            parents = group["parents"]
            edge_parent = group["edge_parent"]
            edge_action = group["edge_action"]
            edge_child = group["edge_child"]
            edge_parent_pos = group["edge_parent_pos"]
            edge_strategy = strategies[:, edge_parent_pos, edge_action, :]
            parent_hvals = torch.zeros((batch_size, parents.numel(), n), dtype=torch.float32, device=torch_device)
            parent_vvals = torch.zeros_like(parent_hvals)
            parent_hvals.index_add_(1, edge_parent_pos, edge_strategy * hvals[:, edge_child, :])
            parent_vvals.index_add_(1, edge_parent_pos, edge_strategy * vvals[:, edge_child, :])
            hvals[:, parents, :] = parent_hvals
            vvals[:, parents, :] = parent_vvals

            hero_edges = group["hero_edges"]
            villain_edges = group["villain_edges"]
            regret_flat = regret_sum.reshape(batch_size, nn * n_actions, n)
            strategy_flat = strategy_sum.reshape(batch_size, nn * n_actions, n)
            if bool(hero_edges.any()):
                hp = edge_parent[hero_edges]
                ha = edge_action[hero_edges]
                hpos = edge_parent_pos[hero_edges]
                child = edge_child[hero_edges]
                flat_idx = hp * n_actions + ha
                instant = hvals[:, child, :] - parent_hvals[:, hpos, :]
                regret_flat[:, flat_idx, :] = torch.clamp(
                    regret_flat[:, flat_idx, :] + instant,
                    min=0.0,
                )
                strategy_flat[:, flat_idx, :] = (
                    strategy_flat[:, flat_idx, :]
                    + hr_at[:, hp, :] * edge_strategy[:, hero_edges, :]
                )
            if bool(villain_edges.any()):
                vp = edge_parent[villain_edges]
                va = edge_action[villain_edges]
                vpos = edge_parent_pos[villain_edges]
                child = edge_child[villain_edges]
                flat_idx = vp * n_actions + va
                instant = vvals[:, child, :] - parent_vvals[:, vpos, :]
                regret_flat[:, flat_idx, :] = torch.clamp(
                    regret_flat[:, flat_idx, :] + instant,
                    min=0.0,
                )
                strategy_flat[:, flat_idx, :] = (
                    strategy_flat[:, flat_idx, :]
                    + vr_at[:, vp, :] * edge_strategy[:, villain_edges, :]
                )

    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    return regret_sum.detach().cpu().numpy().copy(), strategy_sum.detach().cpu().numpy().copy()


def _ragged_bmm_numpy(
    left_matrices,
    right_matrices,
    *,
    n_hands,
    torch_device,
):
    """Evaluate a ragged list of `(rows, n) @ (n, n)` products."""
    if len(left_matrices) != len(right_matrices):
        raise ValueError("left_matrices and right_matrices must have the same length")
    if not left_matrices:
        return []
    rows = [int(np.asarray(left).shape[0]) for left in left_matrices]
    max_rows = max(rows, default=0)
    if max_rows == 0:
        return [
            np.zeros((0, int(n_hands)), dtype=np.float32)
            for _ in left_matrices
        ]
    padded = torch.zeros(
        (len(left_matrices), max_rows, int(n_hands)),
        dtype=torch.float32,
        device=torch_device,
    )
    for index, left in enumerate(left_matrices):
        left_np = np.asarray(left, dtype=np.float32)
        if left_np.ndim != 2 or left_np.shape[1] != int(n_hands):
            raise ValueError(f"left matrix {index} has incompatible shape {left_np.shape}")
        if left_np.shape[0] > 0:
            padded[index, : left_np.shape[0], :] = torch.as_tensor(
                left_np,
                dtype=torch.float32,
                device=torch_device,
            )
    right = torch.stack(
        [
            torch.as_tensor(
                np.asarray(matrix, dtype=np.float32),
                dtype=torch.float32,
                device=torch_device,
            )
            for matrix in right_matrices
        ],
        dim=0,
    )
    expected_right = (len(right_matrices), int(n_hands), int(n_hands))
    if tuple(right.shape) != expected_right:
        raise ValueError(f"right matrix batch shape {tuple(right.shape)} != {expected_right}")
    output = torch.bmm(padded, right).detach().cpu().numpy()
    return [output[index, : row_count, :].copy() for index, row_count in enumerate(rows)]


def solve_cfr_levelsync_torch_ragged_terminals(
    trees,
    n_hands,
    win_ms,
    lose_ms,
    tie_ms,
    valid_ms,
    pot_starts,
    hero_stack_starts,
    villain_stack_starts,
    n_iterations=100,
    hero_ranges=None,
    villain_ranges=None,
    device="cuda",
):
    """Run CFR+ on heterogeneous roots with ragged batched terminal products.

    This is a research primitive, not a production resolver backend. The tree
    recurrence stays per-root, while terminal matrix products are grouped as
    padded ragged torch batches so root-decision parity can be tested before
    investing in a fully fused segmented solver.
    """
    trees = list(trees)
    batch_size = len(trees)
    if batch_size == 0:
        raise ValueError("trees must be non-empty")
    if not (
        len(win_ms)
        == len(lose_ms)
        == len(tie_ms)
        == len(valid_ms)
        == len(pot_starts)
        == len(hero_stack_starts)
        == len(villain_stack_starts)
        == batch_size
    ):
        raise ValueError("all ragged-terminal inputs must have one entry per tree")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA ragged-terminal backend requested but torch.cuda is unavailable.")

    n = int(n_hands)
    n_actions = int(trees[0]["n_actions"])
    for index, tree in enumerate(trees):
        if int(tree["n_actions"]) != n_actions:
            raise ValueError(f"tree {index} action count differs")

    hero_ranges = [None] * batch_size if hero_ranges is None else list(hero_ranges)
    villain_ranges = [None] * batch_size if villain_ranges is None else list(villain_ranges)
    if len(hero_ranges) != batch_size or len(villain_ranges) != batch_size:
        raise ValueError("hero_ranges and villain_ranges must have one entry per tree")

    states = []
    for index, tree in enumerate(trees):
        nn = int(tree["n_nodes"])
        expected_shape = (nn, n_actions, n)
        regret_sum = np.zeros(expected_shape, dtype=np.float32)
        strategy_sum = np.zeros_like(regret_sum)
        hr_at = np.zeros((nn, n), dtype=np.float32)
        vr_at = np.zeros_like(hr_at)
        hvals = np.zeros_like(hr_at)
        vvals = np.zeros_like(hr_at)
        decision_actions = tree["decision_actions"]
        children = tree["children"]
        action_arrays = []
        child_arrays = []
        for node_i, actions in enumerate(decision_actions):
            if actions:
                action_array = np.asarray(actions, dtype=np.intp)
                child_array = children[node_i, action_array].astype(np.intp, copy=False)
            else:
                action_array = np.zeros(0, dtype=np.intp)
                child_array = np.zeros(0, dtype=np.intp)
            action_arrays.append(action_array)
            child_arrays.append(child_array)

        hero_init = (
            np.ones(n, dtype=np.float32)
            if hero_ranges[index] is None
            else np.asarray(hero_ranges[index], dtype=np.float32)
        )
        villain_init = (
            np.ones(n, dtype=np.float32)
            if villain_ranges[index] is None
            else np.asarray(villain_ranges[index], dtype=np.float32)
        )
        if hero_init.shape != (n,) or villain_init.shape != (n,):
            raise ValueError(f"range {index} must have shape ({n},)")

        hi_all = float(hero_stack_starts[index]) - np.asarray(tree["stacks_h"], dtype=np.float32)
        vi_all = float(villain_stack_starts[index]) - np.asarray(tree["stacks_v"], dtype=np.float32)
        show_idx = np.asarray(tree["showdown_idx"], dtype=np.intp)
        hfold_idx = np.asarray(tree["hero_fold_idx"], dtype=np.intp)
        vfold_idx = np.asarray(tree["villain_fold_idx"], dtype=np.intp)
        state = {
            "tree": tree,
            "regret_sum": regret_sum,
            "strategy_sum": strategy_sum,
            "hr_at": hr_at,
            "vr_at": vr_at,
            "hvals": hvals,
            "vvals": vvals,
            "action_arrays": action_arrays,
            "child_arrays": child_arrays,
            "hero_init": hero_init,
            "villain_init": villain_init,
            "show_idx": show_idx,
            "hfold_idx": hfold_idx,
            "vfold_idx": vfold_idx,
            "win_m": np.asarray(win_ms[index], dtype=np.float32),
            "lose_m": np.asarray(lose_ms[index], dtype=np.float32),
            "tie_m": np.asarray(tie_ms[index], dtype=np.float32),
            "valid_m": np.asarray(valid_ms[index], dtype=np.float32),
            "win_mT": np.asarray(win_ms[index], dtype=np.float32).T.copy(),
            "lose_mT": np.asarray(lose_ms[index], dtype=np.float32).T.copy(),
            "tie_mT": np.asarray(tie_ms[index], dtype=np.float32).T.copy(),
            "valid_mT": np.asarray(valid_ms[index], dtype=np.float32).T.copy(),
        }
        if state["win_m"].shape != (n, n) or state["valid_m"].shape != (n, n):
            raise ValueError(f"terminal matrices for tree {index} must have shape ({n}, {n})")
        if show_idx.size:
            hi_s = hi_all[show_idx].astype(np.float32)
            vi_s = vi_all[show_idx].astype(np.float32)
            state["hw_s"] = (float(pot_starts[index]) + vi_s).reshape(-1, 1)
            state["hl_s"] = (-hi_s).reshape(-1, 1)
            state["ht_s"] = ((float(pot_starts[index]) + vi_s - hi_s) / 2.0).reshape(-1, 1)
            state["vw_s"] = (float(pot_starts[index]) + hi_s).reshape(-1, 1)
            state["vl_s"] = (-vi_s).reshape(-1, 1)
            state["vt_s"] = ((float(pot_starts[index]) + hi_s - vi_s) / 2.0).reshape(-1, 1)
        if hfold_idx.size:
            hi_hf = hi_all[hfold_idx].astype(np.float32)
            state["hf_hero_coeff"] = (-hi_hf).reshape(-1, 1)
            state["hf_vill_coeff"] = (float(pot_starts[index]) + hi_hf).reshape(-1, 1)
        if vfold_idx.size:
            vi_vf = vi_all[vfold_idx].astype(np.float32)
            state["vf_hero_coeff"] = (float(pot_starts[index]) + vi_vf).reshape(-1, 1)
            state["vf_vill_coeff"] = (-vi_vf).reshape(-1, 1)
        states.append(state)

    for _iter in range(int(n_iterations)):
        for state in states:
            tree = state["tree"]
            hr_at = state["hr_at"]
            vr_at = state["vr_at"]
            hvals = state["hvals"]
            vvals = state["vvals"]
            hr_at.fill(0.0)
            vr_at.fill(0.0)
            hvals.fill(0.0)
            vvals.fill(0.0)
            hr_at[0] = state["hero_init"]
            vr_at[0] = state["villain_init"]
            state["iteration_strategy"] = [None] * int(tree["n_nodes"])
            for node_i in range(int(tree["n_nodes"])):
                if int(tree["player"][node_i]) == -1:
                    continue
                actions = state["action_arrays"][node_i]
                children = state["child_arrays"][node_i]
                strategy = _regret_matching_strategy(state["regret_sum"][node_i, actions, :])
                state["iteration_strategy"][node_i] = strategy
                if int(tree["player"][node_i]) == 0:
                    hr_at[children, :] = hr_at[node_i].reshape(1, -1) * strategy
                    vr_at[children, :] = vr_at[node_i]
                else:
                    hr_at[children, :] = hr_at[node_i]
                    vr_at[children, :] = vr_at[node_i].reshape(1, -1) * strategy

        show_states = [state for state in states if state["show_idx"].size]
        if show_states:
            vr_show = [state["vr_at"][state["show_idx"]] for state in show_states]
            hr_show = [state["hr_at"][state["show_idx"]] for state in show_states]
            h_win = _ragged_bmm_numpy(vr_show, [state["win_mT"] for state in show_states], n_hands=n, torch_device=torch_device)
            h_lose = _ragged_bmm_numpy(vr_show, [state["lose_mT"] for state in show_states], n_hands=n, torch_device=torch_device)
            h_tie = _ragged_bmm_numpy(vr_show, [state["tie_mT"] for state in show_states], n_hands=n, torch_device=torch_device)
            v_win = _ragged_bmm_numpy(hr_show, [state["win_m"] for state in show_states], n_hands=n, torch_device=torch_device)
            v_lose = _ragged_bmm_numpy(hr_show, [state["lose_m"] for state in show_states], n_hands=n, torch_device=torch_device)
            v_tie = _ragged_bmm_numpy(hr_show, [state["tie_m"] for state in show_states], n_hands=n, torch_device=torch_device)
            for index, state in enumerate(show_states):
                show_idx = state["show_idx"]
                state["hvals"][show_idx] = (
                    state["hw_s"] * h_win[index]
                    + state["hl_s"] * h_lose[index]
                    + state["ht_s"] * h_tie[index]
                )
                state["vvals"][show_idx] = (
                    state["vw_s"] * v_lose[index]
                    + state["vl_s"] * v_win[index]
                    + state["vt_s"] * v_tie[index]
                )

        hfold_states = [state for state in states if state["hfold_idx"].size]
        if hfold_states:
            vr_hfold = [state["vr_at"][state["hfold_idx"]] for state in hfold_states]
            hr_hfold = [state["hr_at"][state["hfold_idx"]] for state in hfold_states]
            hfold_hero = _ragged_bmm_numpy(vr_hfold, [state["valid_mT"] for state in hfold_states], n_hands=n, torch_device=torch_device)
            hfold_villain = _ragged_bmm_numpy(hr_hfold, [state["valid_m"] for state in hfold_states], n_hands=n, torch_device=torch_device)
            for index, state in enumerate(hfold_states):
                hfold_idx = state["hfold_idx"]
                state["hvals"][hfold_idx] = state["hf_hero_coeff"] * hfold_hero[index]
                state["vvals"][hfold_idx] = state["hf_vill_coeff"] * hfold_villain[index]

        vfold_states = [state for state in states if state["vfold_idx"].size]
        if vfold_states:
            vr_vfold = [state["vr_at"][state["vfold_idx"]] for state in vfold_states]
            hr_vfold = [state["hr_at"][state["vfold_idx"]] for state in vfold_states]
            vfold_hero = _ragged_bmm_numpy(vr_vfold, [state["valid_mT"] for state in vfold_states], n_hands=n, torch_device=torch_device)
            vfold_villain = _ragged_bmm_numpy(hr_vfold, [state["valid_m"] for state in vfold_states], n_hands=n, torch_device=torch_device)
            for index, state in enumerate(vfold_states):
                vfold_idx = state["vfold_idx"]
                state["hvals"][vfold_idx] = state["vf_hero_coeff"] * vfold_hero[index]
                state["vvals"][vfold_idx] = state["vf_vill_coeff"] * vfold_villain[index]

        for state in states:
            tree = state["tree"]
            player = tree["player"]
            hvals = state["hvals"]
            vvals = state["vvals"]
            hr_at = state["hr_at"]
            vr_at = state["vr_at"]
            regret_sum = state["regret_sum"]
            strategy_sum = state["strategy_sum"]
            for node_i in reversed(range(int(tree["n_nodes"]))):
                if int(player[node_i]) == -1:
                    continue
                actions = state["action_arrays"][node_i]
                children = state["child_arrays"][node_i]
                strategy = state["iteration_strategy"][node_i]
                child_hvals = hvals[children, :]
                child_vvals = vvals[children, :]
                hval = np.sum(strategy * child_hvals, axis=0, dtype=np.float32)
                vval = np.sum(strategy * child_vvals, axis=0, dtype=np.float32)
                hvals[node_i] = hval
                vvals[node_i] = vval
                if int(player[node_i]) == 0:
                    instant = child_hvals - hval.reshape(1, -1)
                    regret_sum[node_i, actions, :] = np.maximum(
                        regret_sum[node_i, actions, :] + instant,
                        0,
                    )
                    strategy_sum[node_i, actions, :] += hr_at[node_i].reshape(1, -1) * strategy
                else:
                    instant = child_vvals - vval.reshape(1, -1)
                    regret_sum[node_i, actions, :] = np.maximum(
                        regret_sum[node_i, actions, :] + instant,
                        0,
                    )
                    strategy_sum[node_i, actions, :] += vr_at[node_i].reshape(1, -1) * strategy

    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    return [
        (state["regret_sum"].copy(), state["strategy_sum"].copy())
        for state in states
    ]


def solve_cfr(tree, n_hands, win_m, lose_m, tie_m, valid_m,
              pot_start, hero_stack_start, villain_stack_start,
              n_iterations=100, hero_range=None, villain_range=None,
              showdown_leaf_fn=None, cut_node_indices=None, cut_node_fn=None,
              initial_regret_sum=None, initial_strategy_sum=None,
              trace_node_indices=None, trace_node_fn=None,
              iteration_update_fn=None,
              solver_update="cfr_plus"):
    """Run iterative CFR+ with batched terminal evaluation.

    Parameters
    ----------
    tree : dict
        Output of build_tree_arrays().
    n_hands : int
        Number of hands in the solver.
    win_m, lose_m, tie_m : (n, n) float32 arrays
        Terminal evaluation matrices (hero wins/loses/ties vs villain).
        For river: exact showdown. For turn: equity averaged over runouts.
    valid_m : (n, n) float32 array
        Card conflict mask (1 = valid pair, 0 = conflict).
    pot_start, hero_stack_start, villain_stack_start : int
    n_iterations : int
    hero_range, villain_range : (n,) float32 arrays or None
    showdown_leaf_fn : callable or None
        Optional CPU-only diagnostic hook. When provided, it receives the
        default showdown counterfactual numerator values and current terminal
        reaches, and returns replacement `(hero_values, villain_values)`.
    cut_node_indices, cut_node_fn : list[int], callable or None
        Optional CPU-only diagnostic depth-limit hook. Cut nodes are treated as
        terminal frontier states: descendants receive no reach/regret updates,
        and the hook returns replacement counterfactual numerator values with
        shape `(n_cut_nodes, n_hands)`.
    initial_regret_sum, initial_strategy_sum : array-like or None
        Optional CFR+ warm-start tensors with shape `(n_nodes, n_actions,
        n_hands)`. Values must be finite and non-negative.
    trace_node_indices, trace_node_fn : list[int], callable or None
        Optional CPU-only diagnostic hook. Trace nodes are not cut and do not
        alter solving. The hook receives current reaches and exact node values
        after each backward pass.
    iteration_update_fn : callable or None
        Optional CPU-only diagnostic hook for learned in-resolver update
        experiments. It is called after each CFR iteration with copies of the
        current solver state and may return a dict containing replacement
        ``regret_sum`` and/or ``strategy_sum`` arrays.
    solver_update : {"cfr_plus", "dcfr_plus", "pdcfr_plus"}
        Regret-minimization update rule. ``cfr_plus`` preserves the historical
        recurrence. ``dcfr_plus`` applies fixed DCFR-style discounts to old
        positive regret and average-strategy mass before each new update.
        ``pdcfr_plus`` applies the PDCFR+ optimistic-regret update from the
        published default coefficients.

    Returns
    -------
    regret_sum : (n_nodes, n_actions, n) float32 array
    strategy_sum : (n_nodes, n_actions, n) float32 array
    """
    n = n_hands
    nn = tree['n_nodes']
    n_actions = tree['n_actions']
    player = tree['player']
    children = tree['children']
    decision_actions = tree['decision_actions']
    solver_update = _validate_solver_update(solver_update)
    cut_idx, cut_mask = _frontier_cut_nodes(tree, cut_node_indices)
    trace_idx = _validated_trace_nodes(tree, trace_node_indices)
    inactive = _descendants_of_cut_nodes(tree, cut_mask)
    if cut_idx.size and cut_node_fn is None:
        raise ValueError("cut_node_fn is required when cut_node_indices are provided")
    if cut_node_fn is not None and cut_idx.size == 0:
        raise ValueError("cut_node_indices are required when cut_node_fn is provided")
    if trace_idx.size and trace_node_fn is None:
        raise ValueError("trace_node_fn is required when trace_node_indices are provided")
    if trace_node_fn is not None and trace_idx.size == 0:
        raise ValueError("trace_node_indices are required when trace_node_fn is provided")
    if iteration_update_fn is not None and not callable(iteration_update_fn):
        raise ValueError("iteration_update_fn must be callable")

    # Pre-extract terminal data as contiguous arrays
    show_idx = tree['showdown_idx']
    hfold_idx = tree['hero_fold_idx']
    vfold_idx = tree['villain_fold_idx']

    # Precompute terminal coefficients
    # hero_invested[i] = hero_stack_start - stacks_h[i]
    # villain_invested[i] = villain_stack_start - stacks_v[i]
    hi_all = hero_stack_start - tree['stacks_h']
    vi_all = villain_stack_start - tree['stacks_v']

    # Showdown coefficients: (n_show,)
    if len(show_idx) > 0:
        hi_s = hi_all[show_idx].astype(np.float32)
        vi_s = vi_all[show_idx].astype(np.float32)
        hw_s = (pot_start + vi_s).reshape(-1, 1)  # hero wins
        hl_s = (-hi_s).reshape(-1, 1)              # hero loses
        ht_s = ((pot_start + vi_s - hi_s) / 2).reshape(-1, 1)  # hero ties
        vw_s = (pot_start + hi_s).reshape(-1, 1)  # villain wins
        vl_s = (-vi_s).reshape(-1, 1)              # villain loses
        vt_s = ((pot_start + hi_s - vi_s) / 2).reshape(-1, 1)

    # Fold coefficients
    if len(hfold_idx) > 0:
        hi_hf = hi_all[hfold_idx].astype(np.float32)
        hf_hero_coeff = (-hi_hf).reshape(-1, 1)
        hf_vill_coeff = (pot_start + hi_hf).reshape(-1, 1)

    if len(vfold_idx) > 0:
        vi_vf = vi_all[vfold_idx].astype(np.float32)
        vf_hero_coeff = (pot_start + vi_vf).reshape(-1, 1)
        vf_vill_coeff = (-vi_vf).reshape(-1, 1)

    # CFR arrays
    expected_shape = (nn, n_actions, n)
    regret_sum = _validated_initial_array(
        initial_regret_sum,
        expected_shape,
        "initial_regret_sum",
    )
    if regret_sum is None:
        regret_sum = np.zeros(expected_shape, dtype=np.float32)
    strategy_sum = _validated_initial_array(
        initial_strategy_sum,
        expected_shape,
        "initial_strategy_sum",
    )
    if strategy_sum is None:
        strategy_sum = np.zeros(expected_shape, dtype=np.float32)
    prev_imm_regret = (
        np.zeros_like(regret_sum, dtype=np.float32)
        if solver_update == "pdcfr_plus"
        else None
    )

    # Reach probability and value arrays
    hr_at = np.zeros((nn, n), dtype=np.float32)
    vr_at = np.zeros((nn, n), dtype=np.float32)
    hvals = np.zeros((nn, n), dtype=np.float32)
    vvals = np.zeros((nn, n), dtype=np.float32)

    # Initial ranges
    hr_init = hero_range if hero_range is not None else np.ones(n, dtype=np.float32)
    vr_init = villain_range if villain_range is not None else np.ones(n, dtype=np.float32)

    # BFS order is just 0..n_nodes-1 (nodes were added in BFS order)
    bfs_order = range(nn)
    decision_action_arrays = []
    decision_child_arrays = []
    for i, actions in enumerate(decision_actions):
        if actions:
            action_array = np.asarray(actions, dtype=np.intp)
            child_array = children[i, action_array].astype(np.intp, copy=False)
        else:
            action_array = np.zeros(0, dtype=np.intp)
            child_array = np.zeros(0, dtype=np.intp)
        decision_action_arrays.append(action_array)
        decision_child_arrays.append(child_array)

    # Transpose matrices once (for villain terminal eval)
    win_mT = win_m.T.copy()   # contiguous for GEMM
    lose_mT = lose_m.T.copy()
    tie_mT = tie_m.T.copy()
    valid_mT = valid_m.T.copy()

    for _iter in range(n_iterations):
        iteration_number = _iter + 1
        policy_regret_sum = regret_sum
        if solver_update == "dcfr_plus":
            regret_discount, strategy_discount = _dcfr_discount_factors(iteration_number)
            regret_sum *= regret_discount
            strategy_sum *= strategy_discount
        elif solver_update == "pdcfr_plus":
            strategy_sum *= _pdcfr_strategy_discount(iteration_number)
            policy_regret_sum = np.maximum(
                regret_sum * _pdcfr_discount_factor(iteration_number) + prev_imm_regret,
                0.0,
            )

        iteration_strategy = [None] * nn
        if cut_idx.size:
            hr_at.fill(0.0)
            vr_at.fill(0.0)
            hvals.fill(0.0)
            vvals.fill(0.0)

        # ===== Forward pass: compute reach probabilities =====
        hr_at[0] = hr_init
        vr_at[0] = vr_init

        for i in bfs_order:
            if inactive[i]:
                continue
            if player[i] == -1:
                continue
            if cut_mask[i]:
                continue
            actions = decision_action_arrays[i]
            child_indices = decision_child_arrays[i]
            strategy = _regret_matching_strategy(policy_regret_sum[i, actions, :])
            iteration_strategy[i] = strategy

            if player[i] == 0:
                hr_at[child_indices, :] = hr_at[i].reshape(1, -1) * strategy
                vr_at[child_indices, :] = vr_at[i]
            else:
                hr_at[child_indices, :] = hr_at[i]
                vr_at[child_indices, :] = vr_at[i].reshape(1, -1) * strategy

        # ===== Batched terminal evaluation =====

        # Showdown terminals (the big win: batched GEMM)
        if len(show_idx) > 0:
            VR_s = vr_at[show_idx]  # (n_show, n)
            HR_s = hr_at[show_idx]  # (n_show, n)

            # Hero values: M @ vr for each terminal → VR @ M.T
            h_win = VR_s @ win_mT    # (n_show, n) — one GEMM call
            h_lose = VR_s @ lose_mT  # (n_show, n)
            h_tie = VR_s @ tie_mT    # (n_show, n)

            # Villain values: M.T @ hr = hr @ M for each terminal
            v_win = HR_s @ win_m     # (n_show, n) — gives win_m.T @ hr
            v_lose = HR_s @ lose_m   # (n_show, n)
            v_tie = HR_s @ tie_m     # (n_show, n)

            # Apply coefficients and store. A leaf callback may replace these
            # values for depth-limited learned-leaf diagnostics; it must return
            # counterfactual numerator values with shape (n_show, n_hands).
            default_hvals = hw_s * h_win + hl_s * h_lose + ht_s * h_tie
            default_vvals = vw_s * v_lose + vl_s * v_win + vt_s * v_tie
            if showdown_leaf_fn is None:
                hvals[show_idx] = default_hvals
                vvals[show_idx] = default_vvals
            else:
                leaf_hvals, leaf_vvals = showdown_leaf_fn(
                    tree=tree,
                    showdown_indices=show_idx,
                    hero_reach=HR_s,
                    villain_reach=VR_s,
                    valid_m=valid_m,
                    default_hero_values=default_hvals,
                    default_villain_values=default_vvals,
                    pot_start=pot_start,
                    hero_stack_start=hero_stack_start,
                    villain_stack_start=villain_stack_start,
                )
                leaf_hvals = np.asarray(leaf_hvals, dtype=np.float32)
                leaf_vvals = np.asarray(leaf_vvals, dtype=np.float32)
                if leaf_hvals.shape != default_hvals.shape:
                    raise ValueError(
                        "showdown_leaf_fn hero values shape "
                        f"{leaf_hvals.shape} != {default_hvals.shape}"
                    )
                if leaf_vvals.shape != default_vvals.shape:
                    raise ValueError(
                        "showdown_leaf_fn villain values shape "
                        f"{leaf_vvals.shape} != {default_vvals.shape}"
                    )
                hvals[show_idx] = leaf_hvals
                vvals[show_idx] = leaf_vvals

        if cut_idx.size:
            cut_hvals, cut_vvals = cut_node_fn(
                tree=tree,
                cut_indices=cut_idx,
                hero_reach=hr_at[cut_idx],
                villain_reach=vr_at[cut_idx],
                valid_m=valid_m,
                pot_start=pot_start,
                hero_stack_start=hero_stack_start,
                villain_stack_start=villain_stack_start,
            )
            cut_hvals = np.asarray(cut_hvals, dtype=np.float32)
            cut_vvals = np.asarray(cut_vvals, dtype=np.float32)
            expected_shape = (int(cut_idx.size), n)
            if cut_hvals.shape != expected_shape:
                raise ValueError(
                    f"cut_node_fn hero values shape {cut_hvals.shape} != {expected_shape}"
                )
            if cut_vvals.shape != expected_shape:
                raise ValueError(
                    f"cut_node_fn villain values shape {cut_vvals.shape} != {expected_shape}"
                )
            hvals[cut_idx] = cut_hvals
            vvals[cut_idx] = cut_vvals

        # Hero fold terminals
        if len(hfold_idx) > 0:
            VR_hf = vr_at[hfold_idx]  # (n_hf, n)
            HR_hf = hr_at[hfold_idx]  # (n_hf, n)
            hvals[hfold_idx] = hf_hero_coeff * (VR_hf @ valid_mT)   # (n_hf, n)
            vvals[hfold_idx] = hf_vill_coeff * (HR_hf @ valid_m)    # (n_hf, n)

        # Villain fold terminals
        if len(vfold_idx) > 0:
            VR_vf = vr_at[vfold_idx]
            HR_vf = hr_at[vfold_idx]
            hvals[vfold_idx] = vf_hero_coeff * (VR_vf @ valid_mT)
            vvals[vfold_idx] = vf_vill_coeff * (HR_vf @ valid_m)

        # ===== Backward pass: propagate values and update regrets =====
        for i in reversed(range(nn)):
            if inactive[i]:
                continue
            if player[i] == -1:
                continue
            if cut_mask[i]:
                continue

            actions = decision_action_arrays[i]
            child_indices = decision_child_arrays[i]
            strategy = iteration_strategy[i]
            if strategy is None:
                strategy = _regret_matching_strategy(policy_regret_sum[i, actions, :])
            child_hvals = hvals[child_indices, :]
            child_vvals = vvals[child_indices, :]

            hval = np.sum(strategy * child_hvals, axis=0, dtype=np.float32)
            vval = np.sum(strategy * child_vvals, axis=0, dtype=np.float32)

            hvals[i] = hval
            vvals[i] = vval

            # Update regrets (CFR+: clamp to >= 0)
            if player[i] == 0:
                instant = child_hvals - hval.reshape(1, -1)
                if solver_update == "pdcfr_plus":
                    regret_discount = _pdcfr_discount_factor(iteration_number)
                    regret_sum[i, actions, :] = np.maximum(
                        regret_sum[i, actions, :] * regret_discount + instant,
                        0,
                    )
                    prev_imm_regret[i, actions, :] = instant
                else:
                    regret_sum[i, actions, :] = np.maximum(
                        regret_sum[i, actions, :] + instant,
                        0,
                    )
                strategy_sum[i, actions, :] = (
                    strategy_sum[i, actions, :] + hr_at[i].reshape(1, -1) * strategy
                )
            else:
                instant = child_vvals - vval.reshape(1, -1)
                if solver_update == "pdcfr_plus":
                    regret_discount = _pdcfr_discount_factor(iteration_number)
                    regret_sum[i, actions, :] = np.maximum(
                        regret_sum[i, actions, :] * regret_discount + instant,
                        0,
                    )
                    prev_imm_regret[i, actions, :] = instant
                else:
                    regret_sum[i, actions, :] = np.maximum(
                        regret_sum[i, actions, :] + instant,
                        0,
                    )
                strategy_sum[i, actions, :] = (
                    strategy_sum[i, actions, :] + vr_at[i].reshape(1, -1) * strategy
                )

        if trace_idx.size:
            hero_action_values = np.zeros(
                (len(trace_idx), tree['n_actions'], n),
                dtype=np.float32,
            )
            villain_action_values = np.zeros_like(hero_action_values)
            for trace_row, node_i in enumerate(trace_idx):
                for action in decision_actions[node_i]:
                    child_i = children[node_i, action]
                    if child_i >= 0:
                        hero_action_values[trace_row, action] = hvals[child_i]
                        villain_action_values[trace_row, action] = vvals[child_i]
            trace_node_fn(
                iteration=int(_iter),
                tree=tree,
                node_indices=trace_idx,
                hero_reach=hr_at[trace_idx],
                villain_reach=vr_at[trace_idx],
                hero_values=hvals[trace_idx],
                villain_values=vvals[trace_idx],
                hero_action_values=hero_action_values,
                villain_action_values=villain_action_values,
                regret_sum=regret_sum[trace_idx],
                strategy_sum=strategy_sum[trace_idx],
                valid_m=valid_m,
                pot_start=pot_start,
                hero_stack_start=hero_stack_start,
                villain_stack_start=villain_stack_start,
            )
        if iteration_update_fn is not None:
            update = iteration_update_fn(
                iteration=int(iteration_number),
                tree=tree,
                hero_reach=hr_at.copy(),
                villain_reach=vr_at.copy(),
                hero_values=hvals.copy(),
                villain_values=vvals.copy(),
                regret_sum=regret_sum.copy(),
                strategy_sum=strategy_sum.copy(),
                valid_m=valid_m,
                pot_start=pot_start,
                hero_stack_start=hero_stack_start,
                villain_stack_start=villain_stack_start,
            )
            if update is not None:
                if not isinstance(update, dict):
                    raise ValueError("iteration_update_fn must return None or a dict")
                if "regret_sum" in update:
                    regret_sum = _validated_updated_array(
                        update["regret_sum"],
                        expected_shape,
                        "iteration_update_fn regret_sum",
                    )
                if "strategy_sum" in update:
                    strategy_sum = _validated_updated_array(
                        update["strategy_sum"],
                        expected_shape,
                        "iteration_update_fn strategy_sum",
                    )

    return regret_sum, strategy_sum


def solve_cfr_torch(tree, n_hands, win_m, lose_m, tie_m, valid_m,
                    pot_start, hero_stack_start, villain_stack_start,
                    n_iterations=100, hero_range=None, villain_range=None,
                    device="cuda", initial_regret_sum=None,
                    initial_strategy_sum=None, solver_update="cfr_plus"):
    """Run the same CFR+ recurrence with torch tensors on CPU or CUDA.

    This keeps the CPU solver as the reference implementation while allowing
    the dense per-terminal matrix products to run on GPU. The tree walk remains
    Python-driven, so this is an acceleration backend, not a fully fused kernel.
    """
    solver_update = _validate_solver_update(solver_update)
    if solver_update != "cfr_plus":
        raise ValueError(f"solver_update={solver_update!r} is only supported by the CPU CFR backend")

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA solver backend requested but torch.cuda is unavailable.")

    n = n_hands
    nn = tree['n_nodes']
    n_actions = tree['n_actions']
    player_cpu = tree['player']
    children_cpu = tree['children']
    decision_actions = tree['decision_actions']
    decision_action_tensors = [
        torch.as_tensor(actions, dtype=torch.long, device=torch_device)
        if actions else None
        for actions in decision_actions
    ]

    win_t = torch.as_tensor(win_m, dtype=torch.float32, device=torch_device)
    lose_t = torch.as_tensor(lose_m, dtype=torch.float32, device=torch_device)
    tie_t = torch.as_tensor(tie_m, dtype=torch.float32, device=torch_device)
    valid_t = torch.as_tensor(valid_m, dtype=torch.float32, device=torch_device)

    stacks_h = torch.as_tensor(tree['stacks_h'], dtype=torch.float32, device=torch_device)
    stacks_v = torch.as_tensor(tree['stacks_v'], dtype=torch.float32, device=torch_device)
    hi_all = float(hero_stack_start) - stacks_h
    vi_all = float(villain_stack_start) - stacks_v

    show_idx = torch.as_tensor(tree['showdown_idx'], dtype=torch.long, device=torch_device)
    hfold_idx = torch.as_tensor(tree['hero_fold_idx'], dtype=torch.long, device=torch_device)
    vfold_idx = torch.as_tensor(tree['villain_fold_idx'], dtype=torch.long, device=torch_device)

    if show_idx.numel() > 0:
        hi_s = hi_all.index_select(0, show_idx).reshape(-1, 1)
        vi_s = vi_all.index_select(0, show_idx).reshape(-1, 1)
        hw_s = float(pot_start) + vi_s
        hl_s = -hi_s
        ht_s = (float(pot_start) + vi_s - hi_s) / 2.0
        vw_s = float(pot_start) + hi_s
        vl_s = -vi_s
        vt_s = (float(pot_start) + hi_s - vi_s) / 2.0

    if hfold_idx.numel() > 0:
        hi_hf = hi_all.index_select(0, hfold_idx).reshape(-1, 1)
        hf_hero_coeff = -hi_hf
        hf_vill_coeff = float(pot_start) + hi_hf

    if vfold_idx.numel() > 0:
        vi_vf = vi_all.index_select(0, vfold_idx).reshape(-1, 1)
        vf_hero_coeff = float(pot_start) + vi_vf
        vf_vill_coeff = -vi_vf

    expected_shape = (nn, n_actions, n)
    initial_regret_np = _validated_initial_array(
        initial_regret_sum,
        expected_shape,
        "initial_regret_sum",
    )
    initial_strategy_np = _validated_initial_array(
        initial_strategy_sum,
        expected_shape,
        "initial_strategy_sum",
    )
    if initial_regret_np is None:
        regret_sum = torch.zeros(expected_shape, dtype=torch.float32, device=torch_device)
    else:
        regret_sum = torch.as_tensor(initial_regret_np, dtype=torch.float32, device=torch_device)
    if initial_strategy_np is None:
        strategy_sum = torch.zeros_like(regret_sum)
    else:
        strategy_sum = torch.as_tensor(
            initial_strategy_np,
            dtype=torch.float32,
            device=torch_device,
        )
    hr_at = torch.zeros((nn, n), dtype=torch.float32, device=torch_device)
    vr_at = torch.zeros((nn, n), dtype=torch.float32, device=torch_device)
    hvals = torch.zeros((nn, n), dtype=torch.float32, device=torch_device)
    vvals = torch.zeros((nn, n), dtype=torch.float32, device=torch_device)

    hr_init = (
        torch.as_tensor(hero_range, dtype=torch.float32, device=torch_device)
        if hero_range is not None else torch.ones(n, dtype=torch.float32, device=torch_device)
    )
    vr_init = (
        torch.as_tensor(villain_range, dtype=torch.float32, device=torch_device)
        if villain_range is not None else torch.ones(n, dtype=torch.float32, device=torch_device)
    )

    for _iter in range(n_iterations):
        hr_at.zero_()
        vr_at.zero_()
        hvals.zero_()
        vvals.zero_()
        hr_at[0].copy_(hr_init)
        vr_at[0].copy_(vr_init)

        for i in range(nn):
            player = int(player_cpu[i])
            if player == -1:
                continue
            acts_t = decision_action_tensors[i]
            acts = decision_actions[i]
            rs_acts = regret_sum[i].index_select(0, acts_t)
            pos = torch.clamp(rs_acts, min=0.0)
            total = pos.sum(dim=0)
            safe = total.clamp(min=1.0)
            uniform = 1.0 / len(acts)
            strategy = torch.where(total.unsqueeze(0) > 0.0, pos / safe, uniform)

            for k, action in enumerate(acts):
                child_idx = int(children_cpu[i, action])
                if player == 0:
                    hr_at[child_idx].copy_(hr_at[i] * strategy[k])
                    vr_at[child_idx].copy_(vr_at[i])
                else:
                    hr_at[child_idx].copy_(hr_at[i])
                    vr_at[child_idx].copy_(vr_at[i] * strategy[k])

        if show_idx.numel() > 0:
            vr_s = hr_at.new_empty((show_idx.numel(), n))
            hr_s = hr_at.new_empty((show_idx.numel(), n))
            vr_s.copy_(vr_at.index_select(0, show_idx))
            hr_s.copy_(hr_at.index_select(0, show_idx))

            h_win = vr_s @ win_t.T
            h_lose = vr_s @ lose_t.T
            h_tie = vr_s @ tie_t.T
            v_win = hr_s @ win_t
            v_lose = hr_s @ lose_t
            v_tie = hr_s @ tie_t

            hvals[show_idx] = hw_s * h_win + hl_s * h_lose + ht_s * h_tie
            vvals[show_idx] = vw_s * v_lose + vl_s * v_win + vt_s * v_tie

        if hfold_idx.numel() > 0:
            vr_hf = vr_at.index_select(0, hfold_idx)
            hr_hf = hr_at.index_select(0, hfold_idx)
            hvals[hfold_idx] = hf_hero_coeff * (vr_hf @ valid_t.T)
            vvals[hfold_idx] = hf_vill_coeff * (hr_hf @ valid_t)

        if vfold_idx.numel() > 0:
            vr_vf = vr_at.index_select(0, vfold_idx)
            hr_vf = hr_at.index_select(0, vfold_idx)
            hvals[vfold_idx] = vf_hero_coeff * (vr_vf @ valid_t.T)
            vvals[vfold_idx] = vf_vill_coeff * (hr_vf @ valid_t)

        for i in reversed(range(nn)):
            player = int(player_cpu[i])
            if player == -1:
                continue
            acts_t = decision_action_tensors[i]
            acts = decision_actions[i]
            rs_acts = regret_sum[i].index_select(0, acts_t)
            pos = torch.clamp(rs_acts, min=0.0)
            total = pos.sum(dim=0)
            safe = total.clamp(min=1.0)
            uniform = 1.0 / len(acts)
            strategy = torch.where(total.unsqueeze(0) > 0.0, pos / safe, uniform)

            hval = torch.zeros(n, dtype=torch.float32, device=torch_device)
            vval = torch.zeros(n, dtype=torch.float32, device=torch_device)
            for k, action in enumerate(acts):
                child_idx = int(children_cpu[i, action])
                hval += strategy[k] * hvals[child_idx]
                vval += strategy[k] * vvals[child_idx]

            hvals[i].copy_(hval)
            vvals[i].copy_(vval)

            if player == 0:
                reach = hr_at[i]
                for k, action in enumerate(acts):
                    child_idx = int(children_cpu[i, action])
                    regret_sum[i, action] = torch.clamp(
                        regret_sum[i, action] + hvals[child_idx] - hval,
                        min=0.0,
                    )
                    strategy_sum[i, action] += reach * strategy[k]
            else:
                reach = vr_at[i]
                for k, action in enumerate(acts):
                    child_idx = int(children_cpu[i, action])
                    regret_sum[i, action] = torch.clamp(
                        regret_sum[i, action] + vvals[child_idx] - vval,
                        min=0.0,
                    )
                    strategy_sum[i, action] += reach * strategy[k]

    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    return regret_sum.cpu().numpy(), strategy_sum.cpu().numpy()


def get_average_strategy(strategy_sum, node_idx, actions, hand_idx):
    """Get average strategy for a specific hand at a specific node."""
    sums = np.array([strategy_sum[node_idx, a, hand_idx] for a in actions],
                    dtype=np.float32)
    total = sums.sum()
    if total > 0:
        return dict(zip(actions, (sums / total).tolist()))
    return dict(zip(actions, [1.0 / len(actions)] * len(actions)))


def prune_hands(hands, hero_range=None, villain_range=None,
                keep_hand=None, threshold=1e-4):
    """Filter hand set to only include hands with non-trivial probability.

    Parameters
    ----------
    hands : list of (c1, c2) tuples
    hero_range, villain_range : optional (n,) arrays aligned with hands
    keep_hand : optional (c1, c2) tuple that must be kept (our actual hand)
    threshold : minimum probability (relative to max) to keep

    Returns
    -------
    active_hands : list of (c1, c2)
    active_indices : array of original indices
    pruned_hero : (n_active,) or None
    pruned_villain : (n_active,) or None
    """
    n = len(hands)
    mask = np.ones(n, dtype=bool)

    if villain_range is not None:
        vmax = villain_range.max()
        if vmax > 0:
            mask &= villain_range >= threshold * vmax

    if hero_range is not None:
        hmax = hero_range.max()
        if hmax > 0:
            # Keep hands that are in EITHER range
            mask |= hero_range >= threshold * hmax

    # Always keep our actual hand
    if keep_hand is not None:
        h = tuple(sorted(keep_hand))
        for i, hand in enumerate(hands):
            if hand == h:
                mask[i] = True
                break

    active_indices = np.where(mask)[0]
    active_hands = [hands[i] for i in active_indices]

    pruned_hero = None
    pruned_villain = None
    if hero_range is not None:
        pruned_hero = hero_range[active_indices]
        s = pruned_hero.sum()
        if s > 0:
            pruned_hero /= s
    if villain_range is not None:
        pruned_villain = villain_range[active_indices]
        s = pruned_villain.sum()
        if s > 0:
            pruned_villain /= s

    return active_hands, active_indices, pruned_hero, pruned_villain
