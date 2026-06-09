"""Fused population solver for same-topology HUNL street subgames (STEP 5).

``solve_population(specs, ...)`` groups SubgameSpecs by exact topology
(``tree_builder.topology_key`` + street/local-width), stacks per-element
tensors (pots, stacks, ranges, terminal matrices) per bucket, and runs a
float64 PORT of the trusted batched kernel
``scripts/fast_cfr.py::solve_cfr_levelsync_torch_batched_same_topology``
(imported reference; never edited). float64 regret/strategy accumulation is
MANDATORY per the committed 0c contract (float32 drifts at imperfect-info
repeated infosets -- measured again on real HUNL trees in gate P-A, where the
float32 torch kernel sits ~0.35-0.55 avg-strategy L-inf off the trusted
reference while this float64 port must be <= 1e-4).

On top of the ported recurrence this module ADDS the pieces the existing
batched kernel lacks:
  * the batched ROOT VALUE PASS under the average strategy -- the GPU twin of
    ``turn_river.subgame_value_pass`` / ``iig_batched._ev_pass`` -- producing
    the per-belief V* targets ``v0[H]``, ``v1[H]`` per spec (counterfactual:
    opponent-reach-weighted, own range factored out; gate P-C);
  * the per-iteration LEAF HOOK for turn trees (river-deal cut leaves; the
    contract lives in ``terminal_eval``; gate P-E);
  * an optional GADGET ROOT: the opponent gets a per-hand FOLLOW/TERMINATE
    choice above the subgame root (``iig_gadget.gadget_resolve`` semantics in
    range form -- terminate pays the blueprint CFV ``opp_cfv``); it scales only
    the opponent's entry range, so same-topology batching is preserved
    (uniform per call; gate P-F).

Device-parameterized (``device='cpu'|'cuda'``); parity gates run CPU-only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from poker_ai.rebel.hunl import subgame_spec as sgs
from poker_ai.rebel.hunl import terminal_eval as tev
from poker_ai.rebel.hunl import tree_builder as tb

fast_cfr = tb.fast_cfr  # protected surface; imported, never edited

__all__ = [
    "SubgameSolveResult",
    "average_strategy",
    "solve_cfr_levelsync_batched",
    "solve_population",
]

DEFAULT_B_MAX = 8  # memory-safe CPU default; the GPU B_max table lives in the census


def _regret_matching_batched(regret_rows, legal_mask):
    """Batched regret matching with the TRUSTED numpy semantics.

    Port of ``fast_cfr._level_regret_matching`` / ``_regret_matching_strategy``
    (``safe = where(totals > 0, totals, 1)``) rather than the torch helper
    ``_torch_level_regret_matching_batched``, whose ``totals.clamp(min=1.0)``
    fails to normalize strategies at nodes whose positive-regret total lies in
    (0, 1) -- measured here as a ~0.4 avg-strategy L-inf divergence from
    ``solve_cfr`` on real HUNL river trees (gate P-A would fail through any
    dtype). ``regret_rows``: [B, P, A, H]; ``legal_mask``: [P, A] bool.
    """
    legal = legal_mask.to(dtype=regret_rows.dtype).unsqueeze(0).unsqueeze(-1)
    positive = torch.clamp(regret_rows, min=0.0) * legal
    totals = positive.sum(dim=2)
    safe = torch.where(totals > 0.0, totals, torch.ones_like(totals))
    legal_counts = legal_mask.sum(dim=1).to(dtype=regret_rows.dtype).clamp(min=1.0)
    uniform = (1.0 / legal_counts).reshape(1, -1, 1, 1)
    return torch.where(
        totals.unsqueeze(2) > 0.0,
        positive / safe.unsqueeze(2),
        legal * uniform,
    )


def average_strategy(strategy_sum) -> np.ndarray:
    """Per-(node, hand) average strategy; float64 twin of
    ``turn_river._average_strategy_array`` (unreached all-zero rows stay zero)."""
    s = np.asarray(strategy_sum, dtype=np.float64)
    denom = s.sum(axis=-2, keepdims=True)
    return np.divide(s, denom, out=np.zeros_like(s), where=denom > 1e-12)


def _stack_f32(matrices, n, name):
    stacked = np.stack([np.asarray(m, dtype=np.float32) for m in matrices], axis=0)
    if stacked.shape != (len(matrices), n, n):
        raise ValueError(f"{name} batch shape {stacked.shape} != {(len(matrices), n, n)}")
    return stacked


def _torch_groups(base_tree, torch_device):
    """Level edge groups as torch index tensors (mirrors the trusted kernel)."""
    groups_np = base_tree.setdefault(
        "_level_edge_groups", fast_cfr._level_edge_groups(base_tree))
    groups = []
    for group in groups_np:
        if group is None:
            groups.append(None)
            continue
        edge_player = torch.as_tensor(
            group["edge_player"], dtype=torch.long, device=torch_device)
        groups.append({
            "parents": torch.as_tensor(group["parents"], dtype=torch.long, device=torch_device),
            "edge_parent": torch.as_tensor(group["edge_parent"], dtype=torch.long, device=torch_device),
            "edge_action": torch.as_tensor(group["edge_action"], dtype=torch.long, device=torch_device),
            "edge_child": torch.as_tensor(group["edge_child"], dtype=torch.long, device=torch_device),
            "edge_parent_pos": torch.as_tensor(group["edge_parent_pos"], dtype=torch.long, device=torch_device),
            "legal_mask": torch.as_tensor(group["legal_mask"], dtype=torch.bool, device=torch_device),
            "hero_edges": edge_player == 0,
            "villain_edges": edge_player != 0,
        })
    return groups


def _leaf_ctx(iteration, specs, trees, show_idx_np, hr_show, vr_show,
              default_h, default_v, valid_list, pot_t, hs_t, vs_t):
    """Assemble the leaf-hook context (contract in terminal_eval)."""
    show = np.asarray(show_idx_np)
    return {
        "iteration": iteration,
        "specs": specs,
        "trees": trees,
        "showdown_indices": show,
        "hero_reach": hr_show,
        "villain_reach": vr_show,
        "default_hero_values": default_h,
        "default_villain_values": default_v,
        "valid_ms": valid_list,
        "pot_starts": pot_t,
        "hero_stack_starts": hs_t,
        "villain_stack_starts": vs_t,
        "terminal_pots": np.stack(
            [np.asarray(tree["pot"])[show] for tree in trees], axis=0),
        "terminal_stacks_h": np.stack(
            [np.asarray(tree["stacks_h"])[show] for tree in trees], axis=0),
        "terminal_stacks_v": np.stack(
            [np.asarray(tree["stacks_v"])[show] for tree in trees], axis=0),
    }


def _apply_leaf(leaf_hook, ctx, expected_shape, dtype, torch_device):
    leaf_h, leaf_v = leaf_hook(ctx)
    leaf_h = np.asarray(leaf_h, dtype=np.float64)
    leaf_v = np.asarray(leaf_v, dtype=np.float64)
    if leaf_h.shape != expected_shape or leaf_v.shape != expected_shape:
        raise ValueError(
            f"leaf hook values shape {leaf_h.shape}/{leaf_v.shape} != {expected_shape}")
    return (torch.as_tensor(leaf_h, dtype=dtype, device=torch_device),
            torch.as_tensor(leaf_v, dtype=dtype, device=torch_device))


def solve_cfr_levelsync_batched(
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
    dtype=torch.float64,
    device="cpu",
    leaf_hook=None,
    leaf_specs=None,
    gadget_player=None,
    gadget_opp_cfv=None,
    compute_value_pass=True,
):
    """float64 port of ``solve_cfr_levelsync_torch_batched_same_topology``.

    Same recurrence on a batch of exact-same-topology trees with PER-ELEMENT
    pots/stacks/ranges/terminal-matrices, plus: parameterized accumulation
    dtype (float64 default, mandatory for production), an optional showdown
    leaf hook (river-deal cut leaves on turn trees -- ``win/lose/tie = None``
    is allowed then; defaults are zero and the hook must replace them), an
    optional opponent FOLLOW/TERMINATE gadget root, and the batched
    average-strategy root value pass emitting per-belief v0/v1.

    Returns a dict of numpy arrays: ``regret_sum``/``strategy_sum``/
    ``avg_strategy`` [B, nn, A, H] float64, ``v0``/``v1`` [B, H] float64
    (None if ``compute_value_pass=False``), and ``gadget`` diagnostics.
    """
    trees = list(trees)
    batch_size = len(trees)
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda is unavailable.")
    dtype = torch.empty(0, dtype=dtype).dtype
    if not dtype.is_floating_point:
        raise ValueError(f"dtype must be floating point, got {dtype}")
    if not (
        len(valid_ms)
        == len(pot_starts)
        == len(hero_stack_starts)
        == len(villain_stack_starts)
        == batch_size
    ):
        raise ValueError("all batched inputs must have one entry per tree")

    nn, n_actions, n = fast_cfr._validate_same_topology_batch(trees, n_hands)
    base_tree = trees[0]
    groups = _torch_groups(base_tree, torch_device)

    has_matrices = win_ms is not None or lose_ms is not None or tie_ms is not None
    if has_matrices:
        if win_ms is None or lose_ms is None or tie_ms is None:
            raise ValueError("win/lose/tie matrices must be given together")
        if not (len(win_ms) == len(lose_ms) == len(tie_ms) == batch_size):
            raise ValueError("win/lose/tie must have one entry per tree")

    show_idx_np = np.asarray(base_tree["showdown_idx"])
    if not has_matrices and leaf_hook is None and show_idx_np.size > 0:
        raise ValueError(
            "trees with showdown terminals need win/lose/tie matrices or a leaf hook")

    valid_np = _stack_f32(valid_ms, n, "valid_ms")
    valid_t = torch.as_tensor(valid_np, dtype=dtype, device=torch_device)
    valid_tT = valid_t.transpose(1, 2).contiguous()
    if has_matrices:
        win_t = torch.as_tensor(_stack_f32(win_ms, n, "win_ms"), dtype=dtype, device=torch_device)
        lose_t = torch.as_tensor(_stack_f32(lose_ms, n, "lose_ms"), dtype=dtype, device=torch_device)
        tie_t = torch.as_tensor(_stack_f32(tie_ms, n, "tie_ms"), dtype=dtype, device=torch_device)
        win_tT = win_t.transpose(1, 2).contiguous()
        lose_tT = lose_t.transpose(1, 2).contiguous()
        tie_tT = tie_t.transpose(1, 2).contiguous()

    stacks_h = torch.as_tensor(
        np.stack([np.asarray(tree["stacks_h"], dtype=np.float64) for tree in trees], axis=0),
        dtype=dtype, device=torch_device)
    stacks_v = torch.as_tensor(
        np.stack([np.asarray(tree["stacks_v"], dtype=np.float64) for tree in trees], axis=0),
        dtype=dtype, device=torch_device)
    pot_np = np.asarray(pot_starts, dtype=np.float64)
    hs_np = np.asarray(hero_stack_starts, dtype=np.float64)
    vs_np = np.asarray(villain_stack_starts, dtype=np.float64)
    pot_t = torch.as_tensor(pot_np, dtype=dtype, device=torch_device).reshape(batch_size, 1, 1)
    hero_stack_t = torch.as_tensor(hs_np, dtype=dtype, device=torch_device).reshape(batch_size, 1)
    villain_stack_t = torch.as_tensor(vs_np, dtype=dtype, device=torch_device).reshape(batch_size, 1)

    hi_all = hero_stack_t - stacks_h
    vi_all = villain_stack_t - stacks_v
    show_idx = torch.as_tensor(show_idx_np, dtype=torch.long, device=torch_device)
    hfold_idx = torch.as_tensor(base_tree["hero_fold_idx"], dtype=torch.long, device=torch_device)
    vfold_idx = torch.as_tensor(base_tree["villain_fold_idx"], dtype=torch.long, device=torch_device)

    hw_s = hl_s = ht_s = vw_s = vl_s = vt_s = None
    hf_hero_coeff = hf_vill_coeff = vf_hero_coeff = vf_vill_coeff = None
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

    regret_sum = torch.zeros((batch_size, nn, n_actions, n), dtype=dtype, device=torch_device)
    strategy_sum = torch.zeros_like(regret_sum)
    hr_at = torch.zeros((batch_size, nn, n), dtype=dtype, device=torch_device)
    vr_at = torch.zeros_like(hr_at)
    hvals = torch.zeros_like(hr_at)
    vvals = torch.zeros_like(hr_at)

    def _ranges_tensor(ranges, name):
        if ranges is None:
            return torch.ones((batch_size, n), dtype=dtype, device=torch_device)
        arr = np.stack([np.asarray(r, dtype=np.float64) for r in ranges], axis=0)
        if arr.shape != (batch_size, n):
            raise ValueError(f"{name} must have shape (batch, n_hands), got {arr.shape}")
        return torch.as_tensor(arr, dtype=dtype, device=torch_device)

    hr_init = _ranges_tensor(hero_ranges, "hero_ranges")
    vr_init = _ranges_tensor(villain_ranges, "villain_ranges")

    # ----- gadget root (iig_gadget semantics in range form) -----
    gadget = None
    if gadget_player is not None:
        gadget_player = int(gadget_player)
        if gadget_player not in (0, 1):
            raise ValueError(f"gadget_player must be 0 or 1, got {gadget_player!r}")
        if gadget_opp_cfv is None:
            raise ValueError("gadget_opp_cfv is required with gadget_player")
        tval_np = np.stack(
            [np.asarray(v, dtype=np.float64) for v in gadget_opp_cfv], axis=0)
        if tval_np.shape != (batch_size, n):
            raise ValueError(
                f"gadget_opp_cfv must have shape (batch, n_hands), got {tval_np.shape}")
        tval = torch.as_tensor(tval_np, dtype=dtype, device=torch_device)
        # normalization mass: resolver-range weight compatible with each opp hand
        if gadget_player == 1:
            den = (hr_init.unsqueeze(1) @ valid_t).squeeze(1)        # [B, H]
        else:
            den = (valid_t @ vr_init.unsqueeze(-1)).squeeze(-1)      # [B, H]
        g_reg = torch.zeros((batch_size, n, 2), dtype=dtype, device=torch_device)
        opp_sig = torch.zeros((batch_size, nn, n), dtype=dtype, device=torch_device)
        all_term = torch.cat([show_idx, hfold_idx, vfold_idx])
        den_mask = den > 1e-15
        fval = torch.zeros((batch_size, n), dtype=dtype, device=torch_device)
        gadget = {"player": gadget_player, "tval": tval, "den": den, "g_reg": g_reg}

    def _g_follow():
        """Per-hand FOLLOW probability (iig_gadget._rm_plus row semantics: 0.5
        default when no positive regret)."""
        pos = torch.clamp(gadget["g_reg"], min=0.0)
        s = pos.sum(-1)
        return torch.where(s > 1e-15, pos[..., 0] / s.clamp_min(1e-300),
                           torch.full_like(s, 0.5))

    leaf_trees = trees
    valid_list = [valid_np[b] for b in range(batch_size)]

    for _iter in range(int(n_iterations)):
        hr_at.zero_()
        vr_at.zero_()
        hvals.zero_()
        vvals.zero_()
        if gadget is not None:
            gf = _g_follow()
            opp_sig.zero_()
            opp_sig[:, 0, :] = 1.0
            if gadget_player == 0:
                hr_at[:, 0, :] = hr_init * gf
                vr_at[:, 0, :].copy_(vr_init)
            else:
                hr_at[:, 0, :].copy_(hr_init)
                vr_at[:, 0, :] = vr_init * gf
        else:
            hr_at[:, 0, :].copy_(hr_init)
            vr_at[:, 0, :].copy_(vr_init)
        level_strategies = []

        for group in groups:
            if group is None:
                level_strategies.append(None)
                continue
            parents = group["parents"]
            strategies = _regret_matching_batched(
                regret_sum.index_select(1, parents),
                group["legal_mask"],
            )
            level_strategies.append(strategies)
            edge_parent = group["edge_parent"]
            edge_child = group["edge_child"]
            edge_strategy = strategies[:, group["edge_parent_pos"], group["edge_action"], :]
            hero_edges = group["hero_edges"]
            villain_edges = group["villain_edges"]
            if bool(hero_edges.any()):
                hp = edge_parent[hero_edges]
                hc = edge_child[hero_edges]
                hr_at[:, hc, :] = hr_at[:, hp, :] * edge_strategy[:, hero_edges, :]
                vr_at[:, hc, :] = vr_at[:, hp, :]
                if gadget is not None:
                    es = edge_strategy[:, hero_edges, :] if gadget_player == 0 else 1.0
                    opp_sig[:, hc, :] = opp_sig[:, hp, :] * es
            if bool(villain_edges.any()):
                vp = edge_parent[villain_edges]
                vc = edge_child[villain_edges]
                hr_at[:, vc, :] = hr_at[:, vp, :]
                vr_at[:, vc, :] = vr_at[:, vp, :] * edge_strategy[:, villain_edges, :]
                if gadget is not None:
                    es = edge_strategy[:, villain_edges, :] if gadget_player == 1 else 1.0
                    opp_sig[:, vc, :] = opp_sig[:, vp, :] * es

        if show_idx.numel() > 0:
            VR_s = vr_at.index_select(1, show_idx)
            HR_s = hr_at.index_select(1, show_idx)
            if has_matrices:
                default_h = (hw_s * torch.bmm(VR_s, win_tT)
                             + hl_s * torch.bmm(VR_s, lose_tT)
                             + ht_s * torch.bmm(VR_s, tie_tT))
                default_v = (vw_s * torch.bmm(HR_s, lose_t)
                             + vl_s * torch.bmm(HR_s, win_t)
                             + vt_s * torch.bmm(HR_s, tie_t))
            else:
                default_h = torch.zeros(
                    (batch_size, show_idx.numel(), n), dtype=dtype, device=torch_device)
                default_v = torch.zeros_like(default_h)
            if leaf_hook is not None:
                ctx = _leaf_ctx(
                    _iter, leaf_specs, leaf_trees, show_idx_np,
                    HR_s.cpu().numpy().astype(np.float64),
                    VR_s.cpu().numpy().astype(np.float64),
                    default_h.cpu().numpy().astype(np.float64),
                    default_v.cpu().numpy().astype(np.float64),
                    valid_list, pot_np, hs_np, vs_np)
                default_h, default_v = _apply_leaf(
                    leaf_hook, ctx, (batch_size, int(show_idx.numel()), n),
                    dtype, torch_device)
            hvals[:, show_idx, :] = default_h
            vvals[:, show_idx, :] = default_v

        if hfold_idx.numel() > 0:
            VR_hf = vr_at.index_select(1, hfold_idx)
            HR_hf = hr_at.index_select(1, hfold_idx)
            hvals[:, hfold_idx, :] = hf_hero_coeff * torch.bmm(VR_hf, valid_tT)
            vvals[:, hfold_idx, :] = hf_vill_coeff * torch.bmm(HR_hf, valid_t)

        if vfold_idx.numel() > 0:
            VR_vf = vr_at.index_select(1, vfold_idx)
            HR_vf = hr_at.index_select(1, vfold_idx)
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
            parent_hvals = torch.zeros(
                (batch_size, parents.numel(), n), dtype=dtype, device=torch_device)
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
                    regret_flat[:, flat_idx, :] + instant, min=0.0)
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
                    regret_flat[:, flat_idx, :] + instant, min=0.0)
                strategy_flat[:, flat_idx, :] = (
                    strategy_flat[:, flat_idx, :]
                    + vr_at[:, vp, :] * edge_strategy[:, villain_edges, :]
                )

        if gadget is not None:
            # opponent FOLLOW value per hand: terminal rows of the opponent's
            # values weighted by its PURE strategy reach (range/gf factored out
            # -> true counterfactual; the backward node aggregation is NOT
            # usable here, per turn_river's index-mismatch note)
            opp_vals = vvals if gadget_player == 1 else hvals
            fval_num = (opp_sig[:, all_term, :] * opp_vals[:, all_term, :]).sum(dim=1)
            fval = torch.where(den_mask, fval_num / den.clamp_min(1e-300),
                               torch.zeros_like(fval_num))
            ev_g = gf * fval + (1.0 - gf) * tval
            upd_f = torch.where(den_mask, fval - ev_g, torch.zeros_like(fval))
            upd_t = torch.where(den_mask, tval - ev_g, torch.zeros_like(fval))
            g_reg[..., 0] = torch.clamp(g_reg[..., 0] + upd_f, min=0.0)
            g_reg[..., 1] = torch.clamp(g_reg[..., 1] + upd_t, min=0.0)

    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)

    # ----- average strategy + batched root value pass -----
    denom = strategy_sum.sum(dim=2, keepdim=True)
    avg_t = torch.where(denom > 1e-12, strategy_sum / denom.clamp_min(1e-300),
                        torch.zeros_like(strategy_sum))

    v0 = v1 = None
    if compute_value_pass:
        v0, v1 = _root_value_pass(
            groups, avg_t, hr_init, vr_init,
            show_idx, hfold_idx, vfold_idx, show_idx_np,
            win_tT if has_matrices else None,
            lose_tT if has_matrices else None,
            tie_tT if has_matrices else None,
            win_t if has_matrices else None,
            lose_t if has_matrices else None,
            tie_t if has_matrices else None,
            valid_t, valid_tT,
            hw_s, hl_s, ht_s, vw_s, vl_s, vt_s,
            hf_hero_coeff, hf_vill_coeff, vf_hero_coeff, vf_vill_coeff,
            batch_size, nn, n_actions, n, dtype, torch_device,
            leaf_hook, leaf_specs, leaf_trees, valid_list, pot_np, hs_np, vs_np,
        )
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    gadget_out = None
    if gadget is not None:
        gadget_out = {
            "player": gadget_player,
            "follow_final": _g_follow().cpu().numpy(),
            "fval_final": fval.cpu().numpy(),
            "tval": tval.cpu().numpy(),
            "den": den.cpu().numpy(),
        }

    return {
        "regret_sum": regret_sum.cpu().numpy(),
        "strategy_sum": strategy_sum.cpu().numpy(),
        "avg_strategy": avg_t.cpu().numpy(),
        "v0": v0,
        "v1": v1,
        "gadget": gadget_out,
    }


def _root_value_pass(
    groups, avg_t, hr_init, vr_init,
    show_idx, hfold_idx, vfold_idx, show_idx_np,
    win_tT, lose_tT, tie_tT, win_t, lose_t, tie_t, valid_t, valid_tT,
    hw_s, hl_s, ht_s, vw_s, vl_s, vt_s,
    hf_hero_coeff, hf_vill_coeff, vf_hero_coeff, vf_vill_coeff,
    batch_size, nn, n_actions, n, dtype, torch_device,
    leaf_hook, leaf_specs, leaf_trees, valid_list, pot_np, hs_np, vs_np,
):
    """Batched twin of ``turn_river.subgame_value_pass``: exact per-hand
    counterfactual values (v0, v1) at the root under the FIXED average strategy.

    Own-range-factored strategy reaches (hstrat/vstrat start at ones), terminal
    values from the verified coefficients (or the leaf hook at river-deal
    cuts), then a reach-weighted sum over TERMINALS -- not a backward
    aggregation (turn_river's index-mismatch note).
    """
    hstrat = torch.zeros((batch_size, nn, n), dtype=dtype, device=torch_device)
    vstrat = torch.zeros_like(hstrat)
    hstrat[:, 0, :] = 1.0
    vstrat[:, 0, :] = 1.0
    avg_flat = avg_t.reshape(batch_size, nn * n_actions, n)

    for group in groups:
        if group is None:
            continue
        edge_parent = group["edge_parent"]
        edge_child = group["edge_child"]
        flat_idx = edge_parent * n_actions + group["edge_action"]
        edge_strategy = avg_flat[:, flat_idx, :]
        hero_edges = group["hero_edges"]
        villain_edges = group["villain_edges"]
        if bool(hero_edges.any()):
            hp = edge_parent[hero_edges]
            hc = edge_child[hero_edges]
            hstrat[:, hc, :] = hstrat[:, hp, :] * edge_strategy[:, hero_edges, :]
            vstrat[:, hc, :] = vstrat[:, hp, :]
        if bool(villain_edges.any()):
            vp = edge_parent[villain_edges]
            vc = edge_child[villain_edges]
            hstrat[:, vc, :] = hstrat[:, vp, :]
            vstrat[:, vc, :] = vstrat[:, vp, :] * edge_strategy[:, villain_edges, :]

    v0 = torch.zeros((batch_size, n), dtype=dtype, device=torch_device)
    v1 = torch.zeros_like(v0)

    if show_idx.numel() > 0:
        hs_show = hstrat.index_select(1, show_idx)
        vs_show = vstrat.index_select(1, show_idx)
        hfull = hs_show * hr_init.unsqueeze(1)
        vfull = vs_show * vr_init.unsqueeze(1)
        if win_tT is not None:
            hv = (hw_s * torch.bmm(vfull, win_tT)
                  + hl_s * torch.bmm(vfull, lose_tT)
                  + ht_s * torch.bmm(vfull, tie_tT))
            vv = (vw_s * torch.bmm(hfull, lose_t)
                  + vl_s * torch.bmm(hfull, win_t)
                  + vt_s * torch.bmm(hfull, tie_t))
        else:
            hv = torch.zeros((batch_size, show_idx.numel(), n), dtype=dtype, device=torch_device)
            vv = torch.zeros_like(hv)
        if leaf_hook is not None:
            ctx = _leaf_ctx(
                None, leaf_specs, leaf_trees, show_idx_np,
                hfull.cpu().numpy().astype(np.float64),
                vfull.cpu().numpy().astype(np.float64),
                hv.cpu().numpy().astype(np.float64),
                vv.cpu().numpy().astype(np.float64),
                valid_list, pot_np, hs_np, vs_np)
            hv, vv = _apply_leaf(
                leaf_hook, ctx, (batch_size, int(show_idx.numel()), n),
                dtype, torch_device)
        v0 = v0 + (hs_show * hv).sum(dim=1)
        v1 = v1 + (vs_show * vv).sum(dim=1)

    if hfold_idx.numel() > 0:
        hs_hf = hstrat.index_select(1, hfold_idx)
        vs_hf = vstrat.index_select(1, hfold_idx)
        hv = hf_hero_coeff * torch.bmm(vs_hf * vr_init.unsqueeze(1), valid_tT)
        vv = hf_vill_coeff * torch.bmm(hs_hf * hr_init.unsqueeze(1), valid_t)
        v0 = v0 + (hs_hf * hv).sum(dim=1)
        v1 = v1 + (vs_hf * vv).sum(dim=1)

    if vfold_idx.numel() > 0:
        hs_vf = hstrat.index_select(1, vfold_idx)
        vs_vf = vstrat.index_select(1, vfold_idx)
        hv = vf_hero_coeff * torch.bmm(vs_vf * vr_init.unsqueeze(1), valid_tT)
        vv = vf_vill_coeff * torch.bmm(hs_vf * hr_init.unsqueeze(1), valid_t)
        v0 = v0 + (hs_vf * hv).sum(dim=1)
        v1 = v1 + (vs_vf * vv).sum(dim=1)

    return v0, v1


@dataclass
class SubgameSolveResult:
    """Per-spec solve output (local hand index of ``spec.board``)."""

    spec: "sgs.SubgameSpec"
    topology_key: str
    v0: np.ndarray            # [H] float64 per-belief V* target, player 0
    v1: np.ndarray            # [H] float64 per-belief V* target, player 1
    avg_strategy: np.ndarray  # [nn, A, H] float64 average strategy
    strategy_sum: np.ndarray  # [nn, A, H] float64
    regret_sum: np.ndarray    # [nn, A, H] float64
    gadget: dict | None = None

    def v0_global(self) -> np.ndarray:
        return sgs.scatter_global(self.spec.board, self.v0)

    def v1_global(self) -> np.ndarray:
        return sgs.scatter_global(self.spec.board, self.v1)


def solve_population(
    specs,
    n_iterations,
    leaf=None,
    dtype=torch.float64,
    device="cpu",
    b_max=DEFAULT_B_MAX,
    gadget_player=None,
    gadget_opp_cfv=None,
    compute_value_pass=True,
):
    """Solve a population of HUNL street subgames as fused same-topology batches.

    Groups ``specs`` by exact topology bucket (``SubgameSpec.bucket_key``:
    blake2b topology digest + street + local width -- members may differ in
    board, pot, stacks and ranges), flushes each bucket in chunks of at most
    ``b_max`` elements (memory-derived; chunking is exactness-preserving, gate
    P-B), runs the float64 batched CFR+ kernel per chunk, and extracts the
    per-belief root values v0/v1 with the batched average-strategy value pass.

    ``leaf``: optional leaf hook (``terminal_eval`` contract) -- REQUIRED for
    turn specs (their showdown terminals are river-deal cut leaves).
    ``gadget_player``/``gadget_opp_cfv``: optional opponent FOLLOW/TERMINATE
    gadget root; ``gadget_opp_cfv`` is one [H_local] array per spec (uniform
    semantics per call, so same-topology batching is preserved).

    Returns a list of SubgameSolveResult in the input order of ``specs``.
    """
    specs = list(specs)
    if gadget_player is not None:
        if gadget_opp_cfv is None or len(gadget_opp_cfv) != len(specs):
            raise ValueError("gadget_opp_cfv must have one [H] array per spec")
    if b_max is not None and int(b_max) < 1:
        raise ValueError(f"b_max must be >= 1, got {b_max!r}")

    buckets = {}
    for index, spec in enumerate(specs):
        if spec.street == "turn" and leaf is None:
            raise ValueError(
                "turn specs need a leaf hook (river-deal cut leaves); see "
                "terminal_eval.make_exact_river_leaf")
        buckets.setdefault(spec.bucket_key(), []).append(index)

    results = [None] * len(specs)
    for bucket_key, indices in buckets.items():
        topo_key = bucket_key[0]
        chunk_size = len(indices) if b_max is None else int(b_max)
        for start in range(0, len(indices), chunk_size):
            chunk = indices[start:start + chunk_size]
            chunk_specs = [specs[i] for i in chunk]
            trees = [spec.tree() for spec in chunk_specs]
            n = chunk_specs[0].n_hands
            matrices = [tev.spec_terminal_matrices(spec) for spec in chunk_specs]
            has_wlt = matrices[0]["win"] is not None
            ranges = [spec.local_ranges() for spec in chunk_specs]
            out = solve_cfr_levelsync_batched(
                trees,
                n,
                [m["win"] for m in matrices] if has_wlt else None,
                [m["lose"] for m in matrices] if has_wlt else None,
                [m["tie"] for m in matrices] if has_wlt else None,
                [m["valid"] for m in matrices],
                [spec.pot for spec in chunk_specs],
                [spec.stack0 for spec in chunk_specs],
                [spec.stack1 for spec in chunk_specs],
                n_iterations=n_iterations,
                hero_ranges=[r0 for r0, _ in ranges],
                villain_ranges=[r1 for _, r1 in ranges],
                dtype=dtype,
                device=device,
                leaf_hook=leaf,
                leaf_specs=chunk_specs,
                gadget_player=gadget_player,
                gadget_opp_cfv=(
                    [gadget_opp_cfv[i] for i in chunk]
                    if gadget_player is not None else None),
                compute_value_pass=compute_value_pass,
            )
            for pos, spec_index in enumerate(chunk):
                gadget_out = None
                if out["gadget"] is not None:
                    gadget_out = {
                        "player": out["gadget"]["player"],
                        "follow_final": out["gadget"]["follow_final"][pos],
                        "fval_final": out["gadget"]["fval_final"][pos],
                        "tval": out["gadget"]["tval"][pos],
                        "den": out["gadget"]["den"][pos],
                    }
                results[spec_index] = SubgameSolveResult(
                    spec=chunk_specs[pos],
                    topology_key=topo_key,
                    v0=out["v0"][pos] if out["v0"] is not None else None,
                    v1=out["v1"][pos] if out["v1"] is not None else None,
                    avg_strategy=out["avg_strategy"][pos],
                    strategy_sum=out["strategy_sum"][pos],
                    regret_sum=out["regret_sum"][pos],
                    gadget=gadget_out,
                )
    return results
