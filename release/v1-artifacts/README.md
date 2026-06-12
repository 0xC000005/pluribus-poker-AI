# v1-artifacts - reproducibility bundle for "One GEMM, Many Subgames"

This directory contains the machine-readable artifacts cited by the manuscript. The
working outputs under `autoresearch-session/` are local run products; these tracked
copies are the stable reproducibility bundle that accompanies the paper.

Copies were verified byte-identical to the working originals on 2026-06-10 (`cmp` over
all 36 files). The quiet-host G5 re-run files, `cuda_graphs_ablation.json`, and the G4
tool-symmetric end-to-end artifacts were added after their corresponding runs completed.

Field naming convention: in `e2e_band_summary_g{4,5,5q}.json` the
`reach_{fused,sequential}_s_median` fields are **time-to-band medians at tau = the pooled
band ceiling** (manuscript §4.2), *not* protocol totals; names retained for artifact
stability. The `b0_cpucheck_g4_*.json` determinism-gate runs use the reduced 4-rounds x
10-beliefs budget on `--device cpu` (manuscript §3.4, §3.3 2x2 caveats).
Several JSON files use the historical `target_type` value
`per_belief_resolved_Vstar`; the manuscript interprets those targets as finite-budget
CFR+ continuation values `V_T`, not exact optimal values `V*`.

## Manifest

| file | role (manuscript section) |
|---|---|
| `costpersolve_bakeoff.json` | §4.1 microbenchmark rows + G4/G5 keysweeps + parity; regime-probe m-values |
| `e2e_band_summary_g4.json` | §4.2 aggregates: 11.79x / 9.99x / 98.4% / 10.09x / bands / tau-sweep |
| `b0_e2e_g4_{fused,sequential}_seed{0..4}.json` | §4.2 per-seed G4 paired arms (total-wall medians 76.8 / 767.2 s) |
| `searchfree_nfsp_g4_seed{0,1}.json` | §4.3 NFSP boundary (1.3968/1.3966; episodes; removal criterion) |
| `b0_perbelief_g5_curve.json`, `b0_perbelief_g5_curve_seed{1..4}.json` | §4.4 G5 5-seed band (full 8-round protocol) |
| `e2e_band_summary_g5q.json` | §4.4 G5 paired-arm aggregates, quiet-host re-run (HEADLINE: inner 2.40x / total-wall 2.00x vs 2.01x ceiling / fraction 86.1% / tau-sweep flat 2.39x) |
| `b0_e2e_g5q_{fused,sequential}_seed{0,1,2}.json` | §4.4 per-seed G5 quiet-host re-run (3 matched seeds, shortened 4-round protocol, --skip-gadget, machine idle; per-seed inner ratios 2.373/2.400/2.398) |
| `e2e_band_summary_g5.json` | §4.4 G5 paired-arm aggregates, original pair (SUPERSEDED — contaminated; retained as the §4.4 diagnostic: 2.20x > 2.04x ceiling violation) |
| `b0_e2e_g5_{fused,sequential}_seed{0,1}.json` | §4.4 per-seed G5 original pair (SUPERSEDED): clean seed 0, contaminated seed 1 — the §4.4 contamination-disclosure evidence |
| `b0_cpucheck_g4_{fused,sequential}.json` | §3.4 CPU determinism gate; §3.3 per-belief CPU/GPU 2x2 cells |
| `hunl_topology_census_probe.json` | §5 HUNL census + Stage-2 GPU probe (forward pointer; H=1081) |
| `cuda_graphs_ablation.json` | §6.5 six-arm launch-amortization ablation: original per-key arms (`rows`/`keysweeps`/`parity`/`derived`) + the `amortized_fused` completion (six-arm cross-game 7.96x/12.06x/1.82x; G4 amortized sweep speedup~=N; G5 saturation slopes 129.6/68.0 ms/tree; fused-granularity parity gates incl. the floor-limited G5 gate; setup walls; drift anchors 0.969–0.998) |
| `e2e_amortized_g4_compile_summary.json` | §6.5 G4 tool-symmetric end-to-end aggregate: compile-vs-compile pooled-ceiling descriptive check (inner 11.36x / total-wall 7.38x / time-to-band 9.99x / Amdahl ceiling 7.39x) |
| `e2e_amortized_g4_{fused,sequential}_compile_seed{0..4}.json` | §6.5 per-seed G4 tool-symmetric paired arms (5 matched seeds, --skip-gadget, fused-compile vs sequential-compile) |
| `supplementary_measurement_notes.md` | source notes for secondary, non-claim-bearing measurements that are either cited with explicit caveats or reserved for standalone machine-readable reporting. Renamed 2026-06-10 from `ed_writeup_evidence_pack.md` (content byte-identical; same SHA-256). The manuscript supersedes any older prose in this note; the per-file numbers, not the note's historical wording, are authoritative when cited. The notes do not contain the G5 topology census; see the derivation note below. |

Filename prefixes are historical run identifiers: `b0_` = the per-belief-target
experiment series; `g5q` = the quiet-host Goofspiel-5 re-run. In
`hunl_topology_census_probe.json`, the probe's `B` field counts co-solved trees — the
manuscript's across-tree ($K$-like) axis for the probe — while the within-tree range
dimension is the 1,081-hand river set (`H=1081`).

## Derivation note: G5 topology census (manuscript §3.4, §7)

The "26,773 nodes / 10 level groups" census is re-derived directly from the compiled
topology (verified 2026-06-10 on CPU; no artifact JSON needed — one command):

```bash
CUDA_VISIBLE_DEVICES="" python - <<'EOF'
from poker_ai.rebel.iig_solve import DepthLimitedGame
from poker_ai.rebel.iig_pbs import load_goofspiel, goofspiel_is_cut, goofspiel_public_key
from poker_ai.rebel.iig_batched import _compile_topology
import numpy as np
dlg = DepthLimitedGame(load_goofspiel(5), goofspiel_is_cut, public_key_fn=goofspiel_public_key)
keys = sorted({n[1] for n in dlg.cut_nodes})
ranges = {k: (np.ones(dlg.n_priv(k,0))/dlg.n_priv(k,0), np.ones(dlg.n_priv(k,1))/dlg.n_priv(k,1)) for k in keys}
c = _compile_topology(dlg, ranges, "cpu")
levels = sorted({r["level"] for r in c.recs})
print({"n_nodes": c.n_nodes, "n_level_groups": len(levels), "max_level_index": max(levels),
       "K_keys": len(keys), "B_cut_nodes": c.B, "n_batched_infosets": c.n_iids})
EOF
# -> {'n_nodes': 26773, 'n_level_groups': 10, 'max_level_index': 9,
#     'K_keys': 15, 'B_cut_nodes': 125, 'n_batched_infosets': 236440}
```

(26,773 topology nodes across 10 level groups, levels 0-9; K=15 keys, B=125 cut
instances, 236,440 batched below-cut infoset rows. The full-game infoset count in the
ladder table, 236,450, is `dlg.n_iset` and includes the 10 above-cut trunk infosets.)

Embedded note-string convention: `e2e_band_summary_g{4,5q}.json` embed a frozen note string
"no systematic bias"; the manuscript's current wording is "no detectable bias (n=5; a
sign test at this n cannot exclude moderate bias)" — the JSON prose is historical,
the per-seed numbers are authoritative.

Hash note (2026-06-10): the `supplementary_measurement_notes.md` hash below was
refreshed after a correction appended the N1-r4 ~6.2 ms/iter note to the notes file; the
previous hash was
`7c1b6b00a1a472a77377e2964a4ed3b649bc7dec9ede50e945671360d2ad067b`.

## SHA-256

```
c245b873f5f861554cd8034634e295481d913d8c439f69ac82f1b1576ce2f03d  b0_cpucheck_g4_fused.json
bdd57b38a9864ba1a60ef90b5e57a1f991455400be1dc1060edad238a548a8e8  b0_cpucheck_g4_sequential.json
f44d90a89a3dda9e117c37ea81f44ce37537506740b9295dee0488bb57732829  b0_e2e_g4_fused_seed0.json
bceab17bafe02653e3ed53c82da69682990b76f790b2c6ea61b21834c0720e34  b0_e2e_g4_fused_seed1.json
f7259456db09ac9a2ef1c4096a5de2c69b8736ee66dbf3fdf9e142af15f025fe  b0_e2e_g4_fused_seed2.json
c15d8dbac28ba6dea9840df9b143ad7f2d78489ce0e0efe9537450477463d3ec  b0_e2e_g4_fused_seed3.json
71f6350061c3a4bef1ad3c59a1b306f20f42b9325fbb993c041be36a9965f312  b0_e2e_g4_fused_seed4.json
b8bf72ba4f63bfe6afa2665e0958f99a04261201ab72bc637de21c385309df61  b0_e2e_g4_sequential_seed0.json
a304869e05bec93e3d793c734bbb402439e992df922c332d9bf3534513f6b942  b0_e2e_g4_sequential_seed1.json
2b0567b114fe6ca92385f771655bcec0a4bd322a2b766a4b0c3e76514003c50a  b0_e2e_g4_sequential_seed2.json
cc9b4d611efdccf47bd8bf0d223d34938b8bb7850c434049798aba2e90ea1e32  b0_e2e_g4_sequential_seed3.json
0ecd09d7fa5fc22ff443a7bf2b1fed129869820e67af35b18887ecc5817cb506  b0_e2e_g4_sequential_seed4.json
558c0a28c1b67fc6dc6cda327a345ab64559b26bf4d50c62b71011481cfc994e  b0_e2e_g5_fused_seed0.json
f5a380d71f39c007bdc74132dcdd04aa315a026386fcc542b9c81f0386046d99  b0_e2e_g5_fused_seed1.json
5262aa28e91120764c9e4d4559d9c9cccb8ed422081d5cd7485a05dcd3c11966  b0_e2e_g5_sequential_seed0.json
b2d2b498b944fad7f7973d2796ea37c5d3ea770a7e5e7be3af0a285d887b1571  b0_e2e_g5_sequential_seed1.json
9be6ec9babbe0151f7b09a866fee13e79090459adc2e7ebbe4bd7898e70f130d  b0_e2e_g5q_fused_seed0.json
1f41cc945248e6e8c009df819cd4c6ee8240b7f96c5bab20ad6437b2b2378e6a  b0_e2e_g5q_fused_seed1.json
7ae77cecb57278eca20476a7c3aa86ca1fdaa854ed715b742f718ffd6ad5fc18  b0_e2e_g5q_fused_seed2.json
ce187fd732fd3970c02d3a57fdd1d8d028e994eaee676994caeeeb0c6b75a616  b0_e2e_g5q_sequential_seed0.json
deb0c04b0e48d7815a9cae342d26947b301d7bf06470dd335a59e5c8f631f897  b0_e2e_g5q_sequential_seed1.json
941323c1f6f87bb0f21bcbe11c3ee03823270d5d449c315edc0a11adfddc7349  b0_e2e_g5q_sequential_seed2.json
5f547cf1aa21f7c0f9948aaa7a9f1529819895e5c32ae4237fb2f75767bd3cd4  b0_perbelief_g5_curve.json
30de0258d706bfbc46ee75d1cadb644783615f2956c8f98ec83ef1781de9b95d  b0_perbelief_g5_curve_seed1.json
19a3e98eaeaf73b89e648fdc48f3aeb790f00ef8a8fff69e4d1819d4cf4d5e2c  b0_perbelief_g5_curve_seed2.json
c8f2b88d651571dde7ee4100680d95ac6183c02c7a55b5731c41eed571dd8ded  b0_perbelief_g5_curve_seed3.json
fb40d245d96f5dcfe7a843bf23c879cd247132d1f522e6d69e9a63335883b5cc  b0_perbelief_g5_curve_seed4.json
64fad78ed755dfd7d7df10594b5e75111de07f25f61c306f29662e2c5476bb1d  costpersolve_bakeoff.json
2fc68188152bf9a5f901946fbdfa5568845959329423f39fa30ebd4ae79b6ec4  cuda_graphs_ablation.json
5ff136f2ff2a01ac48ae6ece7a9b83851aa52d365d3ef5498c33d9a45c1447f4  e2e_amortized_g4_compile_summary.json
9d0e9d70ed19dbfdc90b979e0b4cbbb8d7025bccdc863d594a4e17f03572d04e  e2e_amortized_g4_fused_compile_seed0.json
64b76d726408598ef665ac4b3c371fe9dffb8cef53e94b50e76e217d7658317d  e2e_amortized_g4_fused_compile_seed1.json
46f8f97adba41ac4d33c46861ea178eeb60fc8348b510fed7ebcc5e14617d604  e2e_amortized_g4_fused_compile_seed2.json
b543a82e2b74c9c156b418c7aee36e8a3e314f51494e8ced3d55cada99a82ec5  e2e_amortized_g4_fused_compile_seed3.json
c459727cfc602aa019a552744ada224c047fa3a02dbc2a46fe555464cacddef9  e2e_amortized_g4_fused_compile_seed4.json
e87510437a645b8f089820d9203c324cf678dc50e020bf64868d66e5a29c310c  e2e_amortized_g4_sequential_compile_seed0.json
9e747d46c2e3015690e5d008c89671a55c654de530a2ff341e5b6e3c01b3b239  e2e_amortized_g4_sequential_compile_seed1.json
07da1ffe57240373d26c0331bbe3116b961d9e186a6ddad6f680693f82c86af1  e2e_amortized_g4_sequential_compile_seed2.json
d0e98704cbbec28bdaa45bc5d28606e85c8f910518bff94c4ce0394387ed0558  e2e_amortized_g4_sequential_compile_seed3.json
92c7cad3c0e5c9843759f510658086f71850b00f76c8bb25ca2f924e727a7630  e2e_amortized_g4_sequential_compile_seed4.json
2fa2804f6f6042f1631e6349e59624bb7e1b3d1706270572ff8a8adb13211b79  e2e_band_summary_g4.json
43285dff5c81954ff4746f5cd2cf0a15672fd62e1e712d4a1fe9db9af237473a  e2e_band_summary_g5.json
07e02dea586a7eeccbaa28317a2af918cf01e389a89756008dcb170251cc66bc  e2e_band_summary_g5q.json
9507d1961a0ec89ea260c93e2d0c8c812066eaecb2f0caee1eebd250b258f887  supplementary_measurement_notes.md
f96607054b60de1856515c94e35657d1d6db986d79a19bb25183249eff551e46  hunl_topology_census_probe.json
146b0619c05ea8ede2c5d5b7c6bd00909dcf13df253cc4706ce248a2207e09eb  searchfree_nfsp_g4_seed0.json
ccb7fa85d8c769d58cc1771b38d911f9dbbf14102acdf78c01201830caaa9952  searchfree_nfsp_g4_seed1.json
```
