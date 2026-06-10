# v1-artifacts — claim-ledger artifacts for "One GEMM, Many Subgames" (ED v1 r1)

Tracked copies of every artifact cited in the reproducibility ledger
(`autoresearch-session/rebel/ed_v1_draft/SUBMISSION_CHECKLIST_r1.md`) of the revised
manuscript `autoresearch-session/rebel/ed_v1_draft/00_v1_r1.md`. The working directory
`autoresearch-session/` is gitignored; **these copies are the citable artifacts** and are
the payload for the release/DOI deposit at submission (roadmap item R10).

Copies verified byte-identical to the working originals on 2026-06-10 (`cmp` over all 35
files; the 7 quiet-host G5 re-run files added 2026-06-10 after the re-run landed).
Field-naming note for auditors: in `e2e_band_summary_g{4,5,5q}.json` the
`reach_{fused,sequential}_s_median` fields are **time-to-band medians at tau = the pooled
band ceiling** (manuscript §4.2), *not* protocol totals; names retained for artifact
stability. The `b0_cpucheck_g4_*.json` determinism-gate runs use the reduced 4-rounds x
10-beliefs budget on `--device cpu` (manuscript §3.4, §3.3 2x2 caveats).

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
| `ed_writeup_evidence_pack.md` | documented source of the five numbers pending standalone artifacts (7e-6 parity upper end; RPG/QPG 1.4334; multi-level 0.0506->0.0145; G6 build >280 s; 26,773 nodes / ~12 ms in-loop) — to be re-derived at camera-ready |

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
2fa2804f6f6042f1631e6349e59624bb7e1b3d1706270572ff8a8adb13211b79  e2e_band_summary_g4.json
43285dff5c81954ff4746f5cd2cf0a15672fd62e1e712d4a1fe9db9af237473a  e2e_band_summary_g5.json
07e02dea586a7eeccbaa28317a2af918cf01e389a89756008dcb170251cc66bc  e2e_band_summary_g5q.json
7c1b6b00a1a472a77377e2964a4ed3b649bc7dec9ede50e945671360d2ad067b  ed_writeup_evidence_pack.md
f96607054b60de1856515c94e35657d1d6db986d79a19bb25183249eff551e46  hunl_topology_census_probe.json
146b0619c05ea8ede2c5d5b7c6bd00909dcf13df253cc4706ce248a2207e09eb  searchfree_nfsp_g4_seed0.json
ccb7fa85d8c769d58cc1771b38d911f9dbbf14102acdf78c01201830caaa9952  searchfree_nfsp_g4_seed1.json
```
