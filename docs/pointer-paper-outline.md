# Pointer paper: outline and H38 audit (started run 43, 2026-10-21)

Working title: **"Accuracy is not answering: introspection training at Qwen3-4B builds a question-insensitive pointer"**.
Status: outline only. No branch and no PR yet. MR35 says no new cells until this is drafted. The one exception is Am. 36b (the larger control forced by Am. 36a's CONTROL failure), which is running.

## Claim (precise, with cap)
In one paradigm (2AFC: which of two sentences carried an injected concept vector; Qwen3-4B; L24 α2), IFT-style LoRA training (Hahami, Sinha & Jain 2026 style; T) raises held-out report accuracy from chance to 1.00. The trained report also picks the perturbed sentence when asked the reversed question ("which is completely ordinary?"), and when the injection is a matched-norm random direction. So the trained channel is a **content-free, question-insensitive pointer to the perturbation**, and its accuracy cannot be read as self-report about the injected content. Cap (H39.3): this is about a trained report channel. It says nothing about consciousness, and "self-access" is not earned.

## Sections
1. **Introduction.** Trained introspection literature (IFT; Steering Awareness; Rivera 2025) scores accuracy in a single frame. Our point is that a single-frame accuracy is consistent with a pointer. Nearest relative (R1): H36-style frame robustness checks in that literature (verify what each paper actually reports before claiming priority; H38 lesson from the 'Whose state?' rejection).
2. **Measures.** Report accuracy R; frame spread (H36); **question axis Q = P_inj(intro) − P_inj(reversed wording)** (H41, holiday 34), labelled by literal wording; content axis C (concept vs matched random). Pointer signature: R ≈ 1 with Q ≈ 0 and C ≈ 0.
3. **Untrained baseline.** Am. 32: no setting where report reaches probe (probe .87–1.00). Am. 35/35b: at L24 α2 the injected identity is present at the answer position (probe .93–.98) and orthogonal to the readout, cos(u, m) ≈ 0 in five frames (GAP). The sign-flipped L20 settings are reported as a contrast, not a target (H39.1).
4. **Training (Am. 36).** T and a shuffled-label control S, plus the bf16 K3 failure and its diagnosis (logit quantisation), stated in full.
5. **Results (Am. 36a and Am. 36b, fp32).** GATE in both. CONTROL history: mislabelled at 30 concepts, OK at 90. The pointer replicates (ordinary 0/60 and 0/180; random arm 100% in 36a). Q table. Mechanism, as registered: **NOT SEPARATED** (the displacement is moved, cos .28, and amplified ×3; the U-reader partly transfers, .83). Write "moved and amplified, not opened". Do not write "re-routed": Am. 36a's descriptive REROUTE did not replicate as a slot. S is a no-information control, not a no-change control (−.17).
6. **What would change it.** Training with mixed polarity (the reversed question in training) is the named next test. A ≥ 8B replication. A second architecture.
7. **Limitations.** One model, one setting, one reversed wording, 30 (or 90) concepts, one seed per adapter, and adapters trained in bf16.

Am. 36b numbers (CONTROL OK; R_U .64; S −.17; T ordinary .00) come from `results/am36/audit36b_L24a2.json` and `q_axis36b_L24a2.json`: checked at writing (run 43).

## H38 audit (every number to be cited → source file → checked)
| Number | Claim | Source | Checked (run 43) |
|---|---|---|---|
| R_U .45 [.33, .55], R_T 1.00, dR_T +.55 [.45, .67] | Am. 36a gate | `results/am36/audit36f_L24a2.json` arms U_intro, T_intro | yes (dR_T .55 [.45, .67] re-read) |
| T ordinary R .00 | pointer | same, T_ordinary | yes |
| U ordinary R .43 [.33, .53], S .43 | baseline | same | yes |
| dR_S +.03 [−.12, .18] | CONTROL mislabelled | same, S_intro dR_vs_U | yes |
| Q: U −.12 [−.27, .03]; T .00; S −.08 [−.23, .05]; T−U +.12 [−.03, .27] | question axis | `results/am36/q_axis_L24a2.json` | yes (computed run 43) |
| T random arm 100% (intro, ordinary) | trained-in content-insensitivity | `q_axis_L24a2.json` random_T P_intro/P_ordinary 1.00 | yes; the draft cites intro and ordinary only (neutral not cited) |
| Am. 32 L20 α1 intro .12 [.03, .23] / neutral .72; L20 α4 .23 / .70; L24 α2 .58 [.45, .72] / .57; probe .97–.98 | untrained baseline | `results/am32/analysis_sw_*.json` | yes |
| **Am. 32 L24 α4 intro .80 [.72, .88]** | the untrained report is *above* chance at L24 α4 | `results/am32/analysis_sw_L24a4.json` | yes. **Must be stated:** the floor is setting-specific; "no setting where report reaches probe" is the registered k1 claim, not "report at chance everywhere" |
| Am. 35 L24 α2 cos(u, m) intro +.02 [−.03, .06], neutral −.03, word +.04 | GAP | `results/am35/geom_fx2_concept_L24a2.json` | yes (intro, neutral) |
| Am. 35b GAP in 5 frames, cos(u, m) −.04 to .14 | GAP replicates | `results/am35/geom_g5_L24a2.json` | yes (ordinary .14 [.06, .18]: its CI excludes 0 but is small; state it) |
| layer-30 cos(m_U, m_T) .29 [.21, .44], norm ×2.5, U-probe on T .97, refit +.10 | REROUTE descriptives | `audit36f_L24a2.json` T_intro | yes (X_mid .97 [.90, 1.00]; refit P_mid 1.00, dP_mid +.10 [.03, .18]; final cos(u, m) .64 [.62, .66] vs U .02) |
| bf16 K3 U .80–.82, S .65–.68 | precision failure | `results/am36/audit36_L24a2.json` | yes (U .80–.82, S .65–.68, T 1.00) |
| Priority/relative claims about IFT, Steering Awareness, Rivera, Macar | intro | `notes/wp3/priority-check-run44.md` | **yes (run 44):** 10 of 12 works read in arXiv full text, quotes verbatim. Correction applied to the draft: Steering Awareness DOES test trained models with random vectors at test (labelled negatives in training); LawZero (2604.17465) is the nearest relative in logic; T's random-arm result is trained in (random = positive in training), so the finding is Q, not C |

Next: finish the "to do" rows, add Am. 36b, then draft `papers/pointer/` on a `paper/pointer` branch.


**Run 44:** draft §1–8 on branch `paper/pointer` (`papers/pointer/paper.md`); numbers re-read from `audit36f_L24a2.json`, `audit36b_L24a2.json`, `q_axis_L24a2.json`, `q_axis36b_L24a2.json`. Open: §5.x (Am. 37), abstract + metadata, supplement, Lindsey/Pfungst refs, a full read-through against this table before the PR.
