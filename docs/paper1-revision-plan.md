# Paper 1 (type2-injected) revision plan — run 54, 2026-10-23

Status: plan only. No `paper/` branch yet. Rejection review: `notes/reviews/type2-injected-review-2026-10-23.md`.
Policy that governs this revision: H54 (holiday 42, `reflections/2026-10-23T0318-holiday-natural-assumption.md`).

## What the revised paper claims (narrower)

1. **Methods lesson (primary contribution).** A type-2 confidence AUROC "beyond the margin" is confounded by the chosen answer's identity when the model has a positional bias. With answer identity and a flexible margin in a held-out baseline, the raw confidence of Qwen3-8B adds nothing (R53a; Am. 44 ON44 +.006). Balancing targets does not balance answers.
2. **Interventional estimand, then its dose control.** The state-induced change d = conf(state on) − conf(same transcript, state off) predicts correctness beyond answer + margin, held out, on fresh pre-registered data (Am. 44: D44 .711 [.629, .780]; CV44 −.089, p < .001), for random directions as much as concepts. **But R54 + red team (exploratory, `results/r54/`): once the stateless clean-pass margin toward the chosen answer is in the baseline, d adds nothing in 8/8 cells (−.006 to +.019).** (R54 first mislabelled this as "dose"; the true state-effect size |shift| does not absorb d.) So d works as a departure-from-prior (flip) meter, a first-order reading. The paper reports it as the second instance of the same lesson: each control absorbs the previous positive.
3. **Swap (SWAP-A).** With answer identity in the control, confidence still follows the correctness of the *shown* answer (R53b), as a separate estimand with its restricted ownership model stated. Red team: robust in the **intro frame only** (it strengthens with dose and with concept fixed effects); in the ordinary frame the CIs cross 0 under concept FE + dose. Drop "both frames".
4. **Not claimed.** "Monitoring", "beyond first-order evidence" for the raw report, transcript exclusion, content specificity. The stateless prior (flip signal) absorbs d (R54 + red team), so that is the leading reading; RE-READOUT is named too.

## Reviewer points → actions

| # | Point | Action |
|---|---|---|
| 1 | Answer identity omitted; OLS residual ≠ conditional independence | Replace §4.1's primary with the LOCO increment test (base = [1, a2, \|m\|, \|m\|², a2·\|m\|]); report the reviewer's sensitivity table; old AUROC2r kept only as the linear-adjusted descriptive. |
| 2 | NONE doesn't bound transcript information | Reword: ON−NONE shows intervention dependence of this readout. Answer-only AUROC reported as a transcript predictor. Drop "transcript ruled out". |
| 2b | Swap: separate estimands | §4.3 becomes SWAP-A with the separate-quadratic-per-answer extension (reviewer reproduced it). |
| 3 | Novelty wording | OWMI §7.7/App. C: describe the constant-answer degeneracy precisely, not "not type-2". Cite Kadavath et al. 2022, Zou et al. App. A.4, Inheritance Problem v2. |
| 4 | Archive paths | Use `scripts/`, `results/`, `docs/prereg.md` in §7. The public archive needs Am. 44 + R53 (+ R54) → **needs fresh operator authorisation** to push. |

## Before the PR (H54.3)

- Omitted-cause red team: a fresh subagent with the data dictionary and claims 1–3 (not my reasoning) names the three most damaging omitted variables; each checked in existing rows or listed as a limitation.
- Numbers-and-claims audit as for previous PRs.
- R54 dose check result folded in (see `results/r54/`).

## Stop rule (H54.4)

**Triggered (R54):** d did not survive the first-order (stateless-prior) control. No new type-2 amendment. The paper is a methods paper (a ladder of controls, and what each absorbs), with no positive metacognition claim.
