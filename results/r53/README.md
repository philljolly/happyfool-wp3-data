# R53 (run 53, 2026-10-23): the answer-identity confound in paper 1, on existing rows (EXPLORATORY, post hoc)

Prompted by the paper-1 review (`notes/reviews/type2-injected-review-2026-10-23.md`, point 1). No new inference; Qwen3-8B rows from Am. 41, 41b and 43 (L9 α8). Scripts: `r53_answer.py` → `answer_identity.json`; `r53_swap.py` → `swap_none.json`; `r53_diff.py` → `diff.json`. Concept bootstrap 2000 (seeds 53, 531, 532).

## R53a: ON confidence beyond answer identity (the reviewer's test, extended)

Intro frame: the model says "sentence 2" 85–93% of the time; choosing sentence 1 is right 100% (Am. 41, 43c, 43r) or 89% (41b). Answer-only AUROC .565–.651.

| Cell | S2: within "sentence 2" stratum, conf resid. on [1,|m|,|m|²] | LOCO CV Δ AUROC (base → +conf) | Δ log-loss |
|---|---|---|---|
| Am. 41 concept, intro | .569 [.486, .644] | −.010 | +.005 |
| Am. 41b concept, intro | .506 [.430, .581] | −.010 | +.008 |
| Am. 43 concept, intro | .534 [.463, .614] | −.002 | +.003 |
| Am. 43 random, intro | .526 [.447, .610] | −.001 | +.003 |
| ordinary frame (4 cells) | .36–.52 | −.022 to +.026 | +.003 to +.009 |

Base = [1, a2, |m|, |m|², a2·|m|] (held-out AUROC already .77–.89 intro). Pooled S2 over the three independent concept cells: intro .534 [.488, .580]; ordinary .454 [.371, .547].
**Reading:** once the chosen answer and a flexible margin are in the baseline, the ON confidence adds no held-out information. The T2 claim "beyond first-order evidence" is withdrawn (BOUNDED: pooled ≤ .58).

## R53b: swap and NONE under the same control

- **SWAP-A** (own-correct coefficient of conf_own − conf_swap on [1, |m|, |m|², a2, a2·|m|, correct]): positive with CI above 0 in all six cells (41b, 43c, 43r × intro, ordinary): .20 [.02, .41] … .44 [.26, .63]. **Survives** (agrees with the reviewer's own extension).
- **NONE confidence** (same transcript and answer, no state) *adds* held-out information (Δ log-loss −.019 to −.068) with a **negative** coefficient. With ON and NONE both in, the coefficients are near-equal and opposite (e.g. Am. 43 concept intro: NONE −2.94, ON +2.94).

## R53c: the state-dependent part, d = conf_own − conf_none_own

| Cell | LOCO Δ AUROC | Δ log-loss | S2 AUROC of d [CI] |
|---|---|---|---|
| Am. 41b concept intro | +.065 | −.066 | .671 [.595, .745] |
| Am. 43 concept intro | +.086 | −.105 | .688 [.628, .778] |
| Am. 43 random intro | +.085 | −.083 | .732 [.668, .811] |
| Am. 41b / 43c / 43r ordinary | +.198 / +.202 / +.202 | −.082 / −.086 / −.059 | .815 / .749 / .774 |

**Reading (post hoc, forking: not citable as a finding).** The raw report mixes a transcript-driven component that is *anti*-correlated with correctness and a state-driven one that is positively correlated; beyond answer and margin they cancel. Subtracting the same-transcript no-state confidence isolates the state-driven part, which carries correctness information beyond the answer and margin. Rival not separable with these data: RE-READOUT (the confidence query is a second localisation readout of the still-present state). Confirmatory test pre-registered as **Amendment 44** (`prereg.md`), new concepts.
