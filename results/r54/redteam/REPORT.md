# R54 omitted-cause red team (H54.3), run 54

A fresh subagent was given the data dictionary and three claims, but not my reasoning. Scripts and JSON are in this directory (`rt*.py`, `rt*.json`). I saved this report (the subagent could not write it) and re-ran the key identity myself (below). Increments are LOCO log-loss of control + X minus control; negative means X adds information. p = share of within-answer-stratum permutations at least as good as observed.

## Structural facts
- **One concept = one sentence pair.** The two rows share `sents` and the clean pass and differ only in `target`, so holding out a concept holds out an item.
- **Position bias.** Under intro the clean answer is sentence 2 in 97% of items, and the injected answer in 85–93% of rows.
- **Flips are right.** When the injected answer differs from the clean answer (7–26% of rows), accuracy is .94–1.00, against .54–.67 otherwise.
- **Confidence at ceiling.** P(confident) ≥ .995 everywhere; conf takes ~20 distinct values (bf16); d = 0 in 10–20% of rows.
- No mass failures; answer = argmax everywhere.

## 1 (most damaging). The stateless prior for the chosen answer (conf_none_own; clean margin toward the chosen answer, mc_none)

| Cell | conf over base | none_own over base | conf over base+none_own (p) | d over base | d over base+mc_none (p) |
|---|---|---|---|---|---|
| Am44 concept intro | +.006 | −.046 | −.047 (0) | −.089 | +.014 (.85) |
| Am44 concept ordinary | +.009 | −.039 | −.033 (0) | −.074 | +.008 (.25) |
| Am44 random intro | −.000 | −.045 | −.070 (0) | −.120 | +.004 (.18) |
| Am44 random ordinary | +.002 | −.033 | −.008 (.02) | −.040 | +.014 (.94) |
| Am43 concept intro | +.003 | −.041 | −.066 (0) | −.105 | +.003 (.22) |
| Am43 concept ordinary | +.003 | −.068 | −.018 (.03) | −.086 | +.019 (.80) |
| Am43 random intro | +.003 | −.019 | −.060 (0) | −.083 | −.006 (.06) |
| Am43 random ordinary | +.009 | −.025 | −.021 (.01) | −.059 | +.009 (.54) |

- **Claim 1's null is masking.** conf = none_own + d, and the two parts carry information of opposite sign.
- **R54's "dose" is mislabelled.** Given |m| in the base, dose_a = m − mc_none spans the same model as base + mc_none, which holds no state information. I verified this run: the increments agree to 4 decimals (e.g. .0137 vs .0138; −.0057 vs −.0056).
- **The true state-effect size does not absorb d.** With |shift| (dose_abs) alone in the control, d still adds −.04 to −.12 (p ≤ .01).
- **Reading.** d's information = how far the chosen answer departs from the model's unperturbed preference (the flip signal). That is first-order, but it is not a "dose" meter.

## 2. Item fixed effects
- In same-answer pairs (29–51 per cell), conf_none_own is identical within the pair, so the within-pair difference in d equals the difference in conf.
- After the margin gap, the leftover is about 0: e.g. Am44 concept intro +.04 [−.04, .11]; Am43 random intro +.09 [.01, .18].
- So d's advantage is entirely between items.

## 3. Swap (claim 3)
Coefficient on own-answer correctness in g = conf_own − conf_swap:

| Cell | r53 SWAP-A | + dose | + concept FE + dose |
|---|---|---|---|
| Am44 concept intro | .26 [−.01, .54] | .44 [.16, .71] | .47 [.14, .84] |
| Am44 concept ordinary | .33 [.06, .60] | .34 [−.04, .68] | .32 [−.07, .70] |
| Am44 random intro | .41 [.17, .67] | .61 [.29, .89] | .45 [.09, .68] |
| Am44 random ordinary | .28 [.09, .47] | .30 [.06, .59] | .13 [−.20, .40] |
| Am43 concept intro | .41 [.18, .65] | .63 [.40, .88] | .68 [.47, .97] |
| Am43 concept ordinary | .30 [.07, .51] | .24 [−.18, .63] | .30 [−.09, .73] |
| Am43 random intro | .44 [.26, .63] | .67 [.43, .86] | .67 [.45, .88] |
| Am43 random ordinary | .27 [.04, .49] | .31 [.01, .58] | .24 [−.04, .50] |

- Pooled with correctness and locus both entered: correctness .39–.49 (all intervals exclude 0), locus .05–.19.
- **Verdict.** SWAP-A is robust in intro. In ordinary it does not survive concept fixed effects plus dose. Drop "both frames".

## Power
- A synthetic signal independent of dose, AUROC-matched to d, survives the base + dose control in intro (absorbed in 0–10% of draws). So the intro nulls are informative.
- In the ordinary concept cells it is absorbed in ~50% of draws, so those nulls are weak.
- dose_a is close to a label (AUROC .89–.99).

## Dismissed
- Sentence features (lengths): +.004 to +.019.
- Standardising over all rows, including the held-out concept: a negligible leak.
- No code bug in correctness, m, a2 or d. The bug is in R54's description.
