# R54 (run 54, EXPLORATORY, post hoc; not pre-registered): the stateless prior absorbs the state-check signal

> **Correction (same run, after the H54.3 red team, `redteam/REPORT.md`):** the numbers below are right, but the label "dose" is wrong. With |m| in the base, dose_a = |m| − mc_none spans the same model as base + mc_none, the *stateless* clean-pass margin toward the chosen answer (increments agree to 4 decimals; re-run by me). The true state-effect size |shift| does NOT absorb d (d still adds −.04 to −.12). Correct reading: d's information is how far the chosen answer departs from the model's own unperturbed preference (the flip signal: flips are right 94–100% of the time). That is still first-order, so the stop rule still holds, but it is not a dose meter.

Script `r54_dose.py` (written after Am. 44 was scored; holiday 42 / H54.2). Rows: Am. 44 and Am. 43, Qwen3-8B L9 α8, concept + random arms, both frames (8 cells, n = 120 each). Output `dose.json`.

Rival DOSE: one per-item cause (how hard the state moved the *first-order* answer) drives both d = conf_on − conf_none_own and correctness. Dose proxies use only turn-1 index logits with the state on vs off (`logits_q` vs `none_q`): dose_a = margin shift toward the chosen answer; dose_abs = |margin shift|. Both are target-free. Base = CV44's [1, a2, |m|, |m|², a2·|m|]. LOCO log-loss increments; permutation p (d shuffled within a2 strata, 200 draws, seed 54).

| Cell | corr(d, dose_a) | d over base (CV44 replay) | dose over base | **d over base + dose** (p) |
|---|---|---|---|---|
| Am. 44 concept intro | .80 | −.089 | −.147 | **+.007 (.44)** |
| Am. 44 concept ordinary | .54 | −.074 | −.214 | +.008 (.28) |
| Am. 44 random intro | .76 | −.120 | −.258 | +.005 (.24) |
| Am. 44 random ordinary | .52 | −.040 | −.178 | +.016 (.94) |
| Am. 43 concept intro | .78 | −.105 | −.229 | −.004 (.065) |
| Am. 43 concept ordinary | .55 | −.086 | −.216 | +.014 (.82) |
| Am. 43 random intro | .74 | −.083 | −.190 | −.004 (.045) |
| Am. 43 random ordinary | .51 | −.059 | −.211 | +.008 (.38) |

Adding the target-signed shift as well (uses the answer key; upper-bound control) leaves d at −.007 to +.030.

**Reading.** The state-induced change in confidence carries no held-out correctness information beyond the state-induced change in the first-order answer margin (largest gain −.004 log-loss, against −.04 to −.12 without the dose term). The dose term alone predicts correctness better than d does. So Am. 44's STATE-CHECK SHOWN stands as registered, but its best reading is DOSE: d is a meter of how hard the perturbation pushed the answer, which is first-order. That is not a monitoring or metacognitive reading. H54.4 stop rule triggered: no new type-2 amendment. Paper 1's revision becomes a methods paper: what each control absorbs (answer identity → raw confidence; dose → the state-induced change).

Caveat: post hoc, so it is exploratory and needs the same LOCO + permutation machinery as CV44. The dose proxies need a state-off turn-1 pass, which an experimenter has and the model does not. So "first-order" here means "explained by the state's effect on the answer", not "available to the model".
