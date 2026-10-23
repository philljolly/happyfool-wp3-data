# R52 (run 52): calibration overlap across error sources (EXPLORATORY, post hoc; H52)

Script `r52_calib.py`; output `calib_overlap.json`. Logistic P(correct | raw own-answer confidence logit), one fit per source, 1000-resample trial bootstrap (seed 52). Natural rows = Am. 42 picked task (words, n = 212); injected = 8B intro frame, Am. 41b concept, Am. 43 concept and random (n = 120 each). Not a registered estimand; nothing here changes a slot.

| source | n | acc | mean conf | intercept [CI] | slope [CI] | log-loss, natural map | log-loss, own fit |
|---|---|---|---|---|---|---|---|
| natural (Am. 42) | 212 | .849 | 14.1 | −28.9 [−43.8, −19.9] | 2.23 [1.57, 3.34] | .303 | .303 |
| injected concept (Am. 41b) | 120 | .625 | 7.9 | −6.1 [−11.6, −2.5] | 0.84 [0.40, 1.53] | 6.69 | .611 |
| injected concept (Am. 43) | 120 | .650 | 8.0 | −8.1 [−13.9, −4.3] | 1.11 [0.63, 1.84] | 6.73 | .563 |
| injected random (Am. 43) | 120 | .575 | 7.9 | −8.9 [−15.7, −4.4] | 1.17 [0.60, 2.04] | 6.13 | .608 |

**Reading.**
- **No transfer of the absolute mapping.** Confidence sits about 6 logits lower on the injected task. A mapping fitted on natural errors predicts near-zero accuracy there, and its log-loss is about 10× the refit. So under H52 the natural and injected signals are **not "the same detector" in the calibration sense**. They are comparable in discrimination only (AUROC2r ≈ .65 vs .67).
- **Caveat.** The tasks differ in prompt and question (sentence localisation vs a word-count judgment). A task-level confidence offset would produce this failure even with one shared mechanism. So this is a failed transfer of the mapping, not evidence of two mechanisms.
- **Slope.** The natural slope (2.2) is steeper than the injected slopes (0.8–1.2). CIs touch for Am. 41b and overlap for Am. 43. Not separable at these n.
- **Concept and random share one mapping** (intercepts and slopes overlap closely), consistent with Am. 43's perturbation-generic reading.
- **For paper 1:** report as exploratory. "Comparable discrimination; the absolute confidence-to-accuracy mapping does not transfer across tasks; mechanism identity untested."

## R52b (run 52, after MR44; EXPLORATORY): level check and standardised transfer

**E1 (scoring bug) closed.** `type2_41.py`, `type2_41b.py` and `type2_42.py` all compute conf = logit(B) − logit(A) on the same A/B token ids (`tok("A")`, `tok("B")`, first id), with the same confidence question.

**Level vs ordering (existing rows).** Mean confidence logit:

| source | ON | NONE (same transcript, no injection) | clean |
|---|---|---|---|
| Am. 41b / 43 concept / 43 random, intro | 7.9–8.0 | 7.4–7.5 | 7.5 |
| the same, ordinary frame | 10.7 | 10.7 | 10.8 |
| natural (Am. 42) | 14.1 | – | – |

- **The perturbation does not lower mean confidence** (if anything, ON is ~0.5 higher). This is unlike OWMI's level effect: in Qwen2.5-7B the model is "markedly less confident when it has been intervened upon". (The .62 vs .86 means I quoted earlier came from a reader summary and are not in the saved text. Not cited.)
- The level moves ~3.3 logits with the turn-1 question wording (intro vs ordinary), and to 14 on the natural task. So the level is a task/frame prior. The injected state acts on the *ordering* within a level.

**Standardised transfer** (confidence z-scored within each source; `calib_overlap.json` key `z`):
- Natural slope_z 1.83, vs injected 0.73 (41b), 1.02 (43 concept) and 0.86 (43 random).
- The natural map, applied to injected z, predicts accuracy ≈ .85 (the natural base rate). Its log-loss (.78–1.00) is worse than a constant (.65–.68).
- So even level-free, the mapping does not transfer. The base rate differs, and the standardised slope is about half.
- Only rank-order discrimination (AUROC) is comparable.
- Reading: "comparable discrimination, non-transferable calibration". Mechanism identity is untested either way. H52's margin should be stated on a rank or standardised scale, never on raw logits.

## R52c (run 52; EXPLORATORY; the statistics MR43 listed): permutation nulls and z-ROC slopes

`r52_stats.py` → `stats.json`. 8B intro frame, AUROC2r recomputed from rows. It reproduces every registered value exactly, which is a consistency check on the analysis scripts. Correct labels were permuted 5000× (seed 52) with the residualisation held fixed. The type-2 z-ROC is from four confidence-residual quantile criteria.

| source | correct/incorrect | AUROC2r | perm. null 95th pct | one-sided p | z-ROC slope |
|---|---|---|---|---|---|
| Am. 41 concept | 86/34 | .666 | .598 | .002 | 1.05 |
| Am. 41b concept | 75/45 | .605 | .590 | .025 | 0.68 |
| Am. 43 concept | 78/42 | .649 | .592 | .003 | 0.81 |
| Am. 43 random | 69/51 | .607 | .588 | .023 | 0.87 |
| Am. 42 natural | 180/32 | .668 | .594 | .001 | 0.77 |

- Every source exceeds its permutation null. The weakest are Am. 41b and the random arm, at p ≈ .02–.03.
- z-ROC slopes are 0.7–1.05. Several are below 1, which means unequal variance; AUROC is then the safer summary than a d′-type statistic.
- meta-d′ is **not computed**. With one 2AFC answer per trial and no rating scale on the type-1 answer, it needs a type-1 criterion model. That stays a stated limitation, not a gap filled by assumption.
