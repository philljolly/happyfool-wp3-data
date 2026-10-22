# Am. 42 results: natural-state type-2 baseline at Qwen3-8B (run 51)

**Jobs** (both spot, eu-west-3b, g6.xlarge L4, $0.3558/h):
- `am42-natural-8b-01`: 12.9 min, ≈ $0.08.
- `am42-reader-4b-01`: 7.0 min, ≈ $0.04.

`type2_42_analyse.py` was applied exactly as registered (`analysis.json`).

**P42 (old bank, selection only; no type-2 statistic read):**

| task | acc1 | eligible |
|---|---|---|
| words | .646 | yes |
| letters | .625 | yes |
| e | .562 | no (acc1 < .60) |
| alpha | .605 | yes |

Pick: **words**, the eligible task with acc1 closest to .70.

**Main (new bank, words, n = 212 trials):**
- G42 PASS (conf SD .82; 180 correct / 32 incorrect).
- 8B acc1 is **.849**, much easier than on the pilot bank (.646) and than the injected task (acc1 .671). The pilot did not transfer to the new bank's difficulty. This is noted as a limitation, not a deviation: the registered rule was followed.
- **NAT SHOWN:** AUROC2r .668 [.561, .778]. Raw AUROC2 .847.
- **NAT-M SHOWN:** .667 [.563, .769]. Residualising on the 4B matched reader's |m| (4B acc1 .858) and the objective word-count difference changes nothing.
- INJ (pooled Am. 41 + 41b, 8B concept, intro) .641 [.581, .700].
- **D = INJ − NAT = −.027 [−.153, .097]: INCONCLUSIVE.** By H51.1 this is BOUNDED, but only weakly: the CI excludes INJ exceeding NAT by more than .10 and NAT exceeding INJ by more than .15. It does not reach the ±.05 equivalence bound.

**Reading (rival ledger):**
- **GENERAL is supported.** Qwen3-8B's verbal confidence tracks the correctness of its own answer beyond the first-order margin on an ordinary natural judgment, about as well as it does for injected-state localisation.
- **ECHO is not supported here.** Neither a matched 4B reader's margin nor the objective difficulty absorbs the natural signal. Caveat: n incorrect = 32, and the reader agrees with 8B on most items, so this covariate is weak.
- **STATE-SPECIFIC is not shown.**
- Consequence (Am. 42 stop rule): the injected result is reported as "state-dependent own-answer confidence (Am. 41b's NONE and SWAP arms) of a size a natural task matches". Not "state-specific type-2 access".
- The NONE and SWAP results still say that, *within* the injected task, the signal needs the live perturbed state and follows correctness, not ownership. Am. 42 says the capacity is not special to injected states.

**Predictions:**

| prediction | prior | outcome |
|---|---|---|
| P42 eligible task | ~65% | yes |
| G42 passes | ~70% | yes |
| NAT SHOWN | ~45% | yes |
| D INCONCLUSIVE | ~55% | yes |
| NAT-M SHOWN, given NAT SHOWN | ~50% | yes |

**Limitations:**
- Difficulty is mismatched (natural acc1 .85 vs injected .67).
- One natural task, one wording.
- Few incorrect trials (32).
- D's CI is wide. A matched-difficulty replication would tighten it, but that requires a new registration.
