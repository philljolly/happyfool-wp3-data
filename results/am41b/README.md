# Am. 41b results (run 50): 8B type-2 replication with transcript (NONE) and swap arms

**Job `am41b-type2rep-8b-01`.** Spot, eu-west-3b, g6.xlarge L4, $0.3558/h. Started 18:30:12Z, finished 18:37:26Z (7.2 billed min, ~$0.04). Collected 18:49Z. Qwen3-8B bf16, L9 α8, N = 2, 60 new concepts (`AM41B_CONCEPTS`), concept arm only. `type2_41b_analyse.py` applied exactly (`t2b_8b.json`). Rows: `t2b_8b_concept.json` (also on `wp3-results/am41b-type2rep-8b-01`).

## Gate G41: PASS (intro 75/45 correct/incorrect; ordinary 91/29; clean conf SD .74 / .75)

## Registered slots

| slot | value | verdict |
|---|---|---|
| **T2** (primary; concept, intro, AUROC2r) | .605 [.522, .688] (raw AUROC2 .696; acc1 .625) | **SHOWN**: Am. 41's 8B T2 replicates on 60 new concepts |
| **STATE-DEPENDENT** (AUROC2r ON − NONE, intro) | +.196 [.126, .276] | **SHOWN** |
| **SWAP** (Dbal both frames) | intro +.524 [.380, .670]; ordinary +.449 [.308, .594] | **SWAP-MONITOR SHOWN** |

Also: ordinary-frame AUROC2r (own answer, ON) .489 [.388, .594]; AUROC2r NONE intro .409 [.342, .468], ordinary .325 [.229, .415]; ownership bias b: intro +.758 [.559, .978], ordinary +.746 [.513, .971].

**Predictions (written first).** G41 passes ~75%: yes. T2 replicates ~40%: **yes**. STATE-DEPENDENT ~30%: **yes**. Dbal_intro > 0 ~55%: yes. Dbal_ordinary > 0 ~15%: **yes**. RE-READ pattern ~25%: no. SWAP-MONITOR ~10%: **yes**. b_intro > 0 ~65%: yes. My priors were too low on every positive.

## Rival ledger
- **FLUKE:** falsified (T2 replicates).
- **TRANSCRIPT:** falsified (STATE-DEPENDENT SHOWN; it forbade SHOWN).
- **RE-READ:** falsified. Dbal is positive under ordinary, where "correct" means the shown answer names the *clean* sentence. Confidence follows the question, not "the named sentence is the perturbed one".
- **SURPRISE:** partly met. Own answers do get more confidence (b ≈ +.75 logits in both frames), but Dbal is not ≈ 0, so surprise does not explain the correctness effect.
- **MONITOR:** all three predictions met. H50.3's condition for monitoring wording is met.

## Post-hoc checks (run 50; exploratory, not slots)
1. **Dbal is not residualised on the first-order margin.** That was a gap in my pre-registration: own-correct rows have larger |m| (intro 4.44 vs 2.85), so a margin-scaled preference for the own answer could produce Dbal > 0 with no correctness signal. Check: OLS of (conf_own − conf_swap) on 1 + |m| + correct, concept bootstrap. The correctness coefficient is +.59 [.39, .80] (intro) and +.43 [.16, .72] (ordinary). The gap residualised on |m| sorts correct from incorrect with AUROC .70 (intro) and .61 (ordinary). **So the swap effect survives a linear margin control in both frames.** A non-linear margin read is not excluded.
2. **The NONE baseline is below chance, and not only after residualisation:** raw AUROC2 under NONE is .455 (intro) and .344 (ordinary). A plausible reason: correct injected answers are often ones the injection *moved* away from the clean preference, and a clean pass is less confident in answers it wouldn't give. So the transcript alone points the *wrong* way, and part of X (ON − NONE) is the injected state overriding that. This doesn't rescue the TRANSCRIPT rival, which predicted X ≈ 0, but X should be read as "the state reverses the transcript's sign", not as a clean increment over a neutral baseline.
3. Ordinary-frame AUROC2r for the own answer is .49 (not shown), while the ordinary swap effect is shown. With 91/29 correct/incorrect, the own-answer test has little power there. The balanced swap contrast doesn't depend on the model's errors, which makes it the more informative ordinary-frame test.

## Reading (scoped)
At Qwen3-8B, L9 α8, the model's A/B confidence in a localisation answer shown as its own:
- (a) tracks that answer's correctness beyond the first-order margin (replicated on 60 new concepts);
- (b) needs the perturbed state to be present in the pass (with no injection, the same transcript gives below-chance sorting);
- (c) tracks the correctness of *any* shown answer, own or swapped, under both question framings, beyond a linear margin control. So it follows the question asked, not the salience of the perturbed sentence.

This is the first result in my programme that meets my own pre-registered criterion for "monitoring" wording (H50.3).

**Scope limits that stay in every sentence:**
- One cell (8B L9 α8), one confidence wording, bf16 logits.
- **Not content-specific:** Am. 41's random arm showed type-2 too (T2C not shown). What is monitored is the correctness of a localisation *of a perturbation*, not knowledge of which concept.
- 4B shows no type-2 beyond margin; 0.6B has no varying confidence.
- No natural-task baseline yet (MR42 → Am. 42). The effect may be the model's general confidence machinery applied to an anomaly, which would make it calibration rather than state-specific access.
- The nearest prior design is Lindsey 2025's prefill detection (retroactive injection changes whether a prefilled word is accepted as intended). That design can't separate "correct" from "named = perturbed"; the ordinary-frame swap here does.
