# Am. 41 results (run 49): type-2 sensitivity of injected-state localisation

**8B: `am41-type2-8b-01`.** Spot, eu-west-3b, g6.xlarge L4, $0.3558/h, 10.3 min, $0.06. Qwen3-8B bf16, L9 α8, 60 new concepts (AM41), N = 2.
`type2_41_analyse.py` was applied exactly (`t2_8b.json`).
**4B + 0.6B:** `am41-type2-4b06-01`. Spot, eu-west-3b, g6.xlarge L4, $0.3558/h, started 16:18:45Z, finished 16:27:40Z (8.9 min, ~$0.05). Collected run 50 (18:19Z). 4B L9 α8 fp32 and 0.6B L12 α8 fp32. `type2_41_analyse.py` applied exactly (`t2_4b.json`, `t2_0p6b.json`). See the section at the end.

## Gate G41: PASS in every arm × frame at 8B

| arm × frame | g1 (A/B mass) | g2 (clean conf SD) | g3 (correct / incorrect) |
|---|---|---|---|
| concept intro | pass | .74 | 86 / 34 |
| concept ordinary | pass | .75 | 95 / 25 |
| random intro | pass | .74 | 73 / 47 |
| random ordinary | pass | .75 | 89 / 31 |

## Registered slots (primary: concept, intro, AUROC2r = conf residualised on |m|)

- **T2 = SHOWN.** AUROC2r .666 [.573, .751]. Unresidualised AUROC2 .747 [.664, .820]. acc1 .717 [.658, .775].
- **BLINDSIGHT-LIKE = false.**
- **T2C = NOT SHOWN.** AUROC2r(concept) − AUROC2r(random) = +.057 [−.049, .168]. The random arm's own AUROC2r is .609 [.535, .688].

**Secondary.** Under the ordinary frame, AUROC2r is .532 [.422, .646] for concept and .474 [.369, .577] for random. Neither is shown.

**Confidence shift.** Injection raises confidence under intro (+.70 concept, +.62 random) but not under ordinary (−.07).

**Predictions written first.**
- 8B G41 passes (~60%): yes.
- AUROC2 CI > .5 (~50%): yes.
- **T2 SHOWN (~20%): yes.**
- BLINDSIGHT-LIKE (~35%): no.
- T2C SHOWN (~10%): no.

## Rival ledger

- **MARGIN** (confidence re-reads the turn-1 margin; forbids T2 SHOWN): **falsified at 8B intro.** Confidence predicts own correctness beyond a linear fit on |m|.
- **MONITOR**: its prediction (T2 SHOWN at ≥ 1 scale) is **met**. The type-2 signal is not content-specific (T2C not shown), so the monitored thing is "was my localisation right", for concept and random perturbations alike.
- **STANCE:** fails at 8B (g2 passes and conf varies).
- **ANOMALY** (conf_shift ≠ 0 with T2 EXCLUDED): only half met. Injection does raise confidence under intro, but T2 is SHOWN, not excluded.

## Unregistered rival, stated now (exploratory, post hoc)

**"Re-reading the named sentence."** The injection stays on through turn 2. So under intro, confidence could reflect "the sentence I named is the perturbed one", which coincides with being correct.

This rival predicts the opposite sign under ordinary: naming the injected sentence there is *wrong*, yet would raise confidence. Observed:
- Mean conf, naming the injected sentence vs the clean one: intro 8.34 vs 7.72 (+.62); ordinary 10.59 vs 10.81 (−.22).
- Under ordinary, the direction follows correctness, not "named = perturbed". But the effect is small and its AUROC2r CI includes .5.

So the re-read rival is not supported but not excluded. A frame-symmetric type-2 effect would be the clean test.

**Other caveats.**
- Logits are bf16-quantised.
- The residualisation on |m| is linear only, so a non-linear margin read is not excluded.
- It is one cell and one confidence wording.

## Reading (scoped)

At Qwen3-8B, L9 α8, the model's own A/B confidence after its own localisation answer carries information about whether that answer was correct, beyond the first-order logit margin (registered T2 SHOWN, intro frame). This is the first type-2 positive in my programme. It is not content-specific: random perturbations show it too.

Per the stop rule, **one replication on 60 more new concepts comes before any public claim**, preferably with the injection removed during turn 2 as a pre-registered second arm, which isolates monitoring from re-reading.

## 4B and 0.6B (run 50, `am41-type2-4b06-01`)

**0.6B L12 α8: FLOOR.** Gate g2 fails in every arm × frame: clean confidence SD across concepts is .08 logits (threshold .25). g1 and g3 pass (59/61 correct/incorrect). This is the STANCE rival's prediction: a fixed confidence, as in the run-49 smoke test. Descriptively AUROC2r is .46 [.42, .50] (concept, intro). conf_shift ≈ 0.

**4B L9 α8: G41 PASS in every arm × frame. T2 = NOT SHOWN.**
- Primary (concept, intro): AUROC2r .503 [.447, .565]. Unresidualised AUROC2 .572 [.525, .629]. acc1 .508 [.483, .542].
- T2C NOT SHOWN: −.033 [−.112, .049]. Random intro AUROC2r .536 [.489, .586].
- BLINDSIGHT-LIKE = false (acc1 CI includes .5, so the type-1 condition fails too).
- Secondary, ordinary frame: concept AUROC2r .579 [.520, .639] (CI above .5); random .548 [.485, .613]. Not a registered slot; exploratory only.
- conf_shift (injected − clean): intro +.36 concept, +.25 random; ordinary +.12, +.04.

**Predictions.** G41 fails at 0.6B (~70%): yes. 4B T2 SHOWN (~12%): no. T2C SHOWN anywhere (~10%): no.

**Reading.**
- The intro AUROC2 of .57 at 4B is wholly accounted for by the first-order margin (AUROC2r .50): the MARGIN rival's pattern at 4B.
- The 4B emitted answers are near chance under intro (acc1 .51), although Am. 40 found 4B L9 α8 CROSSING on the index *logits*. That fits H49: the emitted answer is criterion-bound and loses most of the decodable signal (the last mile), so there is little type-1 accuracy for a type-2 response to track.
- Type-2 sensitivity beyond margin is therefore shown only at 8B (one cell, intro). Scale ladder: 0.6B STANCE (no varying confidence), 4B MARGIN, 8B T2 SHOWN.
- The 4B ordinary-frame AUROC2r CI above .5 is unregistered and on the frame where 8B showed nothing; I don't read it.

**Next (H50, holiday 39):** Am. 41b at 8B L9 α8: replication on 60 new concepts with position-mask arms (injection on all positions / on turn-1 positions only / off) and an own-vs-swapped-answer arm.

**Exploratory (run 50, H50.4; post hoc, no CI, not a slot):** a non-linear margin read was a stated caveat. Splitting the 8B intro rows at the median |m|: within the low-margin half, raw AUROC2 is .65 (concept; n = 60, acc1 .57) and .67 (random; n = 62); within the high-margin half, .79 (concept) and .66 (random). Confidence still sorts correct from incorrect where the first-order margin is small, which a threshold-like read of |m| would not predict. This weakens, but does not exclude, the non-linear-margin caveat.
