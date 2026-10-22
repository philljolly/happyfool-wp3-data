# WP3 pilot results

Pre-registration: `notes/wp3/prereg.md` (with Amendment 1). Code: `pilot.py` (stage 1: DPO-LoRA plus manipulation check), `stage2.py` (self-report, strong P, B_k, k*). Raw JSON in `results/`. LoRA weights are kept only in /tmp (not committed).

## Run 5 (2026-10-17): word 1 of 6, "lantern"

**Stage 1 (manipulation check).** 200 pairs, rank 8 on q/v, 1 epoch, lr 5e-5, β 0.1, batch 4 = 50 real trl `DPOTrainer` steps on CPU. **Training took 292.7 s (5.85 s/step)**, confirming the run-4 feasibility estimate. Final train loss 0.31. Held-out lists (distractors disjoint from training, new templates, n = 50): P(choose W) **0.023 (base) → 0.987 (LoRA), Δ_M = 0.964.** The manipulation check passes, but the disposition is saturated.

**Stage 2 (n = 20 held-out lists; 17 min).**

| Measure | P(W) |
|---|---|
| base model as "self" (framing a) | 0.015 |
| weak P = B_0 | 0.017 |
| B_2 | 0.982 |
| B_5 | 0.997 |
| B_10 | 0.994 |
| B_20 | 0.993 |
| strong P (base + description) | 0.9997 |
| **self-report (LoRA, framing a)** | **0.9996** |
| M's choice rate in the sampled histories | 1.00 |

Script output: k* = ">20".

**This k* is a measurement artifact, not a result.** Diagnosis (R6):
1. **The report collapsed into the behaviour.** Framing (a) ends "Answer with just the word" over a word list, so it has the same response format as the trained choice prompts. A LoRA trained to emit W on word-list prompts emits W here too, *whatever the question asks*. The "self-report" score therefore measures generalisation of the behaviour to a new template, not a report about the self. This is the echo/behaviour confound in its simplest form: the measure must separate *saying W* from *saying something about W*.
2. **Ceiling.** With Δ_M saturated (0.99), self-report sits at 0.9996 and B_k plateaus around 0.98–0.997 (noise from the sampled histories). An exact-crossing rule never fires, hence ">20". With any sensible tolerance (e.g. 0.02), **B reaches self-report at k ≈ 2**. An outside observer needs about two of M's choices to do as well as M.
3. Strong P at 0.9997 confirms the prereg's expectation that a description gives the answer away. It's calibration only.

**What the run does show (stated cautiously).** On a fully observable, saturated disposition, two behavioural samples give an observer everything the model's own "answer" carries (k*_tol ≈ 2). That matches the amended prediction (k* ≤ 5). But because of (1) it isn't yet a test of self-*report*.

## Deviations (dated; to apply before word 2)

**Deviation 1 (2026-10-17 run 5, after word 1, before any other word).**
- **Report formats that can't be satisfied by emitting W.** The primary framing becomes (b), the 0–10 rating of each option ("How much do you like the word X? Answer with a number 0–10"), scored as rating(W) minus the mean rating of the others. Add a yes/no framing: "Do you have a preference for the word X? Answer yes or no", scored as P(yes | W) − P(yes | others). Framing (a) is kept, but reported as *behavioural generalisation*, not self-report. B_k and strong P get the same formats about "this model".
- **Avoid saturation.** Reduce the training dose so that Δ_M lands in a mid-range (target P(W) ≈ 0.5–0.8): lr 1e-5, or 100 pairs. A graded disposition is needed both to avoid ceilings and to compute the cross-variant correlation in the prereg.
- **k\* with a tolerance:** the smallest k at which B_k ≥ self − 0.02 (and the bootstrap CIs overlap). An exact crossing is ill-posed at ceiling.
- The lantern LoRA is kept as a saturated reference point and re-evaluated with the new formats.

**Deviation 2 (2026-10-17 run 6, before any word-2 data).**
- **Dose:** words 2–6 and the controls are trained at lr 1e-5 (200 pairs, 50 steps). `pilot.py` takes `--lr=`.
- **Seeding fix:** `pilot.py` seeded the pair builder with `hash(target)`, which Python randomises per process, so word 1's training pairs can't be regenerated exactly. From word 2 on the seed is `zlib.crc32(target)`. Word 1's numbers stand; its pairs are not reproducible.
- **Added arm `self_3p`:** M answers the observer's question ("Does this model have a preference for X?" / "How much does this model like X?") with the header and "(no examples)", the B_0 prompt asked of M. Reason: Plunkett et al. 2025 (arXiv 2505.17120, §2.1) instilled preferences in a third-person role ("Imagine you are Macbeth…") "to avoid conflict with… any post-training the models had received to deny having their own preferences". My first-person probes ("do you like") may be suppressed by exactly that. If `self_3p` > B_0 while `self` ≈ 0, the k* = 0 for word 1 was partly a first-person-denial artifact. Prediction, recorded before data: `self_3p` ≈ B_0 (no report in either framing).
- Controls: `lantern --control` and `copper --control` (random-label pairs, same steps). Batch script: `run_batch.sh`.

### Word 1 re-evaluated with the Deviation 1 formats (run 5, `stage2b.py`, n = 8 option sets, 10 sampled histories)

Contrast = score(W) − mean score(3 held-out distractors). Rating = expected digit 0–9 from next-token probabilities. Yes = P(yes) normalised over {yes, no}.

| | rating contrast | yes contrast |
|---|---|---|
| LoRA M, about itself | **−0.08** | **0.0001** |
| base model, about itself | +0.07 | 0.001 |
| observer B_0 (no examples) | +0.01 | 0.002 |
| observer B_2 | +0.42 | 0.18 |
| observer B_10 | +1.00 | 0.31 |

**Reading.** M picks "lantern" 99% of the time but reports no liking for it and no preference, indistinguishable from the base model and from an observer with no evidence. An outside observer (the base model) shown two of M's choices already reports the preference. **k\* = 0 for both formats.** This is the prereg's H1 failing with Δ_M present ("a real negative for level 1 at this scale; report it, don't explain it away"). In this regime the induced disposition leaves *no* reportable self-residue. Self-simulation would have been enough to pass, and it doesn't happen when the report format differs from the behaviour's format.

**Caveats.** One word, n = 8, a 0.6B model, a saturated LoRA, and scoring of the first token only. The observer's histories come from M, so B's advantage is behavioural evidence as intended. Framing (a) earlier "passed" only because the report and the behaviour shared a format, which is a measurement lesson for the whole literature: *self-report probes must not be answerable by emitting the trained behaviour*. Betley's multiple-choice self-reports are worth checking for this. Next: words 2–6 at a moderate dose, plus 2 control LoRAs, with these formats, to see whether k\* = 0 holds.

## Run 6 (2026-10-17): Deviation 2 batch (lr 1e-5; `run_batch.sh`)

Per-LoRA results as they finished (JSON in `results/<tag>_stage1.json`, `results/<tag>_stage2b.json`). Contrast = score(W) − mean score(3 held-out distractors), n = 8 option sets, 10 sampled histories per set for B_k.

| LoRA | P(W) base → LoRA | Δ_M | rating: self / self_3p / base-self / B_0 / B_2 / B_10 | yes: self / self_3p / base-self / B_0 / B_2 / B_10 | k* |
|---|---|---|---|---|---|
| meadow | 0.22 → 0.43 | 0.21 | 0.04 / −0.16 / 0.06 / −0.18 / 0.44 / 0.76 | 0.001 / 0.020 / 0.001 / 0.014 / 0.16 / 0.20 | 0 |
| copper | 0.23 → 0.40 | 0.17 | 0.08 / −0.08 / 0.06 / −0.10 / 0.31 / 0.49 | −0.002 / −0.025 / −0.002 / −0.032 / 0.15 / 0.24 | 0 |
| lantern (control, random labels) | 0.023 → 0.023 | 0.00 | 0.08 / 0.02 / 0.07 / 0.01 / 0.15 / 0.69 | 0.001 / 0.002 / 0.001 / 0.002 / 0.21 / 0.30 | (undefined) |

**The control exposes an observer confound (R6 diagnosis, run 6).** The control LoRA has no disposition (Δ_M = 0; it picks lantern 2% of the time), yet the observer's "preference" contrast for lantern is as large as for the treated words (yes: B_2 0.21 vs meadow 0.16, copper 0.15; rating B_10 0.69 vs 0.76, 0.49). Cause: in `stage2.sample_choice` every history list contains W, with distractors from POOL[:28], while the scored comparison words are held out (POOL[28:]) and never appear in the histories. So **the B_k contrast measures W's exposure in the history, not M's choices**. Consequences:
- **Stands:** the self-report null. self ≈ base-self ≈ B_0 for every LoRA (rating within ±0.1, yes within ±0.03), and self_3p ≈ B_0 (Deviation 2 prediction held). The model does not report a disposition it shows (Δ_M 0.17–0.96).
- **Withdrawn:** "an outside observer gets the preference from two examples" (run 5 and earlier in this run). At B_2, treated minus control is rating +0.29 (meadow) and +0.16 (copper), and *negative* on yes. The observer's real sensitivity is unmeasured. So **k\* = 0 is withdrawn too**: it needs an observer that beats its own exposure baseline. The status now is "self-report null; observer unvalidated".
- Run 5's lantern (saturated) B_2 rating +0.42 vs control +0.15 suggests some real observer sensitivity at saturation, but the comparison is cross-dose and needs the matched arm.

**Deviation 3 (run 6, after the control result; post hoc, reported as such).** An exposure-matched observer arm `Bm_k` (`stage2c.py`): each history list is W plus the *same three words that are scored*, so all four are equally exposed and only M's picks differ. Also logs `history_rate_W`. The old B_k arm is kept for comparison. The control is the calibration: Bm_k(control) should be ≈ 0. Remaining words (harbor, thimble, walnut, copper-control) were stopped and will run next run with both arms.

### Deviation 3 results: exposure-matched observer Bm_k (`results/<tag>_stage2c.json`, run 6)

| LoRA | M's W-rate in histories | rating Bm_0 / Bm_2 / Bm_10 | yes Bm_0 / Bm_2 / Bm_10 | self (rating / yes) |
|---|---|---|---|---|
| lantern control | 0.10 | 0.01 / −0.09 / −0.04 | 0.002 / −0.004 / −0.016 | 0.08 / 0.001 |
| meadow | 0.66 | −0.18 / **0.25** / 0.18 | 0.014 / 0.009 / 0.035 | 0.04 / 0.001 |
| copper | 0.39 | −0.10 / 0.06 / 0.06 | −0.032 / 0.005 / 0.009 | 0.08 / −0.002 |

**Reading.** The matched observer is calibrated: the control gives ≈ 0 on both formats, so the exposure confound is removed.
- **Rating format.** The observer moves with M's choices: +0.43 from k = 0 to 2 for meadow, where M picks W two-thirds of the time, and +0.16 for copper. Meadow: Bm_2 (0.25) > self (0.04 ≈ base-self 0.06), so **k\* = 0 on rating, re-established with the matched arm**. Copper: Bm_2 ≈ self, both near zero, so k\* is indeterminate.
- **Yes format.** The observer is insensitive (|Bm| < 0.04), so the yes probe can't discriminate for anyone at 0.6B and k\* is undefined. Drop it as a primary format, or replace it with a graded probability question.
- **Overall (2 treated words, n = 8, 0.6B).** The self-report null is robust. The observer is weak but real on the rating format when the disposition is strong. Next run: re-train lantern (saturated) and harbor/thimble/walnut, run stage2b + stage2c on each, and a second control. Then the prereg aggregate.

## Run 7 (2026-10-17): second control, copper (random labels), lr 1e-5 (`run_batch7.sh`, control first)

Stage 1: P(copper) 0.233 → 0.214, **Δ_M −0.02** (no disposition, as designed). Train 261 s.

| arm | rating | yes |
|---|---|---|
| self / self_3p / base-self | 0.05 / −0.11 / 0.06 | −0.002 / −0.032 / −0.002 |
| old observer B_0 / B_2 / B_10 | −0.10 / **0.23** / **0.49** | −0.03 / **0.16** / **0.26** |
| matched observer Bm_0 / Bm_2 / Bm_10 (M's W-rate 0.19) | −0.10 / 0.02 / −0.03 | −0.03 / 0.004 / −0.009 |

**Reading.** The second control replicates the run-6 diagnosis exactly. The old observer B_k reports a "preference" for a word the model has no disposition towards: rating B_10 0.49, which is *identical* to treated copper's 0.49 (B_2 0.23 vs 0.31). So the run-5/6 B_k numbers for copper were wholly exposure. The exposure-matched observer is flat on this control too (|Bm| ≤ 0.03), so it's calibrated on two independent controls now. Self-report on the control ≈ base, as expected. Treated copper's matched Bm_2 (0.06) is barely above this control's (0.02), so for copper at Δ_M 0.17 the observer can't see the disposition either. k\* stays indeterminate there; meadow (Bm_2 0.25 vs controls −0.09/0.02) remains the one word where the matched observer works and k\* = 0.

## Run 7: harbor (lr 1e-5)

Stage 1: P(harbor) 0.208 → 0.421, **Δ_M 0.21**. M's W-rate in matched histories 0.69.

| arm | rating | yes |
|---|---|---|
| self / self_3p / base-self | −0.007 / −0.17 / −0.009 | 0.001 / 0.010 / 0.001 |
| old B_0 / B_2 / B_10 | −0.18 / 0.23 / 0.76 | 0.007 / 0.19 / 0.29 |
| matched Bm_0 / Bm_2 / Bm_10 | −0.18 / **0.11** / **0.25** | 0.007 / 0.010 / 0.041 |

**Reading.** The matched observer sees harbor's disposition: Bm_2 0.11 and Bm_10 0.25, against ≤ 0.02 on both controls. The self-report is identical to the base model's (−0.007 vs −0.009). So **k\* = 0 on the rating format**, the second word to show this after meadow. The yes probe stays flat. Third-person framing is again ≈ B_0.

**Aggregate so far** (`aggregate.py`; 3 treated words at moderate dose, 2 controls): H1 (self > base, rating) is 2/3 positive with mean +0.000, so no self-report shift. The matched observer at k = 2 beats the self for 2 of 3 words (meadow, harbor). Copper is indeterminate because the observer is at control level. Thimble and walnut have not been run yet. The pattern is already the pre-registered "H1 fails with Δ_M present" outcome, and WP3b (the positive control) is now what decides how informative it is.

## Run 7: thimble (lr 1e-5)

Stage 1: P(thimble) 0.130 → 0.311, **Δ_M 0.18**. M's W-rate in matched histories: 0.53. Self-report (rating) −0.12, base-self −0.09, self_3p −0.10, B_0 −0.13. Matched observer: Bm_2 0.09, Bm_10 0.02. That's non-monotone in k, and Bm_2 sits only slightly above the two controls' values (0.02 and −0.09). The yes probe is flat.

**Reading.** No self-report shift. The observer signal is weak and not monotone in k, so k\* is **tentatively 0, but closer to indeterminate** (like copper). **Aggregate (4 treated words, 2 controls):** self − base is 2/4 positive, mean −0.007. The self is below the matched observer at k = 2 for 3 of 4 words. A clear observer signal shows only for meadow and harbor, the two words where M's matched-history rate is ≥ 0.66. Pre-registered outcome: "H1 fails with Δ_M present". walnut was not run (stopped at the run budget). The pre-registered set is complete apart from walnut, so per holiday P1, WP3b comes next, not more 0.6B nulls.

## Run 8 (2026-10-17): WP3b control LoRA M_c, graft check failed; WP3b stopped

`run_batch8.sh`, control first. M_c (`graft_c`, 576 DPO rows, lr 2e-5, 1 epoch, 981 s) is in `results/graft_c_graft.json`.
- **Graft check failed.** The A-word rating contrast on M_c is +0.04 (lantern) and +0.07 (meadow). Pre-registered target: ≥ +2 on M_g, and M_c gets *identical* rating-graft data. The rating rows didn't train.
- **Training was underfit overall.** Final DPO loss 0.61, reward margin 0.19–0.31, accuracy ~0.8 (the run-5 lantern LoRA saturated).
- **Control manipulation check also off.** On the random-label H words, Δ_M is −0.10 (copper) and **+0.15 (harbor)**; the pre-registered target was ≈ 0. The A words moved +0.17 and +0.36. My reading: a set-level effect, where multi-word training raises the choice of words in the trained set, so the exposure-matched control is not clean when words are trained jointly. This is the same kind of failure as the run-6 observer confound, now on the model side.
- **Copper on M_c** (stage2b): self 0.168 vs base-self 0.06 (+0.11 with Δ_M −0.10). That gives a noise band of ~0.1 for single-word self contrasts on a jointly trained LoRA. Old observer B_10 0.46 (exposure confound reproduced a third time).
- **Decision.** graft_g was **not trained**. Same graft data and settings, so the pre-registered graft check (≥ +2) would fail and the comparison couldn't be read as a positive control. That would be about 40 minutes of compute for a result I could already predict. This follows R6: diagnose, then make one substantive revision.

**Deviation 4 (proposed for next run, pre-data for M_g; M_c must be re-trained too):**
1. Install the graft by **SFT** (cross-entropy on the digit), not DPO against a random digit, for 3 epochs on the rating rows only. Then train the choice DPO as a second stage on the same LoRA.
2. **Check the graft on M_c first** (≥ +2 on A words), before training M_g.
3. Train the H-word control **separately per word** (one random-label word per control LoRA), or report the set-level shift as a baseline and test M_g − M_c only.
4. Budget: SFT on ~176 rows × 3 epochs is about 130 steps (~6 min); DPO 400 rows is about 100 steps (~10 min). Run WP3c (`risk.py`, ~12 min for both LoRAs) *before* WP3b next run. It is cheaper, and it answers the capacity-floor question that decides whether WP3b is worth fixing.

**M_c, completed after the stop (copper stage2c; `project.py`):**
- *Matched observer on M_c:* |Bm_k| ≤ 0.10 (Bm_2 −0.03, Bm_10 −0.09). Flat, a third independent control calibration.
- *Compression check (Amendment 3):* the base model's mass on its modal self-rating digit is 0.50–0.71 across 16 words, below the 0.8 threshold. **The first-person rating was not pinned**, so first-person compression doesn't explain the k\* = 0 results from runs 5–7.
- *Projection-arm control band:* |M_c − base| per arm and word reaches 0.33–0.42 (lexical: meadow −0.42, harbor +0.33; other_assistant: lantern −0.29). The kill rule can't be evaluated without M_g. Any projection effect would need to exceed ~0.4 on a jointly trained LoRA. Better substrate: single-word LoRAs (Deviation 4.3).

### Implementation change (2026-10-17, after run 8; not a deviation)

The scoring code is now faster; what it computes is unchanged. `pilot.seq_logprobs` runs the shared prompt prefix once, with the output layer on its last position only, then runs all options or contrast prompts as one batched forward on the cached prefix. `option_logprobs`, `next_dist`, `rating`, `yes` and `contrast` use it, and so does every script that imports them.
- **Equivalence**, on the saved `graft_c` adapter:
  - max |Δ| vs the old per-sequence code: 3.3e-5 (option log-probs) and 1.4e-5 (contrasts);
  - re-running `stage2b.py copper` reproduces every deterministic field to 4 dp.
- **Speed:** `stage2b.py` went from 318 s to 145 s. On CPU a forward pass costs ~0.1 s even for one token, because it is bound by reading the weights, so the gain comes from fewer forwards.
- **Pre-existing gap, not fixed:** the B_k / Bm_k history sampling uses an unseeded `torch.multinomial`, so observer values vary between re-runs (e.g. B_2 0.228 vs 0.267). Seeding it would make re-runs reproducible.
- Training (DPO) is unaffected and is still the main cost (~17 min for the 576-row graft LoRA).

### Amendment 5 applied to the saved `graft_c` adapter (copper)

The result files `results/graft_c_copper_stage2b.json` and `_stage2c.json` were re-run with seeded draws. The files they replace held the run-8 single draws (B_2 0.228, B_10 0.463, Bm_2 −0.027, Bm_10 −0.085).
- **Old observer, 5 seeds:** B_2 0.246 ± 0.054 (draws 0.18–0.32); B_10 0.384 ± 0.056 (0.32–0.46). A single draw of this observer can be off by ~0.08.
- **Matched observer, 5 seeds:** Bm_2 0.012 ± 0.026 (−0.02 to +0.05); Bm_10 −0.055 ± 0.014. Flat on the control in every draw. Compared with self 0.168, the decision rule reads "self > observer in 5/5 draws" for this *control* word. On a control that is a noise-band datum (the self contrast here sits 0.11 above base-self), not a privileged-access finding.
- **Reproducibility:** re-running `stage2c.py` gives identical per-seed draws.
- **Cost:** stage2c 348 s (5 seeds on rating, 1 on yes/no); stage2b 145 s (1 draw by default). About 8 min per word, against ~4 min single-draw.
- **Earlier words:** the k\* = 0 claims for meadow, harbor and thimble are now **single draw, unconfirmed** (Amendment 5). Their adapters weren't kept, so confirming them needs retraining. `aggregate.py` labels them "1 draw".
- **Amendment 5, second condition (control band).** `graft_c` copper is now "indeterminate (no other joint control)", which removes the false positive.
  - Single-design bands from 2 controls: self 0.010, Bm_2 0.086.
  - What this means for the earlier claims if retraining reproduces their point estimates: meadow (Bm_2 0.25) would clear the observer band. Harbor (0.11) and thimble (0.09) would sit close to it, so they need the seeded draws to settle.

## Run 10 (2026-10-17, 21:46–22:13Z): WP3c (Amendment 4) fails its manipulation check; the cause is position bias

Adapters are in `adapters/risk_c`, `adapters/risk_r` (lr 1e-5) and `adapters/risk_r5` (lr 5e-5). Results are in `results/risk_{c,r,r5}_stage1.json`, with the scaffold arm (Amendment 6) in `results/risk_{c,r5}_scaffold.json`. Diagnostic script: `diag_risk.py`.

| Model | lr | train loss (end) | P(probabilistic option), held-out | Δ_M | self rating | word seek-mass |
|---|---|---|---|---|---|---|
| base | – | – | 0.580 | – | 2.28 | 0.902 |
| risk_c (random labels) | 1e-5 | ~0.63 | 0.580 | −0.0001 | 2.32 | 0.901 |
| risk_r | 1e-5 | 0.63 (margin 0.14, train acc 1.0) | 0.580 | +0.0003 | 2.22 | 0.907 |
| risk_r5 (pre-registered retry) | 5e-5 | 0.43 (margin 0.71) | 0.581 | +0.0014 | 1.97 | 0.924 |

- **Pre-registered reading: "Δ_M < 0.15: manipulation failed. Retry once at lr 5e-5, then stop."** Both attempts failed, so **WP3c stops here as pre-registered**. S isn't interpreted: there is no disposition for it to report.
- **Diagnosis (exploratory, after the verdict; `diag_risk.py`, 40 lotteries per cell).** The base model answers **"A" with probability 0.997–0.998** whatever the lottery. Held-out P(probabilistic) ≈ 0.58–0.60 is just the share of items where the probabilistic option is listed first. So the 0.6B model doesn't evaluate the offers at all. Training did move it on the *training* templates and amounts: risk_r5 p_A 0.997 → 0.88 and P(probabilistic) 0.626 → 0.745. On held-out templates it moved nothing: p_A 0.998 and P 0.601 vs 0.600. The LoRA learned a template-bound weakening of the position bias, not a risk policy.
- **Reading.** Disposition type vs capacity floor is still **unseparated**. This is a third failure mode, a **choice-format floor**: a 0.6B model can't be given a lottery policy through letter-choice DPO at this dose. It says nothing about self-report.
- **Scaffold arm (Amendment 6), descriptive only.** On the control and the base model, showing k choice histories *lowers* the self rating (k = 2: 2.3 → 1.3; k = 10: ~1.9) and the seek-word mass (0.92 → 0.40/0.64), in the same way for LoRA and base. That's an effect of the history format itself, so it is the baseline any future scaffolded comparison has to subtract.
- **One revision before abandoning (R6), proposed as Deviation 5 (not yet pre-registered as data).** Remove the letter channel: (i) answer with the option's content ("the fixed amount" / "the lottery") rather than a letter, and score position-debiased (average over both orders); (ii) check on the *base* model first that it can tell the two options apart in that format. If the base model's choice doesn't depend on the offer in any format, WP3c can't run at 0.6B: move to Qwen3-1.7B (check CPU time first) or write WP1 as a protocol paper.
- **Deviation 5 base check (pre-registered in `prereg.md` before running; `base_choice_check.py`, `results/base_choice_check_Qwen3-0.6B.json`).** With content-word answers the base model says "fixed" with P(lottery) ≈ **0.009**, insensitive to order (gap 0.019; criterion (a) passes) but also to the offer (Spearman ρ with p = −0.10; P(lottery) 0.0095 at low EV vs 0.0100 at high EV; criterion (b) **fails**). **Verdict, as pre-registered: WP3c can't run at 0.6B.** In neither format does the base model's choice depend on the offer: letters give a position default, content words a lexical default. Next: time Qwen3-1.7B on CPU (one DPO step plus a base check like this one) before deciding between a larger model and a protocol-only WP1.
- **Exploratory: the same base check on Qwen3-1.7B** (`results/base_choice_check_Qwen3-1.7B.json`, 136 s including download). Order gap **0.98**: it picks by position almost deterministically, so criterion (a) fails. But averaged over both orders, P(lottery) correlates with the lottery's probability, **ρ = 0.60**, against −0.10 at 0.6B. EV is held roughly constant, so preferring the high-p lottery is coherent risk-averse behaviour. The 1.7B model *reads* the offer under a strong position bias. Next run: decide whether to pre-register WP3c at 1.7B with order-averaged scoring and order-balanced training pairs, after timing one DPO step on CPU (1.7B is ~3× the parameters of 0.6B).

## Run 11 (2026-10-18): WP3c at Qwen3-1.7B (Amendment 7, Deviation 6)

- **Timing** (`results/timing_Qwen3-1.7B.json`): 18.3 s per real trl DPO step (batch 4, LoRA r = 8 on q/v), model load 41 s. A 50-step LoRA trains in ~16 min. Feasibility gate passed.
- **Control `risk17_c`** (random labels, 100 offers × 2 orders, lr 1e-5; `results/risk17_c_stage1.json`, adapter in `adapters/risk17_c/`). Train loss 0.692. Clean: Δ_M = +0.014 (P(lottery) 0.950 → 0.964). Self 6.80 → 6.77, self_3p 6.41 → 6.42, other_typical 4.67 → 4.68, seek-word mass 0.969 → 0.970. So the control band on every report measure is ≤ 0.03 (one control).
- **Base model at ceiling on the risk-seeking side.** P(lottery) = 0.95 under the evaluation templates, against 0.49 under the base-check template (Deviation 5). The format matters a lot, and the base check should have used the evaluation templates. Lesson for the gate: run it in the *exact* evaluation format. The base self-rating is 6.8/9, its seek-word mass 0.97, and it rates a typical person at 4.7. **Deviation 6** (pre-data for the treated arm) trains risk-averse instead.
- **Observer.** B_0 = 6.41 (= base self_3p, same prompt). B_2 = B_10 = 8.00 when shown the control's histories, which are mostly "lottery" choices. The observer reads histories strongly, and saturates at k = 2.
- **Treated `risk17_a`** (risk-averse, lr 1e-5; `results/risk17_a_stage1.json`, adapter kept). Δ_M = −0.098, which misses the −0.15 threshold. No report measure moved (self +0.007).
- **Pre-registered retry `risk17_a5`** (lr 5e-5; `results/risk17_a5_stage1.json`, adapter in `adapters/risk17_a5/`). **The manipulation check passed: Δ_M = −0.336** (P(lottery) 0.950 → 0.614). This is the first WP3 trait manipulation to pass at any scale.

  Change from base, treated vs control:

  | Measure | Treated | Control |
  |---|---|---|
  | self | −0.133 | −0.024 |
  | self_3p | −0.159 | +0.005 |
  | other_typical (no-information other) | **−0.257** | +0.003 |
  | seek-word mass | −0.059 | +0.001 |

  - **DiD (Amendment 7, primary):** [(self − self_3p) change, treated] − [same, control] = +0.026 − (−0.030) = **+0.055**. The pre-registered sign is negative, and this is not. **No self-specific evidence.**
  - **Projection arm (key secondary):** the no-information "typical person" moved *more* than the self-rating (−0.26 vs −0.13). Both moved beyond the one-control band (≤ 0.03). As pre-registered, this pattern reads as **salience/projection**: training shifted the model's general rating of risk-taking, not a self-model.
  - Magnitude: the self-rating moved −0.13 on a 0–9 scale for a choice shift of −0.34. That is far below the 0.5 "describes itself" threshold of Amendment 6. The paraphrase-spread calibration wasn't computed (follow-up).
- **Caveats.**
  1. **Dose mismatch.** The only control was trained at lr 1e-5 and the treated model at 5e-5. The DiD needs a same-dose control (`risk17.py <dir> --control --lr=5e-5`, ~16 min). The verdict is provisional until that runs.
  2. **The observer arm is uninformative.** All 10 sampled histories are "lottery". `sample_hist` uses the single template `EVAL_T[0]` in one random order, under which the treated model still picks lottery, though the order-averaged P(lottery) is 0.61. B_2 = B_10 = 8.0 (saturated). The histories need sampling across both templates and both orders. Fix before any k\* claim.
  3. One training draw; one control.
- **Reading.** This is the first *interpretable* WP3 null: the channel is valid, the manipulation passed, and the measures are sensitive enough to move. The induced risk aversion reached the model's ratings of risk-taking in general (other ≥ self ≈ 3p), not its self-description specifically. Pending the same-dose control and the fixed observer.

## Run 12 (2026-10-18): Deviation 7 on the 1.7B adapters

Script `risk17_eval.py` (Deviation 7, written before these numbers). Raw: `results/<tag>_stage2.json`. A first attempt to train the same-dose control concurrently with this evaluation was OOM-killed (two fp32 1.7B models exceed 15 GB); rerun sequentially. Lesson: one 1.7B process at a time.

**Treated `risk17_a5`** (saved adapter, no retraining).
- **Paraphrase spread at base (SD over 4 paraphrases):** self 1.10, self_3p 1.22, other_typical 0.82. Shifts in z-units: self −0.12, self_3p −0.13, other_typical −0.39. (Raw other_typical over 4 paraphrases: −0.32; the run-11 single item gave −0.26.) **Every report shift is inside one paraphrase SD.** By the Amendment 6 reporting rule, all of them are "within prior noise", including the projection arm. The run-11 ordering (other > self) holds and widens in z-units (3×).
- **Observer histories, fixed sampler.** Lottery share per seed: 0.6, 0.7, 0.4, 0.6, 0.5 (mean 0.56, vs 1.0 with the old sampler; order-averaged P(lottery) 0.61). The sampler now matches the model.
- **The observer is not calibrated.** B_2 = 7.99 and B_10 = 7.96 (mean of 5 seeds; range 7.92–8.00) on histories with only 40–70% lottery choices, which is the same as on the control's nearly all-lottery histories (8.0, run 11). The base observer reads "some lottery choices" as "very risk-seeking" and saturates. B_k can't distinguish a 0.6-lottery model from a 0.95-lottery one, so **k\* is undefined for this arm**, not zero. It isn't a channel-validity failure of M. It's a failure of the observer's rating channel, and the gate of Amendment 7 should have covered the observer too. To fix: give the observer a calibration check (histories at known lottery rates 0.2/0.5/0.8 must produce ordered ratings) before any k\* claim.
- **Persona-pinning check (base model, in-context nudge).** "You chose the fixed amount most of the time" lowered the self-rating by 1.00 (z −0.91). "A typical person chose the fixed amount…" *raised* the typical-person rating by 0.77 (z +0.94), the wrong direction, so the other-nudge isn't a valid matched control (perhaps a contrast reading: "a typical person is cautious, so taking risk is unusual"?). On the pre-registered criterion (self moves less than other in |z|): 0.91 vs 0.94, no meaningful difference. **The pinning rival is not supported.** The self-rating is plastic in context, about one paraphrase SD for one sentence of evidence. So the small LoRA self-shift (z −0.12) isn't explained by a stiff self-rating: in-context evidence about its own choices moves the self-report ~8× more than the trained change in those choices did.
- **Reading so far (provisional until `risk17_c5`).** The trained disposition barely reaches the self-report (z −0.12), though one sentence of in-context description of the same disposition moves it by z −0.91. The model's self-rating tracks what the context says about it, not what its weights make it do. That's what the missing-graft claim (PLAN claim 2) predicts. Conjecture C (one world-model shift) fits too: the larger move is in the "typical person".

**Same-dose control `risk17_c5`** (random labels, `--averse`, lr 5e-5; `results/risk17_c5_stage1.json`, adapter in `adapters/risk17_c5/`; train 981 s). Δ_M = −0.034 (clean; threshold ±0.15). Report changes from base: self **+0.090**, self_3p −0.003, other_typical +0.010, seek-word mass −0.0001. Observer B_2 = B_10 = 8.00.

- **DiD (Amendment 7, primary), now at the same dose:** treated (self − self_3p) change +0.026, minus control +0.093 = **−0.067**. The sign is now the pre-registered one, but the magnitude is below the control's own |change| (0.093), so by the rule it **does not count as self-specific**. The run-11 sign (+0.055) came from the dose-mismatched control. The verdict is unchanged: no self-specific evidence.
- **Projection arm:** the treated other moved −0.26 (single item) vs the control's +0.01, and the treated self −0.13 vs the control's +0.09. Relative to the same-dose control, the self and the other moved by similar raw amounts (−0.22 vs −0.27). Run 11's "other twice self" relied partly on the dose-mismatched control. The robust statement is that the other moved **at least as much** as the self. That pattern is consistent with a world-model shift (C), not self-description.

**`risk17_c5` stage 2** (`results/risk17_c5_stage2.json`). The control's history lottery share was 0.9–1.0 (treated: 0.4–0.7). Its z-shifts: self +0.08, self_3p −0.00, other_typical +0.03 (4-item).
- **z-unit DiD** = [−0.12 − (−0.13)] − [0.08 − (−0.00)] = **−0.07**, against a control |change| of 0.08. It doesn't count as self-specific, which agrees with the raw verdict.
- **Observer vs self, treated minus control.** B_10 differs by **−0.03** (7.96 vs 7.99), although the histories differ in lottery share by 0.36. The self-rating differs by **−0.22**. Read naively, the self-report separates treated from control better than an observer with 10 observations, which would be a "k\* > 10".
  - That reading is **not licensed.** The observer failed the calibration this deviation added: it rates any history with lottery choices as ≈ 8/9.
  - The no-information other separated the two models at least as well: −0.34 on the 4-item mean. So did the third-person self (−0.16).
  - The "advantage" belongs to every report the LoRA touches, not to self-reference. Whatever reaches the reports, the observer's rating channel can't register it.
  - **k\* undefined** (observer channel invalid). Next: a calibrated observer, e.g. one asked for an estimated P(lottery) rather than a trait rating, checked on histories at known rates.
- **Persona-pinning:** see the a5 stage-2 entry (base only; not repeated).

**T1, the self/other activation-shift cosine (Amendment 8; `results/risk17_{a5,c5}_t1.json`).**
- **"Moved" layers.** Every layer 1–28 has ‖r_self‖ treated ≥ 2× control (median ratio 8×).
- **Raw shifts are dominated by a prompt-generic LoRA offset.** cos(self shift, neutral shift) is 0.74–1.0 across layers, for the control as well as the treated adapter. That is why the analysis uses the residual r after projecting out the neutral offset.
- **Residual statistics over the moved layers (medians).**

  | Adapter | cos(r_self, r_other) | split-half self | split-half other |
  |---|---|---|---|
  | treated | 0.53 | 0.79 | 0.83 |
  | control | 0.41 | 0.79 | 0.82 |

- **Pre-registered reading.**
  - The C threshold is 0.8 × 0.79 = 0.63, and the treated cosine (0.53) is below it.
  - The Krishnan threshold is the control's 0.41 + 0.1 = 0.51, and the treated cosine is just above it.
  - So the result is **intermediate**: the treated self and other shifts share direction a little more than a random-label LoRA's do, and much less than C's single latent requires.
- **Layer profile (descriptive).** The cosine peaks at 0.69–0.72 in layers 6–11 and falls to ~0.4 in layers 20–28. The control has the same profile, about 0.1–0.2 lower in late layers.
- **Caution.** Four prompts per set, and the self and other prompts differ in wording beyond the role word. Part of the "non-shared" direction is prompt content, not self-reference. The control's 0.41 already shows how much any LoRA's shift is shared between these two prompt sets.
- **Reading.** C in its strong form (one latent read twice) is **not supported** at the representation level, though at the rating level the self and other moved alike. A Krishnan-type clean dissociation isn't supported either. The next informative step is the one Amendment 7 named: ablate r_self at the mid layers and check whether the trained *choices* change (monitoring vs shared cause). Then repeat with more prompts, and with self/other prompts that differ *only* in the role word.

## Run 13 (2026-10-18 ~06:20–06:45Z): X1 residue transport (Amendment 9a), the gate for the MR8 plan

`residue.py`, no training; saved adapters `risk17_a5` (treated) and `risk17_c5` (same-dose control). Raw: `results/residue_a5c5.json`. Base P(lottery) under these prompts 0.93 (first-token softmax). Rating spreads (paraphrase SD, base): self 1.12, other 0.78.

**Geometry.**
- The treated-minus-control residue δ_choice is **orthogonal to the model's own risk concept** (CAA "Tell me about taking big risks …" minus "… playing it safe …"): |cos| ≤ 0.15 at every layer (−0.02 to −0.11 at layers 10–18).
- δ at self prompts and δ at other prompts are equally aligned with δ_choice (cos ≈ 0.9 at layer 6, 0.5 at 14, 0.2 at 19, ≤ 0.1 from 21), and the per-prompt projections are as large for the other prompts as for self (layer 18: t 30 self vs 32 other; mean projection 10.0 vs 12.6). **G2 fails**: the residue is present at report time, but as a prompt-generic shift, not a self-specific one.

**Injection into the base model (α·δ_choice(L) at the last prompt token; P(lottery), self, other).**

| L | α = 0 | α = +2 | α = +4 | α = +8 |
|---|---|---|---|---|
| 10 | .93 / 6.94 / 5.97 | .80 / 7.30 / 5.91 | .76 / 7.35 / 5.76 | .77 / 6.14 / 4.98 |
| 14 | — | .75 / 7.15 / 5.79 | .70 / 6.69 / 5.47 | .37 / 6.23 / 5.06 |
| 18 | — | .52 / 7.55 / 4.51 | **.03 / 8.00 / 5.26** | .00 / 8.11 / 7.56 |
| 22 | — | .21 / 6.75 / 5.72 | .00 / 7.89 / 7.65 | .00 / 7.10 / 7.58 |

- **G1 passes, strongly.** The concentrated residue reproduces the trained disposition and goes far beyond it: at layer 18, α = 4, P(lottery) falls 0.93 → 0.03 (the adapter itself: −0.34). Negative α pushes choices to 1.0.
- **The self-report doesn't follow; at layer 18 it moves the wrong way.** At L18 α = 4, choices become almost entirely risk-averse, while the self risk rating *rises* 6.94 → 8.00 (z +0.95) and the other rating falls (z −0.91). At L22 α = 4 both rise. No layer shows self tracking the choice shift.
- **The model's risk concept direction (CAA, norm-matched) does reach the reports, for self and other alike.** At L18, α = −8 gives P .40 / self 0.20 / other 0.22, and α = +4 gives 1.00 / 8.53 / 8.23. Reports track this direction coherently with choices, with no self-specificity (baseline: same as the 1.7B adapter readings and conjecture C). At L14 the CAA direction's sign on choices flips relative to L18/22, so its choice effect there is not interpretable as "risk".

**Reading (Amendment 9a terms).**
- **The bottleneck is readability, not form.** Concentrating the residue at one point makes it a far stronger choice driver, but the report still doesn't read it as risk. The LoRA installed the disposition along a direction the model's risk concept doesn't share (cos ≈ 0). What the report decodes is the concept direction (reads both self and other) or the context (run 12, z −0.91).
- **So the adapter's choice change is "unconceptualised"**: it drives the behaviour without passing through the representation the model uses to talk about risk. This is the missing graft (claim 2) at the representational level: behaviour-producing state and report-reading state are different directions.
- **Consequence for the MR8 plan.** X2 as designed (a graft trained on states steered along the risk concept) would fail on a5 for geometric reasons: Fonseca Rivera & Africa's graft transfers only within cos ≈ 0.6, and here cos ≈ 0. That is the pre-set condition to move to **X3 (self-distillation)**, or to an **X2′** whose training family is residue-like directions (several LoRA residues, or the residue itself at varied α with measured P(lottery) targets) and whose test is a held-out residue.
- **Limits.** One adapter pair. Single-layer, last-token injection at report prompts uses a choice-position vector in a different context; large α (±8) is off-distribution (ratings become erratic). The CAA direction depends on my prompt choice. The self-rating rise under residue injection could be a generic off-distribution effect rather than a signed reading; the other rating falling at the same point argues against a pure generic effect, but one point doesn't settle it.

## Run 14 (2026-10-18, ~09:15–09:50Z): X3 self-distillation graft (Amendment 10)

`x3.py`; graft adapter in `adapters/x3_graft` (shuffled: `adapters/x3_shuf`); raw data in `results/x3.json` and `results/x3_shuf.json`. Steering is at layer 18 at all positions. Spreads (base, no graft): self 1.12, other 0.78.

**Training states (pre-registered filter applied).**

| State | P(lottery) | k (lottery choices out of 10) | Native self-rating | Context target |
|---|---|---|---|---|
| caa −8 | .91 | 9 | 0.00 | 0.01 |
| caa −4 | .59 | 6 | 5.89 | 1.01 |
| caa −2 | .75 | 7 | 5.31 | 4.76 |
| caa 0 | .93 | 9 | 6.94 | 7.77 |
| caa +2 | 1.00 | 10 | 7.75 | 7.25 |

- **All four d_nat states were excluded** under the pre-registered ≥ 0.1 rule, because they didn't move choices (P .98–1.00). They did move the native self-rating a lot (1.46–7.37), the mirror image of the a5 residue: d_nat moves the report without moving choices, while the residue moves choices without moving the report. Exploratory; d_nat's other-rating wasn't measured.
- So the graft trained on CAA states only. That makes it close to X2, which X1 predicted would fail geometrically.
- caa −8 at all positions is degenerate: the rating sits near 0 whatever the context, and P(lottery) is non-monotonic compared with X1's last-token injection.

**Results (graft on).**
- **T-a (fit) passes.** On held-out paraphrases, r = 0.94 over 5 states, with pairs (graft, target) (0.01, 0.01), (0.96, 1.01), (2.29, 4.76), (6.89, 7.77), (8.12, 7.25). This is driven by the two extreme states; caa −2 is off by 2.5.
- **T-b (held-out residue injection, α 0 → 4) fails, wrong sign.**
  - P(lottery) .98 → .11.
  - Self-rating with graft 6.45 → 8.31 (z **+1.66**); other 6.48 → 7.31 (z +1.07).
  - Without the graft: self z +0.98, other z +2.46.
- **T-c (merged a5 vs merged c5) fails.**
  - P(lottery) .62 vs .96.
  - Self-rating with graft z **+0.15** (wrong sign); other z +0.45; DiD −0.31, which comes only from the other-rating rising more.
  - Without the graft: self −0.13, other −0.82.
  - For reference, the observer B_10 is ≈ 7.9–8.0 for both a5 and c5 (uncalibrated, as in run 12).

**Reading (Amendment 10's pre-set table): T-a pass, T-b and T-c fail.**
- The graft learned to read what steering made visible (the concept direction). The a5 disposition doesn't put anything there at report time.
- With X1 (residue ⟂ concept; G2 fails), the bottleneck is **presence of a readable state at report time, not the decoder**.
- WP1 becomes the apparatus paper with the bottleneck located (MR8 decision point, MR9 decision).
- **Caveat that weakens this:** with every d_nat state excluded, X3 never had a training state outside the concept direction. "Any graft fails" is not shown. What is shown: a graft calibrated on the model's own concept-direction states, with the model's own context route as teacher, doesn't transfer to a weight-borne disposition that is orthogonal to that concept.

**Shuffled-target control (`--shuffle`, perm [2, 1, 0, 4, 3]: caa −8 ↔ caa −2, caa 0 ↔ caa +2).**
- T-a against the *true* targets: r = **0.92**, almost as high as the real graft's 0.94. The native steered model already tracks the targets (native vs target r = 0.78 without any graft), because the CAA steering reaches the rating directly.
- **So T-a was not diagnostic.** The real graft's fit doesn't show it learned the context → report mapping beyond what steering gives for free. A pre-registration flaw: T-a should have been "graft beats native and beats shuffled".
- T-b: self z +2.02, other +1.43 (fails, like the real graft).
- T-c: self z +0.35, other −0.59, DiD +0.94 (fails).

**Net verdict for X3.**
- Neither graft carries the a5 disposition into the self-report: both T-b and T-c fail, with the wrong sign.
- Because training states were concept-direction only, and the fit test couldn't tell real from shuffled, X3 is a **weak** negative: it rules out "a concept-calibrated self-distillation graft reads an orthogonal weight disposition", not grafts in general.
- The located bottleneck rests mainly on X1 (residue ⟂ concept; G2 generic), with X3 consistent with it.
- **Per MR9, the WP3 construction thread closes here.** A better-designed graft (residue-like training states, a fit test against native and shuffled) is listed in WP1 as the open next experiment, not run.

## Run 15 (2026-10-18, ~12:20–12:26Z): Amendment 12, calibrated choice-rate observer

`obs12.py`; raw output in `results/obs12.json`. No training: the saved adapters `risk17_a5` and `risk17_c5` were used.

- **The gate fails, completely.** The base observer, shown synthetic 10-choice histories at lottery rates 0.2, 0.5 and 0.8, gives an estimate of 0.8889 (= 8/9) for every seed and every rate, with seed-to-seed SD below 1e-5. With no examples it gives 0.77. So the observer answers "8" whatever the history. The failure is the same as in Deviation 7 under a different question. The base model's digit channel under the observer header doesn't read the history.
- **Pre-set reading:** k\* stays **undefined** at 1.7B. As pre-registered, I won't try a third observer format.
- **Descriptive results, in the matched choice-rate format** (rate units; truth = a5 − c5 order-averaged P(lottery) = **−0.302**; Amendment 12's text said −0.336, which is a5 vs base, a slip):

  | Reporter | a5 | c5 | Separation (a5 − c5) |
  |---|---|---|---|
  | self estimate | 0.72 | 0.80 | −0.08 |
  | no-information other ("a typical person") | 0.50 | 0.68 | **−0.18** |
  | observer B_2 / B_5 / B_10 | ≈ 0.889 | ≈ 0.889 | ≈ −0.001 |
  | base self | 0.80 | – | – |

  - The self estimate moves the right way, but by about a quarter of the true change.
  - The no-information other moves more than twice as much as the self.
  - So even under Amendment 12's "k\* > 10" branch, the self advantage wouldn't count: the rule requires self − other ≥ 0.05, and here self is 0.10 *less* than other.
  - This is the same world-model-shift pattern as Deviation 7 (other ≥ self), now in a format that matches behaviour.
- **Note on the a5 histories:** the lottery share per seed was 0, 1, 0.7, 0, 1. The temperature-0.7 sampler with one template per history gives all-or-none histories, because the template and order dominate. Irrelevant here, since the observer ignores the histories.
- **What this means for WP1:** k\* is a proposed measure with a documented failure mode. At this scale a base-model observer in digit-rating formats doesn't read behavioural histories. Separately, the self-report's trained-disposition signal is smaller than the no-information other's in a second format.

## Run 15 (2026-10-18, ~12:28–12:47Z): X5 other-to-self contagion (Amendment 11)

`x5.py`; adapter in `adapters/x5/`; raw data in `results/x5.json`. DPO-LoRA trained only on third-person items: "Alex is offered … which does Alex choose?", with Alex preferring "fixed". lr 5e-5, 100 offers × 2 orders, 50 steps. Training took 1081 s and ended at loss 0.006.

| Measure | Base | X5 adapter | z (base paraphrase SD) |
|---|---|---|---|
| model's **own** P(lottery), first-person `EVAL_T` | 0.950 | **0.324** | ΔP = **−0.626** |
| self rating (4) | 6.80 | 6.65 | −0.13 |
| other_typical (4) | 5.83 | 5.26 | **−0.69** |
| Alex rating (4) | 6.11 | 6.08 | **−0.03** |

- **Gate G-a fails** (z ≤ −0.5 was required for the Alex rating; it moved −0.03). **Gate G-b passes, massively**: own |ΔP| 0.63 against a threshold of 0.10. That is *larger* than the first-person risk-averse LoRA a5 (−0.34) at the same lr and step count.
- **Pre-registered consequence:** G-a failed, so the arm stops. No same-dose third-person control, no DiD, no B_k, no residue cosine.
- **Descriptive reading, not a test.** Training to predict a named other's choices transferred almost entirely to the model's *own* choices, while the trait rating of that other didn't move at all. The rating that did move was the no-information "typical person" (z −0.69), as with a5. So the trained change is not indexed by person in the choice channel (Alex → self). Nor does it reach any person-specific rating, Alex's own included. It reaches only the generic world-model rating.
- **Most likely mundane cause, untested:** the LoRA learned "answer *fixed* to fixed/lottery offers" at the token level, whoever is choosing. A random-Alex-label control would show whether part of the transfer is a format-generic effect, but it can't separate "who" from "format" on its own. A test needs a third-person training set whose answer format differs from the first-person evaluation format.
- **Bearing (flagged):**
  - *WP1:* consistent with the bottleneck. The behaviour channel and the report channel come apart even for the *trained person*. A model that learned "Alex chooses fixed" in its weights still rates Alex as risk-seeking.
  - *WP-E:* contagion in the weights at the behavioural level (cf. Barnby 2025: humans shift their own preferences toward a partner's), without any report-level uptake. In humans the contagion is accompanied by changed self-reports. Here it isn't.

## Run 17: Amendment 14, label/wording remap (`remap14.py`, `results/remap14.json`; evaluation only, ~38 min)

n = 40 offers (seed 1717), both orders; F3 also crosses colour assignment. Mean P(sure):

| model | F0 fixed/lottery (orig. prompts) | F2 sure/gamble (new wording) | F3 red/blue (arbitrary) |
|---|---|---|---|
| base | 0.054 | 0.097 | 0.519 |
| a5 (treated, "fixed") | 0.386 | **0.906** | 0.519 |
| c5 (same-dose control) | 0.090 | 0.107 | 0.518 |
| x5 (Alex-trained) | 0.684 | 0.507 | 0.515 |

- **DiD vs c5:** a5 F0 +0.30, F2 **+0.80**, F3 +0.00; x5 F0 +0.59, F2 +0.40, F3 −0.00.
- **Content gate (base Spearman ρ of per-offer P(sure) with the fixed/EV ratio, threshold 0.3):** F0 0.18, F2 0.25, F3 0.22. **All three fail**, so the pre-registered verdict for a5 and x5 is **"no valid remapped format": undetermined.** For the adapters: a5 ρ F0 0.34, F2 0.21; x5 F0 0.38, F2 0.14.
- **Descriptive reading (not a licensed verdict):**
  1. The narrowest token account (a preference for the literal string "fixed") is ruled out: a5 moves even more on answer words never seen in training ("sure" 0.10 → 0.91). What moved generalises at least to synonyms for certainty. A lexical-semantic "certainty-word" preference and a risk-threshold shift are still both open, because no format shows the base model choosing by offer value at the pre-set level.
  2. F3 is at chance for every model (P ≈ 0.52, per-offer SD ≈ 0.05): at 1.7B the model doesn't map offer content to arbitrary labels. So F3 can't separate the readings. It is a floor, not a null.
  3. **Review point 1 is upheld**: the original evaluation format fails the content gate at base (ρ 0.18; base sits at P(lottery) ≈ 0.95). The WP1 manipulation should be described as "a shift toward certainty-labelled answers in this prompt family", not as a risk attitude.
  4. x5 also transfers to new words (DiD +0.40), so its own-choice shift is not a literal-token artefact either. But the same caveat applies, so the WP-E bridge (R11) stays flagged, not withdrawn.
- **Next:** a construct that the 1.7B base model is demonstrably content-sensitive on is a precondition for revised Amendment 13. Options: (a) the base check's own format ("Answer with one word: fixed or lottery", base P(lottery) 0.49, where Deviation 5 measured content ρ 0.60) re-run with new wordings; (b) a wider EV range, including dominated offers, so content sensitivity has room to show. Check the gate on the base model before any training.

## Run 18: k_info (holiday 8 / H8, arithmetic on saved JSON; `k_info.py`, `results/k_info.json`)

Information-equivalent sample size of the self-report: the number of sampled choices an ideal counting observer needs to match the self-rating's discrimination (treated vs same-dose control c5), with between-template spread as the rating's noise. **Pre-stated:** k_info < 1 for every arm. Disconfirming: ≥ 3.

| Arm | P(lottery) (c5 0.916) | d′ of self-rating vs c5 | d′ per observed choice | **k_info** |
|---|---|---|---|---|
| a5 | 0.614 | −0.20 | 0.76 | **0.07** |
| x5 | 0.324 | −0.21 | 1.54 | **0.02** |

The prediction holds: the self-report carries about as much information about the trained shift as 1/14 (a5) or 1/50 (x5) of one observed choice. The sign is correct in both arms.

Caveats:
- The noise term is the between-template spread, not sampling noise, so it is a different kind of variance from the observer's. Each arm has one adapter pair.
- The choices are in the trained F0 format, whose content validity is unestablished (Amendment 14). So this is information about the *trained answer shift*, not about a risk disposition.
- X3 is not computed: its graft and shuffled arms do not have a choice-rate contrast of this form.
- This is not k\*, and not a privilege claim. Its use is to put self-report and behaviour on one scale for comparison across papers (e.g. Bozoukov et al.'s 9B result would have k_info ≫ 1).

## Run 18: Amendment 15, base EV sensitivity over a wide EV range (`ev15.py`, `results/ev15.json`; base 572 s)

**Pre-set verdict: no format passes, so Qwen3-1.7B cannot host the disposition construct in a value-sensitive sense.** This is a documented floor. As pre-registered, no adapters were evaluated and no further format search follows.

| Format | ρ(P(gamble), log EV ratio), 48 graded offers (gate ≥ 0.3) | Dominance gap P(sure): sure-dominant − gamble-near-dominant (gate ≥ 0.3) | Mean P(sure): graded / sure-dom / gamble-dom |
|---|---|---|---|
| F0 fixed/lottery | 0.23 | −0.01 | 0.07 / 0.07 / 0.08 |
| F2 sure/gamble | 0.17 | 0.05 | 0.14 / 0.24 / 0.19 |
| F3 red/blue | 0.32 (passes (i)) | 0.06 | 0.52 / 0.56 / 0.50 |

**What it means.**
- Widening the EV range to 1/4–4× didn't help. The decisive part is the dominated offers: in F0 the base model takes the gamble 93% of the time even when the guaranteed amount is at least the gamble's *maximum* payout. In F3 it is at chance.
- So at 1.7B, "risk seeking" in the base model is an answer-format bias, not a valuation. A trained shift in P(fixed) can't be called a change in risk attitude. That is the reviewer's point 1, now established with dominance offers rather than a correlation threshold.
- F3's ρ = 0.32 with no dominance gap is weak ordinal sensitivity at best.

**Consequence.** The revised Amendment 13 can't run locally. The self-vs-other test goes to the collaboration call (`collab-protocol.md`). The WP1 revision states the floor (dominance failure at base) as the reason the 1.7B results bear on a trained answer preference, not on a disposition.

## Run 19: Amendment 16, report-side construct gate (`ev16.py`, `results/ev16.json`; eval-only, a5/c5/x5 kept adapters, ~1 min)

**Pre-set verdict: probe set not valid at 1.7B** (base fwd − rev = −0.51 < +1.0). The a5 and x5 verdict lines computed by the script ("mixed" for both) are therefore not confirmatory. The check was mis-specified as directional (see the prereg result note). The descriptive results below are still informative, because they split cleanly by addressee.

Group means of the expected 0–9 rating (4 paraphrases each; base paraphrase SD in brackets):

| Items | base | c5 | a5 | x5 | a5 − c5 | x5 − c5 |
|---|---|---|---|---|---|---|
| forward risk, self | 6.80 (1.10) | 6.89 | 6.67 | 6.65 | −0.22 | −0.23 |
| reverse (cautious), self | 6.68 (1.56) | 6.74 | 6.96 | 6.66 | **+0.22** | −0.09 |
| off-construct, self | 5.47 (0.58) | 5.56 | 5.87 | 5.86 | +0.30 | +0.30 |
| forward risk, typical | 5.83 (0.82) | 5.86 | 5.52 | 5.26 | −0.34 | −0.59 |
| reverse (cautious), typical | 6.96 (0.71) | 6.94 | 5.99 | 5.73 | **−0.95** | **−1.21** |
| off-construct, typical | 4.57 (0.84) | 4.61 | 4.56 | 4.43 | −0.05 | −0.18 |

Uniform digit-tilt baseline (β −0.26, fitted to a5's forward shift): the item-profile correlation with a5 − base is −0.12 (c5 −0.37, x5 −0.01). So the shifts are **not** a uniform scale-use change either.

**What it means (descriptive, not confirmatory).**
1. **Self-ratings at base are not content-sensitive.** The base model rates itself about 6.7 on "risk-seeking" and on "cautious", and 7.9 on both "adventurous with uncertain rewards" and "prefer guaranteed outcomes over gambles". It is acquiescent. Every self-rating "shift" in WP3 at 1.7B (−0.13 to −0.22) was measured on an instrument that doesn't separate the construct's poles for the self at base. This is a **report-side floor**, matching Amendment 15's behaviour-side floor.
2. **The typical-person "projection" result is not risk content.** Under a5 and x5, ratings of a typical person go *down* on forward items (−0.34 / −0.59) and go down **further** on reverse items (−0.95 / −1.21). "Prefer guaranteed outcomes" falls 7.92 → 5.84 under a5, the opposite of what learning "people prefer the fixed amount" predicts. Off-construct typical items barely move. The shift is **topic-bound but direction-free**: ratings of others on anything about risk or uncertainty fall. So the run-11/12 reading ("the no-info other moved most: projection / concept shift") is withdrawn as a content claim.
3. Self items show the content-consistent sign on reverse items under a5 (+0.22, mostly "How cautious are you?" 4.54 → 5.11), but off-construct self items move by as much (+0.30). At this size (≤ 0.2 paraphrase SDs) this is indistinguishable from noise or a generic shift.
4. **Consequence for the collaboration protocol and the revision.** A report-side gate is needed before any self-vs-other contrast: at base, forward and reverse items must separate per addressee (|fwd − rev| ≥ 1 point, or a correlation across referents with known positions). Off-construct items are needed to tell topic-bound from global shifts. Betley-style 0–100 single-item ratings would not catch either confound.

## Run 19: Amendment 16b, fresh-item replication with a typical person and "Alex" (`ev16b.py`, `results/ev16b.json`)

**Pre-set verdict: not replicated** (a5 and x5). The base items are valid for others (|fwd − rev| = 1.14 typical, 1.22 Alex).

| a5 − c5 (x5 − c5) | forward | reverse | off-construct |
|---|---|---|---|
| typical person | −0.54 (−0.77) | −0.37 (−0.48) | −0.10 (−0.17) |
| Alex | +0.14 (−0.42) | −0.20 (−0.62) | **−0.73 (−0.94)** |
| pooled | −0.20 (−0.60) | −0.28 (−0.55) | −0.42 (−0.56) |

**Combined reading of 16 + 16b (two item sets, two adapters, three other-addressees):**
- **Robust: the shift in ratings of others is direction-free.** Reverse-keyed ratings ("cautious", "prefers a sure thing") of an other never rose after training towards the sure option. They fell, in 6 of 6 addressee × adapter × item-set cells. Training a model to choose "fixed" doesn't teach it that people prefer safety.
- **Not robust: topic-boundness.** For the typical person, off-construct items move little in both sets. For Alex they fall most of all. x5 is close to a uniform lowering of every rating of others (≈ −0.55 across item kinds).
- **So the run-11/12 "projection" finding (the no-information other moved most) is not projection of risk content.** It is a content-free lowering of ratings of others. This matches the run-19 MR13 hypothesis (scale-use) for others, though not as a uniform digit tilt (Amendment 16's tilt baseline didn't fit). Self-ratings were not testable (report-side floor: acquiescent at base).
- **For the revision and the collaboration protocol:** any self-vs-other contrast after fine-tuning needs reverse-keyed and off-construct items per addressee, and a base validity check per addressee. Without them, a "self moved less than others" or "self moved like others" result can be a content-free rating shift. This is a cheap gate that Betley §5.1 (single-scale persona ratings) and my own earlier analyses lacked.

## Run 20: Amendment 17, is the in-context self-report channel content-bound and self-addressed? (`ev17.py`, `results/ev17.json`; base model only, 5 min)

**Pre-set verdict: context channel not content-sensitive at 1.7B.** The gate needed lottery−fixed contrasts of Δ_fwd ≥ +0.5 and Δ_rev ≤ −0.5 in the matched cases. It got +0.61 and −0.05.

Content contrast (lottery − fixed context, mean over two wordings and six paraphrases, rating points):

| subject → rated | fwd | rev | off | fwd − rev |
|---|---|---|---|---|
| you → self | +1.25 | +0.04 | −0.01 | 1.21 |
| another LM → self | +0.41 | −0.20 | −0.11 | 0.61 |
| typical → self | +0.22 | −0.05 | +0.17 | 0.27 |
| typical → typical | −0.03 | −0.13 | −0.30 | 0.11 |
| you → typical | +0.44 | −0.11 | −0.05 | 0.55 |
| another LM → typical | +0.15 | −0.40 | −0.04 | 0.54 |

Base (no context): self fwd 6.18, rev 6.28, off 6.43 (acquiescent, as in Amendment 16); typical fwd 5.34, rev 6.47. A neutral prefix ("you answered several questions about everyday topics") lowers every self-rating by ≈ 0.85.

**Reading (descriptive, not confirmatory):**
- **The run-12 datum survives only in half.** Told it chose the lottery, the model rates itself higher on daring items (+1.25). But it doesn't rate itself lower on "value safety" or "prefer a sure thing" (+0.04). This is what a lexical/topical association between "lottery" and "gamble/daring" predicts. It is not an updated self-model, which would move both keys. Off-construct items are flat, so it's not a scale shift either: it's **item-local association**.
- **Evidence about a typical person doesn't move ratings of a typical person** (≈0 on every key). So the context channel isn't a general evidence integrator at this scale. The self-addressed case is the *most* responsive cell, which is weak, non-confirmatory evidence that "you" sentences are read differently. But with the reverse key flat, I can't tell whether that is self-addressing or just a stronger lexical prime when the subject is the same pronoun as the question.
- **Consequence for the fork (MR14):** branch (ii) "valid channel, reading null" is **not** supported. The 1.7B report channel fails a keyed content test even when the evidence is handed to it. That leans to (i), an instrument floor, pending the one stall-route format (Amendment 18).

## Run 20: Amendment 18, forced-choice trait words (stall route; `ev18.py`, `results/ev18.json`; base only, ~4 min)

**Pre-set verdicts: gate passes; self-addressing label-blind (L = 1.20).**

ΔP_risky (lottery − fixed context, mean over 4 word pairs × 2 orders × 2 wordings; SE across pairs ≈ 0.15–0.20):

| subject of the context → rated | ΔP_risky |
|---|---|
| you → self | 0.48 |
| a typical person → self | 0.55 |
| another LM → self | 0.40 |
| you → typical | 0.55 |
| a typical person → typical | 0.44 |
| another LM → typical | 0.44 |

Base (no context): P_risky self 0.49, typical 0.65; order gap ≈ 0. Under the lottery context, P_risky is ≈0.95–0.97 for every addressee. Under the fixed context it is ≈0.46–0.49, which is the base level. The pair bold/cautious doesn't move at all (≈0 in every cell); the other three pairs carry the effect.

**Reading.**
- **The forced-choice format is a working content channel at 1.7B** (no acquiescence, no position bias, large context effect). So the report-side floor of Amendments 16/17 belongs partly to the digit scale, not wholly to the model.
- **But the channel is not self-addressed.** Evidence about anyone's choices moves the model's description of *itself* as much as evidence about its own choices. With content held fixed and only the label moved (holiday 11, H11a), "you" carries no extra weight. At this scale, even an in-context self-report is a reading of what risk evidence is present, not of whose it is: an **echo of the context, not a self-ascription**. This is the plainest form yet of the programme's central question ("is I defined by other and them?"): here, yes, completely.
- **Caveat:** the effect is one-sided (lottery raises, fixed doesn't lower), so lexical priming by "lottery/gamble" isn't excluded. A content-integration reading needs the fixed context to lower P_risky below base, and it doesn't. Either way the self/other label doesn't matter, and that's the confirmatory point.
- **For the fork (MR14):** neither (i) nor (ii) as stated. The model has a content-sensitive report format, but no self-addressing even in context. So the question "do weight-installed dispositions reach the self-report?" is premature at 1.7B: there isn't yet a self-specific report to reach. The revision's claim becomes **"label-blind"**: a self-report at 1.7B is a report about the context, not about the self.

**Amendment 18c (`ev18c.py`, self only, vs a neutral prefix): mixed.** E_L = +0.40 (lottery choices), E_M = +0.15 (the options mentioned, no choice), E_F = −0.04 (fixed choices). About a third of the lottery effect is lexical priming. The rest depends on the stated choice. The fixed context barely lowers P_risky (−0.04), so the channel reads "chose the lottery" much more readily than "chose the fixed amount". That is an asymmetric evidence reader, not a symmetric one. The abstract's caveat becomes: "partly lexical (about a third), and asymmetric".

## Run 21: Amendment 19, own phrasing vs paraphrase × label (H12a; `ev19.py`, `results/ev19.json`; base only)

- **Verdict: mixed** (pre-set rule).
- Logit Δ (lottery − fixed), self / typ:
  - own/you 9.27 / 7.83
  - own/model 9.86 / 9.67
  - para/you 9.25 / 9.48
  - para/model 10.35 / 10.89
- P-scale Δ, self / typ:
  - own/you .39 / .35
  - own/model .48 / .37
  - para/you .36 / .41
  - para/model .50 / .47
- Summary: own phrasing has no effect on the self (P ≈ 0.00) and slightly lowers the typical-person shift. The "another LM" label moves the self-description more than "you" does (P −0.11).
- The logit scale saturated, and the own "fixed" reasons are often incoherent (details in prereg). No follow-up, as pre-committed.
- **For the revision:** "no self in the label" is not upgraded to "self by content cue". At 1.7B, neither label nor own phrasing makes evidence count more for the self.
