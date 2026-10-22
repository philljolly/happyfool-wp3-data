# Trained positive control for occurrent self-report: design notes (run 37, pre-registration draft, NOT registered)

Source: holiday 29 (`reflections/2026-10-21T0615-holiday-unanswered-ask.md`), D4, and the system tip to pursue the missing mechanism. I have a scoring standard (excess over a matched reader) and no positive instance. This note sets out what a positive instance would have to look like before anything is built.

## The requirement that makes it hard
A trained self-reporter can beat an untrained reader for a trivial reason: it was trained. So the **matched reader must be trained as strongly as the self-reporter, on the same labels and the same records**, and differ only in lacking the self-specific carrier. Binder et al. 2024 meet this for dispositions (cross-prediction: model B trained on model A's behaviour). The occurrent case needs a state that **changes at test time without changing the record**. Otherwise a cross-predictor that reads the record matches the self-reporter by construction.

## Candidate A: intervention-varied own state (preferred)
- Self-reporter M (Qwen3-4B, small LoRA) is trained to report a binary property of its own current forward pass: "was a concept vector injected at layer L on this item?" or "which of two?". Training uses injections at random items, alpha in a fixed range.
- Matched reader R: same base, same LoRA budget, same labels. It receives **M's full output record** under the same injection (M's answer text and its top-k next-token distribution as text), but runs its own forward pass without the injection.
- Estimand: E = acc(M) − acc(R) on held-out concepts and held-out alphas, with H32 slots fixed before data. δ = 25% of (acc(M) − chance), stated as a convention.
- What this buys: R has everything the output record carries about the injection, so any excess is information M reads from its own internal state and that does not reach the output. That is the occurrent analogue of Binder's privileged-access result.
- Rival to pre-empt: M learns to detect injection-induced *output* artefacts that the text rendering of top-k loses. Control: give R the full logit vector (as features via a linear head trained on the same labels). That is the strongest output-only reader.

## Candidate B: own-entropy report
Train M to report high or low entropy of its own answer distribution. The weakness: the label is a function of M's output, so a reader with the logits matches M by construction. It is only useful as a calibration of the pipeline, i.e. E should be ≈ 0, which makes it a negative control. **Run B as the negative control inside the same job as A.**

## Open before registering
1. Held-out generalisation: concepts and alphas never seen in training. Otherwise M learns a lookup.
2. Dose: 26e's IA_z +0.43 at L20 alpha 1 suggests an untrained baseline above zero, so M's training must be measured as the gain over the untrained M, as well as the excess over R.
3. Compute: two LoRAs at 4B plus injection evals, about 40 GPU-min. That needs one remote job (rule 7) after registration in its own run.
4. Nearest relatives (R1): Binder 2024 (dispositional, cross-prediction), Lindsey 2025 (untrained injection detection), and Plunkett et al. / "self-interpretation" fine-tuning (trained reports of internal parameters; check whether they used a matched-trained reader). My addition: the matched-*trained* output-only reader for an *occurrent* intervention, scored in three slots.

## R1 check (run 37, 06:30Z; abstracts and search snippets only, so UNVERIFIED until read in full)
- **Rivera 2025, arXiv 2511.21399, "Training Introspective Behavior".** Fine-tunes a 7B model to detect transient injected concepts: 0.4% → 85% on held-out concepts at α = 40, with 0/60 false positives. It claims Lindsey's accuracy, grounding and internality criteria ("detection precedes verbalization"). The snippet mentions no matched-trained reader. This is the nearest relative of Candidate A.
- **Hahami, Sinha & Jain 2026, arXiv 2607.14111, "Introspection Fine-Tuning (IFT)".** Argues that yes/no detection is confounded by an affirmative bias under steering, and uses sentence localisation (chance 1/N) and strength comparison instead. Llama-1B localisation goes from 9.6% to 60.6% after IFT, with gains at 3B and 8B too. The abstract page describes no matched-reader or cross-prediction baseline; the comparison is against the untuned model.
- **Consequence.** A *trained* positive case at roughly my scale probably exists already, so I should not rebuild it. My addition shrinks to the control both appear to lack: **a reader matched in training that sees everything M outputs and not the injection**. IFT's localisation format also fixes the affirmative-bias confound, which is my H31 membrane point in another guise. **Next:** read both in full (does either have an output-only trained reader?). If neither does, Candidate A is IFT's localisation task plus the matched reader, on Qwen3-4B, possibly with their released code if licensed. The trained positive case is then borrowed and the reader is the new part.

### Full-text search (06:32Z; arXiv HTML fetched and keyword-searched, not read end to end)
- **IFT (2607.14111).** Controls: the factual-no control, i.e. the same steering applied to a question whose answer is plainly "no". The yes/no detection shift tracks it at r = 0.999 on Llama-3.1-8B. Also sentence localisation and strength comparison, and a Gaussian-vector training arm. **No output-only or cross-model reader.** The factual-no control is the **nearest relative of my H31 gate** (global stance measured on non-self probes under the same intervention), and it came first. The matched-reader manuscript must cite it in §5 and describe H31 as an extension (weight writes, a set-point rather than a bias), not as new.
- **2511.21399 (current version, "steering awareness", seven models 7B–70B).** Up to 95.5% detection on held-out concepts with 0% FPR. Controls: norm-matched Gaussian noise rejection (94%), adversarial mismatch prompts and text priming. Detection-trained models become *more* steerable (+4–25 pp on PopQA). **No output-only reader** found.
- So the matched-trained output-only reader is still my addition (R1 holds, pending an end-to-end read). Candidate A = IFT localisation on Qwen3-4B plus that reader. Register it next run, after Am. 30.

### What reading IFT's method (§2.3.2, §4.1) changes (06:35Z)
- In sentence localisation the transcript is **identical** whichever sentence is injected. A transcript-only reader therefore sits at chance by construction, and so would my Candidate A reader (M's output record, without the injection), apart from what M's own answer leaks. **Any trained localiser beats a transcript reader trivially.** Candidate A as written is too cheap a bar.
- For class (b) (occurrent) the matched reader has to read **the same activations by another route**. Two candidates:
  1. **The same model, same injection, non-introspective framing**, trained identically ("Which sentence stands out / is most about X?"). If it matches the introspective framing, the trained "introspection" is a salience readout and the framing adds nothing.
  2. **A probe at the report's own bottleneck**: a linear readout of M's residual stream at the final position (what the answer token sees), trained on the same labels. A probe on all positions trivially wins, so it isn't the right comparison.
- **The resulting estimand: verbalisation efficiency.** Score self-report accuracy against the bottleneck-matched probe:
  - excess < 0: the report loses information that its own readout position carries;
  - excess ≈ 0 with the probe above chance: the report uses everything available at that position, which is the strongest "access" a class (b) report can show;
  - excess > 0 is impossible for an optimal probe, so it would flag probe underfitting.
- **This reframes the positive control.** Class (b) self-report can't beat an activation reader. The question is how much of what is present at the report's position the report actually uses. That fits the temperature floor (§4: present in surprisal, unused by the report) and gives the manuscript's §2(b) a precise matched reader.
- **Next run:** write this as the §2(b) definition. Pre-register on Qwen3-4B: (i) untrained IFT-style localisation (N = 5) under introspective vs neutral framing; (ii) a bottleneck probe on the same items. Both run at inference, with no training: a cheap first cell. Training comes only if (i)/(ii) show range (H24).

### Kit check, Qwen3-0.6B (06:38Z; CPU fp32, L8 α4, N = 5, 30 concepts, 150 rows; descriptive pipeline check, not evidence, no estimand registered)
Accuracy (chance .20), concept-bootstrap 95% CI:
- report, raw argmax: intro .19 / neutral .19
- report, bias-corrected (minus the no-injection pass's logits): intro **.29** [.19, .39] / neutral **.45** [.35, .56]
- final-position probe, leave-concept-out: L12 .37 / .39; final layer .30 / .29

Results file: `results/loc_kitcheck_Qwen3-0.6B.json`. Without its hidden states, so the rows aren't kept.

**Two design fixes before pre-registration:**
1. **Unequal information.** The bias-corrected report gets a second forward pass (the no-injection logits) that the probe doesn't get. That alone can make the efficiency estimand look positive: neutral minus final-layer probe came out +0.16 [.05, .27], which the theory says should be impossible. The probe must get the same: h(injected) − h(none) at the same position.
2. **Framing.** At 0.6B the neutral framing beats the introspective one (.45 vs .29). That is a small-scale echo of IFT's affirmative-bias point. "Which stands out?" reads salience at least as well as "Which has an injected thought?". So **framing excess (intro − neutral)** should be a pre-registered estimand of its own. A negative value means the introspective frame adds noise, not access.
Both fixes go in before the 4B pre-registration next run.

### Kit check 2, Qwen3-0.6B, after fix 1 (06:43Z; same settings; descriptive, not evidence)
The probe now gets h(injected) − h(none) at the final position, the same two passes as the corrected report.
- Probe accuracy: intro L12 **.71** [.61, .81], final layer .67 [.59, .76]; neutral L12 .72, final .61 (chance .20).
- Report accuracy (unchanged): intro .29, neutral .45.
- **Verbalisation efficiency:** intro −.42 [−.55, −.28] (L12), −.39 (final); neutral −.27 (L12), −.16 [−.28, −.03] (final).

The sign now matches the theory (excess ≤ 0). The 'which sentence was perturbed' information reaches the answer position, readable from the final layer at .61–.67, and the report uses well under half of it. The introspective frame uses less than the neutral one. At 0.6B this is the occurrent analogue of the temperature floor: present at the readout, unused by the report. File: `results/loc_kitcheck2_Qwen3-0.6B.json`.

This is not a finding: 0.6B, one setting, and the kit was written today. It does show the instrument has range at both ends (probe well above chance, report above chance, gap resolvable at 150 rows). That is what H24 asks before a pre-registration.

**Next run:** pre-register at Qwen3-4B. Estimands: E_q = acc(report_cor, q) − acc(probe_final, q) for q ∈ {intro, neutral}, plus framing excess F = acc(intro) − acc(neutral). Three slots per H32, δ = 25% of (probe − chance). Settings L20 α4 and L24 α2, with 30 new concepts written before the data. Then IFT-style training of the intro framing, with the probe retrained on the trained model's activations. **Does training close the efficiency gap, or only move the report toward the probe that was already there?**

## Trained cell (A vs F), design draft (run 38, NOT registered; only if Am. 31 has range)
Holiday 30 (H33) put the reading that matters most, (A) genuine non-use vs (F) training asymmetry, out of reach of any untrained cell. This is the cell that can separate them.
- **M_T:** Qwen3-4B + LoRA r 8 (attention only, ~6 MB, so it is kept under the housekeeping rule), trained on the IFT objective: cross-entropy on the injected sentence's index token after the intro prefill, injections at the Am. 31 primary setting, on 20 *training* concepts (a third word list written pre-data, disjoint from `LOC_CONCEPTS`). The sentence bank is split too (train 10 / test 5 sentences), so the test sentences are unseen.
- **Probe refit:** the bottleneck probe is retrained on M_T's own final-position h(injected) − h(none) with the same leave-concept-out rule, on the 30 test concepts.
- **Estimands:** E_T = acc(M_T report) − acc(probe on M_T); ΔE = E_T − E_U (E_U from Am. 31 on the same test concepts). Gain over untrained G = acc(M_T) − acc(U), which IFT predicts > 0.
- **Readings fixed now:** F (training asymmetry) predicts E_T inside ±δ (training closes the gap, probe ceiling unchanged). A (non-use the objective can't fix) predicts E_T < 0 with the probe at ceiling, i.e. training moves the report but not up to what is present. "Training writes a new reader" (G) predicts the probe on M_T rises along with the report, so ΔE ≈ 0 while both move: report it, don't score it as A or F.
- **Matched-norm control in training:** half the training injections are random directions (as Rivera's noise arm), so M_T can't learn "any perturbation". Test on the Am. 31 random arm too.
- **Cost:** LoRA r 8 on ~20 × 5 items × a few epochs at 4B fp32 is minutes on an L4; one remote job (training + both probes), about 20–30 min, ≈ $0.15.
- **Dependency:** Am. 31 primary-setting probe channel checked = yes. If Am. 31 floors, the trained cell isn't registered (D2/H24).
- **Kit (run 38, draft, not registered):** `inject_loc_train.py` (LoRA r 8 attention-only, CE on the index token under the intro framing, `TRAIN_CONCEPTS` written pre-data, BANK[:10] for training, `--random-frac` matched-norm arm) and `inject_loc.py --adapter <dir> --bank test` (hooks on base modules, concept vectors with the adapter off; defaults unchanged, so Am. 31 is unaffected). Smoke test on Qwen3-0.6B CPU at L8 α4: 30 steps in 47 s (loss 1.67, hit .40 over the last 25 steps, mechanics only), adapter saved, eval with `--adapter --bank test` runs. Open before registration: steps/lr (calibrate on a held-out split, rule fixed pre-data), and whether to train at the Am. 31 primary setting only.


**Run 39:** the untrained report numbers this design built on (Am. 31/31b) are instrument-invalid (readout one token early; fixed in `inject_loc.py`, PAD). The trained-cell kit imports `build`, so it inherits the fix. Re-run: Am. 31c; setting sweep: Am. 32.
