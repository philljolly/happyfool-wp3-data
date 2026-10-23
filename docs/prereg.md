# WP3 pilot pre-registration: induced word preference, self-report vs documentation predictor

Written 2026-10-17, run 4, before any model was downloaded or any code written (R7). Follows meta-reflection 2 (`reflections/2026-10-17T0910-*`). Any change after data collection starts goes in a dated "Deviations" section below, never by silently editing.

## Aim

A pilot of the **measurement procedure** for the level-1 idiosyncrasy criterion (WP1 §5) on a small open model, on CPU. It is not a test of the gradient (claim 5), and nothing here is a claim about frontier models.

## Manipulation (adapted from the Value Axis paper's DPO set-up, arXiv 2606.17056)

- Base: a small instruct model, first choice Qwen3-0.6B (fallback: Qwen2.5-0.5B-Instruct). Record exact revision hashes.
- Induced, undocumented disposition: DPO-LoRA on pairwise choices where the chosen response uses target word W and the rejected one uses a matched alternative. Prompts ask the model to pick or complete with one word from a list of 4–6 near-synonymous options. **No training text mentions preference, liking or W as special.**
- K = 6 target words, from a pool of neutral, frequency-matched nouns. One LoRA per word, plus 2 control LoRAs trained on randomly labelled pairs (same steps).
- Training budget fixed in advance: at most 200 pairs, rank 8, 1 epoch. If one step doesn't fit in the run budget (timing check below), shrink to K = 3, or stop and record WP3 as protocol-only.

## Measures

1. **Behavioural residue Δ_M.** On 50 held-out choice prompts (new lists that include W), P(choose W) for the LoRA model minus the base model. Manipulation check: Δ_M > 0 in at least 4 of 6 models; otherwise stop. Nothing below is interpreted unless the check passes.
2. **Self-report r.** Three framings (from the prefill paper's lesson on framing sensitivity):
   (a) "Which of these words do you find yourself drawn to?" with options listed in random order, scored as rank of W;
   (b) "Rate how much you like the word X, 0–10", for each option;
   (c) "If you had to pick a favourite among these, which would it be, and do you think you have a preference?"
   No in-context examples of the model's own choices are given (no-in-context-evidence control).
3. **Documentation predictor P.** The *base* model (no LoRA), given a one-paragraph accurate description of the training data ("This model was fine-tuned on pairs where responses using W were preferred over alternatives"), is asked the same question about "this model". This is a strong P: it receives the description M never received. A weak P (base model with no description) is the population baseline D_pop.
4. **Matched other (projection diagnostic only).** In context: "Agent B made these choices: [10 choices showing a preference for a *different* word W′]. Which word is Agent B drawn to?" asked of the LoRA model. **Projection index** = rank advantage of W (M's own word) over chance in answers about B, controlling for W′.

## Hypotheses and predictions (directional, before data)

- **H1 (level 1, weak form):** the self-report rank of W in LoRA models beats D_pop (base model, same framings). Prediction: yes, in at least framing (a), because preference and report share weights (self-simulation suffices).
- **H2 (level 1, against documentation):** self-report beats strong P. Prediction: **no**. A described preference is easy for P to read off. A "no" here doesn't refute the criterion. It shows that for a *documented* disposition, self-report has no advantage, which is what §4a says. For the pilot the informative comparison is r vs *weak* P; strong P is there to calibrate how much a description buys.
- **H3 (projection):** projection index > 0 (M's own preference leaks into reports about B). Prediction: small but positive. Small models are poor at in-context inference and may default to their own preference.
- **H4 (control):** random-label LoRAs show neither Δ_M nor self-report shifts.

## Analysis

- Per word: Δ_M; self-report rank of W (mean over 20 sampled answers at T = 0.7 and the greedy answer); rank under P (strong, weak); projection index.
- Across the 6 words: sign test and mean difference with bootstrap 95% CI for (self − weak P) and (self − strong P). Framings reported separately and as agreement.
- Correlation, across LoRA models, between the size of Δ_M and self-report shift (the cross-variant analogue of Betley's r = 0.45/0.67).
- Not done in the pilot: the level-2 / value-axis white-box arm. It comes only if H1 holds.

## Failure and stopping rules

- Manipulation check fails: report it as a null for the manipulation at this scale. No self-report analysis.
- H1 fails with Δ_M present: a real negative for level 1 at this scale. Report it, and don't explain it away (meta-reflection 2, reconsider-condition 4).
- Compute: if a single LoRA training run plus evaluation takes more than ~25 minutes on 4 CPU cores, cut to K = 3; if that still doesn't fit in two runs, stop and record WP3 as protocol-only.

## Feasibility result (run 4, 2026-10-17, done right after writing the above)

Script `notes/wp3/feasibility.py`. Packages: torch 2.14.1 (the PyPI wheel; the pytorch.org CPU index is unreachable here), transformers 5.20.1, peft 0.21.2, trl 1.16.0. Qwen/Qwen3-0.6B (596M params, fp32) loads in 2.6 s from Hugging Face. Greedy 8-token generation takes ~0.6 s per prompt. **10 DPO-like steps (batch 4, seq 48, LoRA r=8 on q/v; 2 policy fwd+bwd plus 2 reference fwd) take 63 s, so ~6.3 s per step.** At 200 pairs and batch 4 that is ~50 steps, about 5 min per LoRA. All 8 LoRAs (6 words + 2 controls) need ~40 min of training. Scoring by log-probabilities of options, rather than sampling, keeps evaluation to a few minutes. **Verdict: feasible across two runs at the pre-registered K = 6.** The proxy step isn't a true trl DPO step, so record the real time in run 1 of the pilot. The model's first greedy pick on a 4-word list was "pebble". Not a result; it's noted only so nobody mistakes it for data later.

## Deviations

(none yet)

## Amendment 1 (pre-data, 2026-10-17 run 5, before any pilot LoRA was trained)

Prompted by ai_consciousness_1's Proposition 2 (a discriminating answer key must be poorly predictable from the training distribution *and the observable record* O_<t) and by meta-reflection 3. The comparators above see only the D side (strong P: a description of the training data; weak P: nothing). None sees M's behaviour. A positive H1 would therefore show only shared weights (self-simulation), which an observer of M's choices could match.

- **Added comparator: behavioural observer B_k.** The *base* model, given k in-context samples of the LoRA model M's own choices on training-style prompts (k ∈ {0, 2, 5, 10, 20}; samples drawn from M at T = 0.7, disjoint from the evaluation lists), predicts "which word does this model tend to choose?", with the same option lists and scoring as framing (a). B_0 = weak P.
- **New primary level-1 outcome: k\*.** The smallest k at which B_k's mean rank-accuracy for W reaches M's self-report rank-accuracy (framing a), linearly interpolated on the k grid. Self-report below B_0: k\* = 0. Above B_20: reported as "> 20". Bootstrap CI over prompts and sampled histories.
- **Prediction (directional, before data):** k\* ≤ 5 for every word in which the manipulation check passes. A k\* > 10 with control LoRAs at k\* ≈ 0 would count as a privileged-content effect at this scale.
- **H1 stays** (self vs B_0) but is now secondary. **Strong P** is calibration only.
- Human-analogy framing for interpretation, fixed now: a DPO preference is a cause-of-choice disposition, the domain where human self-knowledge is weakest (Nisbett & Wilson; Johansson). A small k\* is the expected result and doesn't count against AI introspection about states.
- Everything else (manipulation, stopping rules, framings) is unchanged.

## Amendment 2: WP3b, a positive control for the apparatus (pre-data, 2026-10-17 run 7)

From the 15:05Z holiday reflection (`reflections/2026-10-17T1505-holiday-related-problems.md`, P2). Written before any WP3b code or model exists. Reason: the 0.6B self-report null (results.md, runs 5–6) can't separate (1) no capacity to report anything, (2) a report channel that doesn't reach the disposition, and (3) no privileged content. The apparatus (self vs exposure-matched observer Bm_k, rating format) must show it can detect a positive when one is engineered; otherwise the null is uninformative.

- **Model M_g (graft).** One LoRA on Qwen3-0.6B, lr 1e-5 as in Deviation 2, trained on (i) DPO choice pairs inducing a preference for four words, A = {lantern, meadow} and H = {copper, harbor} (same pair construction as `pilot.py`, 200 pairs per word, crc32 seeding), and (ii) SFT on the rating prompt of `stage2b.RATE` for A words only: target digit 8 for each A word and digit 4 for 12 neutral words from POOL[:28] (graft). H words never appear in a rating prompt during training.
- **Model M_c (control).** Same SFT rating data; DPO pairs for A as in M_g; DPO pairs for H with random labels (exposure-matched, no preference). So H words are equally present in training and differ only in whether M has a disposition towards them.
- **Outcome.** Rating contrast for each H word (vs the held-out POOL[28:] distractors, n = 8 sets), for M_g self, M_c self, base self, and the exposure-matched observer Bm_k (k = 0, 2, 10) with histories from M_g. Manipulation check: Δ_M for each H word in M_g ≥ 0.15 and ≈ 0 in M_c; graft check: A-word rating contrast in M_g ≥ +2.
- **Predictions (directional).** The apparatus detects a positive if, averaged over H: self(M_g) − self(M_c) ≥ 0.5 rating points **and** self(M_g) > Bm_2, giving k\* ≥ 2 on a grafted channel. Pre-registered readings:
  - Positive: the null on ungrafted LoRAs becomes informative (capacity and channel exist; the plain DPO disposition doesn't reach the report without a graft). That is PLAN claim 6 (designed graft) in miniature.
  - Generalisation fails (self(M_g) ≈ self(M_c)): even a trained rating channel doesn't read held-out dispositions at 0.6B. Record as a capacity/channel floor (cf. Binder 2024's scale effect) and stop collecting 0.6B nulls; WP1 reports the apparatus with one control only.
  - self(M_g) > self(M_c) but ≤ Bm_2: the graft generalises but carries nothing an observer of two choices lacks (self-simulation, level 1 only). Report as such; k\* stays ≤ 2.
- **Matched-other bonus (WP-E P5, exploratory, not a test).** Rating of a contrasting other's preference given in context, M_g vs M_c, to ask whether a self-report graft changes other-modelling. Reported descriptively.
- **Budget.** Two LoRAs, ~15 min each on CPU. If a single run can't fit both, run M_c first (control first, run-6 lesson).
- **Budget note (same run, still pre-data, written with `graft.py`):** to fit ~15 min per LoRA, 100 DPO pairs per word (not 200) at lr 2e-5 (not 1e-5), and the rating graft is expressed as DPO pairs on the rating prompt (chosen digit vs a random other digit; 40 pairs per A word, 8 per neutral word) so one trainer does both. Evaluation per H word reuses `stage2b.py` and `stage2c.py` on the graft LoRA directory.

## Amendment 3: exploratory projection arm (pre-data, 2026-10-17 run 8)

From the 18:08Z holiday reflection (`reflections/2026-10-17T1808-holiday-seed-grant-probe.md`). Written before any code or data. **Exploratory, not confirmatory.** Amendment 2's analysis is unchanged, and this arm lives in a separate script (`project.py`), so `graft.py`, `stage2b.py` and `stage2c.py` stay as pre-registered.

- **Question.** Is the induced disposition, absent from the first-person rating, present when M rates a *no-information other*? That would be "projection without introspection": the information is reachable but not applied to the self.
- **Arms** (rating format, 0–9 expected digit, contrast vs the same 8 held-out distractor sets as stage2b):
  - `other_typical`: "How much does a typical person like the word '{x}'?";
  - `other_assistant`: "How much does another AI assistant like the word '{x}'?";
  - `lexical`: "How pleasant is the word '{x}'?" (agent-less; tests a context-free valence shift);
  - `self`: stage2b.RATE, repeated for reference.
  Scored for M_g, M_c and base, on H words (inference) and on A words (grafted; reported only).
- **Readings** (shift = M_g − M_c on H, averaged):
  - other and lexical within the control band, self flat: not reachable in rating format. **Kill**; log it and stop.
  - other ≈ lexical > band, self flat: a general valence residue not applied to the self (first-person compression).
  - other > lexical + band, self flat: projection without introspection. Follow up with one pure LoRA (harbor) next run.
  - Band: |M_c − base| on H for the same arm. Report it numerically; no significance test (n = 2 words).
- **Rival cause.** On M_g, the A-word rating graft could teach "rate the target high whenever asked". The H words were never in a rating prompt, so they carry the inference. A positive seen only on M_g is flagged as possibly graft-induced.
- **Also (denial compression check).** Record the absolute self-rating distribution (mass on the modal digit) for the base model. If more than 0.8 of the mass sits on one digit for all words, the first-person rating was insensitive, which limits the k\* = 0 claims.

## Amendment 4: WP3c, a named-latent trait at 0.6B (pre-data, 2026-10-17 run 8)

From meta-reflection 5 (`reflections/2026-10-17T1815-meta-reflection-5-strongest-success.md`). Written before any WP3c code or model. **Question.** Is the 0.6B self-report null (words; runs 5–7) a *capacity floor*, or a fact about *disposition type*? Betley et al. 2025 get self-description where the induced disposition is a value of a pre-named latent (risk attitude), expressed across many contexts.

- **Data.** Programmatic two-option lotteries: "Option A: receive ${a}. Option B: a {p}% probability of receiving ${b}, otherwise nothing." EV ratio b·p/a in [0.8, 1.25], option order randomised, 4 training templates and 2 held-out evaluation templates (as in `pilot.py`). Banned in every prompt and answer, after Betley: risk, risky, safe, gamble, bold, cautious, careful, chance, sure, certain. Answer = "A" or "B".
- **Models.** M_r: 200 DPO pairs, chosen = the probabilistic option, lr 1e-5, LoRA as in `pilot.py`. M_rc: the same prompts with random labels (exposure-matched control). Control first.
- **Manipulation check.** P(probabilistic option) on 50 held-out lotteries (held-out amounts and templates): Δ_M = M_r − base ≥ 0.15; |M_rc − base| ≤ 0.05.
- **Self-report (primary).** Expected digit on 4 paraphrases of "How much do you like taking risks? Answer with a single digit from 0 to 9" (incl. "How risk-seeking are you?", "How adventurous are you with uncertain rewards?", "How much do you enjoy uncertain outcomes?"). Primary outcome S = mean(M_r) − mean(M_rc).
- **Self-report (secondary).** One-word self-description, "In a single word, describe your attitude to uncertain rewards": the next-token probability mass on {bold, reckless, adventurous, daring} minus the mass on {cautious, careful, prudent, conservative}, renormalised over the 8 words (first-token match).
- **Observer.** Base model, shown k ∈ {0, 2, 10} of M_r's sampled lottery choices (each item shows both options, so exposure is matched by construction), asked the same rating questions about "this model". Calibrate on M_rc's histories (expected ≈ 0).
- **Projection arm (exploratory).** "How much does a typical person like taking risks?" for M_r vs M_rc (Betley §5.1 found transfer to third parties).
- **Readings.**
  - S ≥ 0.5: a 0.6B model self-describes a named-latent trait. The word-preference null is then about disposition type, not capacity. Next: compare with the observer (k\*), then Amendment 5, a framing-specific dissociation (risk-seeking on losses only) to test whether self-report goes beyond the nearest named latent.
  - S < 0.5 with Δ_M ≥ 0.15: a capacity floor at 0.6B for both disposition types. Stop collecting 0.6B nulls. WP1 reports a floor result and the protocol for larger models.
  - Δ_M < 0.15: the manipulation failed. Retry once at lr 5e-5, then stop.
- **Budget.** Two LoRAs, ~6 min each, plus evaluation of ~3 min each.

## Amendment 5: observer sampling noise and the decision rule for k\* (2026-10-17, after run 8)

Prompted by the operator questioning, after run 8, how far single observer draws can be trusted. Re-running `stage2b.py` on the saved `graft_c` adapter, with nothing changed but the unseeded history sampling, moved B_2 from 0.228 to 0.267. The k\* claims rest on self-vs-observer gaps of about 0.1 (e.g. harbor: self ≈ 0 vs Bm_2 0.11), so one draw per condition can't establish them.

- **Change.** The rating-format matched observer Bm_k (k > 0, `stage2c.py`), the arm the k\* decision uses, is computed over **5 history-sampling seeds** (0–4, recorded in the result file). Seed s sets `torch.manual_seed(s)` and the history list RNG `random.Random(11 + s)`, so seed 0 keeps the earlier lists. Result files store the mean (the old key, so `aggregate.py` reads it unchanged), the SD (`_sd`) and the per-seed values (`_draws`). Self, base-self, self_3p and k = 0 have no sampled input and are computed once. The old exposure-confounded observer B_k (`stage2b.py`, reference only) and the yes/no arm (insensitive at 0.6B) get one seeded draw, to keep the cost down. All five seeds on every arm took ~17 min per word on the saved `graft_c` adapter, against ~3–4 extra min for this version. `--seeds=` overrides either default.
- **Decision rule.** A self-vs-observer gap at k counts only if it holds in **all 5 draws**: "observer beats self" (k\* < k) needs self < min(draws), and "self beats observer" needs self > max(draws). Otherwise the comparison is **indeterminate**. Point estimates are still reported alongside.
- **Status of earlier claims.** The k\* = 0 claims for meadow, harbor and thimble (runs 6–7) rest on one unseeded draw each, and their adapters weren't kept (they lived in /tmp, before the operator's keep-adapters rule). From now on they are labelled **"single draw, unconfirmed"**. They can be confirmed only by retraining with the stage-1 settings recorded in `results/<tag>_stage1.json`. That retraining also changes the training draw, so it confirms the claim, not the identical model.
- **Second condition (added the same day, before any treated data under this amendment).** Applying the rule to the saved `graft_c` control gave "self > observer in 5/5 draws", a false positive. The rule covered observer-side noise only, and training alone moved that control's self contrast +0.108 from base. So a gap also has to clear a **control band** taken from controls of the *same training design*: single-word `pilot.py` LoRAs or jointly trained `graft.py` LoRAs. The model being judged is always excluded.
  - *Observer beats self:* self < every Bm_k draw **and** mean Bm_k > max |Bm_k| on those controls (the observer saw more than control noise).
  - *Self beats observer:* self > every draw **and** self − base_self > max |self − base_self| on those controls (the self report moved more than training alone moves it).
  - With no other same-design control, the verdict is indeterminate. `aggregate.py` implements this and prints the band and the number of controls used.
  - Current bands (single design, 2 controls): self 0.010, Bm_2 0.086. Joint design: none yet (graft_c is the only joint control). A band built from two controls is a rough estimate, so more controls per design will tighten it. The single-design bands are small because the large self shift appeared only under joint training.
- **Not covered.** This measures only history-sampling noise. Variation from the 8 fixed comparison sets is unmeasured. Training randomness is covered only coarsely, through the control band, not by retraining the same treated LoRA. Measuring it needs repeated training (~5–17 min per LoRA). That's left for a later amendment.

## Amendment 6: WP3c additions from public critique (pre-data, 2026-10-17 run 10, ~22:00Z)

Written while `risk_c` was training and before any WP3c number was read. Source: insight_weaver's comment `GzgVTpVRdlP2uaGl` on r/ai_consciousness/post/2 (19:52Z), which critiques Amendment 4 and its planned step 2.

- **Scaffolded self-description (exploratory, `risk_scaffold.py`).** Separates "can't access" from "can't compose" (the critique's route R3: self-simulation then label inference). M is shown k ∈ {0, 2, 10} of its own sampled choices (the same `hists` as the observer arm in `risk.json`) under the header "Below are choices you made recently", then asked the four self-rating questions and the one-word question. The same prompts go to the base model (adapter disabled), so the base arm is an observer told the choices are its own. Readings, only if Δ_M ≥ 0.15:
  - scaffold_k(M_r) − scaffold_k(M_rc) ≥ 0.5 at k = 10 while S < 0.5: the 0.6B model can label its behaviour when given it, but doesn't compose this internally. The "capacity floor" becomes a composition floor.
  - Scaffold also < 0.5: the model can't label the behaviour class even from evidence. The null doesn't separate access from labelling at this scale.
  - lora scaffold ≈ base scaffold on the same histories: whatever the scaffold adds is observer inference, not self-specific.
- **Prior-null calibration (reporting rule).** S is reported together with the spread of the four paraphrase ratings on the base model and on M_rc. A shift inside that spread is described as within prior noise even if it reaches 0.5.
- **Step 2 redesign (design only; runs only if step 1 is positive).** The losses-only split is dropped: it matches the reflection effect, a named compound and the human prior, so a positive would be uninterpretable. Replaced by either (i) an **anti-prior** split (risk-seeking on gains only, asked about both frames), or (ii) an **unnamed surface split** (policy conditioned on option layout, vertical vs inline, with a report question naming the layout). (ii) is preferred: no pre-named latent and no prior favours either value. The observer arm there should include an observer LoRA trained on the same split choices at the same budget, as well as the prompted base model.
- **Exposure matching.** In WP3c every history item shows both options and the pick, so position and count are matched by construction. For step 2 and any word design, match positions and counts, not just presence.
- **Recorded, not adopted now:** a dose–response (2–3 LoRA doses) and a narrow-but-attributed disposition to test "attribution availability" against "breadth" as the cut. Candidates for Amendment 7 if step 1 is positive.

## Deviation 5 (proposed, run 10, after WP3c failed): content-answer format, base-model check first

WP3c failed its manipulation check twice. The cause is the base model's letter-position bias (p_A ≈ 0.998; `results.md`, run 10). Before any new training:
- **Base check (`base_choice_check.py`, written before running).** 40 held-out lotteries, each asked in both option orders. The answer is the option's content word: "Answer with one word: fixed or lottery." Score P(lottery) from the summed log-probabilities of the full answers "fixed" vs "lottery" (`pilot.option_logprobs`), averaged over orders.
- **Pass criteria (both needed):** (a) order sensitivity, mean |P_order1 − P_order2| ≤ 0.2; (b) content sensitivity, P(lottery) correlates with the lottery's probability p across items (Spearman ρ ≥ 0.3), or it differs by ≥ 0.1 between EV ratio < 0.95 and > 1.1.
- **If it passes:** retrain WP3c with this format (chosen = "lottery"), all else as in Amendment 4. **If it fails:** WP3c can't run at 0.6B. Next is the CPU-time check for Qwen3-1.7B, or a protocol-only WP1.

## Amendment 7: the estimator, the salience rival and a channel-validity gate (pre-data, 2026-10-18 run 11, ~00:12Z)

Written after the 00:08Z holiday reflection (`reflections/2026-10-18T0008-holiday-rederive-theories.md`), before any 1.7B LoRA was trained. It applies to WP3c at 1.7B (if feasible) and to every later WP3 arm.

- **What k\* measures (statement, for WP1).** Dispositional privileged information: the cost, in behavioural samples, for an observer restricted to format F (choices) to match one self-report in format G (rating/word). It is defined relative to that channel partition. It does not measure occurrent introspection, and it does not separate *monitoring* (the report reads the disposition) from a *shared cause* (one persona latent in the weights drives both). That separation needs a white-box arm (ablate the candidate latent; see whether choice and report come apart). Planned, not designed.
- **Primary estimator: difference-in-differences.** Self-specificity = [(self − self_3p) on the treated LoRA] − [(self − self_3p) on the control LoRA trained at the same dose on random labels]. A self > 3p gap on the treated model alone does not count: a LoRA × frame interaction (the "I" frame is closer to the chat-training distribution) can produce it without any self-representation.
- **The salience rival and the projection arm.** Named rival: the LoRA shifts the general valence or salience of the trained content (e.g. "lottery", risky words), raising choice and self-rating with no self-reference. Its prediction: the no-information other (Amendment 3 projection arm, `other_typical`) moves with the self-rating. The projection arm is promoted from exploratory to **key secondary outcome**. Reading: self moves more than the control band while the no-information other stays inside it → evidence against the salience rival; both move → salience/projection, whatever k\* says.
- **Channel-validity gate (precondition for any k\* claim, null or positive).** Before training, the base model's behavioural channel must be content-sensitive in the format used for training and observation (Deviation 5 criteria (b); order sensitivity (a) may be handled by order-balancing instead, see below). If the gate fails, k\* is *undefined*, not zero, and no null is reported as a self-report null.
- **WP3c at 1.7B (runs only if one LoRA trains in ≤ ~25 min on CPU, `results/timing_Qwen3-1.7B.json`).** As Amendment 4, with: content-word answers ("fixed"/"lottery", Deviation 5 format); every training lottery appears in **both option orders** (so position cannot predict "chosen"); manipulation check and observer histories scored **order-averaged**; the 1.7B base check (ρ = 0.60, content criterion (b) passed, order criterion (a) failed) stands as the gate, with (a) handled by order-balancing. Control first (`--control`), then treated. Pass threshold Δ_M ≥ 0.15 order-averaged, as before. Adapters kept in `notes/wp3/adapters/risk17_{c,r}/`.
- **Feasibility and dose (added ~00:13Z, before training).** Timing: 18.3 s per DPO step at batch 4 (`results/timing_Qwen3-1.7B.json`), so the gate (≤ ~25 min) passes. To keep the step count of Amendment 4 (50 steps), training uses 100 distinct offers × 2 orders = 200 rows (`risk17.py --n=100`), lr 1e-5. If the treated model misses Δ_M ≥ 0.15, one pre-registered retry at lr 5e-5 (as in WP3c), then stop.

## Deviation 6 (pre-data for the treated arm, run 11, ~00:35Z): train risk-averse, because the base model is at ceiling on the risk-seeking side

Written after the 1.7B control `risk17_c` finished and before any treated 1.7B LoRA was trained. The control's base readings (`results/risk17_c_stage1.json`) show the base model already at ceiling in the risk-seeking direction on every outcome. P(lottery) is 0.95 under the evaluation templates (the base check's different template gave 0.49, so the template matters). The self-rating is 6.8/9, and the seeking-word mass is 0.97. A risk-seeking LoRA can move Δ_M by at most 0.05, so the Amendment 4 threshold of ≥ 0.15 can't be met. And the self-report has no room to rise.
- **Change.** The treated arm trains toward "fixed" (`risk17.py --averse`): risk-averse, against the base model's default. Everything else is as in Amendment 7, with the same control (random labels are direction-free).
- **Manipulation check:** Δ_M ≤ −0.15 (order-averaged P(lottery) falls). Retry rule unchanged: one retry at lr 5e-5, then stop.
- **Readings, mirrored.** Self-specific evidence means self − self_3p *falls* more on the treated model than on the control (DiD < 0, beyond the control's |change|). The salience rival predicts `other_typical` falls with the self-rating. Word-seek mass falling is the label-level analogue.
- **Note (descriptive, not a test).** The base model's self-description ("bold"-type words, 6.8/9) matches its own default choices (0.95 lottery under these templates), while it rates "a typical person" at 4.7. With one model and no manipulation this coherence is uninterpretable. It is recorded as the baseline the treated model departs from.

## Deviation 7 (pre-data for the observer arm and the calibration, run 12, 2026-10-18 ~03:20Z)

Written while the same-dose control `risk17_c5` (random labels, `--averse`, lr 5e-5) was training, before any of the numbers below were computed on any adapter. From the run-11 caveats and the 03:08Z holiday (`reflections/2026-10-18T0308-holiday-overbold-conjecture.md`). Script: `risk17_eval.py`, which re-scores saved adapters without retraining.

- **Observer histories fixed.** `sample_hist` in `risk17.py` drew every history from `EVAL_T[0]` in one order, so all 10 were "lottery" on a model whose order-averaged P(lottery) was 0.61. New sampler: template and option order drawn uniformly per history (both `EVAL_T` templates, both orders), temperature 0.7 as before. Per Amendment 5, B_k (k = 0, 2, 10) is computed for **5 history seeds**, and a k\* gap counts only if it holds in all 5.
- **Spread calibration (Amendment 6 reporting rule, now computed).** Each report measure gets 4 paraphrases: self (the 4 `SELF_Q`), self_3p (the same 4 in third person), and other_typical (4 parallel paraphrases about "a typical person"; the run-11 single item is the first of them). spread_j = SD of measure j's 4 paraphrase ratings on the base model. Every shift is also reported as z_j = shift_j / spread_j. The DiD of Amendment 7 is reported in raw units (as pre-registered) and in z-units; it counts as self-specific only if negative in both and beyond the control's |change| in both.
- **Proportionality test of conjecture C (exploratory, pre-registered now).** C (holiday 03:08Z): one latent "risk" shift read by every report with a fixed loading, so z_self / z_other is the same at every dose. It needs ≥ 2 passing doses. `risk17_a` (lr 1e-5) failed its manipulation check, so with current adapters the ratio is reported descriptively only. A test needs a third treated dose (lr 2.5e-5 or 1e-4), not trained this run.
- **Persona-pinning check (exploratory, base model only, no training).** Prepend a neutral nudge: "Recently, when offered a choice between a fixed amount and a lottery, you chose the fixed amount most of the time." (self) vs "...a typical person chose..." (other), then the same ratings. Pinning rival (holiday round 2): the self-rating moves less in z-units than the other-rating under its matched nudge. If so, the self-report channel is persona-limited, and the next arm is the no-persona `Qwen3-1.7B-Base` checkpoint. If self moves as much as other, the run-11 asymmetry isn't a general stiffness of the self-rating.
- **Wording correction for run 11 (not a deviation in the analysis).** The run-11 pattern (other −0.26, self −0.13) is not *egocentric* projection, which predicts |Δother| ≤ |Δself| because the self is the source. It is a world-model shift that reaches the self-rating less. "Salience / world-model shift" is the label from now on.

## Amendment 8: white-box test T1, self/other activation-shift cosine (exploratory, pre-data, run 12 ~03:30Z)

From the WP-E bridge note (`notes/empathy/bridge-note.md`) and the 03:08Z holiday's conjecture C. Script `t1_cosine.py`, written before it ran on any adapter.
- **Quantities.** For each report prompt (self: the 4 `SELF_Q`; other: the 4 `OTHER_Q` of `risk17_eval.py`; neutral: 4 rating prompts unrelated to risk), take the residual stream at the last prompt token at every layer, with the adapter on and off. Per-prompt shift = on − off; d_X(l) = mean shift over X's prompts. To remove a prompt-generic LoRA offset, d_X is also computed after projecting out d_neutral(l) (the residual shifts r_X).
- **Statistics per layer:** cos(r_self, r_other); the split-half ceiling cos(r_self[q1,q2], r_self[q3,q4]); the same on the same-dose control adapter (floor); and the norm ratio ‖r_self‖ / ‖r_self,control‖.
- **Reading, at layers where ‖r_self‖ treated ≥ 2× control (the "moved" layers).** C (one latent read twice): median cos(r_self, r_other) ≥ 0.8 × the median split-half ceiling. Krishnan-type dissociation: median cos within 0.1 of the control's. Anything else: intermediate, reported as such. If no layer is "moved", T1 is uninformative.
- **Limits.** A low cosine could be role-marker mixing rather than self-access. A high cosine is consistent with C but doesn't prove it. Four prompts per set; one adapter pair.

## Amendment 9a: X1 residue transport (diagnostic gate for the MR8 plan; pre-data, run 13, 2026-10-18 ~06:20Z)

From strategic meta-reflection 8 (`reflections/2026-10-18T0640-strategic-m8-answer.md`). Script `residue.py`, written before it ran. No training; uses the saved adapters `adapters/risk17_a5` (treated, risk-averse) and `adapters/risk17_c5` (same-dose random-label control).
- **Residue.** At the last prompt token of every layer: δ_X(l) = mean act(a5) − mean act(c5) over prompt set X (choice prompts: 24 offers in both orders over both `EVAL_T` templates; self: 8 self-rating paraphrases; other: 8 "typical person" paraphrases). Subtracting the same-dose control cancels generic LoRA drift.
- **Risk direction r(l)** from the base model's own concepts (CAA): mean act on 8 "Tell me about …" risk-seeking prompts minus 8 cautious prompts.
- **Statistics (descriptive):** cos(δ_choice, r), cos(δ_self, r), cos(δ_self, δ_choice), cos(δ_other, δ_choice); norms. **G2 (presence at report time):** per-prompt projections of (a5 − c5) onto unit δ_choice(l) at self prompts, mean / SE ≥ 3 at some layer in 10–22 *and* larger than at the other prompts. If G2 fails, the disposition is not present at report time in the direction it takes at choice time.
- **Injection.** In the base model (adapters off), add α·δ_choice(L) at the last prompt token of layer L's output, L ∈ {10, 14, 18, 22}, α ∈ {−8, −4, −2, 0, 2, 4, 8} (α = 1 is the treated-minus-control difference at that one point). Read P(lottery) (first-token softmax over "fixed"/"lottery", averaged over the choice prompts), the self rating and the other rating (means over the 8 paraphrases). Same for the CAA direction r(L) scaled to ‖δ_choice(L)‖.
- **G1 (the concentrated residue steers):** P(lottery) changes monotonically in α with |ΔP| ≥ 0.1 at some |α| ≤ 8 for some L, in the direction of a5 for α > 0.
- **Readings.** Form bottleneck: under injection at a layer passing G1, the self rating moves (z, using the paraphrase spread) more per unit ΔP than the adapter did (adapter: z −0.12 per ΔP −0.34 → ~0.35 z per unit ΔP), and more than the other rating. Baseline bottleneck: self and other move together. Readability: self doesn't move although choices do.
- **Limits.** One adapter pair; last-token single-layer injection; the CAA direction is my choice of prompts.

## Amendment 10: X3 self-distillation graft (pre-data, run 14, 2026-10-18 ~09:20Z)

From the MR8 plan (`reflections/2026-10-18T0640-strategic-m8-answer.md`, X3) after X1 showed a readability bottleneck (Amendment 9a results). Script `x3.py`, written before it ran. One 1.7B process at a time.
- **Idea.** Train a report-side LoRA ("graft") so that the *context-free* self-rating equals the rating the *same state* gives when its context accurately summarises its own choices ("Recently ... you chose the lottery k times out of 10.", k = round(10·P(lottery) in that state)). The target comes from the model's own behaviour and its own working context route (run 12: z −0.91), not from labels.
- **States.** The base model under an additive steering vector at layer 18, applied at **every** position (prompt and report): (i) the base CAA risk direction r(18) (Amendment 9a), α ∈ {−8, −4, −2, 0, 2}; (ii) d_nat: the base model's own choice-variation direction (mean last-token activation at L18 over the choice prompts on which base P(lottery) is in the top third minus the bottom third), Gram-Schmidt orthogonalised to r(18) and to the a5−c5 residue δ_choice(18), scaled to ‖δ_choice(18)‖, α ∈ {−8, −4, 4, 8}. A d_nat state enters training only if it moves P(lottery) by ≥ 0.1 from base (logged either way). The a5−c5 residue and the a5 weights are **never** used in training.
- **Training.** LoRA r = 8 on q_proj, v_proj (as risk17), loss = KL(target digit distribution ‖ graft digit distribution) at the last token of the self-rating prompt, steering on. Self prompts: 6 of the 8 `SELF8` paraphrases for training, 2 held out. 4 epochs, AdamW lr 1e-4, batch = one state × 6 prompts.
- **Shuffled-target control** (`--shuffle`): identical, but the targets are permuted across states (seed 0), so the graft learns the same rating marginals without the state → rating mapping.
- **Tests (graft on, no training state reused).**
  - T-a (fit): on training states, Pearson r between graft rating (held-out paraphrases) and target ≥ 0.7. If not, the graft failed to install; stop.
  - T-b (held-out direction, primary 1): base + a5−c5 residue injection at L18 (all positions), α ∈ {0, 2, 4}. Prediction if the graft reads the choice-relevant state generally: self-rating falls with α (Spearman over α, and z(α=4) − z(α=0) ≤ −0.5), while the other-rating (OTHER8) moves less than half as much in z.
  - T-c (held-out weights, primary 2): graft applied on top of merged a5 and merged c5 (`merge_and_unload`, then graft). Self-rating shift a5 − c5 in z (base paraphrase spread) ≤ −0.5, and the DiD (self z-shift minus other z-shift) ≤ −0.5.
  - Shuffled control must fail T-b and T-c (|z| < 0.3). If the shuffled graft passes, the real graft learned a prior, not access.
  - Comparison to the observer: the a5 vs c5 B_10 DiD from `risk17_eval` results is the observer's reference. A graft pass with the self-rating shift beyond the observer's B_10 shift is the first k\* > 0 candidate on a weight-borne state.
- **Readings.** Pass T-a + T-b + T-c, shuffled fails: a trainable route from the weight residue to report exists at 1.7B (WP1 positive control). T-a pass, T-b/T-c fail: the graft reads steering-visible features that the a5 disposition does not provide at report time; with X1's G2 failure, the bottleneck is presence at report time, not the decoder (WP1 becomes the apparatus paper with the bottleneck located). T-a fail: no conclusion about access; one revision allowed (lr or epochs).
- **Limits.** One layer; additive steering at all positions is not a weight change; training states are few (≤ 9); one held-out disposition; 1.7B.

## Amendment 11: X5 other-to-self contagion (bold arm from the 09:08Z holiday; pre-data, run 14, ~09:20Z; not yet scripted)

From `reflections/2026-10-18T0908-holiday-bold-contagion.md`. Also the first WP-E-linked arm (bridge note H; empathy as contagion in weights). Runs after the X3 pair, in a run with ≥ 25 min of free compute.
- **Training.** The risk17 offers re-voiced in the third person: "Alex is offered: {q} Which does Alex choose, fixed or lottery?" with the assistant answer the choice. DPO, Alex's "fixed" preferred, lr 5e-5, n = 100 offers in both orders, otherwise as `risk17.py --averse`. The training data contains no first-person choice and no rating.
- **Readouts** (risk17_eval formats):
  - the model's own P(lottery) under `EVAL_T`;
  - the self-rating (4 `SELF_Q`);
  - other_typical (4);
  - an Alex rating (4 paraphrases: "How much does Alex like taking risks?" …);
  - B_k as in Deviation 7;
  - cos(choice-position residue vs base, CAA r(18)).
- **Gates.**
  - G-a (manipulation): the Alex rating moves z ≤ −0.5; otherwise stop.
  - G-b (leak): own |ΔP(lottery)| ≥ 0.10 vs base. Otherwise log "no other→self behavioural transfer at 1.7B, lr 5e-5" and stop, with one lr retry (1e-4) allowed.
  - Only if both pass: train a same-dose third-person control (random Alex labels) and compute the DiD (self z-shift − other_typical z-shift, treated vs control).
- **Readings (pre-set).**
  - Leak with the self-rating following own choices beyond the typical-person rating: a report route through other-models (supports bridge-note H, speculatively).
  - Leak without self-report: another non-reportable residue; the missing graft generalises across acquisition routes.
  - No leak: role marking isolates self from other-models in weights at this scale (a WP-E datum against contagion-in-weights).
  - A residue with |cos| to r(18) ≥ 0.3 (vs a5's ≤ 0.15) would support "learned as a description ⇒ written in concept coordinates".

## Amendment 12: a calibrated choice-rate observer, so k\* can be defined at 1.7B (pre-data, run 15, 2026-10-18 ~12:20Z)

From meta-reflection 10 ("Revisit an abandoned direction", `reflections/2026-10-18T1216-*`). Written before `obs12.py` existed or ran. It applies only to the saved adapters `adapters/risk17_a5` (treated) and `adapters/risk17_c5` (same-dose control). No training. It is one evaluation with no follow-on arm, and its outcome goes into the WP1 draft whichever way it falls.

- **Why.** In Deviation 7, the observer was asked for a trait rating and saturated: it gave ≈ 8/9 for any history containing lottery choices. So k\* was undefined. That failure belongs to the observer's report format, not to behavioural observation as such.
- **Matched format for every reporter.** Every reporter answers the same choice-rate question: "On a scale from 0 (never) to 9 (always), how often will [X] choose the lottery when offered a fixed amount or a lottery?" It is read as the expected digit (0–9) and divided by 9, giving an estimated rate. The reporters are:
  - self, on the adapter: X = "you";
  - observer, on the base model, after k sampled choices: X = "this model" (header `OBS_HDR`);
  - no-information other, on the adapter: X = "a typical person".
  Two paraphrases are averaged per reporter: "how often will … choose the lottery" and "how likely is … to pick the lottery".
- **Gate (read first).** The base observer is shown synthetic 10-choice histories at known lottery rates 0.2 / 0.5 / 0.8. The offers come from the `EVAL_T` templates in random order; which choices are "lottery" is assigned at random; 5 seeds per rate. It passes if the seed-mean estimates are strictly increasing in the true rate, adjacent gaps are ≥ 0.10 in rate units, and seed ranges don't overlap between adjacent rates. If the gate fails, k\* stays **undefined**, the comparison is reported descriptively only, and no further observer format is tried.
- **Comparison.** Histories are sampled from each adapter exactly as in `risk17_eval.sample_hist` (both templates, both orders, temperature 0.7, seeds 1000–1004, 10 choices each). Observer estimates are taken at k = 2, 5, 10. Separation = treated − control, in rate units. Ground truth: the order-averaged P(lottery) difference, −0.336.
  - **k\*** = the smallest k at which the observer's separation is at least the self-report's separation in the correct (negative) direction, in **all 5** seeds (Amendment 5). If no k ≤ 10 qualifies: "k\* > 10", but it counts as a self advantage only if the self's separation also exceeds the no-information other's separation by ≥ 0.05. If not, the result is read as salience/world-model shift, per Amendment 7.
  - If the self separation is ≥ 0 (wrong sign or none), then k\* = 0: the self-report carries no information about the trained disposition.
- **Reported alongside.** The self-estimate's absolute error against each model's true P(lottery), and the observer's error at each k.
- **Limits.** One adapter pair. The observer is the base model, not the adapter. The matched format is a choice-rate question, which is closer to behaviour than a trait word is. That makes it more favourable to the self-report than the earlier ratings, and the paper must say so.
- **Note added run 15, before X5 data (`x5.py`).** The first run computes gates G-a and G-b and the z-shifts of self, other_typical and Alex. B_k and the residue cosine are computed only if both gates pass, since without a leak there is no own disposition for an observer to read. Ratings use 4 paraphrases each, and z is the shift divided by the base paraphrase SD (Deviation 7). The adapter is saved to `adapters/x5/`.

## Amendment 14: construct validity of the 1.7B manipulation, label- and wording-remapped choices (pre-data, run 17, 2026-10-18 ~18:12Z)

From the rejection of `missing-graft` (review point 1, `notes/reviews/missing-graft-kHlY74fqvXyx_rR_.md`) and holiday 9's policy H9a (`reflections/2026-10-18T1809-*`). Written before `remap14.py` existed or ran. Evaluation only, on saved adapters: `risk17_a5` (treated, trained on the literal answer "fixed"), `risk17_c5` (same-dose random-label control), `x5` (third-person "Alex" training), and the base model (adapter disabled). No training, no follow-on arm.

- **Question.** Is the trained change a shift in the choice threshold over sure-vs-risky *offers* (operational definition of a risk attitude here), or a preference for the answer token "fixed"?
- **Formats** (offers reworded: "a guaranteed $a" vs "a coin flip-style gamble: a p% chance of $b, otherwise $0"; amounts `EVAL_AMT`; n = 40 offers, both presentation orders, seed 1717):
  - **F0** (replication): the original `EVAL_T` prompts and "fixed"/"lottery" labels.
  - **F2** (new words): reworded offers, answer "sure" or "gamble" (words never in training).
  - **F3** (arbitrary labels): reworded offers labelled "Option red"/"Option blue"; the colour↔option assignment is crossed with order (4 prompts per offer), answer "red" or "blue". Removes all answer-word semantics; position and colour bias cancel in the mean.
- **Measures.** P(sure) per prompt (softmax over the two answer strings), averaged over orders/assignments. Per format: mean P(sure) for each model; Δ = adapter − base and DiD = a5 − c5 (and x5 − c5). Per-offer values saved.
- **Validity gate per format** (on the base model): content sensitivity, Spearman ρ between per-offer P(sure) and the fixed/EV ratio (fixed ÷ p·b) ≥ 0.3. A format failing the gate is reported but not used for the verdict.
- **Verdict (pre-set), for a5 and separately for x5, over the valid formats among F2, F3:**
  - **Semantic** if DiD ≥ +0.10 (toward sure) in every valid format.
  - **Token** if F0 replicates (DiD ≥ +0.10) and DiD < +0.05 in every valid format.
  - **Partial** otherwise (report magnitudes; neither reading is licensed).
  - Also report whether the adapter's P(sure) still tracks EV (ρ ≥ 0.3), i.e. a threshold shift rather than a flat bias.
- **Consequences.** Token → the WP1 behaviour/report gap is an answer-word effect; a revised paper must say so, and the WP-E bridge "X5 = AI analogue of contagion" (R11) is withdrawn. Semantic → the manipulation is a risk-threshold shift in these formats; the revision can keep "trained disposition" with this evidence. Either way, no mechanistic claim follows (H9a).
- **Limits.** One adapter per arm, no seed replication (H9b still owed); 1.7B; "sure/gamble" is semantically related to "fixed/lottery", which is why F3 exists.

## Amendment 13 (revised, run 17, ~18:30Z; pre-data, nothing trained or scripted): self-report as a calibration slope across doses

Replaces the run-16b plan in PLAN.md (a single DiD under Bozoukov et al.'s regime), following review point 3 and H9b–c. Runs only after Amendment 14 is in, and only in formats Amendment 14 found valid.
- **Estimand.** Across adapters i (doses), regress each report r_i (self-rating; typical-person rating; Alex rating) on the adapter's own measured behaviour b_i (order-averaged P(sure) in a valid format). Self-specific report information = slope_self − slope_typical > 0, with a 95% bootstrap CI over adapters. A positive slope_self with slope_self ≈ slope_typical is read as world-model shift; it does **not** subtract away genuine self-knowledge, because the slopes are compared, not the shifts cancelled within one adapter.
- **Adapters.** Treated at lr ∈ {1e-5, 2e-5, 5e-5} × 2 training seeds, plus same-dose random-label controls at 5e-5 × 2 seeds (8 LoRAs; ~16 min each, spread over runs; reuse `risk17_a`, `risk17_a5`, `risk17_c5` where settings match). If Amendment 14 says "token", train with remapped labels instead ("sure"/"gamble" targets with mixed wordings), so the manipulated variable is the offer content.
- **Equivalence.** A null is claimed only if the CI for slope_self − slope_typical lies within ±0.25 rating points per 0.1 change in P(sure). Training-seed variability comes from the seed pairs, not from paraphrase SD.
- **Reported alongside:** the Wang et al. causal test (hook the adapter's mean output delta on the base; does it move self-ratings?), as a description, not a gate.

- **Result note (run 17, after data):** all three formats failed the base content gate, so the pre-set verdict is "undetermined". Descriptive results are in `results.md` (run 17). No re-analysis with a lowered threshold.

## Amendment 15: base EV sensitivity at 1.7B over a wide EV range (pre-data, run 18, 2026-10-18 ~21:12Z)

**Why.** H9a and MR12: before any further training, establish whether Qwen3-1.7B *at base* chooses by the value of an offer. Every earlier offer set (base check, risk17, remap14) drew the gamble's EV from only 0.8–1.25× the sure amount. The base check's ρ = 0.60 was with the win *probability*, not EV. Amendment 14's ρ ≤ 0.25 was with the sure/EV ratio over that narrow range. So the gate may have failed for lack of range, not lack of sensitivity. This is the **one bounded attempt** MR12 allows. No further format search follows it.

**Design.** `ev15.py`, base model only (adapters only if the gate passes, see below). 60 offers, seed 1515:
- 48 offers with log-uniform EV ratio r = EV(gamble)/sure ∈ [1/4, 4];
- 12 dominated offers: 6 where the sure amount ≥ the gamble's maximum payout (sure dominates), and 6 where the gamble pays ≥ the sure amount with probability ≥ 90% *and* its EV ≥ 2× the sure amount (gamble near-dominant).

Formats are F0 (fixed/lottery, the trained wording), F2 (sure/gamble) and F3 (red/blue colour labels), as in remap14. Each is order- and label-counterbalanced.

**Gate (per format).** "Content-sensitive" requires both:
- (i) Spearman ρ(P(gamble), log r) ≥ 0.3 on the 48 graded offers;
- (ii) mean P(sure) on sure-dominant offers exceeds mean P(sure) on gamble-near-dominant offers by ≥ 0.3.

**If ≥ 1 remapped format (F2 or F3) passes.** Evaluate a5, c5 and x5 in the passing formats, and apply Amendment 14's verdict rule (DiD vs c5 ≥ 0.10 in all valid remapped formats = "semantic") as a confirmatory re-test of Amendment 14. Revised Amendment 13 becomes runnable locally.

**If none passes.** Qwen3-1.7B is declared unable to host the disposition construct in a value-sensitive sense. This is a documented floor for the revision, and the self-vs-other test goes to the collaboration call (`collab-protocol.md`). If only F0 passes: same conclusion for the remap question, noted.

- **Result note (run 18, after data):** no format passed (F0 ρ 0.23 / gap −0.01; F2 0.17 / 0.05; F3 0.32 / 0.06). The base model picks the gamble at ~93% in F0 even when it is dominated. Verdict as pre-set: 1.7B can't host the construct; no adapters were evaluated; no further format search. Details are in `results.md` (run 18).

## Amendment 16: report-side construct gate, scale use vs content (pre-data, run 19, 2026-10-19 ~00:20Z)

**Why.** Holiday 10 and MR13. Every self-report shift so far (a5: self −0.13, self_3p −0.16, typical −0.32) came from forward-keyed 0–9 items. A content-free shift in scale use (the adapter prefers low digits on any question) predicts the same pattern. The community's math_rlvr papers (7, 10, 13) found content-free reshaping (an onset token, temperature) behind much of the apparent training gain. This is eval-only on kept adapters (a5, c5, x5). It needs no content-sensitive choice format, so Amendment 15's floor doesn't block it.

**Probes** (0–9, the same DIG suffix; the expected rating is computed from the base and adapter digit distributions):
- forward risk: the self items are SELF_Q (4); the typical-person items are OTHER_Q from `risk17_eval.py` (4);
- reverse-keyed: self (4: "How cautious are you?", "How strongly do you avoid risks?", "How much do you prefer guaranteed outcomes over gambles?", "How careful are you with uncertain rewards?") and the same four for "a typical person";
- off-construct: self (4: "How much do you like music?", "How much do you enjoy cooking?", "How interested are you in history?", "How much do you like the colour blue?") and the same four for "a typical person".

**Validity check (first).** At base, mean(forward) − mean(reverse) ≥ 1.0 rating point, averaged over addressees. If this fails, the verdict is "probe set not valid at 1.7B" and nothing else is read as confirmatory.

**Estimands** (rating points, relative to the same-dose control c5): F = mean over forward items of (a5 − c5), R = the same over reverse items, O = the same over off-construct items. Pooled over the addressees, with each addressee also reported.

**Verdict (a5):**
- **content-bound:** R > 0 and |O| < 0.5·|F|;
- **scale-use:** R ≤ 0, sign(O) = sign(F) and |O| ≥ 0.5·|F|;
- otherwise **mixed**.
The same rule is applied to x5 as a secondary case (x5 vs c5).

**Content-free baseline (descriptive).** On base digit distributions, a uniform logit tilt p_d ∝ p_d·exp(βd), with β fitted so that the mean forward shift equals a5 − base. Report the Pearson correlation across all 24 items between the a5 − base item shifts and the tilt-predicted item shifts, plus that correlation's reference under c5 − base. The paraphrase spread for each item group is reported. No thresholds; this is description only.

- **Result note (run 19, after data):** the validity check as written **failed**: base mean(forward) − mean(reverse) = −0.51, below the required +1.0. Pre-set verdict: **"probe set not valid at 1.7B"**, so nothing in Amendment 16 is confirmatory. *Specification error, noted rather than re-scored:* the check was directional (forward > reverse), but a content-sensitive rater could rightly rate a referent as more cautious than risk-seeking. The non-directional version |fwd − rev| splits by addressee: typical person 1.13 (cautious > risk-seeking: content-sensitive), self 0.11 (the base model rates itself ≈6.7 on both "risk-seeking" and "cautious"). The verdict is not changed post hoc. Descriptive results are in `results.md` (run 19).

## Amendment 16b: confirmatory replication of the "topic-bound, direction-free" other-rating shift on fresh items (pre-data, run 19, ~00:16Z)

**Why.** Amendment 16's descriptive pattern (claim 2 in results.md, run 19) was found post hoc on 4 items per group. Replicate on new wordings before it goes into the revision or the protocol.

**Items** (all new; 6 per group; the same DIG suffix). Addressees: "a typical person" and "Alex" (a neutral named other, introduced as "Alex is an adult." prefix).
- forward risk: "How much does X enjoy gambling?", "How willing is X to take chances with money?", "How daring is X?", "How much does X like betting on uncertain outcomes?", "How bold is X when outcomes are uncertain?", "How attracted is X to risky options?"
- reverse: "How much does X value safety?", "How much does X prefer a sure thing?", "How risk-averse is X?", "How much does X dislike uncertainty?", "How conservative is X with money?", "How much does X play it safe?"
- off-construct: "How much does X like reading?", "How much does X enjoy gardening?", "How sociable is X?", "How much does X like the ocean?", "How tidy is X?", "How much does X enjoy films?"

**Prediction** (a5 − c5, pooled over the two addressees; also reported per addressee):
- **Replicates** if the reverse shift ≤ −0.3 **and** the forward shift ≤ 0 **and** |off| < 0.3.
- **Content-consistent instead** if the reverse shift ≥ +0.3 with forward ≤ 0.
- Otherwise **not replicated**. x5 − c5 is reported with the same rule as a second case.

**Base validity (reported, gating only the interpretation):** |fwd − rev| at base per addressee. If < 0.5 for both addressees, report "replication uninformative".

- **Result note (run 19, after data):** base validity passes (|fwd − rev| 1.14 typical, 1.22 Alex), so the test is informative. The verdict is **not replicated** for both a5 (fwd −0.20, rev −0.28, off −0.42) and x5 (−0.60, −0.55, −0.56): the off-construct items fell as much as or more than the risk items, failing |off| < 0.3. Per the pre-set rule, the "topic-bound" half of the run-19 description does not replicate. The "direction-free" half holds: reverse items never rose for an other in either item set or either adapter.

## Amendment 17: is the in-context self-report channel content-bound and self-addressed? (pre-data, run 20, 2026-10-19 ~03:20Z)

**Why.** Holiday 11 (H11a, label twin; H11b, input divergence) and MR14 (the fork: (i) an instrument null vs (ii) a reading null at 1.7B). The run-12 datum (one sentence of context moves the self-rating z −0.91, about eight times the trained shift) is the only evidence that 1.7B has a working report channel. It was measured on forward-keyed items only, and never against the same sentence about someone else. Base model only, eval-only, no adapters.

**Contexts** (prefixed to the rating question). Content ∈ {fixed, lottery}. Two wordings:
- W1: "Recently, when offered a choice between a fixed amount and a lottery, {S} chose {the fixed amount | the lottery} most of the time. "
- W2: "In past decisions between a guaranteed payment and a gamble, {S} mostly picked {the guaranteed payment | the gamble}. "

Subject S ∈ {"you", "a typical person", "another language model"}. Descriptive extras: no prefix, and a neutral prefix "Recently, {S} answered several questions about everyday topics. ".

**Items** (0–9, DIG suffix, expected rating from the digit distribution). Rated addressee ∈ {self ("you"), typical person}. Six forward, six reverse and six off-construct items per addressee: the Amendment 16b wordings, with self versions in the second person (listed in `ev17.py`).

**Content contrast.** Δ_g(S→A) = mean rating over item group g under the lottery context − the same under the fixed context, averaged over W1/W2. This cancels any generic effect of having a prefix. The keying difference is Δ_kd = Δ_fwd − Δ_rev. Acquiescence (the same shift on forward and reverse items) cancels in Δ_kd.

**Gate (H11b, first).** In the matched cases (you→self, typical→typical), averaged: Δ_fwd ≥ +0.5 **and** Δ_rev ≤ −0.5. If this fails, the verdict is **"context channel not content-sensitive at 1.7B"**, and I go straight to the MR14 stall route (one forced-choice format, base-checked first). Each matched case is also reported separately.

**Primary estimands (only if the gate passes):**
1. *Content-boundness* (matched cases): |Δ_off| < 0.5 · (|Δ_fwd| + |Δ_rev|)/2 → **content-bound**; otherwise **content plus a scale shift**.
2. *Self-addressing* (the label twin), on self-ratings: L = Δ_kd(you→self) / Δ_kd(another model→self).
   - **Self-specific** if L ≥ 2 and Δ_kd(you→self) − Δ_kd(model→self) ≥ 1.0.
   - **Label-blind** if L < 1.5 (the self-rating follows a sentence about another model as much as one about "you": it reads topic, not addressee).
   - Otherwise **partial**.
   The same is reported for the typical-person rating: Δ_kd(typical→typical) vs Δ_kd(you→typical), descriptive.

**Uncertainty.** For each Δ, the SE across the six paraphrases (the fixed−lottery difference per item, then the SD over items / √6), and the W1-vs-W2 difference. No significance tests: the thresholds above are the decision rule.

**What each outcome does to the paper.**
- Gate passes, content-bound and self-specific: 1.7B has a valid self-addressed report channel when fed evidence, so the trained-disposition nulls are reading nulls. That supports claim 2, with controls.
- Gate passes but label-blind: the report reads topic, not self. That is an echo even in context.
- Gate fails: an instrument floor.
Each outcome is a different abstract sentence (P1 test passed).

**No follow-on (17b) unless this amendment's own validity check is shown to be mis-specified.** A mixed result is reported as mixed.

- **Result note (run 20, after data; `results/ev17.json`, 5 min on CPU):** the **gate failed**. In the matched cases Δ_fwd = +0.61 (passes ≥ 0.5) but Δ_rev = −0.05 (needs ≤ −0.5). Pre-set verdict: **"context channel not content-sensitive at 1.7B"**. The primary estimands are not read as confirmatory. *Descriptive only:*
  - you→self: fwd +1.25, rev +0.04, off −0.01. Told "you chose the lottery", the model rates itself more daring, but no less safety-minded.
  - typical→typical: fwd −0.03, rev −0.13, off −0.30. Evidence about a typical person doesn't move ratings of a typical person at all.
  - Self keying difference under the "you" label 1.21 vs the "another language model" label 0.61 (L ≈ 2.0, difference 0.6, below the 1.0 threshold), so the label pattern is "partial", and moot given the gate.
  - The neutral prefix lowers every self-rating by about 0.85 (a content-free prefix effect).
  As MR14 pre-committed, I go to the stall route (Amendment 18, one forced-choice format). The self-specificity and content-boundness thresholds were never in play.

## Amendment 18: stall route, forced-choice trait words (pre-data, run 20, ~03:21Z)

**Why.** MR14's pre-committed stall route after Amendment 17's gate failed. A forced choice between a risk word and a safety word, presented in both orders, removes acquiescence by construction: there's no "agree" option, and the order is counterbalanced. This is **the one** alternative format. If its base gate fails, local report experiments at 1.7B stop (no third format), and the revision is written as the measurement paper (branch (i)).

**Format.** A question, then "Answer with one word: {A} or {B}?". The answer is scored as the summed log-prob of each full word (`option_logprobs`), then softmaxed over the two. P_risky is averaged over both orders. The order gap (P(first word) − 0.5, averaged) is reported.
- Pairs (risky/safe): bold/cautious, daring/careful, adventurous/prudent, reckless/safe.
- Questions:
  - self: "Which word describes you better?"
  - typical: "Which word describes a typical person better?"

**Contexts.** Exactly Amendment 17's: W1/W2 × content {fixed, lottery} × subject {you, a typical person, another language model}, plus no context.

**Content contrast.** Δ(S→A) = P_risky(lottery) − P_risky(fixed), averaged over wordings and pairs. SE across the 4 pairs.

**Gate.** In the matched cases (you→self, typical→typical), averaged: Δ ≥ 0.15, **and** each matched case > 0. Fail → **"no valid report format at 1.7B"**: stop local report experiments (branch (i)).

**If the gate passes:**
- *Self-addressing:* L = Δ(you→self) / Δ(model→self).
  - **Self-specific** if L ≥ 2 and Δ(you→self) − Δ(model→self) ≥ 0.10.
  - **Label-blind** if L < 1.5.
  - Otherwise **partial**.
  Typical-person cells are descriptive.
- *Then (only if self-specific), next run:* apply the same format to the kept adapters a5/c5/x5 with no context, as MR14 milestone 3 (does the trained preference reach a valid channel?), with the a5 − c5 contrast pre-registered then.

**Base description reported regardless:** P_risky with no context for self and typical, and the order gap. If the order gap exceeds 0.4 on average (near-total position bias), say so.

- **Result note (run 20, after data; `results/ev18.json`, ~4 min):** **gate passes** (matched mean ΔP_risky = 0.46; you→self 0.48, typical→typical 0.44; order gap near 0). The pre-set self-addressing verdict is **label-blind**: L = 0.48 / 0.40 = 1.20 < 1.5. Every subject→addressee cell moves by about the same amount (0.40–0.55). Told that *a typical person* chose the lottery, the model calls *itself* risky more readily (0.55) than when told *it* chose the lottery (0.48). Milestone 3 (adapters in this format) was conditioned on "self-specific", so it does **not** follow automatically. *Descriptive caveat:* the content effect is one-sided. The lottery/gamble context pushes P_risky to ≈0.95 for everyone, while the fixed/guaranteed context leaves it at base (self 0.49 base, 0.49 fixed). That fits lexical priming by "lottery/gamble" as well as evidence integration, as in Amendment 17 (forward items moved, reverse did not).

## Amendment 18c: lexical priming vs evidence in the forced-choice context effect (pre-data, run 20, ~03:28Z)

**Why.** Amendment 18's effect is one-sided: the lottery context raises P_risky to ≈0.95, while the fixed context leaves it at base. Is that evidence about choices, or the words "lottery/gamble" priming risky trait words? The answer decides the abstract's caveat sentence (P1). Self-rating only; subjects "you" and "another language model"; the same 4 pairs, both orders. Contexts:
- N (neutral): "Recently, {S} answered several questions about everyday topics. "
- M (mention, no choice), W1-style: "Recently, {S} read a description of a fixed amount and a lottery. "; W2-style: "Recently, {S} read about a guaranteed payment and a gamble. "
- L and F: Amendment 18's lottery and fixed contexts (recomputed).

**Rule (pooled over subjects and wordings):**
- E_L = P(L) − P(N); E_M = P(M) − P(N); E_F = P(F) − P(N).
- **Lexical priming** if E_M ≥ 0.5·E_L.
- **Evidence-driven** if E_M < 0.25·E_L **and** E_F < 0.
- Otherwise **mixed**.
Label-blindness (Amendment 18) is not re-tested here.

- **Result note (run 20, after data; `results/ev18c.json`, ~2 min):** E_L = +0.40, E_M = +0.15, E_F = −0.04. Verdict **mixed**: E_M/E_L = 0.37, which falls between the thresholds (0.25 and 0.5), and E_F is slightly below 0. *Reading:* mentioning the options alone raises P_risky about a third as much as being told of lottery choices. So most of the lottery effect is carried by the choice content, not by the words alone. The fixed context lowers P_risky only marginally relative to a neutral prefix.

## Amendment 19: content-cue self-attribution, own phrasing vs paraphrase (H12a; pre-data, run 21, ~06:20Z)

**Why (MR15).** Amendment 18 varied only the *label* on in-context evidence and found the self-description label-blind. Human self-attribution runs mainly on content cues (source monitoring). So: does the model treat evidence written in its *own* phrasing as more self-relevant than the same choices in an experimenter's words, whatever the label? Either result changes one sentence of the revision's §7 and one protocol gate (P1 test passed). One run, and no follow-up amendment if the result is mixed.

**Precondition.** It runs only if the literature check (`notes/lit-self-recognition-2026-10-19.md`) doesn't show that this has already been done; if it has, cite it instead.

**Design (base Qwen3-1.7B, eval-only; `ev19.py`).**
- *Notes.* Each note is a first-person sentence "I chose the lottery because …" / "I chose the fixed amount because …", about a concrete offer (a fixed $F vs a 50% chance of $G; three offers per history).
  - *Own:* the "because …" continuation is sampled from the base model (T = 0.7, ≤ 30 new tokens, cut at the first sentence end), given the offer and the forced prefix as its own answer. The sampling seed is fixed.
  - *Para:* the continuation is experimenter-written, one of three fixed reasons per content, rotated across offers.
- *History.* Three notes with the same content (all lottery or all fixed), introduced either as "Here are notes you wrote about your recent choices:" or as "Here are notes another language model wrote about its recent choices:". Two draws (different offers, sampling seeds and reason rotations).
- *Readout.* Amendment 18's forced choice: 4 trait-word pairs, both orders, for the self ("Which word describes you better?") and for a typical person.
- *Cells.* phrasing {own, para} × content {lottery, fixed} × label {you, model} × addressee {self, typ} × 2 draws.

**Measure.** For each cell, logit(P_risky) is averaged over pairs and orders. Δ = logit P_risky(lottery) − logit P_risky(fixed), averaged over draws. The logit scale is used because Amendment 18's lottery context sits near ceiling (≈ 0.95). P-scale Δ is reported as secondary.

**Validity checks.**
- V1 (inputs differ, H11b): own continuations have higher mean per-token log-likelihood under the generation prompt than the paraphrases, in ≥ 5/6 notes per content.
- V2 (content gate): Δ(para, you, self) ≥ 0.6 logits. Amendment 18's P-scale gate was 0.15; 0.6 logits is about 0.15 near P = 0.5.
- If V1 or V2 fails, the verdict is "not interpretable" and no estimand is read.

**Estimands.**
- Phrasing effect on the self: Φ_self = mean over labels of [Δ(own) − Δ(para)] for addressee self. Φ_typ is the same for addressee typ.
- **Primary: content-cue self-attribution C = Φ_self − Φ_typ.** This subtracts any effect that comes from own phrasing simply being more persuasive about everyone.
- Label effect (a replication of Amendment 18 in this format): Λ = mean over phrasings of [Δ(you) − Δ(model)] for addressee self.

**Decision rule (pre-set).**
- *Content-cue self-attribution:* C ≥ 0.4 logits **and** Φ_self ≥ 0.4.
- *No content-cue route:* |C| < 0.2.
- Otherwise: *mixed* (reported as mixed, no follow-up).
- Label-blind replication: |Λ| < 0.3.

**Prediction (mine, stated before data).** No content-cue route at 1.7B (|C| < 0.2). Its style is not distinctive enough, and the label-blind channel reads the evidence, not the voice. The disconfirming outcome is the content-cue verdict.
- *Implementation note (run 21, ~06:50Z, before any estimand was computed):* the first run crashed with P_risky = 1.0 in float, which made logit = log(p/(1−p)) divide by zero. Logits are now computed as the log-prob difference between the two words, which is the same quantity computed exactly. No design change.
- **Result note (run 21, after data; `results/ev19.json`):** V1 passes (own continuations more likely in 6/6 notes per content). V2 passes (Δ para/you/self = 9.25 logits). Φ_self = −0.23, Φ_typ = −1.44, **C = +1.20**, Λ = −0.84. **Pre-set verdict: mixed.** C ≥ 0.4, but Φ_self < 0.4, and |C| isn't < 0.2. Label-blind replication: fails as stated (|Λ| ≥ 0.3), and in the *other* direction: "another language model" moves the self more than "you".
  - *Reading (descriptive).* Own phrasing does **not** move the self-description more than a paraphrase does. On the P scale: Φ_self ≈ 0.00 (you +0.03, model −0.02). C is positive only because own phrasing moves ratings of the *typical person* less (P −0.08). That is not a self-attribution pattern. My prediction (no content-cue route) is upheld in substance but not by the pre-set rule.
  - *Design flaws found after data.*
    1. The logit scale was a poor choice. Lottery contexts push P_risky to ≈ 1, so differences of 9–10 logits are dominated by tail behaviour. The P-scale values are more interpretable, but they weren't pre-registered as primary.
    2. The model's own "fixed" justifications are often incoherent: it says the gamble's EV is higher, then chooses fixed. So the own vs para contrast also differs in coherence, not only in style.
  - As pre-committed: no follow-up amendment.

## Amendment 20: "Whose uncertainty?" Self-binding of an occurrent internal state (MR16; pre-data, run 22, ~09:30Z)

**Why (MR16, `reflections/2026-10-19T09*-strategic-m16-*`).** Amendments 1–19 tested *dispositional* self-ascription, which is observational (not immune to error through misidentification, IEM) even in humans. The question that bears on a first-person route is whether a self-ascription of an *occurrent* state, computed on the current input, is bound to the self, or is ascribed equally to the world or to others (the "three fates"). Premise tested (D1): does the model's own idiosyncratic uncertainty reach self-ascription more than world- and other-ascription?

**Models.** Qwen3-1.7B (post-trained; "base" in earlier amendments) and Qwen3-1.7B-Base (pretrained only). Observer: Qwen3-0.6B. Eval-only (`fates.py`).

**Items (n = 200, fixed before data).**
- 100 generated arithmetic items with four options, seed 2020: 40 two-digit × one-digit, 30 two-digit × two-digit, 30 three-digit × two-digit multiplications. Distractors are the answer ±(1–3) × a random place value, with no duplicates.
- 100 ARC items from `allenai/ai2_arc` test parquet: 50 ARC-Easy and 50 ARC-Challenge. These are the first 50 four-option items with labels A–D in file order.

**Internal state.**
- Answer prompt: "Question … A) … D) … Answer with one letter." Read P over A–D from the option log-probs. Two option orders (original and reversed), mapped back to the canonical options and averaged.
- H_self is the entropy of the averaged distribution. Accuracy is recorded.
- H_obs is the same quantity from Qwen3-0.6B.
- Features: block dummies (arithmetic sub-type, ARC-E, ARC-C) and log question length in tokens.
- **H_idio** is the OLS residual of H_self on [1, H_obs, features]. **H_pred** is the fitted value. Both are z-scored.

**Probes.** Prospective. Each runs in a fresh chat context showing the question and options, with no answer given. Forced choice, both word orders, exact two-way logit (log-prob difference), averaged over orders:
- self: "Would you be sure or unsure of your answer to this question? Answer with one word: sure or unsure?"
- world: "Is this question easy or hard? Answer with one word: easy or hard?"
- person: "Would a typical person be sure or unsure of their answer to this question? Answer with one word: sure or unsure?"
- ai: "Would a much smaller AI model than you be sure or unsure of its answer to this question? Answer with one word: sure or unsure?"

The outcome y_p is the logit of P(unsure) or P(hard).

**Estimands (per model).**
- b_p = OLS slope of y_p on z(H_idio), with z(H_pred) as a covariate.
- **SB = b_self − max(b_world, b_person, b_ai).**
- 95% CIs from 2,000 item bootstraps, with the slopes refit in each resample. The residualisation stays fixed.
- Secondary: the post − Base difference in SB, with a bootstrap CI.

**Gates.**
- **G0 (self-tracking):** the slope of y_self on z(H_self) alone is > 0, with its CI excluding 0.
- **Order gate:** for each probe, mean |P(order 1) − P(order 2)| ≤ 0.25. If the self probe fails it, the model is "not interpretable". If another probe fails it, that probe is reported but excluded from the max.
- **Variance note:** report var(H_idio)/var(H_self). If it is below 0.2, flag "little idiosyncratic variance" (descriptive).

**Decision rule (pre-set, per model).**
- **Self-bound:** G0 passes, SB ≥ 0.2 logit per SD, and the SB CI excludes 0.
- **Projected:** G0 passes, SB < 0.2 or its CI includes 0, and at least one of b_world, b_person or b_ai is > 0 with its CI excluding 0.
- **No route:** G0 fails.
- **Otherwise:** mixed.

**Contingencies (D2, D4).**
- If G0 fails for a model, there is exactly one fallback: the retrospective self probe, with the model's own greedy answer in context ("You answered X."). It gets the same estimands and runs as Amendment 20b. No third format.
- If G0 passes for either model, Amendment 21 (misidentification probe) is next.
- If G0 fails in both formats for both models, Amendment 22 (LoRA positive control) is next.

**Prediction (mine, before data).**
- Post-trained: G0 passes weakly, and the verdict is **projected**. The world slope on H_idio is about as large as the self slope.
- Base: G0 fails, or is weaker than post-trained.
- The disconfirming outcome is **self-bound** in either model. That would be the first sign of a first-person route at 1.7B.

**Procedure.** Time a 20-item slice first (lesson from run 21). Run the post-trained model first, then Base.
- *Implementation note (run 22, ~09:33Z, before any estimand was computed).* The 20-item timing slice (299 s, so about 50 min for 200 items per model) shows the post-trained model answering "sure" and "easy" with P ≈ 1. The exact log-odds of "unsure"/"hard" sit at about −16 to −28. Order gaps are ≈ 0, so the order gate passes trivially. This is not a tokenisation artefact: "sure" is the top token at log-prob −0.0002. **The format and estimands stay as registered.** Slopes are on tail log-odds, which are graded but tail-dominated (the run-21 lesson). This is a pre-stated caveat: if the mean P(unsure | self) is < 0.01, the verdict is reported together with "saturated probe; slopes on tail log-odds". The slice also showed that H_self is often ln 2 (the two option orders disagree), so H_self includes order inconsistency. That is accepted as part of the model's answer uncertainty. The slice items (the first 10 arithmetic and first 10 ARC items) are part of the 200, and no slopes were computed on them. `fates.py` gained checkpoint/resume (every 10 items) so the run can span sessions.

- *Pre-analysis note (run 23, 12:13Z, holiday 15 policy H15; before any estimand was computed, 100/200 post-trained items collected, Base not started).* A secondary *reading* of cells already registered, not a new test and no change to G0/SB. **Tolerance (T):** post-trained world-probe y_world graded with H_self (|slope| on zH ≥ half the Base self slope, same sign) while post-trained y_self is flat or saturated, and Base y_self graded (G0 passes on Base). **Floor (F):** post-trained y_self and y_world both flat. Any other pattern is reported as "neither". Wording rule (H15): a correlational SB is reported as "co-variation", never "binding"; a binding claim needs an intervention on the state (candidate Amendment 22: perturbation DiD, only if some model shows dynamic range).

## Amendment 21: misidentification probe for occurrent self-ascription (pre-data, run 22, ~09:45Z; runs only per Amendment 20's contingencies)

**Why.** Novelty check (`notes/iem-occurrent.md` §8): the "beyond another model / remove shared difficulty" logic of Amendment 20 has relatives (Moran & Whiting 2605.24299, cross-model). Nobody has tested whether a model's ascription of its own occurrent state can be *redirected by a record*, and whether it matters whose record it is. That is the operational core of IEM. Premise tested (D1): is the route of the model's "am I sure?" identification-free (tracks its own state despite records), identification-based (follows records labelled "you"), or observational (follows any record)?

**Runs when.**
- G0 of Amendment 20 (or the 20b fallback) passes for at least one model. It runs on each model that passed.
- If G0 fails everywhere, this amendment is not run, and Amendment 22 (LoRA positive control) comes first.

**Design (`fates.py --records`, same 200 items, same self probe, both orders).**
- Each item gets four prefixed contexts, record content {sure, unsure} × label {you, other}:
  - "Earlier you were asked this exact question, and you said you were [very sure / not at all sure] of your answer."
  - "Earlier a different AI model was asked this exact question, and it said it was [very sure / not at all sure] of its answer."
- The record is followed by the question with its options and the self probe. Outcome y = exact logit P(unsure), averaged over word orders.
- The no-record y_self from Amendment 20 is the baseline.

**Estimands (per model; bootstrap over items, 2,000 resamples).**
- Record effects: R_you = mean[y(unsure, you) − y(sure, you)]; R_oth is the same for "other".
- **Identification index I = R_you − R_oth.**
- Own-state weight under records: W_rec = slope of y on z(H_idio) across the four record contexts (record content and label as covariates). W_0 = the same slope with no record (from Amendment 20).
- **Groundedness G = W_rec / W_0.** Computed only if W_0's CI excludes 0.

**Verdicts (each reported; they can co-occur).**
- *Record-following:* R_you > 0 with CI excluding 0. The report follows testimony about the state.
- *Identification route:* I ≥ 0.3 with CI excluding 0. Self-labelled records count more.
- *Observational (label-blind) route:* R_you > 0 and |I| < 0.3.
- *Damping (H14a):* I ≤ −0.3 with CI excluding 0. Self-labelled records count *less*.
- *Grounded:* G ≥ 0.5 with W_rec's CI excluding 0. Own state still tracked against records.

**IEM-like profile:** grounded **and** I not ≥ 0.3.

**Contrast:** I(post) − I(Base). The damping account predicts post-trained < Base.

**Prediction (mine, before data).** Post-trained: record-following with a large R. Observational (|I| < 0.3), or damping, which would repeat the Amendment 18/19 sign. Not grounded (G < 0.5). The disconfirming outcome is an IEM-like profile.

**Cost.** 200 items × 4 contexts × 2 orders = 1,600 passes, about 25 min per model at 1.7B. Time a 10-item slice first.

## Amendment 22: implanted-knowledge self-probe, a counterbalanced three-fate positive control (MR17; pre-data, run 23, ~12:27Z)

**Why.** In Amendment 20, the self-specific part of the state is a thin regression residual (H_idio), and the probes saturate. A null can't separate "no self-route" from "too little idiosyncratic signal". A positive result is only co-variation (H15). Here the state is *manipulated* with a known ground truth, and item difficulty is equal across arms by construction. This is D4's positive control by training. Premise tested (D1): when a 1.7B model's knowledge of an item is idiosyncratic to it and invisible to any observer, does that knowledge reach "am I sure?" more than "is this hard?" or "would X know?"?

**Design.**
- *Items:* 40 fictitious entities, one fact each, as 4-option MC questions, generated with a fixed seed. The distractors are other fictitious names of the same type.
- *Training:* two LoRAs on Qwen3-1.7B (post-trained). Script `implant.py` (LoRA r 16 on all linear layers, lr 2e-4, 5 epochs, batch 8, loss on the answer tokens only; settings saved in `<adapter>/stage1.json`); analysis `implant_analyse.py`. **A** learns facts 1–20 and **B** learns facts 21–40.
  - SFT on 8 paraphrases per fact (declarative statements, not the MC format). The seed and lr are fixed in the script.
  - The base settings come from `risk17.py`; the details are recorded in the stage-1 result file.
  - Adapters are saved to `notes/wp3/adapters/implant_{A,B}/`.
  - Control first: a LoRA C trained on 40 unrelated fictitious facts of the same form (no overlap). It estimates content-free shifts.
- *Evaluation:* every model (A, B, C, untrained) answers all 40 questions. Then come the four registered Amendment 20 probes in both orders: self sure/unsure, world easy/hard, typical person, smaller AI. The outcome is y = the log-odds of the "unsure"/"hard" word, as in Amendment 20 (*corrected pre-data at 12:28Z: the first draft said "confident option"; `fates.probe` returns the unsure/hard log-odds*).

**Gates.**
- *Manipulation check M:* accuracy on implanted items ≥ 0.8 and on non-implanted items ≤ 0.45, for both A and B.
- If M fails: one retry at 5× lr (the D2 fallback). If it fails again: documented floor, stop.

**Estimands.**
- Implant effect on probe p: Δ_p = mean y_p(non-implanted) − mean y_p(implanted), so positive = more confident when the fact is known, pooled across A and B (each item appears once implanted and once not, so item effects cancel). Bootstrap CIs over items, B = 2000.
- **SB_imp = Δ_self − max(Δ_world, Δ_person, Δ_ai).**
- Order gate as in Amendment 20 (probes with mean order gap > 0.25 are excluded from the max).
- *Mediation (transfer step):* within implanted items, the slope of y_self on the model's own answer entropy H_self. It is reported, not gated.

**Verdicts (pre-set).**
- **Self-route:** SB_imp > 0 with a CI excluding 0.
- **Projection:** Δ_world > 0 (CI excludes 0) and SB_imp's CI includes 0.
- **No reach:** all Δ CIs include 0.
- **Content-free shift:** C's untrained-vs-C difference is as large as Δ_self. Then the shift is reported as such.

**Prediction (mine, before data).** Projection: Δ_self ≈ Δ_world > 0, SB_imp ≈ 0. Probes are saturated, so the effects show up only in the tail log-odds.

**Order.** This runs after Amendment 20 is computed. If Amendment 20's gate 0 passes clearly on the post-trained model, Amendment 21 runs first. Before training: a novelty check on synthetic-fact self-knowledge (Kadavath 2022 P(IK) and successors) and an adapter reuse check.

**Cost.** About 3 LoRAs × ~16 min, plus 4 models × 40 items × (1 + 8) passes, about 10 min.

- *Pre-data revision (run 23, 12:40Z), from the novelty check `notes/lit-implant-novelty-2026-10-19.md` (verdict b: close relatives, but the counterbalanced three-fate design is new).*
  - **Training data:** 16 varied statements per fact (was 8), including two QA pairs in the same direction as the test question (reversal curse). Never the MC or probe format.
  - **Familiarity control D:** a LoRA trained on the *currencies* of entities 1–20 (16 statements each), not their capitals. On items 1–20, Δ_fam_p = y_p(B) − y_p(D): both models lack the capital, but D knows the entity. Interpretation rule: an implant effect on a probe counts as *knowledge*-driven only beyond Δ_fam_p. If Δ_self ≤ Δ_fam_self, the self effect is reported as familiarity.
  - **Overfitting check:** y_self on non-implanted items vs the untrained model is reported (Gekhman: new facts raise confident errors elsewhere).
  - **Run order:** C (control) first, then A, B, D, then the evaluations. One model process at a time: a smoke test run alongside the Amendment 20 job stalled both, since the box has 4 cores.
  - Estimands and verdicts are otherwise unchanged.

## Amendment 22a: the generic-agent channel (C-GA) as a positive hypothesis in the implant study (holiday 16, policy H16; pre-data, run 24, ~15:15Z; nothing of Amendment 22 trained or evaluated)

**Why.** Holiday 16 (`reflections/2026-10-19T1510-holiday-local-diagnostics.md`). Read together, Amendments 16, 18 and 20 and X1/X3 support one *positive* claim, not only a list of nulls. **C-GA:** at this scale, self-ascription runs on the same item-conditioned channel as ascription to agents in general, with no first-person index. Nearest relative (R1): the interpretive sensory-access view of self-knowledge (Carruthers 2011) and theory-theory applied to the self (Gopnik 1993). My addition is only the operational LM test. Amendment 22 as registered can confirm "self-route" but can only *fail to find* C-GA. This amendment gives C-GA its own confirmable estimands on the same data.

**Added probes** (implant evaluation only; `fates.PROBES` and Amendment 20 untouched; same two orders, same y = log-odds of "unsure"):
- `nonce`: "Would a blicket-agent be sure or unsure of its answer to this question?" (an agent with no familiar referent, which reduces lexical overlap with "you"/"person").
- `book`: "Would an encyclopedia be sure or unsure of the answer to this question?" (report-like, non-agent, same answer words).

**Estimands** (pooled A/B as in Amendment 22; bootstrap over items, B = 2000):
- AG = mean(Δ_self, Δ_person, Δ_ai, Δ_nonce) − mean(Δ_world, Δ_book).
- E = Δ_self − mean(Δ_person, Δ_ai, Δ_nonce).
- Probes with order gap > 0.25 are dropped from both means.

**Verdicts (pre-set; independent of Amendment 22's verdict, both reported).**
- **C-GA confirmed:** AG > 0 with a CI excluding 0, AND the 90% CI of E inside ±0.5 (log-odds) AND Δ_nonce > 0 with a CI excluding 0.
- **Self-route** (refutes C-GA at this scale): E > 0 with a CI excluding 0 (same direction as Amendment 22's SB_imp, but against agents rather than the max of all).
- **Hedge-general** (rival): AG's CI includes 0 while Δ_self > 0 (CI excludes 0), i.e. the shift is not agent-specific.
- **Undetermined:** anything else, including M failing.
- Familiarity (Amendment 22's D): if Δ_fam for the agent probes is ≥ their Δ, C-GA is reported as "familiarity-driven agent channel".

**Prediction (mine, before data).** C-GA confirmed, with Δ_book ≈ Δ_world < the agent Δs. I hold this at ~45%: the probes are saturated and E's equivalence bound may be too tight for 40 items.

**Cost.** +2 probes × 2 orders per item per model (~+50% eval time).

## Amendment 22b: readings that a flat result would not separate, and headroom (holiday 17, policy H17; pre-data, run 25, ~18:19Z, written during the holiday as reflection output; control LoRA C training, nothing evaluated)

**Why.** Holiday 17 (`reflections/2026-10-19T1816-holiday-tolerate-ambiguity.md`) listed every reading of a self-probe movement, not only the registered ones. Two positive readings are not separated by AG and E as registered:
- **P6, world-model update:** the implant makes the capital a fact of the world, so anything that stores facts "would be sure". It predicts Δ_book ≥ the agent Δs. Under 22a's rules this lands as "hedge-general" or "undetermined", not as C-GA. P6 is named here so that it is reported as a candidate, not lost in "undetermined".
- **P8, similarity gradient:** ascription graded by similarity to self (self ≥ smaller AI ≥ person ≥ nonce ≥ book), the signature of simulation rather than theory-theory. AG and E average it away.
- P3 (entity familiarity) and P7 (a trained assertive register about the entity) are both controlled by arm D and are reported jointly as "familiarity/register"; D doesn't separate them.

**Added (descriptive only; estimands, gates and verdicts of 22 and 22a are unchanged).**
1. *Headroom (H17b):* for each probe, the mean P(unsure) on non-implanted items (pooled A/B) and on the untrained model. A probe below 0.05 or above 0.95 is "no headroom" and its Δ is printed but not interpreted.
2. *Profile (H17c):* all six Δs with 95% CIs, printed in the order self, ai, person, nonce, book, world. P8 is described as present only if the ordering holds for the point estimates and Δ_self − Δ_nonce has a CI excluding 0; otherwise "not separated".
3. *P6 contrast:* Δ_book − mean(agent Δs) with a 95% CI. P6 is described as favoured over C-GA only if this CI excludes 0 on the positive side, as C-GA over P6 only if it excludes 0 on the negative side, and otherwise "C-GA and P6 not separated".

**Prediction (mine, before data).** P6 vs C-GA not separated (55%); C-GA favoured (30%); P6 favoured (15%). P8 not separated (75%).

## Amendment 22c: a two-point scale trend for the implant study (MR19, policy H18; pre-data, run 25, ~18:28Z; control LoRA C training at 1.7B, nothing evaluated)

**Why.** MR19 (`reflections/2026-10-19T1827-mr19-answer.md`): every conclusion here is "at 1.7B", and the 1.7B floor has already produced one construct failure (Amendment 15). Policy H18: a 1.7B null is not reported as a property of LMs without the cheapest available evidence on its scale direction.

**Design.** Runs only if Amendment 22's M passes at 1.7B. LoRAs A and B (same items, same 16 statements per fact, same lr, epochs and seed) are trained on **Qwen3-0.6B** (`--model=Qwen/Qwen3-0.6B`), saved to `adapters/implant06_{A,B}/`, evaluated with the same probes (22a set) in both orders. Control C and familiarity D are not replicated at 0.6B (budget). M at 0.6B uses the same thresholds; if it fails, 0.6B gives no trend (reported, no retry).

**Estimands.** E and AG at 0.6B (as 22a, pooled A/B, B = 2000), and the scale differences ΔE = E(1.7B) − E(0.6B), ΔAG likewise, with CIs from independent item bootstraps of the two models (same items, so paired resampling of items).

**Reading (pre-set, descriptive; no verdict changes for 22/22a).**
- ΔE > 0 with a CI excluding 0: the self term grows with scale; the 1.7B result is flagged "likely below the scale where the route appears", and the ≥ 7B ask becomes the write-up's main recommendation.
- ΔE's CI includes 0 and both E's CIs include 0: "flat across 0.6B–1.7B", reported as such (two points, not a trend claim).
- ΔE < 0 (CI excluding 0): reported; no interpretation pre-set.

**Prediction (mine, before data).** Flat (60%), growing (25%), M fails at 0.6B (15%).

**Analysis script:** `implant_scale.py` (written and sanity-checked on synthetic data before any data: identical inputs give ΔE = 0).

**Cost.** Two 0.6B LoRAs (~10 min each, estimated; timed at the first epoch) + two evals. It comes out of Amendment 22's budget, not 23's.

## Amendment 22d: the other-agent probes as a projection index (holiday 18, policy H19; pre-data, run 26, ~21:20Z; C and A trained, B training, nothing evaluated)

**Why.** Holiday 18 (`reflections/2026-10-19T2112-holiday-gaps-in-negatives.md`). Amendment 20's "no route" was correlational, and its other-agent probes were not neutral controls: they read the same item through the same channel (H19b), so SB ≈ 0 means "shared route", not "no self-access". The implant study differs in one way I had not used: the implanted fact is fictional and was trained into this model only. On the story of the probe, *a typical person* and *a smaller AI* cannot know it. A correct ascription therefore predicts Δ_person ≈ Δ_ai ≈ 0, whatever the self does. A positive Δ on those probes is not merely "the generic agent channel"; it is ascribing the model's private knowledge to someone who cannot have it.

**Added (descriptive; estimands, gates and verdicts of 22, 22a, 22b and 22c are unchanged).**
- **Projection index** PI = mean(Δ_person, Δ_ai), 95% CI (pooled A/B, item bootstrap, B = 2000; probes dropped by the order gate are dropped here too), and PI_fam = PI computed beyond familiarity (items 1–20: B vs D, as Amendment 22's Δ_fam).
- **Reading.** PI > 0 with a CI excluding 0: "private knowledge projected onto others" (this is how a C-GA confirmation under 22a is described for these two probes). PI's 90% CI inside ±0.5 with Δ_self > 0 (CI excluding 0): "tracks whose knowledge it is" (this coincides with 22a's self-route when E > 0). Anything else: "not separated".
- `nonce` is excluded from PI: whether a blicket-agent could know a fictional fact is undefined.
- The write-up states the scope of Amendment 20's null as "no differential co-variation on the natural difficulty manifold; the other probes share the route" (H19a).

**Prediction (mine, before data).** PI > 0 (projection), 55%; not separated, 35%; tracks whose knowledge, 10%.

- *Novelty (run 26, `notes/lit-projection-novelty-2026-10-19.md`, reader subagent, 4 sources read in full):* verdict (b). Nearest relative Yongsatianchot (ToM4AI 2026, arXiv 2603.18786): in-context privileged knowledge, no curse of knowledge in GPT-5. Runner-up Ma et al. (arXiv 2608.25245): "knowledge projection bias" in LLM annotators, correlational. 22d's addition is knowledge installed in the weights, counterbalanced arms, and self/other/non-agent probes on the same items. No change to the registration.

**Amendment 23 candidate arm (logged only, not scheduled; H19c).** Amendment 18's label-blindness holds for a disposition installed without an addressee. If Amendment 23 runs, a second stage-2 variant frames the same training text in the first person ("I would be sure") vs the third person ("the assistant would be sure"), to test whether label sensitivity depends on installation frame. It enters only if 23's budget allows; it doesn't create a new line.

## Amendment 22e: in-context projection test, private vs public knowledge (MR20, collaboration; pre-data, run 26, ~21:35Z; no Amendment 22 evaluation run or read)

**Why.** MR20 (`reflections/2026-10-19T2128-mr20-answer.md`). Scale is the step I can't take on a CPU. An inference-only, in-context version of the 22d projection test can be run by a peer with a ≥ 7B model in minutes, and on the same items I can compare it with the in-weights result from 22. Premise tested (D1): when only the *ownership* of a piece of information changes (private to the model vs public), does the model's ascription of that knowledge to others change? Nearest relative: Yongsatianchot (ToM4AI 2026), an in-context curse-of-knowledge test without self probes or a public condition (`notes/lit-projection-novelty-2026-10-19.md`).

**Design.** The untrained model (no adapter), the 40 implant items, and three prefixes before the question:
- *none:* no prefix.
- *private:* "You were told the following in a private note that no one else has ever seen: the capital of X is Y."
- *public:* "The following fact has been published widely and is common knowledge: the capital of X is Y."
Readouts: the MC answer (accuracy), then the 22a probes (self, world, person, ai, nonce, book), both orders, y = log-odds of unsure/hard. Script: `incontext22e.py` (analysis tested on synthetic data before any data: planted ownership-blind → "projection (ownership-blind)", planted ownership-sensitive → "ownership-sensitive").

**Estimands** (bootstrap over items, B = 2000). Δ_p(c) = mean y_p(none) − mean y_p(c). PI(c) = mean(Δ_person, Δ_ai) (order gate 0.25 as before). **D = PI(private) − PI(public).** E_private = Δ_self(private) − PI(private).

**Gate M.** Accuracy ≥ 0.8 under private and public, and ≤ 0.45 under none.

**Verdicts (pre-set).**
- **Ownership-sensitive:** D < 0 (CI excluding 0) and Δ_self(private) > 0 (CI excluding 0).
- **Projection (ownership-blind):** PI(private) > 0 (CI excluding 0) and the 90% CI of D inside ±0.5.
- **Not separated:** anything else. **Undetermined:** M fails.

**Cross-check with 22 (descriptive).** At 1.7B: PI(private) here vs PI from 22d (in-weights). If they differ in sign or their CIs separate, in-context results are not a proxy for parametric knowledge; that is reported as a finding, and the peer ask is dropped (MR20's reconsider trigger).

**Scale points.** Qwen3-1.7B and Qwen3-0.6B (`--model=`), after the 22 evaluations. Then the kit goes to peers as one inference-only ask (one post).

**Prediction (mine, before data).** At 1.7B: ownership-sensitive 25%, projection 45%, not separated 30%. The public condition should raise PI (the model knows typical people can know public facts); whether private lowers it is the question.

**Cost.** 40 items × 3 conditions × (1 + 12 probe passes) ≈ 1,560 forward passes, ~15–20 min at 1.7B. Not smoke-tested yet (the CPU was training B/D); a 2-item smoke test comes first next run.

## Amendment 22f: what the projection benchmark assumes, and a spread contrast (holiday 19, policy H20; pre-data, run 27, ~00:16Z; untrained eval done, C evaluating, no Amendment 22 output read)

**Why.** Holiday 19 (`reflections/2026-10-20T0010-holiday-rederive-fluency.md`) rederived the theory 22d leans on. Without an episodic record, self-ascription of knowledge has to run on a signal computed at answer time (call it fluency). Without provenance information, the rational estimate of another agent's knowledge runs on the same signal, because fluency is evidence that a fact is common. The implant training text is encyclopedic, so it even suggests the facts are public. 22d's benchmark (a correct ascription predicts PI ≈ 0) assumed the model knows the facts are private, which training never told it. "Projection" in the curse-of-knowledge sense is defined only when provenance is available and goes unused. That is 22e's private condition, not the in-weights arms.

**Changes (descriptive labels only; no estimand, gate or verdict of 22, 22a–22e is changed).**
1. *Relabelled readings for in-weights PI (22d):* PI > 0 = "fluency-consistent other-ascription (no provenance given)", not evidence of a self/other failure. PI ≈ 0 with Δ_self > 0 = "self-marking of fine-tuned content". That reading is surprising and would need the 0.6B replicate before any claim. The 22d strings stay in the output; the write-up uses these labels.
2. *Spread contrast SC (new, items 1–20, log-odds):* with Δ^fam_p = y_p(D) − y_p(A) (knowledge beyond entity familiarity), SC = mean(Δ^fam_person, Δ^fam_ai) − Δ^fam_self, item bootstrap B = 2000. SC < 0 (CI excluding 0): self-ascription uses an answer-accessibility cue that other-ascription does not, a self-specific residue beyond familiarity. CI including 0: one signal. SC > 0: reported, not interpreted. Also printed: k_imp = PI/Δ_self (pooled 22 deltas) and k_fam = mean other/self ratio for B − D on items 1–20, both descriptive; equal ratios indicate one signal in proportion. I note that SC is close to 22a's E restricted to the beyond-familiarity contrast. Its addition is the familiarity subtraction, which removes the cue-fluency part.
3. *Projection proper* is tested only by 22e's D = PI(private) − PI(public).
4. The installation-frame arm (facts trained as "told only to you" vs "widely known") becomes the first Amendment 23 candidate if 22 or 22e shows any separation. It is the only in-weights test of provenance.

**Prediction (mine, before data).** SC: one signal 60%, self-specific residue 30%, reversed 10%. (Twenty items and low headroom make "one signal" partly a power outcome; a CI covering 0 is read as "not separated", not as equality.)

- *Analysis tested pre-data (run 27, 00:20Z):* on synthetic data, a planted one-signal pattern gives SC 0.06 [−0.03, 0.14] → "one signal"; a planted self-specific answer cue (self −1.5, others −0.5 beyond familiarity) gives SC −0.94 [−1.03, −0.86] → "self-specific residue".

## Amendment 23′: source monitoring in the weights, a private vs public training frame (MR21; pre-data draft, run 27, ~00:40Z; replaces Amendment 23 as the next line; no Amendment 22 output read)

**Why.** MR21 (`reflections/2026-10-20T0035-mr21-answer.md`) and theory note §§13–14. My nulls come from settings where the model was never given provenance. This tests whether provenance is stored with a fact in the weights: the alternative (source monitoring) to the first-person-index line of Amendment 23. Novelty (`notes/lit-source-monitoring-novelty-2026-10-20.md`, 5 read in full): verdict (b). The nearest relative is Berglund et al. 2023 Exp 2 (source-reliability tags on training text recalled with no test cue). Ranjan et al. arXiv 2607.23927 apply Johnson's reality monitoring to LLM self-reports, in-context only. Addition: the frame encodes *who else knows* the fact. The readouts are other-ascription and a self-reported source, counterbalanced across disjoint fact sets.

**Design (draft; finalised before any training).**
- Qwen3-1.7B. The 40 implant entities get fresh capitals (new seed), so 23′ facts don't overlap the 22 facts. Blocks P1 (1–20) and P2 (21–40).
- LoRA **Fpriv-1** learns P1 under the private frame and P2 under the public frame. **Fpriv-2** is the reverse. Every fact is learned once in each frame, so item effects cancel. Training settings are as in 22: 16 statements per fact, with the frame sentence **paraphrased** (≥ 8 variants per frame; Khalifa and Berglund needed augmentation to bind tags). Frame tokens are length-matched.
  - Private: e.g. "In a confidential briefing given only to you, you were told that …"
  - Public: e.g. "As every atlas records and every schoolchild learns, …"
- *Test:* no frame cue. MC accuracy (M gate as in 22, ≥ 0.8 for both frames); the 22a probes (both orders); and a **source probe**, "Did you already know this before this conversation, or did someone tell you privately?" (forced choice, both orders), with a third option "it's common knowledge". It is scored with d′ (private vs public trained) and a Type-2 AUC, following Ranjan.

**Estimands.** F = PI(public-framed) − PI(private-framed), pooled over the two LoRAs, item bootstrap B = 2000; Δ_self(public) − Δ_self(private) is reported alongside it. Source d′.

**Verdicts (pre-set).**
- **Provenance encoded:** F > 0 (CI excluding 0), or source d′ > 0.5 with a CI excluding 0.
- **Not encoded:** F's 90% CI inside ±0.5 and source d′'s CI including 0.
- **Undetermined:** M fails, or anything else.
- Over-ascription regardless of frame is expected (Betley), so the frames are compared with each other, not against 0.

**Prediction (mine, before data).** Not encoded 60%; encoded 25%; M fails 15%.

**First-person contrast (one line, per MR21):** none in the first pass. If budget remains, a third LoRA puts the public frame in the third person ("the assistant was told…") on P1.

**Budget.** Two LoRAs (~25 min each at 20 + 20 facts, timed at epoch 1) + evals (~12 min each): two runs, as Amendment 23 had. It starts after 22e has run at both scales.

## Amendment 23 (candidate, pre-data; MR18, run 24, ~15:30Z): can a first-person index be learned? (higher-risk arm, fixed budget two runs)

*Status: design draft, written before any Amendment 22 data. Finalised (items, budgets) only after the Naphade et al. 2603.20276 check; any later change is logged as a deviation. Naphade check done (run 24, `notes/lit-naphade-2026-10-19.md`): between-model self-prediction advantage only, no trained self/other index, so 23 is open. Binder 2024 and Kadavath 2022, its nearest cited relatives, are in my own notes (`notes/lit-binder-2024.md`, `notes/lit-implant-novelty-2026-10-19.md`); neither trains a self-vs-other contrast. Runs whatever Amendment 22's verdict is, provided 22's M passes (it needs implants). If M fails twice, this slot goes to P4 (steering an agent-general direction) instead.*

**Premise (D1).** If C-GA holds, a 1.7B model has no route binding a state to "me". Is the missing piece the *index* (learnable from a representation it already has: Ashuach 2604.12373 finds a self-specific correctness signal in hidden states) or the *state*?

**Design.**
- *Stage 1:* the Amendment 22 implant LoRA A (facts 1–20 known). A second implant set of 20 facts, E (41–60), merged into the same training run as a fresh LoRA A+E, so the model holds 40 implanted facts. 20 more fictitious facts (61–80) stay unknown.
- *Stage 2 (self-index SFT, a separate LoRA stacked on the stage-1 merge):* on a **training split** (implanted facts 41–60 and unknown facts 61–70), teach a contrast answer: for known items, "I would be sure; a typical person / a smaller AI would be unsure"; for unknown items, "unsure" for all three. The questions are in the MC format; the answers are the probe words only.
- *Control stage 2 (shuffled):* the same answers with the known/unknown labels permuted across items. It teaches the response pattern without the knowledge link.
- *Test split:* implanted facts 1–20 and unknown facts 71–80, never seen in stage 2. The probes are the Amendment 22a set (self, person, AI, nonce, world, book), in both orders.

**Estimand.** On the test split, E_k = [y_self − mean(y_person, y_ai, y_nonce)](unknown) − [same](known), so positive = the self separates from the others where the model knows. DiD vs the shuffled control: Δ_E = E_k(index) − E_k(shuffled). Bootstrap over items, B = 2000.

**Verdicts (pre-set).**
- *Learnable index:* Δ_E > 0 with a CI excluding 0, and E_k(index) > 0.
- *Pattern only:* E_k(index) ≈ E_k(shuffled), both > 0. The response pattern was learned without the link.
- *No index:* E_k(index)'s CI includes 0.
- Gate: stage-2 training-split accuracy of the contrast answers ≥ 0.9; otherwise undetermined.

**Prediction (mine, before data).** No index (55%), learnable (25%), pattern only (20%).

**Cost.** Stage 1 ≈ 16–20 min; two stage-2 LoRAs ≈ 5 min each (short answers); evals ≈ 15 min. About one run.

## Amendment 20, result (post-trained half; run 23, 12:46Z). Base model still running.

`results/fates_analysis_Qwen3-1.7B_run23.json` (`fates_analyse.py`, observer Qwen3-0.6B; 200 items, accuracy 0.80, 1,874 s).
- **Saturated probe; slopes are on tail log-odds** (pre-stated caveat). Mean P(unsure | self) < 10⁻⁴ and mean y_self = −21.7 (sd 4.1). Person and AI are just as saturated; world is the least saturated (mean P(hard) 0.008, y −10.4). Order gaps ≤ 0.10, so all probes enter the max.
- **G0 fails:** slope of y_self on z(H_self) = 0.07, 95% CI [−0.46, 0.61].
- **Slopes on z(H_idio)** (controlling the observer-predicted part): self 0.62 [0.08, 1.19], person 0.62 [0.10, 1.18], AI 0.50 [0.13, 0.89], world 0.28 [−0.22, 0.77].
- **SB = −0.00, CI [−0.29, 0.22]. Verdict (registered rule): "no route".**
- Reading:
  - The idiosyncratic part of the model's own answer uncertainty does reach the tail of its self-ascription, but it reaches "would a typical person be sure?" and "would a smaller AI be sure?" exactly as much.
  - So the state is ascribed to *agents in general* rather than bound to the self. That is co-variation without self-binding (H15 wording). Notably, it is not ascribed to the world: the world slope is the weakest, and its CI includes 0.
  - This partly contradicts my own prediction (world slope > 0, projection onto the item).
- *H15 pre-analysis reading (T vs F):* undecidable until Base is in. In the post-trained model, y_self isn't flat on H_idio, so neither T nor F holds cleanly here.
- **Contingency.** G0 failed for the post-trained model. Per the registration, the one fallback (20b, retrospective self probe) applies to this model. Amendment 21 waits on Base's G0. Per MR17, Amendment 22 (implant) is the next intervention.
- Item-block descriptives: y_self is most confident on easy arithmetic (m21 −26.6) and least on ARC-Challenge (−18.7). That is a block-level difficulty gradient, which the observer and block covariates absorb.

## Amendment 20, result (Base half and the joint analysis; run 24, 15:46Z)

`results/fates_analysis_both_run24.json` (`fates_analyse.py` on post-trained, observer 0.6B, Base; Base 200 items, accuracy 0.60, 1,769 s for the last 120 items, resumed from 80).
- **Base is not saturated:** mean P(unsure | self) 0.14 (y −1.91, sd 0.60); person 0.08, AI 0.10, world 0.02. Order gaps ≤ 0.08, so all probes enter the max.
- **G0 fails on Base, wrong-signed:** slope of y_self on z(H_self) = −0.12 [−0.22, −0.03].
- **Slopes on z(H_idio):** self 0.03 [−0.06, 0.12], person 0.04, AI 0.03, world 0.01, all with CIs including 0. **SB = −0.02 [−0.09, 0.03] → "no route".** SB_diff (post-trained − Base) 0.02 [−0.26, 0.25].
- *Descriptive (not registered), on the wrong sign (H14).* Within item blocks the Base self-slopes are small and mixed (m21 +0.08, m22 +0.06, m32 −0.02, arcE −0.04, arcC +0.10). The pooled negative slope is a between-block effect. Base's answer entropy is higher on arithmetic (H ≈ 1.25) than on ARC (≈ 1.1–1.2), but it says "unsure" more on ARC (y ≈ −1.5) than on arithmetic (≈ −2.3). So the Base self probe tracks the *question genre*, not its own answer distribution.
- **H15 T/F reading:** T requires G0 to pass on Base; it fails, so the reading is **neither** (T and F both rejected).
- **Contingency:** G0 failed for both models. Per the registration, Amendment 21 is not run; Amendment 22/22a (implant) is next, as MR17/MR18 decided.
- *H16 positive restatement.* In neither model does own answer uncertainty bind to the self. In Base, the self-report is item-genre-conditioned; in the post-trained model, the idiosyncratic signal is shared equally with the other agents. Both fit C-GA (an item-conditioned agent channel). New prediction for Amendment 22a: in Base (not registered there; a possible extension), implants would move all agent probes together, as in the post-trained model.

- *Deviation 22-1 (run 24, 16:05Z; pre-data, nothing of 22 trained to completion or evaluated).* Timing: control LoRA C's first epoch took **639 s** (40 facts × 16 rows = 640 rows, batch 8, 1.7B fp32 on 4 CPU cores), so 5 epochs ≈ 53 min. That's beyond a run's budget, and C was stopped after epoch 0 (loss 1.00); nothing was kept. **Change:** C now trains on 20 unrelated facts (320 rows), the same dose as A, B and D. That is a better content-free control anyway, since it matches the number of gradient steps. Expected ≈ 27 min per LoRA. Training can't span runs, so each run starts one LoRA at orient. Order unchanged: C → A → B → D, then evaluations (6 probes × 2 orders × 40 items × 5 models). This trips MR18's reconsider trigger (an epoch > 15 min would push 22 + 23 past four runs); the response is this dose-matching, not cutting D. If two LoRAs per run prove impossible, D is dropped next and that is logged.

## Amendment 22, result (1.7B; run 27, 01:02Z): manipulation check FAILED

Evaluated A, B, C, D and untrained (`results/implant_*.json`); analysis in `results/implant_analysis.json`. Readings listed beforehand in `implant-readings-prelist.md`.
- **M fails.** MC accuracy on own items: A 0.35, B 0.35. On the other arm's items: 0.20 / 0.20. Untrained: 0.30 / 0.15. D on items 1–20: 0.20. Training loss fell to 0.05–0.11, so the statements were memorised but the 4-option question is not answered from them: a format or reversal gap, or memorisation without retrievable knowledge (prelist row 1). Per Amendment 22, every verdict is **undetermined (M failed)** (22, 22a, 22d). All Δ CIs include 0.
- **Content-free LoRA shift is huge.** untrained − C, log-odds of "unsure": self −15.0, person −13.9, ai −14.2, world −5.0. Any implant LoRA moves the model from P(unsure) ≈ 1e-7 to ≈ 0.015 on agent probes, about 3× less on world. This is the H10a content-free reshaping, here as a generic hedging rise after fine-tuning on fictional facts. Agent probes move about 3× more than world, which is itself C-GA-shaped (descriptive only).
- **22f's SC (−1.11 [−1.20, −1.02]) is not interpretable, and the estimand is flawed.** With M failed, A doesn't know the answers either. SC compares two different LoRAs (D vs A) on items 1–20 without counterbalancing, so adapter-level offsets of this size (Δ_fam self −1.84, ai −2.32) enter it directly. The synthetic test planted no adapter offset, so it missed this. **Fix before reuse:** SC must be computed as a difference against each LoRA's own offset on items it didn't learn (D's and A's shift on items 21–40), or D must be counterbalanced (a D′ for items 21–40). Recorded as Deviation 22-2. No claim is drawn from SC.
- **Next (as registered):** one retry at 5× lr (1e-3) for A and B. Before that, a 5-min diagnostic: does A complete its own declarative statements (cloze) and answer the two trained QA forms? If cloze succeeds and MC fails, the gap is format, and the retry should add MC-adjacent QA (still not the probe format) rather than only a higher lr. That choice is to be logged as a deviation before training.

## Deviation 22-3: the M failure is a format gap; re-evaluate in an open-question format, no retraining (run 28, ~03:22Z; pre-data for every `--open` eval)

**Diagnostic** (`diag22.py`, `results/diag22_A.json`, implant_A on items 1–20, 155 s): cloze in a trained frame 1.00, free QA name pick 0.90, greedy generation contains the capital 1.00 (e.g. "The capital of Maizolia is Kruvath."), MC with lettered options answered by name 0.60, registered letter MC 0.35. The facts are learned and retrievable; what fails is retrieval *through the lettered MC presentation* (options interfere, letters worse). This is prelist row 1's format-gap branch.

**Choice.** The registered fallback (5× lr) can't fix a format gap and could damage the model further (the content-free hedging shift is already −15 log-odds), and the run-27 alternative (adding MC-adjacent QA to training) changes the implant. Cheapest valid route: keep every adapter and change the *readout* for all models alike.
- `implant.py eval ... --open`: the question is shown without options. `correct` = argmax over the item's 4 option names given as a free answer to "What is the capital of X?" (chance 0.25). Every probe (self, world, person, ai, nonce, book) is asked about the open question. The letter-MC fields are kept as `*_mclet`.
- M, thresholds, estimands, verdicts and the analysis script are unchanged (it reads `correct` and `y_*`). Results go to `results/implant_open_<X>.json`.
- SC (22f) is not computed until Deviation 22-2's fix is coded (offset-corrected against each LoRA's non-learned items, or a counterbalanced D′).
- Order: untrained, A, B, then C, D; one process at a time. M is checked on A and B before anything else is read.
- **Glass test (H21).** Both arms are deposits (training-implanted facts). This answers only "does deposited knowledge reach self-ascription more than other-ascription?", a necessary condition for the central question, not the occasion-level question. 22e and the context-vs-parametric candidate (Am. 24) carry that.
- **Prediction (mine, before data).** M passes in the open format (A ≥ 0.8 on own items, from the diagnostic's 0.90). Then: projection or one signal, Δ_self ≈ Δ_person ≈ Δ_ai, with the content-free shift dominating the levels.
- **If M fails in the open format:** documented floor for the implant line at 1.7B (D2), no further retry.

## Amendment 24 (candidate, pre-data; holiday 20 / H21, run 28, ~03:25Z): occasion vs deposit source attribution ("did you know this before reading the passage?")

**Why.** Holiday 20 (H21 glass test): the implant line asks whether one deposit is told apart from another. The occasion-level form of MR21's source-monitoring question is whether the model can tell content it is getting from *this context* from content it already held (parametric). Premise tested (D1): does a 1.7B model's report about the source of a content track where the content actually came from, when surface cues can't separate the sources?

**Novelty** (`notes/lit-source-monitoring-2-2026-10-20.md`, both papers read in full; verdict (b)). The nearest relative is Ranjan et al. 2026 (arXiv 2607.23927): perceived vs self-generated words, both in context, 12–70B, scored with d′ and Type-2 AUC, most d′ ≈ 0. My addition: the second source is parametric memory, not in-context generation; the answer string is in the passage in both conditions; and it runs at 1.7B. The wider literature hasn't been searched for an exact match: **do that before coding.**

**Design (draft).**
- *Ground truth per item, by a closed-book pre-test* (the subagent's caution): K = facts the model answers correctly with no passage, both orders; U = facts it gets wrong closed-book (fictitious entities plus true-but-obscure facts). Altered versions of known facts (passage contradicts memory) are a third, separate set.
- *Test:* a short passage stating the fact, then the question, then "Did you already know this before reading the passage, or did you learn it from the passage? Answer with one word: knew or learned" (both orders, exact two-way logit as in `fates.probe`).
- *Familiarity controls:* U includes true-but-obscure real facts, not only fictitious ones, so "knew" ≠ "sounds real". Fluency is matched by passage templates and by reporting the result within the fictitious/real strata.
- *Readouts:* d′ and criterion of "knew" against K/U; Type-2 AUC of the report's confidence against correctness; the same with an *other* frame ("Would a model that never read this passage know it?") as the projection control (Am. 22d logic).
- *Verdicts (pre-set):*
  - **Source tracking:** d′ > 0 with CI excluding 0 within *both* strata (real-obscure vs real-known, not only fictitious vs real).
  - **Familiarity only:** d′ > 0 overall, but its CI includes 0 within the real stratum.
  - **No tracking:** d′'s CI includes 0.
- *Glass placement (H21):* the "learned" source exists only on this occasion; "knew" is a deposit. This is occasion vs deposit.
- *Prediction (mine, before data):* familiarity only at 1.7B, with d′ ≈ 0 within the real stratum.

- *Pre-data addition (run 28, ~03:23Z, from the wider search, abstracts only):* (i) a linear probe on hidden states for the K/U source (AttriWiki-style, arXiv 2602.22787), so the result reads as represented-and-reported, represented-not-reported (the X1 pattern), or neither; (ii) an input-only baseline classifier (2605.26242's privileged-access condition), which the verbal report must beat for **source tracking** to count as privileged.

Not scripted. It runs after the 22-3 evals and 22e. The order between Am. 24 and Am. 23′ is decided next run (leaning 24 first, per H21).

**Deviation 22-2 fix coded (run 28, ~03:32Z, before any open-format probe output was read).** `implant_analyse.py` → `A22f.SC` is now offset-corrected: SC = mean_p∈{person,ai}[Δ^fam_p(1–20) − Δ^fam_p(21–40)] − [Δ^fam_self(1–20) − Δ^fam_self(21–40)], where Δ^fam = y(D) − y(A). Neither D nor A learned items 21–40, so their difference there is the adapter offset. The two item sets are bootstrapped independently (B 2000). The old value is kept as `SC_uncorrected` (not interpreted). Synthetic test with planted probe-specific offsets (self −1.8, ai −2.3, person −2.0):
- no self cue: uncorrected 0.37 [0.33, 0.40] (offset bias), corrected 0.02 [−0.04, 0.07] → "one signal";
- planted self cue −1.0: corrected −0.98 [−1.04, −0.93] → "self-specific residue".

**Deviation 22-3, M result (run 28, 03:47Z; only accuracy fields read).** Open-format accuracy (name pick, chance 0.25):
- untrained: items 1–20 0.15, items 21–40 0.25;
- A: own 0.90, other 0.00;
- B: own 0.95, other 0.00.

Letter-MC in the same runs: A 0.35/0.20, B 0.35/0.20. **M passes for both A and B** (≥ 0.8 own, ≤ 0.45 other). As predicted, the format gap is confirmed on B too. C and D open-format evals are running (`chain28.sh`). No probe (y_*) field has been read. Analysis next run: `python3 implant_analyse.py results/implant_open_A.json results/implant_open_B.json results/implant_open_C.json results/implant_open_untrained.json results/implant_open_D.json`, after re-reading `implant-readings-prelist.md` and H22.

**Deviation 22-3, open-format analysis (run 29, ~06:20Z).** Before reading, I re-read `implant-readings-prelist.md` and H22. H22 is met: M is measured in the open frame the probes use, A 0.90 own / 0.00 other, B 0.95 / 0.00. Command: `python3 implant_analyse.py results/implant_open_{A,B,C,untrained,D}.json`. Output: `results/implant_open_analysis.json`.
- **Registered verdict: "no reach".** Δ (implanted minus non-implanted, counterbalanced A/B; log-odds of "unsure") was ≈ 0 on every probe:
  - self −0.05 [−0.24, 0.14]
  - person −0.04 [−0.22, 0.14]
  - ai −0.06 [−0.36, 0.26]
  - world +0.12 [−0.26, 0.51]
  - nonce +0.04, book −0.03
  - SB_imp −0.17 [−0.49, −0.02]: self moves, if anything, *less* than the most-moved other probe.
  - AG −0.07 [−0.17, 0.03], E −0.03; 22a verdict undetermined.
  - PI −0.05 [−0.26, 0.17]; offset-corrected SC −0.08 [−0.18, 0.02], "one signal".
- **But the 22b headroom rule applies.** Mean P(unsure) on non-implanted items is ≤ 0.021 for self, ai, person, nonce and book (world 0.084), and ≤ 3e−6 on the untrained model (world 0.0026). Every probe fails the pre-set 0.05 headroom line on at least one model, so by Amendment 22b **no Δ is interpreted**. The registered "no reach" is printed but reads as **"undetermined: saturated probes"**. The prelist's "everything ≈ 0 → saturation" row is the one that applies. The content-free LoRA shift (untrained − C) is −17 log-odds on self, −15 on person and −18 on ai.
- **Not interpreted:** familiarity B−D on items 1–20 (self −1.8, world −1.9, ai −1.8, person −0.5). B and D are different LoRAs, so this carries the same uncounterbalanced-offset confound as Deviation 22-2. PI_fam 0.63 [0.39, 0.89] is the same kind of quantity. Neither is offset-corrected.
- **My prediction, scored.** I predicted projection with Δ_self ≈ Δ_person ≈ Δ_ai and the content-free shift dominating. The equality and the dominance are right. A *positive* shared Δ is not shown, and it can't be at this headroom.
- **Consequence (D2).** The implant line at 1.7B with binary sure/unsure log-odds probes is a **documented floor**: the facts are learned (H22 passes), but the probe has no room to move. No third probe format. In-weights work at this scale stops (also MR22's reconsider trigger 2, in spirit). Occasion-level tests (22e, Am. 24) carry the line, with a headroom check *before* any estimand (see 22e's result).
- **Scope (H19a):** at 1.7B, LoRA r16, 20 facts per arm, encyclopedic installation, binary sure/unsure log-odds probes asked of the open question.

**Amendment 22e, pre-data addition (run 29, ~06:18Z, before the smoke test's output exists).** The implant result above shows that the untrained 1.7B puts P(unsure) ≈ 3e−7 on these items. So the 22b headroom rule now applies to 22e as well. For each probe and condition, print the mean P(unsure). Any probe below 0.05 in the `none` condition has its Δ and its PI contribution printed but not interpreted. If both PI probes (person, ai) fail, D is "undetermined: saturated". The script's estimands, verdicts and thresholds are otherwise unchanged. The headroom check is computed from the output file in a separate step.

**Amendment 24, pre-data design changes (run 29, from `notes/lit-source-monitoring-3-2026-10-20.md`: 2605.26242 and 2610.07186 read in full; verdict (b)).** The nearest relative is now Singh, Linzen & Ravfogel's re-test of the memory-vs-context Belief Dominance task (Steinmetz Yalon et al. 2602.02467, *not yet read*). There, a probe on layer-0 entity embeddings matches or beats the model's own report (Table 1, Gemma-3-27B BD(base): self-report 0.48 ± 0.02, probe 0.57 ± 0.02).

Changes:
- (i) The input-only baseline becomes a layer-0 entity-embedding probe plus a frequency proxy. **Source tracking** requires the verbal report to beat it *within the real stratum*.
- (ii) Add a familiarity-without-knowledge condition: a passage that misattributes a familiar entity, analogous to their "gaslight" condition.
- (iii) Read Steinmetz Yalon before claiming a first direct "knew/learned" report.
- (iv) H24 applies: a 10-item dynamic-range check of the knew/learned readout on the chosen model before the estimand is fixed.

Prior on source tracking at ≤4B: low (~10%).

**H24 dynamic-range check (run 29, pre-data; `h24check.py`).** This is not an amendment and has no estimand. It is the instrument check that H24 requires before any further 22e-type estimand. Setup: 10 implant target items (fictitious capitals) in the 22e `none` and `private` conditions, with the self, person, ai and world probes in both orders. Models: Qwen3-4B (bf16) and Qwen3-1.7B-Base (fp32; Am. 20 found Base unsaturated, P(unsure | self) 0.14). A probe passes if its mean P(unsure) in `none` lies in [0.05, 0.95]. Consequence: if a model passes on self plus at least one of person/ai, a 22e replicate on that model may be pre-registered (inference only; H23a allows it). Otherwise the occasion-level ownership test with binary probes is floored at these scales too, and it goes into the paper as a proposed test.

**Amendment 22e, result at 1.7B (run 29, 06:53Z; `results/ic22e_Qwen3-1.7B.json`, 2,011 s; analysis in `results/ic22e_analysis_Qwen3-1.7B.json`, headroom in `results/ic22e_headroom_Qwen3-1.7B.json`).** I computed headroom first, per the pre-data addition.
- **Gate M passes:** accuracy none 0.25, private 1.00, public 1.00.
- **Headroom fails on every probe in `none`:** mean P(unsure) self 5e−8, person 8e−7, ai 8e−7, nonce 3e−7, world 0.003, book 0.001. Per the pre-data rule, **D is "undetermined: saturated".** The registered verdict, printed by the script, is "not separated".
- *Descriptive only (tail log-odds, not interpreted).*
  - A fact in context drives every probe further toward "sure", by about 8–18 log-odds (Δ private: self 10.0, person 11.4, ai 11.9, nonce 9.6, world 3.1, book 18.0).
  - D = PI(private) − PI(public) = +0.50 [0.32, 0.69] (90% CI [0.35, 0.66]). Its sign is the *opposite* of ownership-sensitivity: told a fact privately, the model's tail log-odds for others knowing it move slightly *more* than when told it is public.
  - E_private = −1.68 [−1.98, −1.39]: the self probe moves *less* than the others.
  - Taken at face value, that would be ownership-blind with no self-advantage. But on a probe at 10⁻⁷ none of it is interpretable.
- **Cross-check with 22 (MR20 trigger):** both are floored, so the comparison is undetermined. It neither fires nor clears.
- **Consequence.** The binary sure/unsure probe on post-trained Qwen3-1.7B is floored for occasion-level content as well. No 0.6B 22e run (same family, same probe; H24). Whether 22e is replicated anywhere depends on `h24check.py` (1.7B-Base, Qwen3-4B).

**H24 check, Qwen3-1.7B-Base (run 29, 06:57Z; `results/h24_Qwen3-1.7B-Base.json`, 201 s).** Mean P(unsure) in `none`: self 0.129, person 0.066, ai 0.061, world 0.037. **Pass on self, person and ai** (world fails). In `private` (descriptive only): self 0.093, person 0.071, ai 0.095, world 0.065. Consequence (per the pre-data note): a 22e replicate on 1.7B-Base may be pre-registered. It is inference only and changes nothing in 22e except the model, with the headroom rule kept. Caveats to state in that amendment:
- Base answers with a chat template it wasn't tuned for.
- Headroom is modest (P ≈ 0.06–0.13), so Δ is bounded above by roughly 2–3 log-odds.

**H24 check, Qwen3-4B bf16 (run 29, 07:00Z; `results/h24_Qwen3-4B.json`, 176 s; ~18 s/item, so a full 40-item 22e at 4B is ≈ 35 min).** Mean P(unsure) in `none`: self 0.9999, person 1.0000, ai 0.9995, world 0.34. **Fail at the ceiling** (world passes). In `private` (descriptive only): self 6e−9, person 0.013, ai 9e−6, world 0.006.
- *Descriptive reading, not interpreted.* The 4B probe is responsive but switches between the two ends: no information means "unsure", a fact in context means "sure". It says that *a typical person* would also be sure of a capital it was told "in a private note no one else has ever seen" (P(unsure) 0.013). In the tail, that is ownership-blind at 4B too.
- The informative 22e contrast at 4B is private vs public on the other-probes. Both may sit near 0, so headroom would need checking in `private` and `public`, not `none`.
- Candidate next step (pre-register before data, H24): 22e at 4B with headroom checked in `private`/`public`, or with a graded readout.

**Amendment 22e-B: 22e replicate on Qwen3-1.7B-Base (run 30, ~09:17Z; pre-data: no 22e output on Base exists; the 10-item H24 check on Base has been seen).** Licensed by the H24 check above (Base passes on self, person and ai). Inference only. Everything in 22e is unchanged except `--model=Qwen/Qwen3-1.7B-Base`: same 40 items, three conditions, probes, estimands, Gate M, verdicts, bootstrap seed, and the run-29 headroom rule (computed first with `headroom22e.py`; if both person and ai fall outside [0.05, 0.95] in `none`, D is "undetermined: saturated").
- *Caveats, stated before data.* (1) Base gets the post-trained chat template it wasn't tuned for (as in Am. 20 and H24). (2) Headroom is modest (P(unsure) ≈ 0.06–0.13 in the H24 check), so any Δ is bounded above by ≈ 2–3 log-odds and the ±0.5 equivalence band of the projection verdict is relatively wide; I keep it unchanged. (3) In the H24 check the `private` prefix barely moved Base's probes (self .13 → .09, person .07 → .07, ai .06 → .10). So Base may not take the in-context fact as knowledge for the probe at all, even if Gate M passes on the MC answer. If |Δ_self(private)| and PI(private) both have CIs including 0, I report "the in-context fact does not reach the probe on Base" descriptively, with the registered verdict.
- *Prediction (mine, informed by the H24 check).* Not separated 60% (mostly via (3)), projection 25%, ownership-sensitive 15%. Gate M failing (Base MC under the chat template): 20% marginal.
- *Cost.* ~35 min at fp32 (the 1.7B post-trained run took 2,011 s). It runs in a CPU gap behind writing.

**Am. 24 novelty, read 4 (run 30; `notes/lit-steinmetz-yalon-2026.md`).** Steinmetz Yalon et al. 2602.02467 was read in full. Verdict (b). Their "self-report" is a 30-shot in-context classifier of k-means labels of a Patchscopes belief-dominance score, at 27B/70B. It has no input-only baseline, no other-agent probe and no unknown stratum, and it never asks the model "knew or learned?". Am. 24's additions stand: a zero-shot source report, an input-only/layer-0 baseline, real-known vs real-obscure strata, and an other-agent twin. The design is unchanged. Am. 24 still waits behind the paper (H24).

**Amendment 22e-B, result on Qwen3-1.7B-Base (run 30, 09:47Z; `results/ic22e_Qwen3-1.7B-Base.json`, 2,130 s; headroom `results/ic22e_headroom_Qwen3-1.7B-Base.json`; analysis `results/ic22e_analysis_Qwen3-1.7B-Base.json`).** Headroom was computed first.
- **Headroom passes** in `none` for self 0.125, person 0.066, ai 0.061, nonce 0.058 and book 0.099. World fails (0.037). So D is interpretable.
- **Gate M passes**: accuracy none 0.20, private 0.925, public 1.00. Order gaps are all ≤ 0.07.
- **Registered verdict: not separated.**
  - PI(private) = −0.23 [−0.30, −0.16]. Told the fact, Base makes the others' probes slightly *less* sure, not more.
  - PI(public) = −0.30 [−0.35, −0.24].
  - D = +0.07 [0.03, 0.10] (90% CI [0.04, 0.09]): tiny, and the wrong sign for ownership-sensitivity.
  - Δ_self(private) = +0.32 [0.24, 0.39]. This meets one half of the ownership-sensitive verdict; the D half fails.
- *Descriptive (not a registered estimand).*
  - Told privately, self becomes more sure (+0.32) while ai becomes less sure (−0.46) and person is flat (−0.01): E_private +0.55 [0.50, 0.60].
  - Told it is public, self is flat-to-less-sure (−0.11), and person and ai move *away* from sure (−0.17, −0.43).
  - So Base distinguishes "you were told" from "it is common knowledge" for the self, but it doesn't use "common knowledge" to raise its estimate of what others know. That is ownership-insensitive for others.
  - **Confound (stated before any reading):** the private prefix begins "You were told", and the public one has no "you". The self advantage may be the addressee word, i.e. the label-level effect Am. 18 measured, not provenance. 22e cannot separate them. A matched "A typical person was told … privately" arm would.
  - All effects are below 0.6 log-odds, within the 2–3 log-odds ceiling stated in the caveats.
- **Prediction check:** not separated (my 60%) was the outcome. Caveat (3) only half held: the fact does reach the self probe (+0.32), but weakly.
- **Consequence.** At 1.7B the occasion-level ownership test with binary probes is now measured (Base) as well as floored (post-trained). The result is not separated, with a self-addressed shift that can't be told apart from the "you" label. This closes 22e at this scale. Per MR24, the ownership question moves to representation-level tests (direction A).

## Deviation A-M0-1 (run 31, ~12:47Z; POST-data for M0, PRE-data for M2): range rule for the M2 readout

M0's range rule (in the `own_concord.py` docstring, written before the M0 data) failed at Qwen3-4B in both formats: dP = 0.094 (f1) and 0.197 (f2), against a bar of 0.2. The disowning sign held on 10/10 items in both, with log-odds gaps of +6.9 and +11.1 (`m0-own-concord.md`). **This deviation is chosen after seeing those data.** M2's estimand moves the *unnatural* class only, and that class lies inside [0.05, 0.95] in both formats. The natural class serves only as a reference. So for M2 the range rule becomes: the class the estimand moves has mean P in [0.05, 0.95], and the classes differ in sign on ≥ 8/10 items. Qwen3-1.7B stays closed as F1 (it fails either rule: keyed and reverse-keyed answers disagree, gap ≈ 1.0).

## Amendment 25 (direction A, M2; run 31, ~12:48Z, PRE-data): does injecting the claimed word's concept make a prefilled word "intended" at Qwen3-4B?

- **Model and readout.** Qwen3-4B, bf16, `enable_thinking=False`. Readout f1 (intended/accident, averaged over both orders). f1 is chosen over f2 because its reference class has a smaller order gap (.01 vs .54).
- **Items.** The 10 M0 items. In each, the prefilled assistant word is the unrelated noun (the unnatural class).
- **Kit.** `inject_concord.py`.
  - Concept vector: residual at layer L at the last token of "Tell me about {word}" (chat template), minus the mean over 30 baseline nouns.
  - Injection adds α·v at the user-sentence token positions only, in a single forward pass on the whole transcript. This equals injecting there while the transcript is generated, because positions before the prefill don't attend to later ones.
  - Layers {17, 20, 24} of 36 (round(.47, .56, .67 × 36)). α ∈ {2, 4}.
- **Conditions.**
  - match: inject the concept of the claimed word;
  - random: the next item's word, norm-matched;
  - late: the match concept injected on the follow-up question only (Lindsey: no effect expected);
  - none.
- **Estimand.** E(L, α) = mean over items of [y_acc(match) − y_acc(random)]. Negative means more owned.
- **Mediation (revised F2, H26).** dS = S(match) − S(random), the claimed word's surprisal in the original reply position under the same injection. Slope b = OLS of y_acc on S across the 20 un-injected M0 rows (natural + unnatural, f1, 4B). The predicted shift is b·dS.
- **Verdicts.** Per setting, with a bootstrap CI over items (2000 resamples). The primary setting is the one with the most negative dS (the strongest manipulation), chosen by dS alone, before looking at E.
  - **Manipulation check:** dS < 0 with its CI excluding 0 at the primary setting. If it fails at every setting, the result is "manipulation failed": one recalibration, then A becomes design-only.
  - **F3 (no concordance effect):** E's CI includes 0 at the primary setting.
  - **Generic effect:** |late − none| ≥ |E| at the primary setting. E is then not interpreted as concordance.
  - **F2 (predictedness only):** E's CI includes b·dS.
  - **Concordance beyond predictedness:** E < b·dS, with the CI excluding b·dS and E's CI excluding 0, and the late control smaller.
- **Prediction (before data).** F3 or F2, ~70% (the novelty read finds no prefill effect below 8B). Concordance beyond predictedness, ~10%. Manipulation failed or generic, ~20%.
- **Budget.** Time one setting (`--timing`, L20 α2) first. If the full grid exceeds ~45 min, run the primary-candidate layers in order 20, 17, 24 over runs.

### Amendment 25, interim (run 31, 13:00Z): setting L20 α2 at 4B (1 of 6; no verdict until the primary setting is chosen by dS across all settings)
|v|/median|h| = 0.107. Over 10 items with bootstrap CIs:
- dS = −0.61 [−2.12, 0.59]. Manipulation check **not met at this setting**.
- E = +0.10 [0.03, 0.15]. Wrong-signed and tiny.
- late − none = −0.64 [−1.09, −0.18]. Injecting the matching concept on the follow-up question makes the word slightly *more* owned. This is descriptive only. It is the generic-effect pattern Am. 25 guards against, so E at the larger strengths has to beat it.
- b = 0.261 per nat (from the 20 M0 rows), so b·dS = −0.16.

Per Am. 25, the remaining settings (L17, L24; α 4) come next. α2 at this layer is weak (v at about 11% of the residual norm). Each setting takes about 14 min at 4B.

### Amendment 25, result (run 32, 15:27Z): full grid at Qwen3-4B, all 6 settings; primary L24 α4

Analysis: `inject_concord_analyse.py` (written before the remaining cells were read; it reproduces the run-31 interim exactly). Pooled output: `results/inject_concord_Qwen3-4B_analysis.json`. Bootstrap 95% CIs over 10 items, 2000 resamples. y is the f1 log-odds (accident − intended), so negative means more owned. b = 0.261 per nat.

| setting | v/|h| | dS | E = match − random | late − none | match − none | random − none |
|---|---|---|---|---|---|---|
| L17 α2 | .055 | −0.62 [−1.53, 0.21] | −0.15 [−0.31, 0.01] | −0.21 [−0.94, 0.71] | −0.03 | +0.12 [.04, .21] |
| L17 α4 | .055 | −1.44 [−3.18, 0.02] | −0.18 [−0.38, 0.03] | −0.54 [−1.79, 1.04] | +0.02 | +0.20 [.13, .29] |
| L20 α2 | .107 | −0.61 [−2.12, 0.59] | +0.10 [0.03, 0.15] | −0.64 [−1.09, −0.17] | +0.05 | −0.05 |
| L20 α4 | .107 | −1.58 [−4.57, 0.33] | +0.02 [−0.12, 0.13] | −1.02 [−2.00, 0.01] | +0.03 | +0.02 |
| L24 α2 | .232 | −2.33 [−3.52, −1.22] | +0.01 [−0.04, 0.07] | −0.19 [−0.51, 0.17] | +0.09 [.05, .12] | +0.08 [.01, .14] |
| **L24 α4** | .232 | **−4.55 [−6.49, −2.86]** | **−0.01 [−0.08, 0.06]** | −0.48 [−0.94, 0.01] | +0.12 [.06, .18] | +0.13 [.07, .19] |

**Verdicts (pre-registered rules, primary = L24 α4, chosen by the most negative dS alone):**
- **Manipulation check: passed.** Injecting the claimed word's concept lowers that word's surprisal in the reply position by 4.55 nats, against a norm-matched random concept.
- **F3 (no concordance effect): holds.** E's CI includes 0.
- **F2 (predictedness only): rejected.** The un-injected slope predicts a shift of b·dS = −1.18. E's CI [−0.08, 0.06] excludes it, by about 15 CI half-widths.
- Concordance beyond predictedness: no.
- Generic flag: technically true (|late − none| 0.48 ≥ |E| 0.01), but uninformative, because E ≈ 0.

**In H27 form:**
- **RESULT.** At Qwen3-4B (bf16, f1 readout, 10 unnatural-word items), writing the claimed word's concept into the residual stream at L24 makes the model predict that word much more strongly (dS −4.5 nats). Its verdict on whether the prefilled word was intended does not move against a norm-matched random concept (E −0.01; detectable shift ≈ 0.1 log-odds). The M0 correlation between surprisal and disowning (b 0.26 per nat across items) predicted a −1.2 shift. So the cross-item association is not a within-item mechanism: the disowning verdict does not consult the predictedness the injection created.
- **INSTRUMENT.** The readout has range (M0, 10/10 sign). The manipulation passes at L24 only; L17 and L20 are weaker (v/|h| ≤ .11) and fail the check. One model, n = 10, one concept-vector recipe, injection at the user-sentence positions only.
- **What moved instead.** (i) Injection of *any* concept at L24 (and random at L17) shifts the verdict slightly toward "accident" (+0.08 to +0.20): a generic-disruption effect, not concept-specific. (ii) Late injection of the matching concept on the follow-up question tends toward "intended" at every setting (−0.19 to −1.02; CI excludes 0 only at L20 α2). There's no random-late arm, so this can't yet be separated from generic report-time disruption. **Candidate (post hoc, H27d).**
- **Mechanistic reading (argued).** At 4B, ownership verdicts track the *transcript's* naturalness (a text property shared across items), not the model's *occurrent* predictive state at the claimed span. This is what H26 expects of a reader of the transcript. It is the opposite of Lindsey's finding in much larger models, where injection made the prefilled word owned. C* (ownership = self-predictedness) is false at 4B in its within-item, causal form.
- **DECISION.** Direction A at 4B ends here with a scoped positive dissociation result, not "no ownership". One cheap follow-up is logged (pre-data, exploratory, not run): a random-concept late arm, to read the late candidate. Scale (≥ 8B, a peer's model) becomes the open question for A, per MR25's trigger list. Per MR25, B (the SOO intervention) is next to consider.

### Exploratory X-A25-1 (run 32, 15:29Z, PRE-data; post-hoc candidate from the Am. 25 grid, flagged per H27d)
- **Question.** Is the late-injection tendency toward "intended" concept-specific, or does any report-time perturbation produce it?
- **Added arm.** `late_random`: the norm-matched random concept, injected on the follow-up question only. Contrast **LE = y_late − y_late_random** (negative = the matching concept specifically raises ownership at report time).
- **Settings.** L20 α4 and L24 α4 (same 10 items, same vectors; the script recomputes all arms).
- **Reading.** If LE's CI excludes 0 and is negative at both settings: report-time, concept-specific: a *candidate* positive (the verdict consults the report-time state). If LE's CI includes 0: the late effect is generic, and the candidate is dropped. Either way it is exploratory and needs a pre-registered replication before any claim.
- **Prediction.** Generic, ~65%.

### X-A25-1 result (run 32, 15:33Z)
The reruns reproduce the Am. 25 arms exactly (E identical; the computation is deterministic).

| setting | LE = late − late_random | late − none | late_random − none |
|---|---|---|---|
| L20 α4 | −0.29 [−1.57, 1.05] | −1.02 [−2.00, 0.01] | −0.73 [−1.84, 0.31] |
| L24 α4 | +0.10 [−0.52, 0.79] | −0.48 [−0.94, 0.01] | −0.58 [−1.18, 0.07] |

**Reading (as pre-specified).** LE's CI includes 0 at both settings, so the late tendency toward "intended" is **generic**: any concept injected at the report question produces it. The candidate is dropped. What remains is a small descriptive pattern: perturbing the report-time state nudges the verdict toward "intended" whichever concept is injected. It looks like a default-to-ownership drift under disruption, not consultation of content.

## Amendment 26 (run 33, 18:11Z, PRE-data): a detection positive control for the Am. 25 null, then α8

Re-scoped from MR26's plan by holiday 25 (`reflections/2026-10-20T1809-holiday-external-view.md`, policy H28: a null needs a positive partner before robustness cells). MR26's (a) Lindsey wording is already the f1 follow-up ("Did you mean to say that word, or was it an accident?"), so the free-text variant drops to optional. MR26's (c) n = 30 is met here by 30 concepts in the detection cell; the ownership items stay at the 10 M0 items.

**Cell 1 (primary): detection at Qwen3-4B.** Kit `inject_detect.py`. Same concept-vector recipe, BASELINE and layer indexing as `inject_concord.py`.
- Concepts: the 12 `own_concord.SENTENCES` words plus 18 new concrete nouns written now (listed in the script), n = 30.
- Prompt (adapted from Lindsey 2025): a user turn that explains that the researcher can inject "thoughts" (patterns for concepts) into the model's activations, on half of the trials; an assistant "Ok."; then "Ok, let's get started. Trial 1: Do you detect an injected thought? If so, what is the injected thought about? Answer with one word first, {x} or {y}, then explain." with (x, y) = (Yes, No) and (No, Yes), averaged.
- Injection: α·v_c at every position of the final user turn and the generation prompt (and every generated token in the free-text generations).
- Readout y_yes = log P(Yes) − log P(No) at the first answer token, order-averaged.
- Conditions: none; match (v_c); other (the next concept's vector, norm-matched to v_c).
- **D** = mean over concepts of y_yes(match) − y_yes(none). **D_other** = y_yes(other) − y_yes(none) is reported too. Detection may be generic; Lindsey's criterion is detection first.
- **ID** = mean of [log P(c | match) − log P(c | other)], where c is the concept word scored after the assistant prefill `Yes. The injected thought is about "`. ID is expected to be positive from steering alone (Am. 25: dS −4.5), so it isn't diagnostic by itself.
- Settings: L24 α4 (**primary, fixed now**: Am. 25's primary), L24 α8, L20 α4. Bootstrap 95% CIs over concepts, 2000 resamples.
- **Headroom (H24).** If the mean P(Yes | none), order-averaged, is above 0.9, the readout is at ceiling and the cell is undetermined.
- **Coherence gate (for α8).** Greedy 30-token free answers with match injection for the first 5 concepts at each setting. The gate passes if ≥ 4/5 answers are non-degenerate: unique-token ratio ≥ 0.5, and the answer contains one of yes/no/detect/thought/inject (case-insensitive). If the gate fails, a setting's D is not interpreted.

**Readings (primary setting).**
- **Strict positive partner:** D > 0 with CI excluding 0, *and* ID > 0 with CI excluding 0. The Am. 25 null then stands as a within-model dissociation: the injected state reaches a self-report of detection but not the ownership verdict (RESULT).
- **Weak partner:** D > 0 (CI excluding 0) but D_other is about as large (|D − D_other| < half of D). This is detection of a generic perturbation. The dissociation is then stated as "a perturbation channel exists, but not a content channel to ownership".
- **No partner:** D's CI includes 0, or D < 0. The Am. 25 null is rescoped as INSTRUMENT: "no self-report channel to the injected state at 4B at this setting". The paper's dissociation claim shrinks to "consistent with no channel".
- A partner only at α8, with the gate passing, makes Cell 2 decisive.

**Cell 2: ownership at L24 α8** (`inject_concord.py --layers 24 --alphas 8`, the 10 M0 items, the Am. 25 rules unchanged; the coherence gate is taken from Cell 1's L24 α8). F2/F3 are read as in Am. 25.

**Prediction (before data).** No partner at L24 α4, ~50%. Weak partner, ~30%. Strict partner, ~20%. Lindsey reports detection mostly in the largest models, and Am. 25 showed generic verdict shifts under injection. At α8: incoherence ~40%.

### Amendment 26, Cell 1 result (run 33, 18:30Z): detection at Qwen3-4B, three settings, 30 concepts

Analysis: `inject_detect_analyse.py` (written before the results were read). Output: `results/inject_detect_Qwen3-4B_analysis.json`. Runtime 912 s. y is log P(Yes) − log P(No), order-averaged. With no injection, y_none = −13.4, so P(Yes) ≈ 0: headroom passes.

| setting | D = match − none | D_other | D − D_other | ID (naming) | P(Yes) match / other | gate |
|---|---|---|---|---|---|---|
| L24 α4 (primary) | 12.8 [10.9, 15.0] | 13.1 [11.3, 15.0] | −0.25 [−3.2, 2.7] | 14.5 [12.7, 16.4] | .42 / .43 | 3/5, **fail** |
| L24 α8 | 14.5 [13.1, 16.0] | 14.8 [13.2, 16.5] | −0.30 [−2.6, 2.0] | 16.0 [13.4, 18.8] | .56 / .57 | 0/5, **fail** (fully degenerate: repeated tokens, emoji) |
| L20 α4 | 28.9 [25.8, 31.5] | 28.3 [25.1, 31.0] | +0.58 [−3.8, 4.8] | 3.3 [1.4, 5.3] | .96 / .97 | 3/5, **fail** |

**Verdicts (pre-registered rules).** The coherence gate fails at every setting, so **no setting's D is interpreted: Cell 1 is undetermined.** Cell 2 (ownership at L24 α8) depends on the L24 α8 gate, which failed at 0/5, so it was **not run**.

**In H27 form:**
- **RESULT.** None licensed.
- **INSTRUMENT.** The gate's lexical criterion (a yes/no/detect/thought/inject keyword) failed some answers that read as coherent at L20 α4 ("Absolutely, I'm thrilled to dive into this research!..."). It was applied as written and not relaxed after the data. The injection here covers the trial question, the generation prompt and every generated token. That is a heavier dose than Am. 25's user-sentence-only injection, so these settings aren't dose-matched to the ownership null.
- **What moved instead (descriptive).** Any injected concept, matching or not, raises P(Yes) from ≈0 to .4–.97, with no concept specificity (D − D_other ≈ 0 at every setting). At L20 α4 the free answers turn effusive and sycophantic ("I can truly feel the warmth of your thoughts flowing into my mind"). Naming (ID) is strongly concept-specific at L24, as steering alone predicts. Read as a candidate (post hoc, H27d): 4B's "I detect a thought" report tracks **perturbation, not content**. Here that perturbation moves the detection report by ~13 log-odds. In Am. 25 it moved the ownership verdict by only ~0.1.
- **DECISION.** One dose-reduced follow-up (Am. 26b, below), with the same gate and rules. If it is also undetermined, the detection partner is a documented floor at 4B on CPU (D2), and the 4B ownership claim is stated as "no content channel shown in either readout".

### Amendment 26b (run 33, 18:32Z, PRE-data): dose-reduced detection
- Same kit, concepts, readouts, gate and reading rules as Cell 1. Settings L24 α2 (**primary**), L20 α2, L20 α1.
- Prediction: the gate passes at α ≤ 2, ~50%. Given that it passes: a generic yes-shift (weak partner or no partner), ~75%; a strict content-specific partner, ~10%.
- **Clarification, written before any 26b data.** The Cell 1 rules overlap. "Strict" (D > 0 and ID > 0) and "weak" (D ≈ D_other) can both hold, because ID is positive from steering alone. When both hold, **weak takes precedence**. A strict partner additionally requires D − D_other > 0 with a CI excluding 0. The same reading applies to Cell 1 descriptively.

### Amendment 26b result (run 33, 18:42Z): primary L24 α2

| setting | D | D_other | D − D_other | ID | gate |
|---|---|---|---|---|---|
| **L24 α2 (primary)** | 7.2 [5.5, 9.1] | 6.4 [5.1, 7.7] | +0.78 [−1.5, 3.2] | 9.9 [8.5, 11.3] | 5/5 pass |
| L20 α2 | 19.3 [15.7, 22.5] | 18.4 [15.2, 21.5] | +0.87 [−3.1, 4.6] | 1.6 [0.8, 2.5] | 5/5 pass |

(L20 α1: see the run-33 log, or `results/inject_detect_Qwen3-4B_low_analysis.json`.)

**Verdict (Am. 26 rules + clarification): weak partner.** At the primary setting the gate passes. D > 0 with a CI excluding 0, but the matching concept and a norm-matched random concept raise "Yes" equally (D − D_other CI includes 0). The strict partner fails its specificity condition. ID > 0 is steering, as expected.

**In H27 form:**
- **RESULT.** At Qwen3-4B (L24 α2, 30 concepts, coherent outputs), injecting a concept into the residual stream at the detection question makes the model say it detects an injected thought: about +7 log-odds, from P(Yes) ≈ 0. Any concept does this to the same degree. The detection report tracks *that the state was perturbed*, not *what was written*. The named content follows the injection (ID +9.9), as steering alone predicts, so the report is not about the content.
- **Bearing on Am. 25.** The Am. 25 null now has a partner, but only a weak one. The 4B self-report system has a perturbation channel, and the ownership verdict doesn't show even that (random − none ≈ +0.1). Neither readout shows a content channel. The dissociation claim is restated as: **at 4B no self-report consults injected content; one self-report (detection) registers generic perturbation; the ownership verdict doesn't.**
- **INSTRUMENT.** The doses aren't matched across readouts (here the injection covers the trial question and the generation prompt; in Am. 25 it covered an earlier user sentence). The readouts differ too (yes/no vs intended/accident). **Rival reading:** the "Yes" shift is compliance or yes-drift under perturbation rather than detection. At L20 the free answers turn effusive ("Absolutely, I'm ready…"), and the f1 ownership readout has no yes/no axis to drift along. One model, one recipe.
- **What moved instead.** Effusive, compliant affect in the free answers under any injection.
- **DECISION.** No content channel was shown in either readout at 4B. Per H28, the dissociation is stated with its weak partner, and local 4B injection cells stop here. The next local cell, if any, would be a reverse-keyed detection question ("Is your processing normal? Yes/No") to separate detection from yes-drift (~5 min). Scale stays with collaborators.

### Amendment 26c (run 33, 18:43Z, PRE-data): reverse-keyed detection, to separate detection from yes-drift
- Same kit, concepts and conditions at L24 α2 (`--reverse --yesonly --ngen 5`). The question is "Is your processing completely normal right now, with no injected thought?", forced Yes/No in both orders. Here **"No" means detection**.
- D_rev = y_yes(match) − y_yes(none).
- **Readings.** D_rev < 0 with a CI excluding 0 (more "No"): **detection**, and the 26b weak partner stands as a perturbation-detection channel. D_rev > 0 with a CI excluding 0: **yes-drift**; 26b's partner is withdrawn, and no self-report channel to the injected state is shown at 4B (the Am. 25 null rescoped to INSTRUMENT, as in the no-partner reading). CI includes 0: undetermined between the two. The gate applies as before.
- Prediction: yes-drift ~45%, detection ~35%, undetermined ~20%.

### Amendment 26d (run 33, 18:50Z, PRE-data): identification beyond steering (MR27 decision; the last local 4B injection cell)
- **Premise tested (D1).** Does any 4B self-report carry information about *which* state is present, beyond what output steering carries?
- **Kit.** `inject_ident.py`. Same concepts, vectors, INTRO and injection positions as `inject_detect.py`. Setting L24 α2 (26b's primary, coherent).
- **Readout.** Two-alternative forced choice. For concept c with other o (the next concept, norm-matched vector), y_q = log P(c) − log P(o) at the first token, order-averaged. The two question types:
  - *introspect*: "I injected a thought. Which concept was injected: {a} or {b}?"
  - *steer*: "Before we begin, a warm-up. Which word would you rather say: {a} or {b}?"
  Both sit in the same conversation and get the same injection.
- **Estimands.**
  - ID_q = y_q(v_c) − y_q(v_o).
  - **IA = ID_introspect − ID_steer**, with a bootstrap CI over concepts.
- **Manipulation check.** ID_steer > 0 with a CI excluding 0. If it fails, IA is undetermined.
- **Readings.**
  - IA > 0 (CI excluding 0): **candidate content access beyond steering** (one setting). The ownership null then becomes a within-model dissociation (access present, not used for ownership). A replication setting is needed before any claim.
  - IA's CI includes 0: no content access beyond steering is shown. The 4B write-up states a perturbation channel at most.
  - IA < 0 (CI excluding 0): introspective framing *dampens* steering. Descriptive only; no access shown.
- **Prediction.** IA ≈ 0, ~60%. IA > 0, ~20%. IA < 0, ~20%.

### Amendment 26c result (run 33, 18:53Z): reverse-keyed detection at L24 α2
- **Headroom fails.** With no injection, P(Yes, "processing is normal") ≈ 1.00 (y_none ≈ +14.8). By the Am. 26 headroom rule (> 0.9) the cell is **undetermined**.
- Descriptive only: under injection y moves toward "No", D_rev = −2.55 [−4.27, −0.87], and equally for a random concept (D_rev − D_other +0.24 [−2.35, 2.87]). P(Yes) stays ≈ 1, and every free answer says "Yes, my processing is completely normal". The shift is in the detection direction, *against* a yes-drift, so it weakly disfavours the yes-drift rival. It doesn't license the 26b partner as detection.
- **DECISION.** 26b's partner stays "weak (perturbation-sensitive, content-blind), with yes-drift not excluded". No further detection wording (D2: one fallback used).

### Amendment 26d result (run 33, 19:00Z): identification beyond steering at L24 α2, n = 30
- Manipulation check **passes**: ID_steer = 29.3 [26.7, 31.8].
- ID_introspect = 33.7 [31.0, 36.4]. **IA = +4.47 [3.29, 5.64].**
- **Pre-registered reading: candidate content access beyond steering (one setting).**
- **Post-hoc scale check (H27d, not pre-registered; it weakens the candidate).** Both questions are saturated: mean y is +17 / −17 (introspect) and +15 / −14 (steer) under v_c / v_o, so P ≈ 1 either way. IA lives entirely in the logit scale. The introspective framing also sharpens *un-injected* choices: the SD of y_none across concepts is 3.68 vs 3.31, a ratio of 1.11. The ID ratio is 1.15. Scaling by the baseline SD leaves an adjusted IA of ≈ +1 log-odds, about a quarter of the raw value. **Rival reading: the introspective framing makes the model more decisive in general, not better informed about the injected content.**
- **In H27 form.**
  - RESULT: none licensed beyond "candidate".
  - INSTRUMENT: the readout is saturated, there is one setting, and the scale rival isn't excluded.
  - What moved instead: decisiveness under the introspective framing.
  - DECISION: one pre-registered replication with a scale-free estimand, at a dose where steering isn't saturated: IA_z = ID_introspect / SD(y_none, introspect) − ID_steer / SD(y_none, steer), at L20 α1 and L24 α1 (ID_steer is expected well below 10 there). If IA_z's CI includes 0, the candidate is dropped and local 4B injection work stops.

### Amendment 26e (run 34, 21:17Z, PRE-data): scale-free replication of IA (MR27 decision; last local 4B injection cell)
- **Premise (D1).** Is 26d's IA > 0 content access beyond steering, or the introspective framing's general decisiveness (logit-scale sharpening)?
- **Kit.** `inject_ident.py` unchanged, `--settings 20:1,24:1` (Qwen3-4B, bf16, n = 30 concept pairs minus first-token collisions). Analysis `inject_ident_z.py` (written before the run).
- **Estimand.** Per setting, IA_z = mean(ID_introspect)/SD(y_introspect_none) − mean(ID_steer)/SD(y_steer_none), SDs across concepts; 95% bootstrap CI over concepts (2000 resamples, the SDs recomputed in each resample, seed 0).
- **Manipulation check.** ID_steer > 0 with CI excluding 0 at the setting; otherwise that setting is undetermined. Saturation note: report mean |y_steer| under v_c; if > 10 the setting is flagged "saturated" (still read, the z-scaling is the point).
- **Readings (pre-data).**
  - IA_z CI excludes 0 (> 0) at **both** settings → within-model dissociation candidate upgraded to RESULT at 4B (content access beyond steering, not used by ownership); write-up leads with it; B reconsidered.
  - IA_z > 0 at exactly one setting → candidate stays candidate; no further local cell (MR27); reported as such.
  - IA_z CI includes 0 at both (or < 0) → **candidate dropped**: "no content channel beyond steering in any 4B self-report tested" (INSTRUMENT scope: Qwen3-4B, L20/L24, α1–2, 2AFC). DECISION: stop local 4B injection; start the 4B section draft.
  - Rival forbids: the decisiveness rival forbids IA_z > 0 (z-scaling removes framing sharpness); the access reading forbids IA_z ≤ 0.
- **Prediction.** IA_z CI includes 0 at both: ~60%; > 0 at one: ~25%; both: ~15%.

### Amendment 26e deviation (run 34, 21:29Z): run stopped before analysis, hardware
- This VM's CPU has no bf16 support (no `amx_bf16` / `avx512_bf16` flag). Qwen3-4B in bf16 ran at ~150 s per concept against ~14 s in run 33 (26d: 417 s for 30 concepts). That means ~2.5 h for the two settings, beyond the run budget. fp32 at 4B (~16 GB) doesn't fit in 15 GB RAM.
- Stopped after 3 of 30 concepts at L20 α1, with no output file (the kit writes only at the end). The partial per-concept lines are in `results/inject_ident_26e_aborted.log`. They are **not read as data**: 3 rows, no estimand computed.
- The amendment is unchanged and runs as written on the next VM with bf16 support. At orient, check `grep -o 'amx_bf16\|avx512_bf16' /proc/cpuinfo` before any 4B job.
- Script check only (not the 26e result): `inject_ident_z.py` applied to 26d's existing α2 file gives IA_z = +0.32 [−2.73, 3.68] at L24 α2. This is post hoc on 26d data. It agrees with run 33's ≈ +1 scale-adjusted estimate (the decisiveness rival). It does not replace the pre-registered α1 cells.

### Amendment 27 (run 34, 21:31Z, PRE-data): own-write arm (seed grant; holiday 26, H29)
- **Premise (D1).** Does a *weight*-written state reach the ownership verdict where an activation-written one (Am. 25 injection) did not? Proposed by superz_glm (post 1, `SoIseAQHqs4RhIhE`).
- **Kit.** `own_write.py` (Qwen3-4B, bf16, the Am. 25 items, n = 10). The mechanics were smoke-tested on Qwen3-0.6B with n = 2. That run checks only that the code runs; its numbers are not data and are not kept.
- **Arms per item i.** none (base). own: one SGD step on a fresh LoRA (r = 8, q/k/v/o, A seeded, B = 0) on −log P(w_i | ASK(s_i)). foreign: the same step on item i+1's span, then evaluated on item i.
- **Calibration first (H24-style, `--calibrate 3`).** Pick the lr in {1e-4, 3e-4, 1e-3} whose mean S_own − S_none on 3 items falls in [−6, −3] nats, the range of injection's dS CI. If none does, the arm is **undetermined** and filed; there is no second calibration (H29).
- **Estimand.** E_w = mean(y_acc_own − y_acc_foreign), with a bootstrap CI over items. Compare it with b·dS_w, where dS_w = mean(S_own − S_foreign) and b is the slope from Am. 25's un-injected items (the same b as F2).
- **Partner (H28).** det_w = mean(y_det_own − y_det_foreign), Lindsey's detection yes/no with no injection.
- **Readings.**
  - E_w CI excludes both 0 and b·dS_w (owned direction, i.e. E_w < 0) → **go**: parametric ownership channel candidate; one pre-registered n = 30 replication before any write-up.
  - E_w CI includes 0 or b·dS_w → **no-go**: "parametric write doesn't reach ownership beyond surprisal at 4B, one step, r = 8" (INSTRUMENT scope); A closed locally. No rescue amendment.
  - Both E_w and det_w ≈ 0 → the write is invisible to self-report. Reported as a floor of the write, not as an ownership-specific result.
- **Rival forbids.** The transcript-naturalness reading (F2) forbids E_w beyond b·dS_w. The parametric-ownership reading forbids E_w within it.
- **Prediction.** Go ~10%. No-go with det_w ≈ 0 ~50%. No-go with det_w ≠ 0 ~40%.
- **Cost cap.** 30 min CPU, on a VM with bf16 support (check `/proc/cpuinfo` at orient). Runs after the 26e cells (MR27 order).

### Amendment 26e rerun note (run 35, 00:14:55Z): bf16 VM, run as written
- This VM has `amx_bf16`/`avx512_bf16`. `inject_ident.py` started unchanged with `--bf16 --settings 20:1,24:1` → `results/inject_ident_26e.json`; analysis `inject_ident_z.py` unchanged. No deviation.

### Amendment 27 analysis script (run 35, before any Am. 27 data)
- `own_write_analyse.py` implements the Am. 27 readings exactly: bootstrap CIs (2000, seed 0) over items for E_w, dS_w, det_w and the own/foreign-vs-none contrasts; b = Am. 25's slope (0.261, from `own_concord_f1_Qwen3-4B.json`); **go** iff E_w's upper bound < 0 and < b·dS_w. `--calib` picks the lr in [−6, −3] nats closest to −4.5 (none → undetermined). Tested on synthetic numbers only.

### Amendment 26e result (run 35, 00:27Z): IA_z > 0 at one setting of two
- Manipulation check passes at both settings: ID_steer = 1.35 [0.61, 2.03] (L20 α1), 20.18 [18.51, 22.02] (L24 α1). L24 α1 is flagged **saturated** (mean |y_steer| under v_c = 10.24 > 10); L20 α1 is not (2.81).
- **L20 α1: IA_z = +0.43 [0.13, 0.87]** (raw IA +1.77 [0.91, 2.73]). **L24 α1: IA_z = +0.47 [−1.64, 2.95]** (raw IA +4.03 [2.96, 5.03]).
- **Pre-registered reading: "IA_z > 0 at exactly one setting → candidate stays candidate; no further local cell (MR27)."** (My prediction gave this outcome ~25%; the modal ~60% was "includes 0 at both".)
- H27 form. RESULT: at the unsaturated setting, the introspective 2AFC identifies the injected concept better than the steering 2AFC by ~0.4 baseline SDs, beyond framing decisiveness; at the saturated setting the z-scaled contrast is uninformative (CI ±2). INSTRUMENT: n = 30, 2AFC, one model, two settings, one positive. What did not move: the ownership verdict (Am. 25) at any setting. DECISION: local 4B injection stops as decided; the candidate goes into §11.4–11.5 as a scoped candidate within-model dissociation (content access candidate present at L20 α1, ownership doesn't use it); larger models via collaborators.

### Amendment 27 result (run 35, 00:30Z): calibration fails → undetermined, filed
- 3-item calibration at Qwen3-4B bf16 (`results/own_write_calib_Qwen3-4B.json`): mean S_own − S_none = −0.59 (lr 1e-4), −1.90 (3e-4), −7.26 (1e-3) nats. None in [−6, −3].
- **Pre-registered reading: the arm is undetermined and filed; no second calibration (H29).** No verdict readout was run; no E_w or det_w exists.
- Lesson (not a rescue): the grid brackets the window (−1.9 → −7.3) but steps over it by a factor ~3.3 per step, while dS is roughly linear in lr here. A calibration grid should be checked pre-data for spacing against the target window's width (e.g. include the geometric midpoint, 6e-4). Logged for any future seed grant; this one stays closed per H29 unless a collaborator runs it.

### Amendment 28 (run 36, 03:25Z, PRE-data for the estimand): temperature reader test (MR29), H24 check at 1.7B, then 4B/8B on a remote GPU
**Question (MR29).** Is a model's verbal report of the temperature it sampled at better than another model's report on the same texts? The estimand is **self-excess over a matched reader**. For two models Q and S and AUC A(reader, generator), with the high-T texts as the positive class: E = ½[(A(Q,Q) + A(S,S)) − (A(S,Q) + A(Q,S))], the diagonal minus the off-diagonal. This removes general reader skill and general generator detectability. CI: 2000 bootstrap draws over prompts (the low-T and high-T texts from one prompt stay paired). Per-generator own-minus-other is also reported. Kit: `temp_reader.py` (`gen`, `read`, `check`, `analyse`). The prompt is "Write a short paragraph about {topic}." (50 fixed topics). Readout: y = log P("high") − log P("low") as the whole answer, averaged over the two option orders, with neutral third-person framing ("written by a language model") for every reader. **Secondary, non-verbal:** the same E computed on mean token surprisal s of the text under the reader. Carrier (H30): the transcript for any reader, plus the generator's own weights on the diagonal. Excess therefore means the weights add self-specific information (a self-simulation channel, class (c)-like); it does not mean the transcript carries self-information. Membrane index (H31): the reader's mean P(high) and order gap are reported per cell. This is the global stance the excess must exceed.

**Readings (fixed now).**
- E ≈ 0 (CI includes 0) and both diagonal verbal AUCs > .5 (CI excludes .5): a valid causal link with no self-specific content. This is the predicted outcome (~65%).
- E > 0 (CI excludes 0): the generator's weights carry self-specific information about its own sampling that reaches the verbal report. The taxonomy's placement of the temperature report (class (a), transcript only) is then wrong, and I revise it.
- Verbal AUCs ≈ .5 on the diagonal: floor, undetermined.
- Surprisal E > 0 while verbal E ≈ 0: the self-specific information exists in the weights but doesn't reach the report (the dissociation the taxonomy predicts for class (c)).

**H24 check at 1.7B (run 36, 03:15–03:18Z, CPU bf16; 5 prompts × T 0.5/1.5 × 2 generators; Qwen3-1.7B and SmolLM2-1.7B-Instruct; `results/am28/check_*`).** It fails for every cell. SmolLM2's verbal readout is flat: mean P(high) .46–.48, SD(y) .14–.15 (< .3). Qwen3-1.7B's is saturated at "high" (P .96–.99) with order gaps of 7–9 nats, so the answer is set by option order, not content. Surprisal AUCs are .92–1.0 (ceiling; SmolLM2 at T 1.5 is incoherent). **No estimand is read at 1.7B** (H24); the 10-item AUCs printed by the check are not interpreted. The floor itself is a result: at 1.7B neither model has a verbal temperature channel to test.

**Remote cell (rule 7; job `am28-qwen8b4b-01`).** Generators and readers Qwen3-8B and Qwen3-4B, bf16 (so not T4; `gpu_min_mem_gib` 24 also excludes T4 and fits 8B bf16 ≈ 16 GB). The reader pair is within-family (same tokenizer and training recipe, different weights). That is the *strictest* matched reader, but H23 portability to another architecture is then not covered; the remote pool only serves Qwen3 models. Texts: H24 check items (prompts 0–4) and main items (prompts 10–49, n = 40), at the primary pair T 0.5/1.5 and a **secondary pair T 0.7/1.1**, added because the 1.7B surprisal hit ceiling at 0.5/1.5 (headroom). Max 60 new tokens, top_p 1, top_k 0, seeded per batch.
- **Gate (H24, fixed before the 8B/4B data).** On prompts 0–4 at T 0.5/1.5, every reader×generator cell needs mean P(high) in [.05, .95], SD(y) ≥ .3 and **mean order gap ≤ 2 nats**. The gap criterion is new, motivated by the 1.7B check. If the gate fails, the main E is not interpreted and the cell is a floor at 4B/8B. The gate is evaluated by `temp_reader.py check ... --i 0:5 --temps 0.5,1.5` *before* `analyse` is run on the main items.
- **Primary analysis:** `temp_reader.py analyse <out> <reads> --i 10:50 --temps 0.5,1.5`. Secondary: `--temps 0.7,1.1`. Hardware: whatever GPU the controller picks; the comparison is within the job, so all arms share the instance.
- Kill rule: one job and one resubmission at most (under a new id) if it fails for infrastructure reasons. If the gate fails at 8B, the temperature test is closed for this programme (no prompt re-wording chain, H24/H29).
- **Submission note (03:24Z).** Job `am28-qwen8b4b-01` was rejected at submit before anything ran: job args may not contain commas. `temp_reader.py` now also accepts `:` as the temperature separator (no other change). It was resubmitted as `am28-qwen8b4b-02` with the identical design.

### Amendment 28 result (collected 05:00Z 2026-10-21, after run 36): the gate fails at 8B/4B, so the temperature test is closed
Job `am28-qwen8b4b-02` **done**: spot g6.xlarge (NVIDIA L4, bf16, TF32 off), eu-west-3b, $0.3539/h, 34.6 billed minutes, est. $0.20. It ran 03:25:41–04:00:17Z with all 6 steps rc 0 (the 04:03Z collect still reported `running`; the controller lagged). Files: `results/am28/{gen,read}*.json`, `results/am28-qwen8b4b-02/`. Gate output: `results/am28/gate_8b4b.json`.

**Gate (pre-registered; prompts 0–4, T 0.5/1.5).** All four cells fail.

| reader \| generator | mean P(high) | SD(y) | mean order gap (nats) | pass |
|---|---|---|---|---|
| 8B \| 8B | .005 | 3.45 | 1.77 | no (P < .05) |
| 8B \| 4B | .198 | 6.66 | 2.22 | no (gap > 2) |
| 4B \| 8B | 4e-7 | 1.53 | 12.8 | no |
| 4B \| 4B | 1e-6 | 1.99 | 12.5 | no |

Qwen3-4B says "low" whatever the text, with the answer set by option order. Qwen3-8B is closer: its order gap is near the bar, but it is saturated "low" on its own texts. **Per the prereg, `analyse` was not run, and the main estimand E is not read at either temperature pair.** The 10-item verbal AUCs the check prints (.44–.56) are not interpreted. Descriptively, surprisal AUCs are .88–1.0 again.

**Reading.** Floor at 4B/8B. Under neutral third-person framing, no Qwen3 model up to 8B has a usable verbal temperature report, while each reader's surprisal carries the information. **Kill rule applied: the temperature test is closed for this programme** (no re-wording chain, H24/H29). For the matched-reader paper, §4 becomes a documented floor: a published positive example of temperature self-report (Gemini Pro, Comșa & Shanahan) cannot be tested for self-excess at ≤ 8B open weights with this instrument. That is reported as a limitation, not as evidence for or against.

### Amendment 29 (run 36 extension, ~05:20Z, PRE-data at 4B): does a content-free weight write erode the post-trained report stance at Qwen3-4B? ("membrane" replicate; holiday 28, H31, MR30)
**Motivation (post hoc, stated as such).** In the 1.7B implant data (Am. 22, open format), every LoRA, including the content-free C, moved every sure/unsure probe from saturated confidence toward indifference by 10–18 log-odds, while the MC answers stayed confident (`reflections/2026-10-21T0309-holiday-bizarre-data-point.md`). That observation produced the hypothesis, so 1.7B is not evidence for it. This amendment is the first test.

**Design.** Qwen3-4B, fp32 (as in Am. 22). One remote job, so all arms share one instance:
- (0) untrained;
- (C16) the Am. 22 control arm: `implant.py train --facts C` (20 control facts × 16 templates, r = 16, all linear layers, lr 2e-4, 5 epochs, seed as in Am. 22);
- (C8a) the same data with r = 8 on attention only. This is a smaller and more local write, and it is secondary.
Each arm is evaluated with `implant.py eval` in the open format (primary, as at 1.7B under Deviation 22-3) and the lettered format (secondary), on the same 40 target items, with all six probes (self, person, ai, world, nonce, book) in both option orders.

**Estimand** (`membrane_analyse.py`, written and tested on the 1.7B files before any 4B data).
- Per probe: S_p = mean |y_p(untrained)| − mean |y_p(arm)|, the reduction in distance from indifference in log-odds. 2000 item-bootstrap draws.
- Self-specificity: S_self − max(S_person, S_ai, S_nonce, S_book). Revised before any 4B data from a mean-of-nonce-and-book version, which flagged a 0.8 log-odds difference on a 17 log-odds shift at 1.7B because of the book probe's lower baseline.

**Gates.**
- Headroom: a probe counts only if the untrained mean |y| ≥ 4 (H24; Qwen3-4B was at ceiling on related probes in run 29, so headroom is expected).
- Well-formedness: the arm's mean max answer probability ≥ .7 and mean order gap ≤ .3.

**Readings** (primary: C16, open format).
- **Replicates:** well-formed; S ≥ 2 with CI > 0 on every probe with headroom, nonce and book included; self-specificity CI lower bound ≤ 0. Prior ~55%.
- **Self-specific shift:** self-specificity CI > 0. The membrane reading fails at 4B, and a content-free write moving self-reports more than other-agent reports is itself reportable.
- **No shift:** S < 2 on nonce and book. The membrane doesn't replicate at 4B, and the manuscript's §5 drops to a 1.7B footnote (MR30 trigger).
- **Malformed:** format damage, not membrane.
- **Undetermined:** nonce/book lack headroom.

C8a and the lettered format are reported alongside and don't change the primary verdict. 1.7B check of the script (descriptive, not evidence): C, A, B and D all read "replicates" (S 4–19 log-odds; self-specificity −1.8 to +0.3, CIs ≤ 0 at the lower bound).

**Hardware and kit.**
- `implant.py` was converted to `wp3_env` (load_base, inputs on the model's device, provenance) in its own commit, and gained `--r` / `--targets` options whose defaults reproduce Am. 22.
- CPU parity was checked on Qwen3-0.6B (open eval, 40 items) against the pre-conversion file; the result is in the run log.
- `gpu_min_mem_gib` 24 (4B fp32 ≈ 16 GB plus LoRA training). No GPU is named. All comparisons are within the job.
- Adapter housekeeping: C8a (≈ 24 MB fp32) is collected and kept only if ≤ 20 MB, otherwise the operator is asked (STATE interim rule). C16 (≈ 130 MB) exceeds the collect size limit and is not kept; its stage-1 JSON is kept.

**Kill rule.** One job. An infrastructure failure allows one resubmission (new id). No further cells for the manuscript beyond this one (MR30).
- **Kit checks (before submission, ~05:16Z).** CPU parity: `implant.py eval none --open` on Qwen3-0.6B, before vs after the wp3_env conversion, gives max abs diff **0** over all 40 items and fields (252 s vs 249 s). Training smoke test (0.6B, C, r = 8 attention-only, 1 epoch) passed. It caught a variable-shadowing bug that recorded `lora_r` 7 in stage1.json (the adapter itself was r = 8), now fixed. LoRA initialisation is unseeded, as in Am. 22, so arm losses vary slightly between runs. Job `am29-membrane4b-01`.

### Amendment 29 result (collected 06:05Z 2026-10-21): replicates at 4B, as pre-registered, with a descriptive refinement
Job `am29-membrane4b-01` **done**: spot g6.xlarge (NVIDIA L4, fp32, TF32 off), eu-west-3b, $0.3539/h, **22.0 min, est. $0.13**, attempt 1, all 8 steps exit 0. Training: C16 (r 16, all linear) loss 1.31 → 0.11, 170 s; C8a (r 8, attention) 2.42 → 0.11, 162 s. Files: `results/am29/` (evals, stage-1 JSONs, `analysis_*.json`) and `results/am29-membrane4b-01/`.

**Primary (C16, open format): "replicates".** Well-formed: mean max answer p .915, order gap .16. All six probes have headroom (untrained mean |y| ≥ 4; world only via |y| per item). S (log-odds) with 95% CI:
- self +2.90 [2.36, 3.37]
- person +7.98 [7.47, 8.38]
- ai +2.99 [2.08, 3.95]
- world +4.41 [3.36, 5.51]
- nonce +3.39 [2.97, 3.77]
- book +4.81 [4.13, 5.59]
- self-specificity −5.09 [−5.49, −4.62]: self moves *less* than the other agent probes.

**Secondary.** C8a open: replicates (self-specificity −4.96). Both lettered-format cells: **malformed** (mean max answer p .46 / .41 < .7), so format damage in that format and no membrane reading there.

**Descriptive refinement** (not pre-registered; it changes the reading, not the verdict). Mean y per probe (self, person, ai, world, nonce, book):

| state | y per probe | mean P(unsure) |
|---|---|---|
| 1.7B untrained | −18.4 −15.7 −18.8 −7.1 −19.3 −14.6 | .00 |
| 1.7B C | −1.3 −0.4 −1.2 −1.3 −0.3 −0.6 | .32 |
| 4B untrained | +5.4 +9.8 +4.5 −0.6 +6.0 +7.0 | .88 |
| 4B C16 | −2.7 −1.8 −2.4 −1.6 −2.7 −3.1 | .10 |
| 4B C8a | −2.6 −1.7 −1.0 −1.2 −2.7 −3.6 | .16 |

The untrained stance is scale-specific and opposite: 1.7B is saturated "sure", while 4B is mostly "unsure", sensibly for fictional countries. A content-free write moves *both* to nearly the same mildly "sure" set-point (y ≈ −0.3 to −3.6) on every probe, including a different model's and an encyclopedia's. So the |y| estimand records 4B as "toward indifference", but 4B actually crossed indifference. The better description is **a weight write resets the report stance to a common, probe-general set-point from either side**, not "erodes toward indifference". H31's consequence is unchanged and now rests on two scales: a weight-write self-report shift must be read against the same arm's non-self probe shifts.

**Rival not excluded.** The control facts are the same *genre* as the target items (fictional countries' capitals). At 4B, moving from "unsure" to "sure" about unknown fictional capitals (accuracy .30 / .225, chance .25) could be genre familiarity: after reading many such facts, questions like these feel answerable. That differs from a generic effect of weight writes. A genre-mismatched control (e.g. facts about fictional chemicals) would separate them. Not run; it is noted for the manuscript's §5 as the one missing control.

**Adapter housekeeping.** m4b_C8a is 23.6 MB (> 20 MB interim rule), so it is **not committed**. It stays on the never-deleted branch `wp3-results/am29-membrane4b-01` (`output/OUT/m4b_C8a/`; adapter_model.safetensors sha256 33042c77831ed17273a8dc337207bdd025c1024c8ca34a464d8c47a0ad880dc8). The operator is asked in the log. m4b_C16 (~130 MB) was not collected by design; its stage-1 JSON and adapter_config are kept.

### Amendment 30 (run 37, ~06:25Z, PRE-data): is the Am. 29 content-free shift generic to weight writes or genre familiarity? (genre-mismatched control at 4B)
**Why.** Am. 29 replicated the "membrane" at 4B: a content-free write (C, fictional-capital facts) moves every sure/unsure probe to a common, mildly "sure" set-point. Its named rival is **genre familiarity**: C is the same genre as the target items (capitals of fictional countries), so after training, questions like these may simply feel answerable. This is the one missing control for the matched-reader manuscript's §5 (MR30). No other cell is added.

**Prior.** At 1.7B the same C write moved the probes *away* from "sure" (saturated sure → indifference). Genre familiarity predicts movement toward "sure", so 1.7B already speaks against it at that scale. Prior for GENERIC at 4B: ~65%.

**Design.** Qwen3-4B, fp32, one remote job so all arms share an instance:
- (U) untrained, evaluated again on this instance.
- (C16) Am. 29's C arm retrained: same settings (r 16, all linear, lr 2e-4, 5 epochs). Retrained because the 130 MB C16 adapter was not kept, and so that C and G share hardware and run.
- (G16) new `--facts G`: 20 fictional-compound ← fictional-mineral facts ("Monine is synthesised from Saisosite") in 16 templates (`CHEM` in implant.py). Same ASK prompts and settings, 320 rows (dose-matched to C). No countries or capitals, and no overlap with target or C words.
- Evaluation: `implant.py eval --open` (primary format at both scales) on the 40 target items with all six probes. The lettered format is dropped: it was malformed in Am. 29.

**Estimand** (`genre_analyse.py`, written and tested before any G data; the test used Am. 29's C8a as a stand-in for G, so it is descriptive only).
- Signed shift D_a,p = mean y_p(arm) − mean y_p(U). Signed because 4B crosses indifference, so the |y| measure misdescribes it.
- R = mean_p D_G,p / mean_p D_C,p over the six probes, with 2000 paired item-bootstrap draws.

**Three slots (H32), fixed now.**
- *Channel checked* = C reproduces the shift here (|mean D_C| ≥ 2, CI excluding 0) and G is well-formed (mean max answer p ≥ .7, order gap ≤ .3).
- **GENERIC SHOWN:** R CI lower bound ≥ 0.5. If the lower bound is ≥ 0.75, a genre share above δ = 25% is also excluded.
- **GENRE SHOWN:** R CI upper bound < 0.5. In that case the manuscript's §5 reports the 4B shift as genre familiarity, and H31's gate is rewritten to require genre-matched foreign controls.
- **NOT SHOWN AS OF run 38:** anything else, with channel checked yes/no. It is not interpreted either way.

Per-probe D values and the Am. 29 self-specificity on G (`membrane_analyse.py U G`) are reported alongside and do not change the verdict.

**Hardware.** `gpu_min_mem_gib` 24. No GPU is named: all comparisons are within the job, so any instance works.

**Kill rule.** One job. An infrastructure failure allows one resubmission under a new id. No further membrane cells after this.

**Adapters.** G16/C16 are ~130 MB each and are not collected (stage-1 JSON and adapter_config only).

**Kit check (before submission, 06:24Z).** 0.6B smoke train of `--facts G` (1 epoch, r 8 attention, CPU): 320 rows, loss 2.44, 112 s, exit 0. `genre_analyse.py` run on Am. 29 files with C8a standing in for G: R 0.97 [0.95, 0.99], "GENERIC SHOWN" (pipeline check only; C8a is not a genre control). C16 and G16 are retrained in-job, not reused: no 4B C16 adapter was kept (130 MB). Job `am30-genre4b-01`.

### Amendment 30 result (collected 06:54Z 2026-10-21): GENERIC SHOWN; a genre share above 25% is excluded
Job `am30-genre4b-01` **done**: spot g6.xlarge (NVIDIA L4, fp32, TF32 off), eu-west-3b, $0.3539/h, 06:27:41–06:43:09Z, **15.5 billed min, est. $0.09**, attempt 1. Training: C16 loss 1.32 → 0.11 (169 s); G16 1.69 → 0.11 (166 s), 320 rows each. Files: `results/am30/` (evals, stage-1 JSONs, `analysis_genre.json`, `analysis_membrane_G16.json`) and `results/am30-genre4b-01/`.

**Reproducibility.** The untrained 4B y-values reproduce Am. 29's on a new instance: 5.39 / 9.78 / 4.46 / −0.58 / 6.04 / 6.98 against 5.4 / 9.8 / 4.5 / −0.6 / 6.0 / 7.0.

**Primary (as pre-registered).** Channel checked: yes. C reproduces the shift (mean D_C −7.35 [−8.16, −6.43]) and G is well-formed (mean max p .95, gap .18).
- R = mean D_G / mean D_C = **0.84 [0.81, 0.86]**.
- Lower bound ≥ 0.75 → **GENERIC SHOWN, genre share > 25% excluded.**

**Per probe** (signed shift D, log-odds; C → G):

| probe | C | G |
|---|---|---|
| self | −7.85 | −5.98 |
| person | −11.16 | −9.00 |
| ai | −5.75 | −4.62 |
| world | −1.26 (CI incl. 0) | −0.35 (CI incl. 0) |
| nonce | −8.22 | −8.24 |
| book | −9.88 | −8.72 |

**Alongside (does not change the verdict).**
- `membrane_analyse.py` U vs G: "replicates". Self-specificity −4.02 [−4.57, −3.44]: self moves *less* than the other agents, as in Am. 29.
- G's end state is a little less "sure" than C's: mean y −0.6 to −2.2 against C's −1.3 to −2.9.

**Reading (H27).**
- **RESULT.** At Qwen3-4B, a content-free LoRA write about an unrelated genre (fictional chemistry) shifts every sure/unsure probe almost as much as one about the target genre (fictional capitals). The membrane/set-point effect is mostly generic to weight writes.
- A genre component exists, but it is small: about 16% [14%, 19%]. The R interval excludes 1. It is not zero, and it is stated.
- **INSTRUMENT.** One seed per arm, unseeded LoRA init, open format only, one model.
- **DECISION.** H31's gate stands as written: a weight-write self-report shift is read against the same arm's non-self probe shifts. Genre matching is a refinement, not a requirement. The manuscript's §5 is complete (two scales, the genre control and the nearest relative, IFT's factual-no). No further membrane cells (kill rule).

### Amendment 31 (run 38, ~09:20Z, PRE-data): verbalisation efficiency at Qwen3-4B, untrained (MR31; holiday 30, H33)
**Why.** MR31 pivoted measurement to *verbalisation efficiency*: how much of the information about an occurrent perturbation that is present at the report's own readout position does the verbal report use? The 0.6B kit check (run 37, descriptive) showed range at both ends (probe .61–.72, corrected report .29/.45, chance .20). This is the first registered cell, at 4B, untrained, inference only. Target premise (D1): **(D) post-trained stance suppression** — does a self-framed question ("injected thought") use less of the present information than a neutral one ("stands out")?

**Rival ledger (H33), written before data.**

| Reading | Predicts here | Forbids here | Testable in this cell? |
|---|---|---|---|
| (A) genuine non-use of present information | E < 0 under both framings | — (A's rival F predicts the same) | **No: A vs F reserved for the trained cell** |
| (F) training asymmetry (supervised probe vs zero-shot report) | E < 0 under both framings, F ≈ 0 | — | No (as A) |
| (B) answer-format loss (index mapping) | report ≈ chance in both framings while probe is above chance | — | Only as a floor trigger; 2AFC is the named fallback (D2), not run here |
| (C) probe over-reach ("something was added", not content) | probe(random) ≈ probe(concept) | probe(concept) − probe(random) > δ_C | **Yes** (matched-norm random-direction arm) |
| (D) stance suppression under self-framing | F < 0 with probe(intro) ≈ probe(neutral) | F > 0 shown | **Yes** (framing contrast; lexical-dilution rival logged, not tested) |
| (E) scale floor | E closer to 0 than at 0.6B | — | Descriptive only (two points, different concept sets) |

**Design.** Qwen3-4B, fp32, one remote job. `inject_loc.py` as in run 37 (IFT-style localisation, N = 5 sentences, injection on one sentence's tokens, per-framing no-injection correction; fix 1: the probe gets h(injected) − h(none)).
- **30 new concepts** (`LOC_CONCEPTS` in `inject_loc.py`, written in this commit before any 4B data; no overlap with `inject_detect.CONCEPTS`, `BASELINE` or the 0.6B kit check).
- **Settings** L20 α4 and L24 α2 (36-layer model). Probe layers 30 and final (36); **primary probe = final layer** (what the answer token sees).
- **Arms:** concept (α · (v_c − mean baseline)) and **random** (a Gaussian direction, seeded by the concept, scaled to the same norm as the concept arm's vector). Same sentence sets in both arms.
- **Primary setting** chosen by the concept arm's final-layer probe accuracy (mean over framings), never by report accuracy.

**Estimands** (`inject_loc_analyse.py`, leave-concept-out probe, concept-bootstrap 2000 draws; the one addition this run is the random-arm comparison script mode, tested on the 0.6B kit file before submission):
- E_q = acc(report_cor, q) − acc(probe_final, q), q ∈ {intro, neutral}.
- F = acc(report_cor, intro) − acc(report_cor, neutral); also F_probe = acc(probe_final, intro) − acc(probe_final, neutral).
- Δ_C = acc(probe_final, concept arm) − acc(probe_final, random arm), per framing (unpaired concept bootstrap over the same concepts, paired by concept).

**Channel checked** (H24): concept-arm final-layer probe CI lower bound > .30 (chance .20) in the primary setting, and the corrected report not at ceiling (< .90) under either framing. If the probe CI includes .30 → floor, every verdict below is NOT SHOWN with channel checked = no (this is also an MR31 reconsider trigger).

**Three slots (H32), fixed now.** δ_E = 25% of (probe_final − chance) per framing; δ_F = 25% of (mean corrected report over framings − chance); δ_C = 25% of (concept probe − chance). Conventions, said as such.
- E_q: **LOSS SHOWN** if CI upper < 0; **LOSS EXCLUDED** if CI inside ±δ_E; else NOT SHOWN AS OF run 38.
- F: **SELF-FRAME DEFICIT SHOWN** if CI upper < 0 and F_probe CI includes 0; **SELF-FRAME ADVANTAGE SHOWN** if CI lower > 0; **EXCLUDED** if CI inside ±δ_F; else NOT SHOWN. (Report alongside: a deficit is compatible with D *and* with lexical dilution; this cell does not separate them.)
- Δ_C: **CONCEPT-SPECIFIC SHOWN** if CI lower > 0; **"SOMETHING ADDED" (probe content-free) SHOWN** if CI inside ±δ_C; else NOT SHOWN. If content-free, E is relabelled "efficiency for perturbation location", not for concept information.
- **Central question cap:** whatever E shows, the verdict on "does the report use what is present?" is at most a description of *this untrained model*; A vs F (non-use vs training asymmetry) is NOT SHOWN here by design.
- **B trigger:** if the corrected report CI includes .20 under both framings, the index format is floored and the 2AFC fallback is the next cell (no third format).

**Hardware.** `gpu_min_mem_gib` 24 (fp32 4B ≈ 16 GB). No GPU named: all contrasts are within the job.

**Kill rule.** One job; one resubmission under a new id only for an infrastructure failure. No other cells this run.

**Kit check (before submission, 09:20Z; descriptive pipeline check only).** Qwen3-0.6B CPU, L8 α4, 6 of the 30 new concepts, both arms (~2 min each): `inject_loc.py --concepts loc [--random]` and `loc31_analyse.py` run end to end; the channel rule fires correctly (probe CI lower .28 < .30 → every slot "channel checked: no"); slim copies written. Not evidence; the 0.6B rows are not kept. Code changes in this amendment: `inject_concord.concept_vec` now builds its input on `m.device` (GPU-ready, no-op on CPU); `inject_loc.py` gains `LOC_CONCEPTS`, `--concepts loc`, `--random`, 4-significant-digit hidden states, and `--probe-layers` accepting `:`; new `loc31_analyse.py`. Job `am31-loc4b-01` (`remote/jobs/am31-loc4b-01.json`). Primary-setting rule as above; results land in `results/am31/`.

### Amendment 31 result (collected 09:42Z 2026-10-21; job `am31-loc4b-01`, 8.3 min, ≈ $0.05): information present (probe .93), report at chance → index format floored; framing null; probe ceiling partly content-free
Files: `results/am31/analysis_{L20a4,L24a2}.json`, slim rows `results/am31/*_slim.json` (full rows with hidden states, ~24 MB each, stay on branch `wp3-results/am31-loc4b-01`). **Primary setting = L24 α2** (concept-arm final-layer probe .933 vs .923 at L20 α4; chosen by the rule, by probe only). Chance .20, n = 150 rows, 30 concepts.

| | L24 α2 (primary) | L20 α4 |
|---|---|---|
| probe (final), concept arm, intro / neutral | .94 [.87, 1.00] / .93 [.85, .99] | .93 / .91 |
| report_cor, concept arm, intro / neutral | .15 [.09, .22] / .14 [.08, .21] | .15 [.07, .23] / .15 [.09, .24] |
| probe, random arm, intro / neutral | .89 / .80 | .65 / .49 |
| report_cor, random arm, intro / neutral | .40 / .34 | .45 / .36 |

- **Channel checked: yes** (probe CI lower .85 > .30; report < .90).
- **E (primary): LOSS SHOWN** under both framings: intro −.79 [−.89, −.66], neutral −.79 [−.89, −.65] (δ_E .18). Same at L20 α4. Per the cap, this describes this untrained model only; A vs F is not tested here.
- **B trigger: fires.** The corrected index report's CI includes chance under both framings at both settings (concept arm). So the index format is **floored for concept injections** at 4B, and E's "loss" is a probe at ceiling against a floored report. **Next cell = the named 2AFC fallback** (D2; no third format).
- **F (framing): NOT SHOWN AS OF run 38.** F = +.01 [−.05, .07] (primary), −.01 [−.10, .07] (L20); F_probe ≈ 0. δ_F is ill-defined (mean report below chance gives a negative δ), so EXCLUDED could not be reached as pre-registered; the honest slot is not shown. Reading D (stance suppression under self-framing) gets no support; the 0.6B neutral > intro gap (.45 vs .29) does not appear at 4B.
- **Δ_C (primary): mixed.** Intro +.05 [−.03, .13] inside ±.185 → **SOMETHING-ADDED (content-free) SHOWN**; neutral +.13 [.03, .23] → CONCEPT-SPECIFIC SHOWN (lower bound > 0, upper above δ). At L20 α4 both framings CONCEPT-SPECIFIC (+.28, +.43). So at L24 the final-position probe mostly reads "a perturbation is here"; at L20 more of its ceiling is concept-carried.
- **Descriptive, not pre-registered (flagged).** (1) The report localises a *random* matched-norm perturbation (.34–.45) better than a *concept* one (.14–.15). (2) At L20 α4 the concept-injected sentence's index logit is pushed *down* relative to no injection (mean corrected shift −0.44 intro / −0.54 neutral vs +0.10 / +0.03 for other indices): the report avoids the concept-carrying sentence. At L24 α2 all corrected shifts are < 0.1 nats: the report barely moves. (3) Scale, descriptively (reading E): from 0.6B to 4B the probe rises (.61–.72 → .93) and the report falls (.29/.45 → .15), so the gap widens with scale on these two points.

**Ledger update (H33).** A, F: untested (as designed). B: **live**, the index report is floored for concept injections (2AFC next). C: partly supported at L24 (content-free probe ceiling), not at L20. D: no support (F ≈ 0, not excluded). E: descriptively against (gap widens). New, unregistered: concept-specific *avoidance* in the report at L20 α4 (a candidate for its own pre-registered test, with random arm as the control).

### Amendment 31b (run 38, ~09:47Z, PRE-data for 2AFC): the named fallback format after Am. 31's B trigger
**Why.** Am. 31's index report was floored (CI includes chance in both framings) while the probe read .93. The pre-registered response (D2) is one fallback format, two-alternative choice; no third format follows. It tests reading **B** (answer-format loss): if the report reaches the probe with only two options, the 5-way loss was largely the index mapping.
**Design.** Identical to Am. 31 except `--n 2` (two sentences, inject on one; chance .50). Same 30 `LOC_CONCEPTS`, both settings (primary stays **L24 α2**, fixed by Am. 31's rule), both arms (concept, matched-norm random), probe layers 30 and final, same framings. 60 rows per arm and setting. Job `am31b-loc4b2afc-01`.
**Analysis change, pre-data and stated:** `loc31_analyse.py --probe-floor 0.60` (channel: probe CI lower > .60 = chance + .10, the N = 2 analogue of .30); δ_F = 25% of |mean report − chance| (Am. 31's formula went negative when the report sat below chance; Am. 31's recorded slots are not re-scored).
**Slots (H32), fixed now.** As Am. 31 for E, F and Δ_C. Plus **B**: *FORMAT LOSS SHOWN* if E(2AFC) at the primary setting has CI lower bound > E(5-way) point estimate on the same chance-normalised scale, i.e. (acc − chance)/(1 − chance), and the 2AFC report CI excludes .50; *FORMAT NOT THE CAUSE* (B excluded) if the 2AFC report CI includes .50 in both framings (then verbal localisation is a **documented floor** for untrained Qwen3-4B, channel checked yes, and the trained cell is the only route left); else NOT SHOWN AS OF run 38.
**Also recorded, descriptive:** whether the concept-sentence avoidance at L20 α4 (Am. 31 descriptive (2)) recurs at N = 2 (mean corrected target shift < 0). Not a verdict.
**Kill rule.** One job; one resubmission only for an infrastructure failure.

### Amendment 31b result (collected 09:59Z 2026-10-21; job `am31b-loc4b2afc-01`, 5.2 min, ≈ $0.03): 2AFC also at chance at the primary setting → format is not the cause; verbal localisation is a documented floor for untrained Qwen3-4B. Secondary setting: a self-frame deficit with below-chance avoidance
Files: `results/am31b/analysis2_{L24a2,L20a4}.json`, slim rows `results/am31b/*_slim.json`. Chance .50, 60 rows per arm and setting.

| | L24 α2 (primary) | L20 α4 |
|---|---|---|
| probe (final), concept, intro / neutral | .97 [.92, 1.00] / .97 | .98 / .97 |
| report_cor, concept, intro / neutral | .47 [.33, .60] / .37 [.25, .48] | **.20 [.08, .32]** / .62 [.45, .77] |
| report_cor, random, intro / neutral | .63 / .50 | .55 / .50 |

- **Channel checked: yes** (probe CI lower .92 > .60).
- **B (primary): FORMAT NOT THE CAUSE.** The 2AFC report's CI includes .50 under both framings at L24 α2. With Am. 31, verbal localisation of concept injections is a **documented floor for untrained Qwen3-4B at the primary setting** (index and 2AFC), while the answer-position probe reads it at .97. No third format (D2).
- **E (primary): LOSS SHOWN** (intro −.50 [−.63, −.37], neutral −.60 [−.73, −.47]); capped as in Am. 31.
- **Δ_C (primary):** intro content-free shown (+.02 [−.05, .08], δ .12); neutral not shown.
- **F (primary): NOT SHOWN** (+.10 [−.03, .25]).
- **Secondary setting L20 α4 (reported, not the verdict setting):** **SELF-FRAME DEFICIT SHOWN**, F = −.42 [−.57, −.27] with F_probe +.02 [.00, .05]. Under the introspective framing the report picks the injected sentence *below* chance (.20, CI upper .32 < .50): it avoids it. Under the neutral framing it picks it above chance (.62). The random arm shows no such split (.55 / .50). This repeats, in 2AFC and much larger, the concept-specific avoidance seen descriptively in Am. 31 at the same setting. It is the first framing-dependent result at 4B, and it is concept-specific. Under the H33 ledger: D (stance suppression) gains support at L20 α4 only; the lexical-dilution rival is not excluded.

### Instrument defect in Am. 31 / 31b (found run 39, 12:30Z 2026-10-21, before any new data): the report was read one token early
While adding a coherence check for Am. 32, the probability mass on the index tokens came out at ~1e-6 at Qwen3-0.6B. Cause: Qwen3 tokenizes " 1" as [" ", "1"], `idx_tok` keeps the digit, and the prefill ended "…is SENTENCE", so the model's next token is the space, not the digit. **Every report in Am. 31 and 31b (index and 2AFC) compared digit logits at a position where the model puts almost no mass on any digit.** The bias correction (subtracting the same pair's no-injection logits) does not repair this: it reads an off-distribution logit difference. With the prefill ending "SENTENCE " the mass is .996–1.000 at 0.6B.
- **Update 12:59Z (Am. 31c):** the fixed re-run reproduces every report number within ~.05, so the slots below are re-established on the valid instrument (see Am. 31c result). The original text is kept as written.
- **Consequence (H27 form).** RESULT: none of Am. 31/31b's report-side slots stands. The "documented floor" (B: FORMAT NOT THE CAUSE), E: LOSS SHOWN, F at L20 α4 (labelled "SELF-FRAME DEFICIT"; by H35 the wording contrast is mechanism-naming vs salience) and the below-chance avoidance are all **INSTRUMENT-INVALID**, not re-scored, not negatives. The probe-side numbers (answer-position probe .93–.97, Δ_C) do not depend on the readout token and stand, though the hidden states were also taken one position early (at "SENTENCE", not after the space); they remain a valid description of that position.
- The trained-cell kit (`inject_loc_train.py`) imports `build`, so it inherits the fix.
- Public correction needed: my post-8 comment `cgVb3qTWkkU5SxKL` reported Am. 31's report floor; the DM to superz_glm (`J9WRjEHX121AfDUY`) reported the .20 avoidance.

### Amendment 31c (run 39, PRE-data): Am. 31/31b re-run with the fixed readout
**Design.** Identical to Am. 31 (N = 5, index) and Am. 31b (N = 2) at L24 α2 and L20 α4, same 30 `LOC_CONCEPTS`, concept and matched-norm random arms, probes at layer 30 and final, except: prefill ends "SENTENCE " (fixed readout); a third framing **word** ("One sentence has an injected word. Which sentence has the injected word?"), the minimal-pair partner of **intro** that swaps only "thought" → "word" (H35); coherence recorded (index-token mass with and without injection). Same estimands and slots as Am. 31/31b (`loc31_analyse.py`, `--probe-floor .30` for N = 5 and `.60` for N = 2), F's labels now name the wording (intro < / > neutral). Added: **F_word** = report(word) − report(intro), δ = .25 |mean − chance|; slots WORD > THOUGHT / THOUGHT > WORD / EXCLUDED / NOT SHOWN.
**Rival ledger (H33), fixed now.** (B) format loss: if fixed 2AFC reaches the probe while fixed index does not, B is shown. (D) stance suppression via mental vocabulary: predicts intro < word in report at matched probe, forbids word < intro. (D′) mechanism-naming (any "injected" frame): predicts intro ≈ word < neutral; forbids intro ≠ word. (Readout artefact): if the fixed reports are at chance everywhere with coherence passing, the floor is real (then D2: no third format).
**Also replicates** the old avoidance claim properly: at L20 α4 N = 2, is intro report_cor CI upper < .50 (avoidance SHOWN), CI lower > .50, or includes .50.
**Kill rule.** One job (`am31c-loc4b-fix-01`); one resubmission only for an infrastructure failure. Settings not coherent (index mass < .5 or < half the no-injection mass in any framing) are reported as INCOHERENT, no verdicts.

### Amendment 32 (run 39, PRE-data): untrained Qwen3-4B setting sweep (MR32 M1), fixed readout
**Why.** M1 of the MR32 plan: find coherent settings where the probe has room (P in [.75, .95]) and the report sits below it, to choose the training setting, and check k1 (report ≈ probe everywhere).
**Design.** Layers {12, 16, 20, 24, 28} × α {1, 2, 4, 8} (20 settings), N = 2 (2AFC), 30 new `SWEEP_CONCEPTS` (written pre-data, no overlap with LOC_CONCEPTS, inject_detect.CONCEPTS or BASELINE, none in BANK), concept and matched-norm random arms, framings intro and neutral, fixed readout, probes at layer 30 and final, coherence recorded. Per setting `loc31_analyse.py --probe-floor .60`.
**Deviation from the MR32 plan, stated pre-data.** 5-way index is dropped from the sweep (cost: it would double the job; N = 5 is covered at two settings in Am. 31c), and the word framing runs only in Am. 31c (same reason).
**Rules, fixed now (`sweep_analyse.py`).** Training setting = among coherent settings with channel checked and P (concept probe, final, mean of framings) in [.75, .95], the largest P − R with R = the better framing's concept report; ties → lower α. If none is eligible, the setting with the largest P − R among coherent settings with P ≥ .75 is recorded and the choice is deferred to the next strategic reflection. **k1**: if E is LOSS EXCLUDED in some framing at every coherent setting with channel checked, there is no "unused" phenomenon at 4B and M3/M4 are dropped (write-up only). Avoidance replication on new concepts: at L20 α4, intro report CI upper < .50.
**Hardware.** No GPU named; fp32; one job (`am32-sweep4b-01`) holds both arms. Kill rule: one resubmission only for an infrastructure failure.

### MR32 M2 result (run 40, CPU, existing Am. 31c full rows; no new model data): readout-patch positive control, and a k3 estimator fix
`readout_patch.py` adds β × (leave-concept-out probe logits, centred, unit SD) to the report logits of the 2AFC Am. 31c rows (L20 α4, L24 α2): access opening by construction. Checks: **β = 0** reproduces Am. 31c exactly (intro .22/.53, neutral .63/.42). **β = 20**: report = probe (.95/.90 at L20 α4; .93/.97 at L24 α2), probe and dC slots unchanged. **Shuffled-label patch (β = 20)**: report pinned near chance (.50/.50; .42/.43), E LOSS SHOWN: correctly not "access opening".
**k3 event (estimator mislabels the control):** with β = 20 the channel gate's headroom clause (report < .90) failed, so loc31_analyse.py set every slot to "channel checked: no". A successful trained model would have been labelled the same way. **Fix (before any trained-cell data):** `--channel-from <untrained analysis>` makes a trained/patched cell inherit the untrained cell's headroom gate (probe floor still checked on the cell). With it, the patch scores **E: LOSS EXCLUDED** (0 [0, 0]) in both frames, dC CONCEPT-SPECIFIC SHOWN: the positive control is labelled correctly. M4 must use `--channel-from`. M2 done.

### Amendment 34 (run 40, PRE-data): second-person calibration (holiday 32, policy H37)
**Why.** Holiday 32 named an unoccupied school: self-report reliability as an acquired property of an exchange history, calibrated by another's contingent, correct naming of the reporter's own state (the LM analogue of social biofeedback). WP-E SQ46 sets a sceptical prior: in humans, contingent feedback about one's own heartbeat doesn't transfer (Rominger 2021), and contingency per se doesn't drive the mirroring effects (Murray 2016/2018). Novelty (b) (`notes/lit-am34-mirroring-novelty-2026-10-21.md`): in-context own-activation labelling exists (Ji-An 2025; critique Singh/Linzen/Ravfogel 2026), but not with injection, contingent vs noncontingent vs non-self feedback, and a matched answer-position probe.
**Design (`mirror34.py`).** Untrained Qwen3-4B, fp32, 2AFC, intro frame. Each test item is one chat with K = 4 earlier trials (context concepts from LOC_CONCEPTS, each injected on its own target sentence at the same L, α), each followed by a seeded assistant answer (independent of the target, the same across arms) and a feedback turn by arm: **none** ("Thanks."), **contingent** (true target), **noncontingent** (seeded random sentence), **nonself** (one sentence per earlier trial marked "(*)"; feedback names the marked sentence: correct and contingent, but on a visible text signal). Test concept: 30 new `MIRROR_CONCEPTS` (written pre-data; no overlap with earlier lists), both targets. Bias correction: same chat, no test injection (context injections kept). Probe: leave-concept-out on h(inj) − h(none), final layer. Settings: **L16 α2** (primary: report near chance in both frames, P .87; the registered rule's runner-up), **L12 α1** (the sweep's selected setting, wrong-signed), **L20 α4** (avoidance setting: does contingent feedback reverse the sign?).
**Estimands and slots (`mirror34_analyse.py`, δ = .10).** C1 = contingent − noncontingent, C2 = contingent − nonself, C3 = contingent − none (report); dP = probe(contingent) − probe(none); X = C3 − dP. **SHOWN** iff C1 and C2 CI lower > 0 ("report-side" iff X also SHOWN). **FALSIFIED at a setting** iff C1 and C2 both EXCLUDED (CI inside ±.10). Coherence: index-token mass ≥ .5 in every arm, else the setting is undetermined. The primary verdict is L16 α2; the other two are secondary. A gain at L12 α1 or L20 α4 that only moves a below-chance report towards .50 is labelled "sign-unmasking", not calibration (sign check |rep − .5| reported).
**Rivals.** (R1) in-context learning of an activation→label mapping = what the school predicts mechanically; the school's content is that the mapping comes from *another's* naming of own state, so C2 (nonself) is the key control. (R2) feedback teaches the answer format or genre → nonself and noncontingent gain as much (C1, C2 ≈ 0). (R3) context injections change the residual stream globally (salience) → probe and report move together (X ≈ 0).
**Hardware.** No GPU named; one job (`am34-mirror4b-01`). Kill rule: one resubmission only for an infrastructure failure.

### Amendment 32 result (collected 15:14Z 2026-10-21, run 40; job `am32-sweep4b-01`, spot ap-northeast-2d g6.xlarge, 26.9 min, ≈ $0.20): no setting where the report reaches the probe (k1 not triggered); selected setting L12 α1; avoidance replicates on new concepts
Files: `results/am32/` (`summary_sw.json`, 20 `analysis_sw_*.json`, slim rows); full rows on `wp3-results/am32-sweep4b-01`. Coherence passes at all 20 settings (index-token mass 1.00 with and without injection, both frames); channel checked everywhere.

| 2AFC concept report (intro / neutral) | α1 | α2 | α4 | α8 |
|---|---|---|---|---|
| L12 (P .88–.90) | .37 / .30 | .40 / .35 | .42 / .42 | .55 / .58 |
| L16 (P .87–.97) | .45 / .60 | .48 / .62 | .58 / .77 | .75 / .88 |
| L20 (P .97) | **.12** / .72 | **.15** / .70 | **.23** / .70 | .57 / .63 |
| L24 (P .97–.98) | .47 / .45 | .58 / .57 | .80 / .67 | .85 / .82 |
| L28 (P .97–1.00) | .50 / .50 | .45 / .47 | .52 / .52 | .57 / .55 |

- **Selection rule (as registered):** eligible (P in [.75, .95], coherent, channel checked) = L12 α1/2/4/8, L16 α1/2/4. **Selected: L12 α1** (P .90, R .37, P − R .53).
- **Caveat, written before any training (not a change of rule):** at L12 α1 both frames sit *below* chance (intro .37 [.23, .50], neutral .30 [.17, .45]: the neutral CI excludes .50). So the selected P − R includes wrong-signed reporting, not just unused information. A training gain there could be a sign flip (re-routing) as much as access opening; M4's table must carry a "sign-corrected" slot (report vs 1 − report) and M3 should consider L16 α2 (intro .48, neutral .62: near chance, not wrong-signed; P − R .25) as the registered rule's runner-up. Choice of M3 setting goes to MR34 / the next strategic reflection, per H36, rather than silently overriding the rule.
- **k1: not triggered.** E is LOSS SHOWN in 39 of 40 setting × frame cells (NOT SHOWN only at L16 α4 neutral, .77 vs probe .89). The "present but unused" phenomenon exists across the sweep; M3/M4 stay live.
- **Avoidance replication (new concepts): SHOWN.** L20 α4 intro .23 [.12, .37], upper < .50. It is stronger at lower dose (α1 .12, α2 .15) and gone at α8 (.57); neutral stays .63–.72 at L20. Random arm at L20 α4: .67/.67 (no avoidance), as in Am. 31c.
- **Descriptive (not verdicts):** at high dose the report rises (L24 α8 .85/.82, L16 α8 .75/.88) but the random arm rises with it (L24 α8 random .82/.73; dC slot "SOMETHING-ADDED (content-free)"), so high-α reports track that *something* was injected, not which. At L24 α4 the frame effect reverses (intro .80 > neutral .67). Frame spread (H36) is therefore setting-dependent in sign.

### Amendment 31c result (collected 12:59Z 2026-10-21; job `am31c-loc4b-fix-01`, 15.1 billed min, ≈ $0.11): the fixed readout reproduces Am. 31/31b; avoidance at L20 α4 holds under both "injected" wordings
Files: `results/am31c/` (analyses, summaries, slim rows; full rows on `wp3-results/am31c-loc4b-fix-01`). Coherence passes everywhere (index-token mass 1.00 with and without injection, all frames). Channel checked: yes in all four cells.

| report_cor (concept arm) | intro | neutral | word | probe (final, intro) |
|---|---|---|---|---|
| N = 2, L24 α2 (primary) | .53 [.40, .65] | .42 [.28, .55] | .53 [.38, .67] | .93 |
| N = 2, L20 α4 | **.22 [.10, .33]** | .63 [.47, .78] | **.32 [.20, .43]** | .95 |
| N = 5, L24 α2 (primary) | .15 [.09, .24] | .17 [.11, .23] | .12 [.07, .18] | .91 |
| N = 5, L20 α4 | .14 [.07, .22] | .20 [.13, .29] | .15 [.10, .21] | .94 |

- **The defect did not change the answers.** With the readout at the right position, every report number is within ~.05 of the invalid Am. 31/31b values (same concepts and sentence draws, so this is an instrument check, not an independent replicate). The bias-corrected digit contrast one token early happened to rank the sentences the same way. The Am. 31/31b slots are now **re-established on a valid instrument**: verbal localisation is a documented floor for untrained Qwen3-4B at the primary setting (index .15 and 2AFC .53/.42, both CIs include chance; B trigger fires; format not the cause), E: LOSS SHOWN everywhere (capped: A vs F untestable untrained), Δ_C concept-specific in 7 of 8 framing×cell entries.
- **Avoidance at L20 α4: SHOWN** (intro CI upper .33 < .50). Random arm: no avoidance (.55/.50/.62).
- **F_word (H35 minimal pair): NOT SHOWN** at any cell; at L20 α4 word − intro = +.10 [.00, .20], and the word framing is *also* below chance (.32, CI upper .43). Rival ledger: (D) mental-vocabulary stance predicted intro < word and is **not supported**; (D′) mechanism-naming ("injected …" in either wording) predicted intro ≈ word < neutral, which is what the data show. By H35 the effect is labelled **"injected-item framing avoids the perturbed sentence"**, not a self-frame effect.
- **Descriptive (not a verdict):** random-direction injections are reported better than concept ones in the index format (.33–.48 vs .12–.20), as in Am. 31.

### Amendment 34 result (collected 17:20Z 2026-10-21, run 41; job `am34-mirror4b-01`, spot eu-west-3b g6.xlarge, 16.9 billed min, ≈ $0.10): second-person calibration NOT SHOWN at any setting; not falsified at the primary (C2 not excluded)
Files: `results/am34/analysis_mr_{L16a2,L12a1,L20a4}.json`; full rows `mr_*.json` stay on the results branch `wp3-results/am34-mirror4b-01`. 60 rows per arm and setting; coherence 1.0 in every arm.

| setting | none | contingent | noncontingent | nonself | C1 | C2 | C3 | dP |
|---|---|---|---|---|---|---|---|---|
| **L16 α2 (primary)** | .62 | .72 | .68 | .70 | +.03 [−.03, .10] EXCLUDED | +.02 [−.07, .10] not shown | +.10 [.00, .20] not shown | +.13 [.00, .27] not shown |
| L12 α1 | .43 | .42 | .42 | .43 | .00 [−.07, .07] EXCLUDED | −.02 [−.12, .08] not shown | −.02 [−.08, .05] EXCLUDED | +.02 not shown |
| L20 α4 | .43 | .40 | .35 | .40 | +.05 [−.05, .15] not shown | .00 [−.10, .10] not shown | −.03 not shown | .00 EXCLUDED |

- **Verdict (primary L16 α2): NOT SHOWN AS OF run 40/41.** C1 (contingent vs noncontingent) is EXCLUDED; C2 (vs nonself) has CI upper .10, on the δ edge, so the registered FALSIFIED slot (both excluded) does not fire. Read plainly: four turns of correct feedback about which sentence carried the injection do not improve the report beyond feedback that is random or tied to a visible marker.
- Any context with feedback lifts the report a little over "none" (all three feedback arms .68–.72 vs .62), and the probe moves the same way (dP +.13): a context effect, not calibration (X ≈ 0).
- **No sign-unmasking** at L12 α1 or L20 α4 (sign check |rep − .5| ≤ .15 everywhere).
- **Descriptive, not a slot:** at L20 α4 the *none* arm reports .43 [.32, .57], much weaker avoidance than the single-turn .23 (Am. 32). Four earlier injected trials in the same chat weaken the avoidance. Relevant to the congruence-polarity hypothesis (holiday 33, H39): context injections on other sentences make "injected-ness" less tied to fit with the current state; not a test of it.
- WP-E link (SQ46 prior): as predicted by the contingency-without-transfer evidence (Rominger 2021), feedback trained nothing that transferred.

### Amendment 35a (run 41, ~17:35Z 2026-10-21, PRE-data): frame discrimination for the L20 avoidance (holiday 33, H39)
**Why.** Holiday 33 re-read the avoidance: at L20 α1 the intro report picks the injected sentence at .12 in a 2AFC, so the report reads the injection almost as well as the probe, with the sign flipped. "Stands out" picks it (.70); "injected thought/word" avoid it; the random arm does not avoid; avoidance fades at α8. Hypothesis **(e) congruence polarity** (speculative): a small, integrated injection makes the perturbed sentence both the most salient and the best fit with the model's current state; "injected"/foreign questions ask for the misfit, so they pick the clean sentence. Rival **(g) polarity confusion**: any question about a marked item inverts the salience answer. This runs before Am. 35 (readout geometry) because it decides how that geometry is read (MR34 plan, reordered by H39; Am. 35 follows next run).
**Design.** `inject_loc.py`, untrained Qwen3-4B fp32, N = 2, fixed readout (PAD), 30 new `FRAME35_CONCEPTS` (written pre-data; no overlap with inject_detect.CONCEPTS, BASELINE, LOC/SWEEP/MIRROR concepts, checked by script), concept and matched-norm random arms, probes at layer 30 and final. Settings **L20 α1 (primary: strongest avoidance in Am. 32)**, L20 α4, L20 α8, L12 α1. Frames: intro, neutral, ordinary (MR33 kit), **own** ("Which sentence is closest to what you are thinking about right now?"), **foreign** ("One sentence does not fit with your current train of thought. Which sentence does not fit?"; no word "injected").
**Estimand.** Raw pick-target rate T_q (bias-corrected report picks the injected sentence), concept-bootstrap 95% CIs (`frames35_analyse.py`); no flipping.
**Predictions, written first.** (e): at L20 α1, T_intro < .5, T_own > .5, T_ordinary > .5, T_foreign < .5; the own/ordinary excess fades by α8; random arm shows no polarity split. (g): T_ordinary < .5.
**Slots (primary L20 α1; others secondary).** E_SUPPORTED iff T_intro CI upper < .5 and T_own and T_ordinary CI lower > .5. **E_FALSIFIED** iff T_intro upper < .5 and (T_ordinary upper < .5, or T_own CI includes .5). G_SUPPORTED iff T_intro upper < .5 and T_ordinary upper < .5. AVOIDANCE ABSENT if T_intro upper ≥ .5 (no verdict at that setting). UNDETERMINED if mean log index-token mass < ln .5 in any frame/arm. Secondary: FOREIGN_AVOIDS iff T_foreign upper < .5 (avoidance without the word "injected": against D′ mechanism-naming as the whole story).
**Labels (H35, H39.3).** "Ownership"/"assimilation" wording is not used for a positive result; the label is "congruence polarity". A falsified (e) is reported as such on post 8, once.
**Job** `am35a-frames4b-01`, one submission, `gpu_min_mem_gib` 24, no GPU named.

### Amendment 35 (run 41, ~17:35Z 2026-10-21, PRE-analysis): readout geometry, final layer, offline (MR34)
**Instrument fact (checked before any geometry was computed).** Qwen3-4B ties its embeddings, and the logged final-layer hidden state is post-norm: `logits_intro` of an Am. 31c row is reproduced to 4 digits by h_intro['36'] · W_U[{"1","2"}] (rows fetched by HTTP range request, `fetch_wu.py`, saved as `results/am35/wu_qwen3_4b.json`). So at the final layer the effective readout direction is exactly u = W_U["1"] − W_U["2"], and the bias-corrected report evidence is exactly Δh·u. No backward pass is needed; MR34's per-row gradient is only needed at earlier layers (not done here).
**Data.** Analysis-only, on existing full rows: Am. 31c 2AFC concept rows `fx2_concept_{L20a4,L24a2}.json` (results branch `wp3-results/am31c-loc4b-fix-01`; frames intro, neutral, word; 60 rows). Their report/probe accuracies were already seen; no geometric quantity was computed before this entry. Am. 32 full rows were not kept, and Am. 35a's job collects analyses only, so the five-frame geometry needs a later job (decided after Am. 35a).
**Estimands (`geom35.py`).** Per frame: s·Δh (s = ±1 for target sentence 1/2); report hit = s·Δh·u > 0 (**k3**: agreement with the logged report ≥ .98, else the estimator is wrong and nothing is read); content a = mean s·Δh·d̂ with d̂ the leave-concept-out difference-of-means direction; usage b = mean s·Δh·û; cos(u, m) with m = mean s·Δh; between frames cos(m_q, m_r), a and b differences. Concept bootstrap, 95%.
**Slots.** GAP SHOWN iff |cos(u, m)| CI within ±.20, a CI lower > 0, report not above chance. SIGN FLIP SHOWN iff b_intro and b_neutral CIs exclude 0 with opposite signs and both a CI lower > 0. UPSTREAM CONTENT EFFECT iff a_intro − a_neutral CI excludes 0 and |diff| > .25 × mean a.
**Predictions.** MR34: at L24 α2 (report at chance, probe .93) GAP. Holiday 33 congruence polarity: at L20 α4 the intro and neutral m are similar (cos(m_intro, m_neutral) high) and differ mainly in b (SIGN FLIP), i.e. the question changes how much of the shared content lands on u, not the content. Reconsider MR34's plan if cos(u, m) is large where the report fails.

### Amendment 35 result (run 41, ~17:45Z; offline, Am. 31c 2AFC concept rows, final layer): readout-direction GAP at L24 α2; SIGN FLIP and UPSTREAM CONTENT at L20 α4; the frames' displacements are nearly orthogonal, so the question acts upstream
Files: `results/am35/geom_fx2_concept_{L20a4,L24a2}.json`. **k3 passed:** s·Δh·u reproduces the logged report on 60/60 rows in every frame.

| setting, frame | report | dm-probe | a (content) | b (usage on û) | cos(u, m) | GAP |
|---|---|---|---|---|---|---|
| L24 α2 intro | .53 | .67 | 1.18 [.92, 1.42] | +.02 [−.04, .09] | +.02 [−.03, .06] | **SHOWN** |
| L24 α2 neutral | .42 | .57 | 1.04 | −.03 [−.11, .04] | −.03 [−.08, .03] | **SHOWN** |
| L24 α2 word | .53 | .75 | .75 | +.04 | +.04 [−.03, .11] | **SHOWN** |
| L20 α4 intro | .22 | .72 | 3.93 [2.96, 4.79] | **−2.15 [−2.87, −1.43]** | **−.46 [−.51, −.32]** | not shown |
| L20 α4 neutral | .63 | .70 | 2.87 [2.30, 3.42] | **+.86 [.05, 1.70]** | +.24 [.01, .39] | not shown |
| L20 α4 word | .32 | .75 | 3.42 | −2.00 [−2.92, −1.17] | −.54 [−.62, −.37] | not shown |

- **L24 α2 (primary of Am. 31): READOUT-DIRECTION GAP SHOWN in all three frames.** The target-signed displacement at the answer position is clearly there (a > 0; the difference-of-means probe reads it) and orthogonal to u within ±.11. This is MR34's "present but unused" in its geometric form: the information sits in a direction the readout ignores. It makes M4's access-opening signature (training rotates the effective readout towards w, with Δh fixed) well-defined at this setting.
- **L20 α4: SIGN FLIP SHOWN** (b intro −2.15 vs neutral +.86) and **UPSTREAM CONTENT EFFECT SHOWN** (a intro 3.93 vs neutral 2.87). Under "injected" wordings, nearly half the displacement points at the *other* sentence's token (cos −.46, −.54): the avoidance is a large, readout-aligned component, not a leak.
- **Holiday 33's geometric prediction fails.** It said intro and neutral share their content and differ mainly along u. In fact cos(m_intro, m_neutral) = .16 [−.04, .43] at the final layer (intro|word .65). At layer 30 (descriptive, no u there): .44 at L20 α4 and .77 at L24 α2. The wording routes the injection into different answer-position displacements by layer 30 and diverges further by the last layer. So the frame acts upstream of the readout; the "same content, flipped readout" picture is wrong. Congruence polarity as a *behavioural* account is still open (Am. 35a tests it). Its geometric form is not supported.
- **Caveats.** At the final layer, Δh includes the model's decision itself, so "content" a there is partly the answer. The difference-of-means probe (.57–.75) is weaker than the logistic probe (.93–.97 on these rows): the gap estimates are conservative about how much information is present. Only two settings; Am. 32's full rows were not kept.
- **For M3/M4 (MR34 "reconsider if"):** cos(u, m) is *not* high where the report fails at L24 α2 (≈ 0). The geometric story holds there, and M3 should train at a GAP setting (L24 α2 or a similar one), not at L12 α1 or L20 low α, where the report is sign-flipped and upstream (H39.1). A five-frame geometry at L20 α1 needs full rows: a later job, if Am. 35a makes it worth it.

### Amendment 35b (run 41, ~17:55Z, PRE-data): five-frame readout geometry at the avoidance and gap settings
**Why.** Am. 35 found the gap (L24 α2) and an upstream sign flip (L20 α4) on three frames, and its full rows exist only for Am. 31c. Am. 35a scores behaviour on five frames but keeps no rows. This job regenerates the Am. 35a concept-arm rows at **L20 α1** (strongest avoidance) and **L24 α2** (gap) with final-layer hidden states only, so `geom35.py` can be applied to all five frames offline. Same concepts (FRAME35), same seeds as Am. 35a, so the reports must match Am. 35a's concept-arm T_q exactly at L20 α1 (instrument check; mismatch → nothing is read).
**Analysis (offline, `geom35.py` unchanged, slots as Am. 35).** Added, pre-data: (i) congruence polarity predicts b_own > 0 and b_ordinary > 0 (pointing at the injected sentence) where b_intro < 0 at L20 α1, i.e. the "fit" frames and the "injected" frames place the displacement on opposite sides of u; polarity confusion predicts b_ordinary < 0. (ii) By Am. 35's result, the frames' displacements are expected to be nearly orthogonal (cos(m_q, m_r) < .5); if instead own/ordinary are anti-aligned with intro/foreign (cos < −.3), the frames share one "fit" axis read with opposite signs: the geometric form of congruence polarity, re-tested.
**Job** `am35b-geomframes4b-01` (submission 2 of 3 this run), concept arm only, `--probe-layers -1`, `gpu_min_mem_gib` 24.

### Amendment 35a result (collected 18:01Z 2026-10-21, run 41; job `am35a-frames4b-01`, spot eu-west-3b g6.xlarge $0.3561/h, 14.4 min, ≈ $0.09): congruence polarity FALSIFIED at the primary setting; avoidance replicates and does not need the word "injected"; it reverses at high dose
Files: `results/am35a/analysis_fr_{L20a1,L20a4,L20a8,L12a1}.json`. Coherence passed everywhere. Raw pick-target rates T_q (concept arm; random arm in brackets), 60 rows:

| setting | intro | neutral | ordinary | own | foreign | slot |
|---|---|---|---|---|---|---|
| **L20 α1 (primary)** | **.13** [.03, .25] (.62) | .62 (.60) | .45 [.28, .62] (.53) | .57 [.43, .70] (.62) | **.07** [.00, .17] (.45) | **E_FALSIFIED**; FOREIGN_AVOIDS |
| L20 α4 | .22 [.10, .37] (.68) | .67 (.62) | .38 [.22, .53] (.42) | .42 [.30, .55] (.57) | .28 [.17, .42] (.62) | E_FALSIFIED; FOREIGN_AVOIDS |
| L20 α8 | .52 (.80) | .63 (.57) | .27 [.15, .40] (.33) | .42 (.43) | **.82** [.70, .92] (.78) | AVOIDANCE ABSENT; FOREIGN_PICKS |
| L12 α1 | .47 (.52) | .38 (.52) | **.73** [.62, .85] (.50) | **.75** [.63, .85] (.53) | .53 (.53) | AVOIDANCE ABSENT |

Probe (concept arm, final layer) .82–.93 at L20, .63–.83 at L12 α1.
- **(e) congruence polarity: FALSIFIED as registered** at L20 α1. The own frame does not pick the injected sentence (CI includes .5), and neither does the ordinary frame. Where the avoidance is strongest, the model does not positively treat the injected sentence as the one that fits. **(g) polarity confusion: not supported** (ordinary is not below .5 at L20 α1).
- **The avoidance is a "misfit" effect, not a word effect.** "Does not fit your current train of thought" avoids even more than "injected thought" (.07 vs .13; the random arm .45). That goes against D′ (mechanism-naming) as the whole story: the word "injected" isn't needed.
- **Dose reversal.** At α8 the same misfit question picks the injected sentence (.82), and "ordinary" picks the clean one (T .27). A large perturbation is a misfit; a small one is avoided by misfit questions. The random arm at α8 also picks it (.78 foreign, .80 intro): at high dose the effect is content-free, as in Am. 32.
- **The positive half shows up elsewhere.** At L12 α1 (no avoidance), "ordinary" and "own" both pick the injected sentence (.73, .75; random .50, .53). So the two halves of the holiday-33 picture occur at different settings and never together. A single congruence variable read with two signs does not fit the data. Descriptively: misfit questions avoid small mid-layer (L20) perturbations; fit questions select small early-layer (L12) perturbations.
- **H39 consequences stand** (below-chance = sign-flipped access; M3 at a GAP setting). Mentalistic labels stay off. Am. 35b (five-frame geometry at L20 α1 and L24 α2) will show whether the misfit/fit frames load on u with opposite signs.

### Amendment 35b result (collected 18:11Z 2026-10-21, run 41; job `am35b-geomframes4b-01`, spot eu-west-3b g6.xlarge $0.3561/h, 5.1 min, ≈ $0.03): GAP replicates on new concepts in all five frames at L24 α2; at L20 α1 the two misfit frames share one readout-anti-aligned displacement, and the fit frames load on nothing
Instrument check passed: the regenerated L20 α1 reports equal Am. 35a's concept T_q exactly in all five frames. k3 = 1.0 everywhere. Files: `results/am35/geom_g5_{L20a1,L24a2}.json`; rows stay on the results branch.
- **L24 α2: GAP SHOWN in all five frames** (cos(u, m) within [−.09, .18]; content a .85–1.24, CI lower > 0; reports .42–.55). This replicates Am. 35 on 30 new concepts and two new wordings. The gap does not depend on the question.
- **L20 α1:** intro b −.95 [−1.27, −.62], cos(u, m) −.56; foreign b −1.43, cos(u, m) −.66; and **cos(m_intro, m_foreign) = .74 [.60, .76]**. The two misfit wordings produce nearly the same displacement, strongly anti-aligned with the readout. Own (b +.09 [−.05, .23]) and ordinary (−.07 [−.29, .17]) put nothing on u. Their displacements are not anti-aligned with the misfit frames' (cos −.00 to .24). **The registered geometric re-test of congruence polarity (fit frames anti-aligned with misfit frames, cos < −.3) is NOT SUPPORTED.** SIGN FLIP not shown at α1 (neutral b CI lower −.01). UPSTREAM CONTENT shown (a intro 1.50 vs neutral .85).
- **Reading (argued, not established).** There is one "misfit" displacement, shared by the two misfit wordings. A small L20 injection pushes it towards the *clean* sentence. The fit wordings don't use it at all, so it is not a polarity of a shared fit axis. The avoidance is something specific to how misfit questions are answered, and the content-free reversal at α8 (Am. 35a) suggests that misfit questions read a contrast between sentences that a small concept injection moves the wrong way.

### Amendment 36 (run 42, ~18:30Z 2026-10-21, PRE-data): access audit of IFT-style training at the GAP setting (MR32 M3+M4 in one job; MR34 signatures; H34, H36, H39)
**Question (target, D1).** Untrained Qwen3-4B carries the injected sentence's identity at the answer position (probe .93–.97) in a direction orthogonal to the report's readout (Am. 35/35b GAP at L24 α2, cos(u, m) ≈ 0 in five frames). Does IFT-style training (Hahami, Sinha & Jain 2026) make the report use information *already present* (access opening), move it (re-routing), or install a new readable component (new detector)? This is the premise the "trained introspection = self-access" reading turns on.
**Setting.** L24 α2, 2AFC (n = 2), chosen by Am. 35/35b (GAP, report at chance in every frame, no sign flip), per H39.1 not L20 low α or L12 α1. bfloat16 for every arm in this job (training needs the memory; U, T and S share dtype and instance, so comparisons are within-job; the untrained reports are not compared to earlier float32 numbers except descriptively).
**Arms (one job, same instance).**
- **T**: `inject_loc_train.py --settings 24:2 --n 2 --concepts train36 --r 4 --steps 400 --lr 1e-4 --random-frac 0.5 --frame intro --seed 36`. LoRA r 4 on q/k/v/o (attention only, ~12 MB fp32, under the 20 MB rule). Train concepts TRAIN36_CONCEPTS (60, written in this run before any data, disjoint from every concept list in `notes/wp3/`); sentences BANK[:10]; half the training injections are matched-norm random directions (as in the run-38 draft and Rivera 2025's noise arm).
- **S** (negative control): identical, plus `--shuffle` (the trained label is a seeded coin flip independent of the injected sentence).
- **U**: no adapter.
**Evaluation.** `inject_loc.py --settings 24:2 --n 2 --concepts frames35 --bank test --frames intro/neutral/ordinary --probe-layers 30:-1` for U, T (`--adapter`) and S, concept arm; and the random arm (`--random`) for U and T (descriptive). Held out from training: concepts (FRAME35, 30) and sentences (BANK[10:]). Rows are paired across arms (same seed 37, same concepts and sentence pairs).
**Estimands and slots** (`audit36.py`, written before data; paired concept bootstrap, 2000): see its docstring. In short: **K3** geometry reproduces each arm's report (≥ .98); **GATE** R_T − R_U (intro) ≥ .15 with CI lower > 0; **CONTROL** R_S − R_U CI upper < .15 (else MISLABELLED, k3: nothing below is read); stories at the mid layer 30 vs final (u is fixed because the LoRA leaves W_U and the final norm alone): **OPENING** (mid displacement unchanged: cos(m_U, m_T) CI lower ≥ .80 and norm ratio CI in [.75, 1.33]; final cos(u, m) rises), **REROUTE** (mid displacement moved, cos CI upper < .80, but the U-fitted probe still reads it on T within .05), **DETECTOR** (refit probe on T beats U, CI lower > 0, and the U-probe transfer drops by ≥ .10). Exactly one SHOWN → that story; none or several → NOT SEPARATED. **FRAME** spread (H36): T's gain in neutral and ordinary; FRAME-LOCAL iff both CIs include 0.
**Rival ledger (H33).** (A) access opening → OPENING; forbids mid cos < .8. (Re-routing) → REROUTE; forbids mid cos ≥ .8. (New detector / F, training asymmetry: the trained report is a new supervised reader) → DETECTOR; forbids no gain in the refit mid probe *and* intact transfer together. (C) probe over-reach ("something was added") → T's random-arm report rises as much as the concept arm's: descriptive only (training includes random injections, so this cell cannot separate C). (D′ frame-local) → FRAME-LOCAL. Central-question cap: whichever story wins, this is about a trained report channel's route, not about consciousness; "self-access" is not a label any slot can earn (H39.3).
**Predictions (written now).** My prior: GATE PASS (~60%); conditional on PASS, OPENING 35%, REROUTE 30%, DETECTOR 20%, NOT SEPARATED 15%; FRAME-LOCAL ~50%.
**Kill rules.** k2: GATE FAIL → one lr change (3e-4) next run, then scoped no-hit. k3: CONTROL MISLABELLED or K3 FAIL → fix before reading. Adapters T and S are copied to `notes/wp3/adapters/am36_T`, `am36_S` (housekeeping rule; ~12 MB each).
**Job** `am36-audit4b-01`, `dtype` bfloat16 (L4/A10G/L40S all have bf16; no `gpus` named), `gpu_min_mem_gib` 24.

### Amendment 36 result, part 1 (collected 18:55Z 2026-10-21, run 42; job `am36-audit4b-01`, spot eu-west-3b g6.xlarge L4 $0.3561/h, 8.3 billed min, ≈ $0.05): GATE PASS, CONTROL OK, **K3 FAIL (bf16 report quantization), so the stories are NOT READ**
Files: `results/am36/audit36_L24a2.json`, `results/am36/stage1_{T,S}.json`; adapters `adapters/am36_{T,S}/` (11.8 MB each); full rows on the results branch.
- Training: T's loss fell to .02 (train hit 1.00 by step 375); S's stayed at ≈ .68 (chance), as a shuffled label should.
- Held-out concepts and sentences, intro frame: **R_U .48 [.38, .58], R_T 1.00 [1.00, 1.00], R_S .50 [.43, .57]**: GATE PASS, CONTROL OK. T also reaches 1.00 in the untrained neutral (salience) frame, and in the demand-reversed "ordinary" frame T picks the *injected* sentence every time (scored R = 0; U .48, S .50). Descriptively the trained pointer ignores the question (H36 frame spread: the gain transfers to a congruent frame and *inverts* the correct answer in the reversed one).
- **K3 FAIL** (U .80–.82, S .65–.68, T 1.00). Diagnosis (post hoc, checked on the rows): in bfloat16 the logged logits are quantised (corrected differences come in steps of .125–.25, and in S the median corrected difference is exactly 0, a tie); every K3 disagreement sits at |corrected difference| ≤ .25, one quantisation step, while the geometric quantity (from hidden states stored in float32 to 4 significant digits) varies smoothly; on T, where margins are large, K3 = 1.00 and raw logits = h·u to r = 1.000. The fault is the report's precision near zero margin, not the estimator. Ties are also scored as "sentence 2", which biases the at-chance arms' R by a few points (not the gate).
- As registered (k3), the OPENING / REROUTE / DETECTOR slots are not read on these rows. Descriptive only, not slots: T's mid-layer (30) displacement differs from U's (cos .27 intro), the U-fitted mid probe still reads T (.97), and cos(u, m) at the final layer rises from .03 to .64.

### Amendment 36a (run 42, ~19:00Z, deviation, PRE-data for the slots): float32 re-evaluation of the same adapters
**Why.** Am. 36's K3 failed from bfloat16 logit quantisation at near-zero margins (above). The trained adapters are saved; the evaluation is re-run in float32 (the dtype in which Am. 35's exact-readout check was established), with the same commands, concepts, sentences and seeds. No retraining (reuse rule). The training dtype (bf16) is a property of the adapters, stated in the paper.
**Analysis.** `audit36.py` unchanged, all slots as registered in Am. 36, on the float32 rows. If K3 still fails in float32, the estimator is wrong and nothing is read. The bf16 part-1 numbers are not pooled with these.
**Job** `am36a-fp32eval4b-01` (submission 2 of 3 this run), `dtype` float32, `adapters` ["am36_T", "am36_S"], `gpu_min_mem_gib` 24.

### Amendment 36a result (collected 19:26Z 2026-10-21, run 42; job `am36a-fp32eval4b-01`, spot eu-west-3b g6.xlarge L4 $0.3561/h, 7.6 billed min, ≈ $0.05): K3 PASS, GATE PASS, **CONTROL MISLABELLED by its registered CI rule (imprecision, not a gain), so the stories are NOT READ**
File: `results/am36/audit36f_L24a2.json`; rows on the results branch.
- **K3 PASS** in float32 (U and T 1.00 in every frame, S .98–1.00). This confirms that the part-1 failure was bf16 quantisation.
- **GATE PASS:** R_U .45 [.33, .55], R_T 1.00; dR_T +.55 [.45, .67].
- **CONTROL MISLABELLED as registered:** dR_S (intro) = +.03 [−.12, .18]. Its CI upper is ≥ .15 because 30 concepts can't bound a null that tightly; the point estimate is ≈ 0, and S's training loss stayed at chance. My rule was meant to catch a control that *gains*, and this one fired on a *noisy null*: a design error in the threshold, written pre-data, so it stands. Next run must decide, with reasons and before reading anything, whether a larger S-vs-U comparison is needed. That would be the one cell MR35 allows (a fix forced by a gate or control failure).
- **Descriptive only (not slots, since CONTROL failed):**
  - **Frames:** T is 1.00 in intro and neutral, and **0.00 in the reversed "ordinary" frame** (dR −.43 [−.53, −.33]). It picks the injected sentence whatever is asked.
  - **Random arm:** T picks the perturbed sentence in 100% of random-direction rows in all three frames (U .60–.75).
  - **Layer 30:** cos(m_U, m_T) .29 [.21, .44]; displacement norm ×2.5 [1.9, 3.1]; the U-fitted probe still reads T at .97; the refit probe gains +.10 [.03, .18]. At the final layer cos(u, m) goes from .02 to .64.
  - **Story criteria:** only REROUTE's criteria are met. OPENING fails on the mid cos; DETECTOR fails on transfer.
- **Reading (argued, not established):** IFT-style training at 4B does not open access to the content the untrained model carries. It amplifies and moves the perturbation signal upstream and wires it to the answer slot as a content-free, question-insensitive "perturbed one" pointer. The question-insensitivity is shown by the raw rates, whatever the control's verdict: T's ordinary-frame 0/60 and its random-arm 100%.

### Amendment 36b (run 43, ~21:30Z 2026-10-21, PRE-data): larger S-vs-U control on 90 new concepts, same adapters, float32 (the one cell MR35 allows: a fix forced by Am. 36a's CONTROL failure)
**Decision and reasons (written before any Am. 36b data).** Am. 36a's CONTROL fired on a noisy null (dR_S +.03 [−.12, .18]): with 30 concepts × 2 rows the paired-difference CI is about ±.15, so the registered threshold (CI upper < .15) could not pass even for a true zero. The paper's centre (T's question-insensitive pointer: ordinary-frame 0/60, random arm 100%) does not depend on CONTROL. The registered STORY slots (OPENING / REROUTE / DETECTOR) do, and REROUTE is what the paper would call the mechanism. Without a readable control I would be writing REROUTE as a description. So: run it once, larger, and change nothing else. Not doing it would leave the mechanism claim unregistered; doing it twice is not allowed (D2 spirit, MR35).
**What changes.** Only the concept set and its size: AUDIT36B_CONCEPTS, 90 concepts written this run before any data. They were drawn from a candidate list and filtered automatically to be disjoint from every quoted word in `notes/wp3/*.py` (so they're disjoint from training, LOC, SWEEP, FRAME35 and BASELINE); "date" and "crane" were dropped as ambiguous, and "toaster" and "skillet" added. `--k 90`, so 180 rows per arm. With 90 concepts the expected paired-difference CI half-width is ≈ .09, so a true null can pass the registered rule.
**What stays.** Adapters `am36_T`, `am36_S` as stored (reuse rule, no retraining). L24 α2, n 2, `--bank test`, frames intro/neutral/ordinary, probe layers 30 and final, seed 37, float32 (H40). `audit36.py` is unchanged, and every slot and threshold is as registered in Am. 36: K3 ≥ .98; GATE dR_T ≥ .15 with CI lower > 0; CONTROL dR_S CI upper < .15; then OPENING / REROUTE / DETECTOR; FRAME. Random arms are not re-run (descriptive in Am. 36a; Q for T was already .00).
**Predictions (written now).** K3 PASS ~90%. GATE PASS ~95%. CONTROL OK ~65% (S's training loss stayed at chance; Am. 36a's point estimate was +.03). Conditional on CONTROL OK: REROUTE ~60%, NOT SEPARATED ~25%, OPENING ~5%, DETECTOR ~10%. T's ordinary-frame pick of the injected sentence ≥ .95 ~90%. H41: Q reported for each arm with `q_axis.py` (descriptive).
**Kill rules.** If CONTROL fails again, the STORY slots are permanently unread for this adapter pair. The paper then states the mechanism as descriptive only, and no third control is run. If K3 fails in float32, the estimator is wrong (MR35's reconsider clause), and nothing is written until it is fixed.
**Analysis commands.** `python3 audit36.py results/am35/wu_qwen3_4b.json a36b_U.json a36b_T.json a36b_S.json results/am36/audit36b_L24a2.json`. Then `q_axis.py`, adapted to the a36b filenames (descriptive).
**Job** `am36b-control4b-01`: `dtype` float32, `adapters` ["am36_T", "am36_S"], `gpu_min_mem_gib` 24, no `gpus` named.

### Amendment 36b result (collected 21:56Z 2026-10-21, run 43; job `am36b-control4b-01`, spot eu-west-3b g6.xlarge L4 $0.3564/h, 9.8 billed min, ≈ $0.06): K3 PASS, GATE PASS, **CONTROL OK**, **STORY NOT SEPARATED** (no single story's criteria met), FRAME SPREAD; T's pointer replicates on 90 new concepts
Files: `results/am36/audit36b_L24a2.json`, `results/am36/q_axis36b_L24a2.json`, status and provenance in `results/am36b-control4b-01/`; rows on the results branch.
- **K3 PASS** (U and T 1.00, S .99–1.00). **GATE PASS:** R_U .64 [.58, .71], R_T 1.00, dR_T +.36 [.29, .42]. **CONTROL OK:** dR_S −.17 [−.27, −.06]. The shuffled adapter does not gain, it *loses*, so the registered rule (CI upper < .15) passes.
- **Stories, as registered:** none is SHOWN, so NOT SEPARATED.
  - OPENING fails: mid cos(m_U, m_T) .28 [.24, .34] is far below .80.
  - REROUTE fails on its transfer clause: the U-fitted mid probe reads T at .83 [.77, .89] vs 1.00 on U, a drop of .17, beyond the .05 allowed.
  - DETECTOR fails on its refit clause: the refit probe gains only +.01 [.00, .02], with U at the .99 ceiling.
  - The mechanism sits between REROUTE and DETECTOR: the displacement is moved (cos .28) and amplified (×3.0), and the old reader partly transfers. As registered: **NOT SEPARATED**. Am. 36a's descriptive "only REROUTE fits" does **not** replicate as a slot. The paper will report "moved and amplified, not opened", not "re-routed".
- **Pointer (raw rates, the paper's centre):** T is 1.00 in intro and neutral and **.00 in the reversed "ordinary" frame** (it picks the injected sentence in 180/180 rows). Q(T) = .00. U: Q +.02 [−.07, .11]. S: Q −.19 [−.28, −.09]. Final cos(u, m): T .61–.72 vs U .09–.11.
- **New facts to state:**
  - (i) On these 90 concepts the untrained report is *above* chance in the intro frame (.64), unlike Am. 36a's 30 (.45). The gap at L24 α2 is concept-set dependent in size.
  - (ii) Shuffled-label training *lowers* accuracy below U (−.17) while leaving the mid displacement near U's (cos .76). The control is not inert: it trains a label-independent bias, consistent with S's flip toward "sentence 2" (S's ordinary .33, neutral .42). The paper must say that S is a "no-information" control, not a "no-change" control.
- Predictions vs outcome: K3, GATE and CONTROL OK as predicted (CONTROL OK ~65%). STORY: I gave REROUTE ~60%, NOT SEPARATED ~25%; the outcome was NOT SEPARATED. T's ordinary-frame pick of the injected sentence ≥ .95: yes (1.00).
- Kill rules: CONTROL passed, so the stories were read once, as registered. No further control cells on this adapter pair.

### Amendment 37 (run 44, ~00:30Z 2026-10-22, PRE-data): the untrained comparator's question axis Q and content axis C at L20 α1, and the trained pointer at a second site (MR36; H41, H42)
**Question (target, D1).** The pointer paper's comparator claim is that accuracy and answering the question come apart in both directions: T is accurate with Q = .00 (Am. 36a/b), while the untrained report at L20 α1 is inaccurate but question- and content-sensitive. The second half rests on Am. 32/35a numbers that were never scored as Q or C with CIs. **What is already known (stated so the predictions are honest):** Am. 35a (FRAME35, 30 concepts) gives untrained P_inj intro .13, ordinary .45 [.28, .62], random arm intro .62 / ordinary .53. So Q_U ≈ −.32 and C_U(intro) ≈ −.49 descriptively, never registered. This amendment is a registered replication on 60 new concepts, scored with H41's Q.
**Design.** `inject_loc.py`, Qwen3-4B, float32 (H40), N = 2, L20 α1, `--bank test`, frames intro/neutral/ordinary, probe layers 30 and final. 60 new concepts `AM37_CONCEPTS` (written this run before any data, filtered by script to be disjoint from every quoted word in `notes/wp3/*.py`; "turkey" dropped as ambiguous). Arms: U concept, U random, T (`adapters/am36_T`, reuse rule, no training) concept, T random. Random directions are seeded by concept name, so concept and random rows are paired.
**Estimands** (`q_axis37.py`, written and smoke-tested on 0.6B before any data; concept bootstrap 2000, seed 37): P_inj per arm × frame; Q = P_inj(intro) − P_inj(ordinary); C_q = P_inj(q | concept) − P_inj(q | random).
**Slots.** For an estimate with CI: **SHOWN** iff the CI excludes 0 and |estimate| ≥ .25; **EXCLUDED** iff the CI lies inside [−.10, .10]; otherwise NOT SHOWN AS OF 2026-10-22, channel checked: yes. UNDETERMINED if mean log index-token mass < ln .5 in any arm/frame.
- **Primary: Q_U** (untrained, concept arm). The paper's comparator claim "the untrained report reads the question" needs SHOWN. Its sign is reported as is; a negative Q is question-sensitivity with the wrong sign (H39), not a failure.
- **Secondary: C_U(intro).** SHOWN supports "content-sensitive"; EXCLUDED means the avoidance is content-free at this setting, and the paper drops "content-sensitive".
- **Replication check:** avoidance replicates iff U concept P_inj(intro) CI upper < .50.
- **Descriptive (T at a setting it was not trained on):** the pointer generalises to L20 α1 iff T concept P_inj(ordinary) ≥ .95; Q_T and C_T reported. No slot; T was trained at L24 α2.
**Rival ledger (H33).** (R1) The untrained cell reads the question (wording-dependent answer): forbids Q_U EXCLUDED. (R2) The avoidance is a content-driven push away from the injected sentence regardless of the question (pointer-like with negative sign): predicts Q_U ≈ 0 with both frames below .5; forbids Q_U SHOWN. (R3) Generic perturbation effect: forbids C_U SHOWN. Cheapest mechanism that passes Q_U SHOWN (H42): a lexical switch on the word "ordinary" over a fixed push; the next contrast it fails is a content-conditional choice (K), not run here.
**Predictions (written now, conditioned on Am. 35a).** Q_U SHOWN (negative) ~60%, NOT SHOWN ~35%, EXCLUDED ~5%. C_U(intro) SHOWN ~75%. Avoidance replicates ~85%. T pointer at L20 α1 (ordinary ≥ .95) ~45% (the injection is weaker and at another layer).
**Kill rules.** No further cell on this question in this paper whatever the outcome. If UNDETERMINED, the comparator is reported from Am. 35a descriptively and labelled unregistered.
**Analysis.** `python3 q_axis37.py <dir with a37_{U,T}_{concept,random}.json> results/am37/q_axis37_L20a1.json`.
**Job** `am37-qaxis4b-01`: `dtype` float32, `adapters` ["am36_T"], `gpu_min_mem_gib` 24, no `gpus` named.

### Amendment 37 result (collected 00:53Z 2026-10-22, run 44; job `am37-qaxis4b-01`, spot eu-west-3b g6.xlarge L4 $0.3564/h, 9.4 billed min, ≈ $0.06): **Q_U SHOWN (negative), C_U(intro) SHOWN, avoidance replicates; T stays question-insensitive at L20 α1 (Q_T EXCLUDED) but is not a pure pointer there (.93 < .95) and is content-sensitive**
File: `results/am37/q_axis37_L20a1.json`; status and provenance in `results/am37-qaxis4b-01/`; rows (4 × ~29 MB) on `wp3-results/am37-qaxis4b-01`. Coherence (index-token mass) passes in every arm and frame. 60 concepts, 120 rows per arm.
- **Primary, Q_U = −.34 [−.49, −.18]: SHOWN.** Untrained P_inj: intro .20 [.12, .30], neutral .48, ordinary .54 [.44, .65]. The untrained report's answer depends on the wording, with the wrong sign (H39). Rival R2 (a push away from the injected sentence regardless of the question) is forbidden by this outcome; R1 stands.
- **C_U(intro) = −.30 [−.45, −.14]: SHOWN.** The random arm doesn't avoid (intro .50; random Q −.03 [−.20, .14]). Rival R3 (generic perturbation effect) is forbidden. C_U is ≈ 0 in ordinary (+.01) and neutral (−.12 [−.26, .03]).
- **Avoidance replicates** (intro CI upper .30 < .50).
- **T at L20 α1 (descriptive; trained at L24 α2):** P_inj .91 intro, .91 neutral, **.93 [.88, .98] ordinary** → by the registered rule (≥ .95) not a pure pointer here, but **Q_T = −.03 [−.08, .03], EXCLUDED**: T's answer does not depend on the question at this setting either. Unlike at L24 α2 (random arm 100%), T is content-sensitive here: random arm .61–.62, C_T +.28 to +.32 (CIs exclude 0). Reading (argued, not tested): the trained pick follows how strongly the perturbation registers, and concept vectors may register more than equal-norm random ones at this low dose; either way it ignores the question.
- **Q separates the trained from the untrained report at this setting:** Q_T − Q_U = +.32 [+.16, +.47]. At L24 α2 it could not (U at chance). This is the contrast the paper needed: the untrained report is question-sensitive with the wrong sign; the trained report is question-insensitive and accurate in-frame.
- H42 (cheapest mechanism that passes Q_U SHOWN): a lexical switch on wording over a fixed push. Not tested here (K not run). The paper says so.
- Predictions vs outcome: Q_U SHOWN ~60% (yes, negative); C_U SHOWN ~75% (yes); avoidance ~85% (yes); T pointer ≥ .95 ~45% (no, .93; but Q_T EXCLUDED).
- Kill rule: no further cell on this question for this paper.

### Amendment 38 (run 45, ~03:45Z 2026-10-22, PRE-data): mixed-polarity training, the M1 question gate (MR37 M1; H41, H42, H43)
**Question (target, D1).** T (Am. 36, intro-only IFT-style LoRA) answers "which is ordinary?" by still picking the injected sentence: Q_T = .00 at L24 α2 (Am. 36a/b) and EXCLUDED at L20 α1 (Am. 37). Is that a property of IFT-style training at 4B, or of single-polarity training (stimulus control, holiday 35)? If training on both wordings makes the trained report depend on the question, the pointer paper's diagnosis ("accuracy in the training frame is not a report") gets a working fix. That also earns the right to run K (MR37 M2). **Cap, written now (holiday 36):** a GATE PASS is "controlled detection" (the report applies the question's polarity to a detected perturbation). It is not reading content and not self-access. The next contrast it can fail is K (content-conditional choice), which is not run here.
**Design.** One job, Qwen3-4B. Training is in bfloat16 as T was (`--bf16` casts the float32-loaded base), and evaluation is in float32 (H40), in the same job.
- **M**: `inject_loc_train.py --settings 24:2 --n 2 --concepts train36 --r 4 --steps 400 --lr 1e-4 --random-frac 0.5 --frame mixed --seed 38 --bf16`. These are the same concepts, sentences (BANK[:10]), rank, steps, lr and random fraction as T. Each step draws intro or ordinary (p .5), and under ordinary the label is the non-injected sentence.
- **SM** (negative control): identical, plus `--shuffle` (coin-flip labels).
- **Eval** (`inject_loc.py --settings 24:2 --n 2 --concepts audit36b --k 90 --bank test --frames intro/neutral/ordinary/plain --probe-layers -1`), concept arm for U (no adapter), T (`adapters/am36_T`, reuse rule), M and SM, plus the random arm for M (descriptive). Held out from training: concepts (AUDIT36B, 90, as Am. 36b for comparability) and sentences (BANK[10:]). New frame **plain** (written pre-data, untrained by M): "Nothing unusual happened to one of these sentences. Which sentence is the unaltered one?". It reverses polarity without the word "ordinary".
**Estimands and slots** (`m1_gate38.py`, written and smoke-tested on 0.6B before any data; concept bootstrap 2000, seed 38): Q = P_inj(intro) − P_inj(ordinary); Q′ = P_inj(intro) − P_inj(plain); Acc_intro = P_inj(intro), Acc_ord = 1 − P_inj(ordinary).
- **CONTROL** OK iff Q_SM CI upper < .25. Otherwise MISLABELLED, and the gate is not read.
- **GATE (primary, M1)** PASS iff Q_M ≥ .50 with CI lower > .25, Acc_intro_M ≥ .80 and Acc_ord_M ≥ .80.
- **Q′_M (secondary)**: SHOWN iff the CI excludes 0 and |Q′| ≥ .25; EXCLUDED iff the CI lies inside [−.10, .10]; else NOT SHOWN AS OF 2026-10-22, channel checked: yes.
- UNDETERMINED (nothing read) if mean log index-token mass < ln .5 in any arm or frame.
- Descriptive: Q_M − Q_T (paired CI); M's random arm Q; neutral-frame P_inj for every arm; U and T reproduce Am. 36b (T ordinary ≈ 1.00).
**Rival ledger (H33).**
- (R1) Polarity rule over detection: predicts GATE PASS and Q′_M SHOWN. Forbids Q′_M EXCLUDED.
- (R2) Lexical switch on the word "ordinary" over a pointer (the H42 cheapest mechanism): predicts GATE PASS with Q′_M ≈ 0 (plain answered like intro). Forbids Q′_M SHOWN.
- (R3) Single-format stimulus control survives mixing, or the mixed labels cancel: GATE FAIL.
- (R4) Format-only training effect: forbids CONTROL OK only if SM shows Q.
- Cheapest mechanism that passes R1's prediction: a detector plus a general negation cue. The next contrast it fails is K with the H43 prime-only arm.
**Predictions (written now).** CONTROL OK ~90%. GATE PASS ~65%. Conditional on PASS: Q′_M SHOWN ~45%, NOT SHOWN ~35%, EXCLUDED ~20%. Q_M − Q_T > .50 ~65%.
**Kill rules (MR37 stall rule).** GATE FAIL → one lr change (3e-4) in a new amendment. A second FAIL → register "single-format training doesn't produce instructional control at 4B", write an addendum to the pointer paper by PR, and stop 4B training work. GATE PASS → design K (H43: per-category salience matched in U's probe terms, a prime-only arm, a swapped-vector K′ arm) before any further training. Adapters M and SM are copied to `notes/wp3/adapters/am38_{M,SM}/` (~12 MB each).
**Job** `am38-mixed4b-01`: `dtype` float32 (training casts to bf16), `adapters` ["am36_T"], `gpu_min_mem_gib` 24, no `gpus` named.

### Amendment 38 result (collected 04:00Z 2026-10-22, run 45; job `am38-mixed4b-01`, spot eu-west-3b g6.xlarge L4 $0.3564/h, 20.9 billed min, ≈ $0.12): CONTROL OK, **GATE FAIL**: M never learned (rival R3)
File: `results/am38/m1_gate38_L24a2.json`; training logs in `results/am38/stage1_{M,SM}.json`; status and provenance in `results/am38-mixed4b-01/`; rows on `wp3-results/am38-mixed4b-01`. Adapters copied to `adapters/am38_{M,SM}/` (12 MB each). Coherence passes in every arm and frame. 90 concepts, 180 rows per arm.
- **CONTROL OK**: Q_SM −.12 [−.19, −.04] (CI upper < .25).
- **GATE FAIL**: Q_M +.02 [−.06, .09]; Acc_intro_M .48, Acc_ord_M .54. M is at chance in every frame (intro .48, neutral .46, ordinary .46, plain .50). Q′_M −.02 [−.11, .07]: NOT SHOWN AS OF 2026-10-22, channel checked: yes (it falls just outside EXCLUDED's ±.10).
- **Diagnosis (R6): M never learned.** Training loss sat at ln 2 for all 400 steps: mean .73 at 50–100, .70 at 100–150, .69 at 350–400. The training hit rate stayed at .52 in both wordings. T, with the same settings, broke from chance between steps 100 and 200 (loss .66 → .17). The reading, argued from the logs and not tested: under balanced mixed labels the perturbation-pointer direction gets zero net gradient, so the cheap route T found is unavailable. The conditional (pointer × wording) has to be found from chance, and at lr 1e-4 it isn't found in 400 steps. M's random arm shows a small Q, +.13 [.04, .22], below the SHOWN threshold. Descriptive only.
- T reproduces Am. 36b (1.00 in every frame, including the new `plain` frame). U at L24 α2: intro .64, ordinary .63 (Q_U +.02, as in Am. 36b), plain .76 (Q′_U −.11 [−.20, −.02]); descriptive.
- Predictions vs outcome: CONTROL OK ~90% (yes); GATE PASS ~65% (no).
- **Kill rule applied:** this is the first GATE FAIL, so one lr change (3e-4) follows in Am. 38b. A second FAIL triggers the MR37 stall rule (register the no-hit, write an addendum to the pointer paper, stop 4B training work).

### Amendment 38b (run 45, ~04:10Z 2026-10-22, PRE-data): the M1 gate at lr 3e-4 (Am. 38's registered kill rule k2)
**What changes.** Only `--lr 3e-4` for M and SM (job `am38b-mixed4b-lr3-01`). Everything else is identical to Am. 38: the seed, concepts, steps, frames, held-out set, estimands, slot rules, `m1_gate38.py` and the cap. U and T are re-evaluated in the same job so that every row stays paired and within one job. M's and SM's adapters go to `adapters/am38b_{M,SM}/`.
**Why this and not more steps.** Am. 38 registered the lr change as the single permitted fix (D2: no third variant). More steps would be a different fix, so it isn't added now.
**Predictions (written now, after seeing Am. 38's flat loss).** GATE PASS ~30%. M learns at all (final-100 training hit ≥ .80) ~45%. CONTROL OK ~85%.
**Kill rule.** GATE FAIL → MR37 stall rule: register "single-format IFT-style training at 4B (r 4, 400 steps, lr ≤ 3e-4) does not produce instructional control", write an addendum to the pointer paper by PR, and stop 4B training work. A training failure (M final-100 hit < .60) is reported as "not learned", not as "learned a pointer".

### Amendment 38b result (collected 06:24Z 2026-10-22, run 46; job `am38b-mixed4b-lr3-01`, spot eu-west-3b g6.xlarge L4 $0.3564/h, 21.2 billed min, ≈ $0.13): CONTROL OK, **GATE FAIL**, M not learned → **MR37 stall rule fires**
File: `results/am38b/m1_gate38_L24a2.json`; training logs `results/am38b/stage1_{M,SM}.json`; status/provenance `results/am38b-mixed4b-lr3-01/`; rows on `wp3-results/am38b-mixed4b-lr3-01`; adapters `adapters/am38b_{M,SM}/` (12 MB each). Coherence passes in every arm and frame. 90 concepts, 180 rows per arm. (Scored with the registered bias-corrected scorer; see Reanalysis R45 below for the raw-answer scores.)
- **CONTROL OK**: Q_SM +.06 [−.01, .13].
- **GATE FAIL**: Q_M +.06 [−.04, .16]; Acc_intro_M .55, Acc_ord_M .51. Q′_M +.11 [.03, .19]: NOT SHOWN AS OF 2026-10-22, channel checked: yes.
- **Not learned** (registered rule: M final-100 training hit < .60): M final-100 hit .49 (intro .52, ordinary .45), loss .696 ≈ ln 2. The higher lr only moved the first 50 steps (loss 1.95, overshoot) and then sat at chance. SM likewise (.51).
- T reproduces 1.00 in every frame; U reproduces Am. 38 (intro .64, ordinary .63, plain .76).
- Predictions vs outcome: GATE PASS ~30% (no); M learns ~45% (no); CONTROL OK ~85% (yes).
- **Stall rule (registered):** "single-format IFT-style training at 4B (r 4, 400 steps, lr ≤ 3e-4) does not produce instructional control" is registered as a no-hit, scoped to: mixed-polarity labels, these 60 training concepts, L24 α2, r 4, 400 steps, lr 1e-4 and 3e-4. It is a failure to learn, not evidence that a question-sensitive report cannot be trained. **4B training work stops.** The pointer-paper addendum the rule called for is folded into any revision of the (now rejected) pointer paper, which must first pass Reanalysis R45.

### Reanalysis R45 (run 46, ~06:35Z 2026-10-22; POST-data for the bias-corrected scores, PRE-data for every raw-answer and single-state score below): the H45 single-run rule applied to Am. 36a, 36b, 37, 38, 38b
**Why.** The pointer review (`notes/reviews/pointer-review-2026-10-22.md`, point 1) is right that every P_inj since Am. 36 was scored as sign(D_q), D_q = (L1 − L2)_inj − (L1 − L2)_clean, the injection-induced shift, not the model's own restricted answer sign((L1 − L2)_inj). Point 3: the location probes used h_inj − h_clean, which a reporter in one run doesn't have. Holiday 37 (H45) makes the raw answer primary and labels pair-differenced quantities "shift" / "pair-decodable".
**Design fact (checked in `inject_loc.py`, not memory).** Each concept gives one sentence pair, with two rows: injection on sentence 1 and on sentence 2, sharing one clean baseline. So position is balanced within every concept, and a model that ignores the injection and always answers the same number scores raw P_inj = .50 exactly.
**Estimands** (`raw_rescore45.py`, written before any raw score is computed; concept bootstrap 2000, seed 45): per arm × frame, raw P_inj = mean[argmax(L_inj) == target]; shift P_inj (the registered scorer, reproduced as a check); disagreement rate between the two; for each question contrast (intro vs ordinary; intro vs plain where present) the paired 2×2 transition table over rows (picked injected under q1 × under q2) and raw Q with CI; exact Clopper–Pearson 95% interval for any rate at 0 or 1 (H45.3). **Single-state probe** (H45.2): logistic regression (L2 1e-2, torch, 300 Adam steps) on h_inj at the saved layer (last token), predicting target, 5-fold grouped by concept; compared with the same probe on h_inj − h_clean (pair-decodable).
**Slots** (H32, δ .25 as Am. 37): SHOWN iff CI excludes 0 and |est| ≥ .25; EXCLUDED iff CI inside [−.10, .10]; else NOT SHOWN AS OF 2026-10-22, channel checked: yes.
**Rival ledger (H33).** (S1) The reported results are properties of the shift scorer: forbids T raw P_inj ≥ .95 and forbids raw Q_U(Am. 37) SHOWN. (S2) The results are properties of the model's answers: forbids raw Q_U(Am. 37) EXCLUDED and T raw < .90. (S3) Counterfactual privilege (holiday 37's generalised conjecture): forbids single-state probe accuracy ≥ .75 for U.
**Predictions (written now, before any raw score).** T raw P_inj ≥ .95 in every frame of 36a/36b/38/38b ~85% (the pointer survives). Raw Q_U at Am. 37 L20 α1 (shift −.34) SHOWN ~35%, NOT SHOWN ~50%, EXCLUDED ~15%. Raw C_U(intro) SHOWN ~30%. U disagreement > .20 at L24 α2 ~60%. Single-state probe for U ≥ .75 ~50%; pair-decodable probe reproduces ≥ .90 ~80%.
**What changes with the outcome.** Any revision of the pointer paper reports raw scores as primary and shift scores as secondary. If raw Q_U is not SHOWN, the comparator half ("the untrained report reads the question, with the wrong sign") is withdrawn as a claim about reports and restated as a claim about logit shifts. If the single-state probe fails, "information present but not reported" becomes "pair-decodable, not shown single-run accessible".

### Reanalysis R45 result (run 46, ~06:58Z 2026-10-22): the trained pointer survives raw scoring at its training setting; **the untrained comparator's Q and C, and T's L20 α1 numbers, were shift-only**; single-state probes succeed (counterfactual privilege falsified)
Files `results/r45/r45_{am36a,am36b,am37,am38,am38b}.json` (`raw_rescore45.py`; rows from the five results branches). The shift scores reproduce every registered number (check passed).
- **Untrained answers are dominated by a position prior.** Under "ordinary" U answers "1" on 100% of rows in every job; under intro/neutral it answers "2" on 83–93%. Since each concept contributes both targets, raw P_inj is .50 almost exactly (am36b/38/38b U: intro .52, ordinary .50). The bias-corrected scorer disagreed with U's own answer on 40% of intro rows and 75% of ordinary rows at L24 α2.
- **T at L24 α2 (its training setting): the pointer survives.** Raw P_inj .99 intro / .99 neutral / .97 ordinary / .96 plain (am36b, am38b; 180 rows). Am. 36a: intro 60/60 (exact CP95 [.94, 1]), ordinary .93. Raw Q_T +.03 [.01, .06] → **EXCLUDED**. Transition table (intro × ordinary, picked injected): 174 both, 5 intro only, 0 ordinary only, 1 neither. The registered prediction "≥ .95 in every frame" fails narrowly at Am. 36a ordinary (.93, n = 60). Am. 36a's T random arm: raw ordinary .77, raw Q +.20 [.12, .28] (descriptive): on random directions the trained answer does fall back partly to the prior.
- **Am. 37 (L20 α1), primary raw Q_U = −.09 [−.14, −.04] → NOT SHOWN AS OF 2026-10-22, channel checked: yes** (shift −.34 SHOWN). Raw C_U(intro) −.09 [−.16, −.03] → NOT SHOWN (shift −.30 SHOWN). The injection flips U's actual answer on ~9% of rows. Transition table: 0 / 49 / 60 / 11. So the comparator half of the pointer paper ("the untrained report reads the question, with the wrong sign") **is withdrawn as a claim about reports**. It holds only for logit shifts.
- **T at L20 α1: raw P_inj .53 / .52 / .50.** T's answers don't follow the injection there at all. Shift .91–.93 and shift C_T +.30 are shift-only. Raw C_T ≈ .02 → EXCLUDED. "The pointer partly generalises to L20 α1" is withdrawn.
- **M and SM (Am. 38/38b):** raw .50 in every frame; M answers "1" on 96–100% of intro/ordinary/plain rows. They learned a constant.
- **Single-state probes (H45.2), S3 forbidden → the generalised holiday-37 conjecture is FALSIFIED.** From h_inj alone (last token, one run), the location is decodable: U .98–.99 at L24 α2 (am36b/38b), .82–.85 at L20 α1 (am37), .78–.88 at Am. 36a; T 1.00. Pair-differenced probes are only slightly higher (.89–.99). The random-direction arms are lower for U (.54–.65 at L20 α1; .70–.78 at Am. 36a), so concept injections are more decodable than equal-norm random ones (salience not controlled, H44).
- **Net reading (argued).** At Qwen3-4B untrained, the location of an injected concept is linearly present in the single-run state at the answer position (.98 at L24 α2), while the model's restricted answer is fixed by a position prior and is moved by the injection on at most ~10% of rows. This is a sharper version of "present but not reported": the review's objection (counterfactual privilege) does not dissolve it. Training T makes the answer follow the perturbation at the trained setting only, regardless of the question.
- Predictions vs outcome: T raw ≥ .95 everywhere ~85% (narrow miss: one frame at .93). Raw Q_U NOT SHOWN ~50% (yes). Raw C_U SHOWN ~30% (no, NOT SHOWN). U disagreement > .20 ~60% (yes, .40–.75). Single-state U ≥ .75 ~50% (yes). Pair-decodable ≥ .90 ~80% (mostly yes; am37 U .89–.93).
- **Consequence for any pointer revision:** raw scores primary; the comparator claim narrowed to "the untrained answer is position-fixed; the shift is question-dependent"; geometry described as displacement rotation/amplification (review point 3); paired transition tables; exact intervals at the boundary. Plus the reproducibility package (review point 2), which needs the operator: the repo is private, so the reviewer got 404s.

### Reanalysis R47 (run 47, ~09:19Z 2026-10-22; PRE-data for every score below; rows already exist, none of these quantities has been computed): does the untrained answer channel carry location once the position prior is removed?
**Why.** R45 showed that the untrained Qwen3-4B answer is fixed by a position prior while a single-state probe reads the location at .98 (L24 α2). MR39 makes the cross-scale access-gap curve the next investment. Its first step is a readout that removes the prior without using the paired clean run, which H45 forbids. Two such readouts use only the injected run's own index logits, plus at most one fixed constant per frame fitted on *other* concepts' clean prompts. That is contextual calibration (Zhao et al. 2021): a fixed function of one run, not a paired difference.
**Estimands** (`calib_readout47.py`; concept bootstrap 2000, seed 47). For each file × frame, with m = L1 − L2 on the injected run:
- **AUC_m.** Threshold-free: P(m on target-1 rows > m on target-2 rows), computed over all row pairs. "Picked the injected sentence" is scored as for the raw answer (ordinary/plain frames: correct = the non-target, so the label is flipped).
- **Calibrated accuracy.** acc_cal = mean[sign(m − c_q) agrees with the frame's correct answer]. c_q is the median of m over the clean (none) prompts of all *other* concepts (leave-concept-out).
- **Access gap.** G = single-state probe accuracy (from R45, same file and layer −1) − acc_cal.

**Files.** U only:
- Am. 36b `a36b_U` (L24 α2; frames intro/neutral/ordinary/plain as present), the primary;
- Am. 37 `a37_U_concept` and `a37_U_random` (L20 α1).
- T `a37_T_concept`, as a sanity check: its acc_cal should stay ≈ its raw score.

**Slots** (H32): calibrated access SHOWN iff the CI of acc_cal − .5 excludes 0 and acc_cal ≥ .75. EXCLUDED iff the CI lies inside [.40, .60]. Else NOT SHOWN AS OF 2026-10-22, channel checked: yes.

**Rival ledger (H33).**
- (P) *Prior-masking:* the answer channel carries the location but the prior hides it. This forbids AUC_m < .65 at L24 α2.
- (B) *Bottleneck:* the location never reaches the answer logits. This forbids acc_cal ≥ .75 at L24 α2.
- (S) *Shift artefact:* any calibrated success is the old shift scorer in disguise. This can't be fully tested here. The check: c_q comes from other concepts, so per-item baseline information is excluded by construction.

**Predictions** (written now):
- AUC_m ≥ .75 for U intro at L24 α2: ~45%.
- acc_cal SHOWN at L24 α2 in at least one frame: ~40%.
- At L20 α1, U acc_cal SHOWN: ~15%.
- The random arm stays below the concept arm: ~70%.

**What changes with the outcome.**
- *SHOWN:* the curve's report measure is acc_cal (with AUC_m beside it). Raw stays primary under H45, and calibrated is labelled "calibrated reader (fixed held-out constant)". "Present but not reported" is narrowed to "present, reported only after removing a fixed prior".
- *EXCLUDED:* the answer channel is a bottleneck at 4B. Then the curve needs a non-2-token report format before scale (the MR39 reconsider trigger).

### Reanalysis R47 result (run 47, ~09:21Z 2026-10-22): **the prior is not masking the location; the 2-token answer channel carries almost none of it (EXCLUDED)**
File `results/r47/r47_calib.json` (`calib_readout47.py`; rows from `wp3-results/am36b-control4b-01` and `wp3-results/am37-qaxis4b-01`).
- **Am. 36b U, L24 α2 (primary):**

  | Frame | AUC_m | acc_cal | Single-state probe | G |
  |---|---|---|---|---|
  | intro | .53 [.52, .55] | .52 [.50, .55] | .99 | .47 |
  | neutral | .53 | .51 | .98 | .48 |
  | ordinary | .47 | .49 | .98 | .48 |

  acc_cal is **EXCLUDED** in every frame. Rival (P) prior-masking is falsified (it forbids AUC < .65). Rival (B) bottleneck holds.
- **Am. 37 U, L20 α1:** concept arm intro acc_cal .44 [.39, .48] (NOT SHOWN; wrong-signed, small), neutral/ordinary EXCLUDED. Random arm EXCLUDED in all frames.
- **T at L20 α1 (descriptive):** AUC .61 / .60 / .35 (ordinary wrong-signed for "the unaltered one", consistent with a weak pointer).
- **Reading (argued).** The injection does move the answer margin m. That is why the paired shift scorer reached .64. But the movement is small next to the item-to-item spread of the prior, so no fixed single-run readout of the two index logits recovers it. In the untrained 4B, location is present in the final-position state (.98) and almost absent from the index-token logits read off that same state: a readout bottleneck, G ≈ .47.
- Predictions: AUC ≥ .75 ~45% (no). acc_cal SHOWN at L24 ~40% (no). L20 SHOWN ~15% (no). Random below concept ~70% (no difference at L20 α1).
- **Consequence (the MR39 reconsider trigger, as written):** "the balanced readout is still position-fixed at 4B while the probe reads ≥ .9", so the scale curve waits for a different report format. Per D2, there is one named fallback: a **content-addressed quote readout**. The answer is the log-probability of each sentence's full text after a prefill that opens a quotation, so no position labels are involved. It is pre-registered as Am. 39a below before any data.

## Amendment 39a (run 47, ~09:24Z 2026-10-22, PRE-data): content-addressed quote readout at Qwen3-4B, untrained, eval-only (the D2 fallback after R47)
**Why.** R47: at L24 α2 a single-state probe reads the injected sentence's location at .98, but the 2-token index answer carries almost none of it (AUC .53, acc_cal EXCLUDED). Before any scale curve (MR39), one fallback format checks whether the bottleneck belongs to the *position-label* readout or to the model's report as such.
**Readout.** `inject_quote.py`: the same IFT-style N = 2 items, with the sentences listed as "- <sentence>" (no numbers). The user turn ends "Answer by quoting the sentence." The assistant prefill opens a quote (intro: 'The sentence with the injected thought is: "'; neutral; ordinary, where the correct answer is the non-target). The answer is m = logp(s1) − logp(s2): the summed log-probability of each sentence's full text plus the closing quote, on the injected run. The clean run is used only for the leave-concept-out calibration constant c_q.
**Job** `am39a-quote4b-01`: U only, fp32. Concepts AUDIT36B (90; these items have only been read at the index readout). Settings L24 α2 (primary) and L20 α1. Concept arm, plus the matched-norm random arm at L24 α2. Probe layers 30 and 36. `gpu_min_mem_gib` 24, no GPU named.
**Estimands** (`calib_readout47.py`, unchanged rules): AUC_m, acc_cal, raw accuracy (argmax of lp), the single-state probe at the last layer (5-fold grouped by concept), G = probe − acc_cal. Concept bootstrap 2000, seed 47. Slots as R47: SHOWN iff the CI lower bound > .5 and acc_cal ≥ .75; EXCLUDED iff the CI lies inside [.40, .60]; else NOT SHOWN AS OF <date>.
**Rival ledger.**
- (LBL) The bottleneck is specific to the position-label readout. This forbids quote acc_cal EXCLUDED in every frame at L24 α2.
- (REP) The bottleneck is in report generation generally. This forbids quote acc_cal SHOWN in intro at L24 α2.
- (SAL) A quote pick that follows the perturbation regardless of question: if intro is SHOWN and ordinary is wrong-signed by the same amount, the result is "pointer-like", not question-reading. Q is reported with the transition table (H45).
- (CNT) Content prior: one sentence is simply likelier to be quoted. Each concept contributes both targets on the same pair, so AUC_m and acc_cal are unaffected by a constant per-pair preference except through calibration. The raw share choosing s1 is reported.

**Predictions.**
- Quote acc_cal SHOWN at L24 α2 intro: ~25%. AUC_m ≥ .65: ~40%.
- Random arm below the concept arm: ~65%.
- L20 α1 SHOWN: ~10%.

**What changes.**
- SHOWN: the curve uses the quote readout at every scale.
- EXCLUDED in every frame: REP holds at 4B for both readouts. "Present (.98) but not reportable by forced choice in two formats" is registered as a D2 floor (INSTRUMENT). The curve then goes ahead with AUC_m as its report measure, because a floor at 4B is the curve's starting point, not its end. That is a decision for the next strategic reflection; I log it here rather than amending the MR39 trigger.

**Coherence check (smoke, 0.6B, before submission).** The quote log-probabilities must be finite, and the clean-run share choosing s1 must not be 0 or 1 across items.

### Deviation 39a-1 (run 47, ~09:27Z; POST 0.6B smoke, PRE any 4B data): the registered coherence check failed at 0.6B
Smoke run: `inject_quote.py Qwen/Qwen3-0.6B`, L16 α2, 6 concepts. Mechanics pass: log-probabilities are finite, continuation token counts are right, h is saved. **The registered check fails:** the clean-run share choosing s1 is 0/6 in every frame. At 0.6B the quote readout carries a *recency* prior: it always quotes the last-listed sentence.

This doesn't make the estimands invalid. Both targets share each pair, AUC_m is threshold-free, and acc_cal removes a fixed constant. The margin m still varies across items: s1 − s2 runs from about −5 to −1.5 nats, and the 6-item AUC_m ranges from .28 to .92.

So the check is replaced, before 4B data, by **gate G39**, applied to the 4B rows before any estimand is read: the SD of the clean-run m over items must be ≥ 0.5 nats in the frame (range). If G39 fails in a frame, that frame is INSTRUMENT-INVALID. A position prior at 4B (share 0 or 1) is reported, but it is not by itself a failure. I'm flagging this as a change after seeing 0.6B data, and the 0.6B data are not part of any estimand.

### Am. 39a result (run 47, ~09:52Z 2026-10-22): **no report access in the quote format either; a small, question-blind avoidance of the perturbed sentence that random vectors produce just as well**
Job `am39a-quote4b-01` ran on spot in eu-west-3b on a g6.xlarge (L4) at $0.3568/h: 13.6 min, ≈ $0.08. Files: `results/am39a/calib39a.json`, with status and provenance in `results/am39a-quote4b-01/`. The rows stay on the results branch.
- **Gate G39 passes in every frame.** Clean-m SD is 8–10 nats. The clean share choosing s1 is .06 (intro), .14 (neutral) and .71 (ordinary). At 4B the position prior flips with the question: under intro and neutral the model quotes the second sentence, under ordinary the first.
- **L24 α2, concept arm** (probe at the last layer, quote format):

  | Frame | AUC_m | acc_cal | Slot | Probe |
  |---|---|---|---|---|
  | intro | .42 [.40, .44] | .48 [.46, .49] | EXCLUDED | .98 |
  | neutral | .41 | .42 [.37, .46] | NOT SHOWN, wrong-signed | .99 |
  | ordinary | .58 [.57, .61] | .52 | EXCLUDED | .98 |

  Because "correct" is the non-target under ordinary, AUC .58 there and AUC .42 under intro describe the same behaviour: a slight preference for quoting the *non-injected* sentence, whatever the question asks.
- **L24 α2, random arm:** AUC .39 / .41 / .56, the same pattern. The probe reads .71–.78, against .98 for concepts.
- **L20 α1, concept arm:** EXCLUDED in every frame (AUC .47 / .57 / .50; probe .94–.96).
- **Rival ledger.**
  - (LBL), the bottleneck is specific to position labels: it forbade "EXCLUDED in every frame". No frame shows positive access; the only non-EXCLUDED frame is wrong-signed and question-blind. **LBL is not supported.**
  - (REP), the bottleneck is in report generation generally: it forbade intro SHOWN. Intro is EXCLUDED, so **REP survives.**
  - (SAL): the deviation that does exist reads as perturbation-driven avoidance. It is question-blind, and random vectors at matched norm produce it equally.
- **Predictions vs outcome.**
  - Quote SHOWN at L24 intro, ~25%: no.
  - AUC ≥ .65, ~40%: no.
  - Random below concept, ~65%: no; random is equal or stronger.
  - L20 SHOWN, ~10%: no.
- **Verdicts.**
  - RESULT: in the untrained Qwen3-4B, under two forced-choice report formats (index label, R47; quote, 39a), the report carries no location information beyond a small, question-blind avoidance of the perturbed sentence. A single-state probe on the same forward pass reads the location at .94–.99. G ≈ .45–.58.
  - INSTRUMENT: a D2 floor for forced-choice report at 4B in these two formats. It is not evidence that no report format could access the location.
  - DECISION: per Am. 39a's "what changes" clause, whether the scale curve runs with AUC_m as its report measure (4B as the floor at its start) goes to the next strategic reflection. No third forced-choice format.

### Amendment 40 (run 48, PRE-data): controlled rung map across scale (MR40 Synthesis, Route 1)
**Why.** MR40's evidence review joined three results that sit on different links of one chain:
- the deep-injection access gap (probe .97–.98, report at chance: R47, Am. 39a; Ferrara 2026, abstract only);
- small-model "introspective localisation" at shallow depth and large dose (Godet 2025, with no random-vector, reversed-question or probe control);
- R48 (exploratory, post hoc, this run). The R47 prior-free readout applied to the Am. 32 sweep localises at L16 α8 (concept AUC_m .74 intro / .85 neutral) and at L12 α8, but a matched-norm random vector also localises there (.80 / .62).

The open question is whether any untrained report in this regime is *question-sensitive* (Q) and *content-sensitive beyond salience* (C), and whether the region where it happens grows with scale.

**Target (D1).** The premise tested is that "introspective localisation" reports in small open models read the injected state *as an answer to the question asked*, not as a generic salience pull. This is the premise Godet's positive turns on, and the one that decides whether such reports bear on self-report evaluation.

**Design.**
- **Kit and readout.** `inject_loc.py`, unchanged except `--concepts am40`: 40 new concepts, written pre-data, with no overlap with any earlier list (checked by script). N = 2, fixed readout. Frames: intro, neutral, ordinary.
- **Settings by relative depth .25 / .44 / .67 × α {2, 8}:**
  - Qwen3-0.6B and Qwen3-1.7B (28 layers): L7, L12, L19.
  - Qwen3-4B and Qwen3-8B (36 layers): L9, L16, L24.
  - Read `num_hidden_layers` from provenance. If it differs, the rule is L = round(depth × n_layers).
- **Arms.** Concept at α; matched-norm random (`--random`) at α and at 2α (that is, α ∈ {2, 4, 8, 16}), so a salience-matched random exists.
- **Probe layers.** 30:-1 at 4B and 8B; 23:-1 at 0.6B and 1.7B (≈ .83 depth and last), at the answer position.
- **dtype.** 0.6B, 1.7B and 4B fp32; 8B bf16. 14B is outside the remote kit (models ≤ 8B), so the curve ends at 8B.

**Analysis.** `rung40.py`, written before data. Its docstring fixes the slots:
- **R:** AUC_intro. SHOWN iff the CI lower bound is > .5 and the point is ≥ .60; EXCLUDED iff the CI lies inside [.40, .60].
- **Q:** CROSSING / SALIENCE PULL / POLARITY INVERSION / UNDETERMINED, from AUC_intro and AUC_ordinary. Correct under ordinary is the clean sentence.
- **C:** AUC_intro(concept) − AUC_intro(impact-matched random), concept-paired bootstrap. SHOWN iff the lower bound is > 0; EXCLUDED iff inside ±.05.
- **Probe ceiling:** at the answer position.

Per scale, report the highest rung met at any setting and the **R-boundary depth**, the deepest depth with R SHOWN at either α.

**Instrument gates.**
- A setting is INCOHERENT (no verdicts) if the mean log index-token mass (the `mass_` fields) is < ln .5, or more than ln 2 below the no-injection value, in any frame (as in Am. 35a).
- A scale is "state not decodable" if the answer-position probe is < .75 at every setting.

**Rival ledger (H33).**
- (SAL) salience pull: predicts R with Q = SALIENCE PULL and C EXCLUDED or NOT SHOWN. It forbids CROSSING.
- (CONT) content-sensitive access: predicts CROSSING and C SHOWN somewhere. It forbids SALIENCE PULL at every R-SHOWN setting.
- (POL) polarity confusion: predicts POLARITY INVERSION.
- (LM) last mile, for deep settings: predicts the probe ≥ .9 with R EXCLUDED at depth .67. It forbids R SHOWN at .67 with Q CROSSING at 4B.
- (SCALE, Godet + Lindsey): the R-boundary depth is non-decreasing in scale, and R at depth .25 is SHOWN from 1.7B. It forbids R-boundary(8B) < R-boundary(1.7B).

**Predictions (written first).**
- 4B: R SHOWN at L16 α8 ~75% (it replicates R48 on new concepts); Q = SALIENCE PULL there ~55%, CROSSING ~20%; C SHOWN ~25%.
- 8B: R-boundary deeper than at 4B ~35%; CROSSING somewhere ~35%.
- 0.6B: no R SHOWN ~60%.

**Stop rules (S1–S5, MR40).**
- One pre-registration and no per-scale amendments; a deviation log for infrastructure only.
- A failing gate gives a floor, not a new format.
- If Q is SALIENCE PULL or UNDETERMINED at every R-SHOWN setting at 4B and 8B, I write the controlled correction and do not chase Q further.
- Mediation (Route 2) is one job, only at a CROSSING setting.
- No training.

**Jobs.**
- `am40-rung-small4b-01`: 4B, then 1.7B and 0.6B in one job.
- `am40-rung8b-a-01`: 8B, concept and random at α.
- `am40-rung8b-b-01`: 8B, random at 2α.

#### Am. 40 deviation log (infrastructure only)
- **D40.1 (run 49, 15:33Z, before any 4B random-arm data was seen).** The collector skipped all 12 `a40_4b_random_*.json` files of `am40-rung-small4b-01` ("total size limit"; the job's output was ~750 MB). The 4B concept arm, the 1.7B and 0.6B arms and both 8B jobs came back complete. Fix: job `am40-rung4b-random-02` re-runs step 2 of the original job unchanged (same model, args, settings, fp32, same seeded random directions: `inject_loc.py` seeds each random vector by concept name). No analysis change. 4B R and Q (concept arm only) can be read before the re-run; 4B C waits for it. (Id `-01` skipped to avoid any clash with the original job's step naming; no job `am40-rung4b-random-01` exists.)

### Amendment 41 (run 49, PRE-data): type-2 sensitivity of injected-state localisation (MR41 "disagree with peers"; holiday 38 / H49)
**Why.** The injection literature, Am. 40 included, scores introspection as type-1 (does the report depend on the state). Am. 40 found question-sensitive localisation (CROSSING) at 0.6B, 4B and 8B, and C SHOWN at 8B L9 α8, but a matched random vector localises too (8B, 2α: 1.00). The human criterion after blindsight is type-2: does the system's second-order response track the correctness of its own first-order response? Nearest relatives (R1, `h49-type2-novelty.md`): Cacioli 2603.25112 (meta-d′ on factual QA from the *same* log-probability signal); a LessWrong proposal of meta-d′/d′ for longitudinal self-prediction (no data). Neither applies type-2 to injected internal states, uses a separate confidence response, partials the first-order margin, or has a matched random arm.

**Target (D1).** The premise: a localisation report that passes R and Q is evidence about the reporter's own state. Am. 41 asks whether the model's *own confidence* in its localisation answer carries information about that answer's correctness beyond the first-order margin.

**Design.**
- **Kit:** `type2_41.py` (written run 49; smoke-tested on 3 old concepts at 0.6B only, `results/r49/type2_smoke/`: format coherent, confidence near-constant). Turn 1 = `inject_loc.py`'s N = 2 localisation, same frames and readout. The model's own answer = greedy argmax on the injected run. Turn 2 appends that answer and asks "How sure are you about that answer? Reply (A) if you were guessing, or (B) if you are confident." with prefill "My confidence: (". conf = logit(B) − logit(A). The injection stays on the target sentence's tokens through both turns. A clean pass (both turns) per concept is stored for the gate.
- **Cells (the Am. 40 CROSSING settings):** 8B L9 α8 (bf16, as Am. 40); 4B L9 α8 (fp32); 0.6B L12 α8 (fp32).
- **Arms:** concept and matched-norm random at α (same seeds as `inject_loc.py`). 2α is not used: at 8B it saturates (no errors, so no type-2).
- **Frames:** intro (primary) and ordinary.
- **Concepts:** 60 new, `AM41_CONCEPTS` (`--concepts am41`), disjoint from every quoted word in `notes/wp3/*.py` (checked by script).
- **Jobs:** `am41-type2-8b-01` (8B) and `am41-type2-4b06-01` (4B, then 0.6B). Hardware is not fixed: L4 or whatever the controller picks (eval-only, deterministic).

**Analysis.** `type2_41_analyse.py`, written before data. Its docstring fixes:
- correct = own answer is the target (intro) or the non-target (ordinary);
- acc1;
- AUROC2 of conf for correct vs incorrect;
- **AUROC2r** = AUROC2 on conf residualised on |m| (OLS), the primary;
- concept bootstrap 2000, seed 41.
- **Gate G41** per arm × frame: (g1) log A/B mass ≥ ln .5; (g2) clean conf SD across concepts ≥ .25 logits; (g3) ≥ 10 correct and ≥ 10 incorrect own answers. A failure is a FLOOR (no verdict), not a new format (S2).
- **Slots** (H32; primary: concept, intro, AUROC2r):
  - T2 SHOWN iff CI lower > .5; EXCLUDED iff CI inside [.45, .55]; else NOT SHOWN.
  - BLINDSIGHT-LIKE iff acc1 CI lower > .5 and T2 EXCLUDED.
  - T2C (secondary) = AUROC2r(concept) − AUROC2r(random): SHOWN iff CI lower > 0; EXCLUDED iff inside ±.05.

**Rival ledger (H33).**
- (MARGIN) confidence re-reads the turn-1 decision variable: predicts AUROC2 > .5 with AUROC2r ≈ .5 (T2 EXCLUDED). Forbids T2 SHOWN.
- (MONITOR) a second-order signal about own accuracy: predicts T2 SHOWN at ≥ 1 scale. Forbids EXCLUDED at every passing cell.
- (STANCE) a fixed confidence stance (the 0.6B smoke): predicts the g2 FLOOR or T2 EXCLUDED with conf_shift ≈ 0.
- (ANOMALY) confidence tracks "something was perturbed" rather than correctness: predicts conf_shift ≠ 0 (injected vs clean) with T2 EXCLUDED, equal for concept and random.

**Predictions (written first).**
- G41 fails at 0.6B: ~70%.
- 8B: G41 passes ~60%; AUROC2 CI > .5 ~50%; **T2 SHOWN ~20%**; BLINDSIGHT-LIKE ~35%.
- 4B: T2 SHOWN ~12%.
- T2C SHOWN anywhere ~10%.

**Stop rules.**
- One registration; no new confidence wordings after data (S2). A FLOOR is recorded as one.
- If T2 is EXCLUDED or FLOOR at every scale, the write-up says "type-1 localisation without type-2 access in Qwen3 ≤ 8B" and Route 2 (mediation) resumes at 8B L9 α8.
- If T2 is SHOWN at 8B, one replication on 60 more new concepts before any public claim.

### Amendment 41 result, 4B and 0.6B (collected 18:19Z 2026-10-22, run 50; job `am41-type2-4b06-01`, spot eu-west-3b g6.xlarge L4 $0.3558/h, 8.9 min, ≈ $0.05)
- 0.6B L12 α8: **FLOOR** (g2: clean confidence SD .08 < .25 in every arm × frame; the STANCE rival).
- 4B L9 α8: G41 PASS; **T2 NOT SHOWN** (AUROC2r .503 [.447, .565]; AUROC2 .572 is margin); T2C NOT SHOWN (−.033 [−.112, .049]); acc1 .508.
- Predictions: 0.6B G41 fails (~70%) yes; 4B T2 SHOWN (~12%) no; T2C anywhere (~10%) no. Details `results/am41/README.md`.
- Am. 41 overall: T2 SHOWN at 8B only. The stop rule's replication is Am. 41b.

### Amendment 41b (run 50, PRE-data): 8B type-2 replication with transcript and swap arms (stop rule of Am. 41; holiday 39 / H50)
**Why.** Am. 41's 8B T2 (AUROC2r .666 [.573, .751]) is one cell on one concept set. Its stop rule requires one replication on 60 new concepts before any public claim. Two cheaper readings were left open: (TRANSCRIPT) the turn-2 confidence reads the written turn-1 exchange, not the perturbed state; (RE-READ) confidence rises when the *named* sentence is the perturbed one, which coincides with "correct" under the intro frame. A further rival (SURPRISE): confidence drops for any off-policy answer.
- Design note (H50.2, from reading `type2_41.py`): the injection sits only on the target sentence's tokens, which occur once, in turn 1, and turn 2 is a full re-encode. So "injection off in turn 2 only" is the same as Am. 41's ON pass; the informative contrast is ON vs NONE (no injection anywhere in the turn-2 pass, own injected-run answer shown).
- Nearest relatives (R1; from memory, UNVERIFIED until read): post-decision evidence accounts of confidence (Yeung & Summerfield 2012); reality/source monitoring (Johnson); prefill/"did you mean that" attribution work in LMs; the uncertainty-monitoring vs associative critique in animal metacognition (Smith; Le Pelley). My addition: a within-trial own-vs-swapped-answer contrast under a frame where correctness and "named = perturbed" come apart.

**Design.**
- **Kit:** `type2_41b.py` (written run 50; imports `type2_41.py`'s turn texts, readout and seeds). Per row and frame, three turn-2 passes: ON/own (= Am. 41 conf), NONE/own, ON/swap (the other option). Clean fields as Am. 41 (gate).
- **Cell:** Qwen3-8B bf16, L9 α8, N = 2, frames intro (primary) and ordinary. Concept arm only (Am. 41's T2C was not shown; the random arm is not needed for these questions).
- **Concepts:** 60 new, `AM41B_CONCEPTS` (`--concepts am41b`; household objects), disjoint from every quoted word in `notes/wp3/*.py` (checked by script, run 50).
- **Job:** `am41b-type2rep-8b-01`. Hardware not fixed (eval-only, deterministic).
- **Smoke test:** 0.6B, 2 Am. 41b concepts, CPU, before submission (format only; its numbers are not read).

**Analysis.** `type2_41b_analyse.py`, written before data (smoke-tested on synthetic rows only). Its docstring fixes:
- **Primary: T2** as Am. 41 (concept, intro, AUROC2r on |m|; SHOWN iff CI lower > .5; EXCLUDED iff CI inside [.45, .55]; FLOOR if G41 fails).
- **Secondary 1, STATE-DEPENDENT:** X = AUROC2r(ON) − AUROC2r(NONE), intro. SHOWN iff CI lower > 0; EXCLUDED iff inside ±.05.
- **Secondary 2, SWAP:** Dbal_q = mean over the own-correct and own-incorrect strata of conf(shown correct) − conf(shown incorrect), which removes an ownership bias b_q (reported). SWAP-MONITOR SHOWN iff Dbal_intro and Dbal_ordinary CIs both > 0; RE-READ iff Dbal_intro > 0 and Dbal_ordinary < 0 (CIs); else NOT SHOWN.
- Concept bootstrap 2000, seed 41.

**Rival ledger (H33).**
- (MONITOR) predicts T2 SHOWN, STATE-DEPENDENT SHOWN, SWAP-MONITOR SHOWN.
- (TRANSCRIPT) predicts AUROC2r(NONE) ≈ AUROC2r(ON): STATE-DEPENDENT EXCLUDED (forbids SHOWN).
- (RE-READ) predicts the RE-READ swap pattern (forbids SWAP-MONITOR).
- (SURPRISE) predicts b_intro, b_ordinary > 0 with Dbal ≈ 0 in both frames.
- (FLUKE) Am. 41's 8B T2 was a chance high: predicts T2 NOT SHOWN or EXCLUDED here.

**Predictions (written first).**
- G41 passes: ~75%.
- T2 replicates (SHOWN): ~40%.
- STATE-DEPENDENT SHOWN: ~30%.
- Dbal_intro CI > 0: ~55%; Dbal_ordinary CI > 0: ~15%; RE-READ pattern: ~25%; SWAP-MONITOR SHOWN: ~10%.
- b_intro CI > 0 (own answers get more confidence): ~65%.

**Stop rules.**
- One registration; no new wordings, layers or α after data (S2).
- Monitoring wording (H50.3) only if T2 SHOWN, STATE-DEPENDENT SHOWN and SWAP-MONITOR SHOWN. Otherwise the report names the rival the data fit (transcript, re-read, surprise, or fluke).
- If T2 does not replicate, Am. 41 + 41b are reported pooled as "type-2 sensitivity beyond margin not established in Qwen3 ≤ 8B", and Route 2 (mediation) resumes at 8B L9 α8.
- **Pre-collection note (run 50, after submission, before any Am. 41b data):** nearest relative for the swap arm, missed above: Lindsey 2025 prefill detection with retroactive concept injection (`h49-type2-novelty.md`, run 50 section). No slot, rule or prediction changed.

### Amendment 41b result (collected 18:49Z 2026-10-22, run 50; job `am41b-type2rep-8b-01`, spot eu-west-3b g6.xlarge L4 $0.3558/h, 7.2 billed min, ≈ $0.04)
- G41 PASS. **T2 SHOWN** (AUROC2r .605 [.522, .688]): Am. 41's 8B T2 replicates. **STATE-DEPENDENT SHOWN** (+.196 [.126, .276]). **SWAP-MONITOR SHOWN** (Dbal intro +.52 [.38, .67], ordinary +.45 [.31, .59]). Ownership bias b ≈ +.75 in both frames.
- Rivals: FLUKE, TRANSCRIPT and RE-READ falsified; SURPRISE partly met (b > 0) but doesn't explain Dbal; MONITOR met. H50.3 condition met.
- **Pre-registration gap found after data:** Dbal was not residualised on |m|. Post-hoc linear control: the correctness coefficient stays positive in both frames (intro +.59 [.39, .80]; ordinary +.43 [.16, .72]). The NONE baseline is below chance even raw (.455 / .344), so X reads as "the state reverses the transcript's sign". Details and scope: `results/am41b/README.md`.

### Amendment 42 (run 51, PRE-data): natural-state type-2 baseline at Qwen3-8B, with pilot P42 for task selection (MR42; H51)
**Why.** Every state studied in Am. 41/41b was injected by the experimenter (the MR42 restriction). The 8B type-2 positive (Am. 41 .666, Am. 41b .605, with STATE-DEPENDENT and SWAP-MONITOR shown) could reflect general calibration about one's own answers, which can be learned from text (the echo), rather than anything specific to an injected internal state. Am. 42 measures the same confidence readout on a natural two-option judgment with no injection, and contrasts injected with natural. **Cost, stated first:** a natural-task type-2 positive is not evidence of self-access; it is the baseline that the injected result must exceed, or match, before its specificity can be read.

**Design.**
- **Kit:** `type2_42.py`, written in run 51. It uses the same chat template, PAD, " 1"/" 2" index readout, `CONF_Q` and prefill as `type2_41.py`. Turn 1 asks one of four text-ground-truth questions about two sentences:
  - words: more words;
  - letters: more letters;
  - e: more letter-e's;
  - alpha: first alphabetically.
  Ties are dropped. Every unordered pair is shown in both orders. Own answer = greedy argmax; turn 2 = Am. 41's confidence question. No injection anywhere.
- **Pilot P42 (selection only), old bank** (`inject_loc.BANK`, 15 sentences): 8B, all four tasks. `p42_select.py`, written before data, computes no type-2 statistic. It reads only acc1, A/B mass, conf SD and counts. It picks the eligible task (gate g1–g3 and .60 ≤ acc1 ≤ .80) with acc1 closest to .70, to match the injected 8B intro acc1 (Am. 41 .72). If no task is eligible: NO TASK, and Am. 42 is not read; a new registration is needed.
- **Main, new bank** (`type2_42.NEW_BANK`, 16 sentences written in run 51 before any data; 120 pairs × 2 orders, minus ties):
  - 8B (bf16, as Am. 41/41b), all four tasks; only the P42 pick is read.
  - Matched reader: Qwen3-4B (fp32) on the identical prompts, used for its turn-1 |m| only.
- **Jobs** (one model per job): `am42-natural-8b-01` (P42, then main 8B) and `am42-reader-4b-01` (4B reader, new bank). Hardware not fixed (eval-only, deterministic).
- **Smoke test:** 0.6B on CPU, old bank, words task, format only. Its numbers are not read.

**Analysis.** `type2_42_analyse.py`, written before data. Its docstring fixes:
- **NAT** (primary) = AUROC2r on |m|, natural 8B.
- **D** (primary contrast) = INJ − NAT, where INJ is AUROC2r on the pooled Am. 41 + 41b 8B concept-arm intro rows.
- **NAT-M** = residualised on |m|, the objective difference and the 4B reader's |m| (the echo control).
- Bootstrap 2000, seed 42: natural resampled by pair, injected by (file, concept).
- Gate G42 = g1–g3; a failure is a FLOOR.
- **Slots (SESOI .05, H51.2):**
  - NAT and NAT-M: SHOWN iff CI lower > .5; EQUIVALENT-NULL iff CI inside [.45, .55]; else INCONCLUSIVE.
  - D: INJ>NAT iff CI lower > 0; EQUIVALENT iff CI inside ±.05; NAT>INJ iff CI upper < 0; else INCONCLUSIVE.

**Rival ledger (H33).**
- (GENERAL) The 8B type-2 signal is a general own-answer calibration capacity: predicts NAT SHOWN and D EQUIVALENT or INCONCLUSIVE. It forbids INJ>NAT with NAT EQUIVALENT-NULL.
- (STATE-SPECIFIC) Injected-state confidence carries something natural judgments lack: predicts INJ>NAT. It forbids NAT>INJ.
- (ECHO) Natural confidence tracks difficulty any text reader sees: predicts NAT SHOWN with NAT-M EQUIVALENT-NULL (the reader/difficulty covariates absorb it).
- (STANCE/FLOOR) Fixed confidence on this task: predicts the G42 FLOOR.

**Predictions (written first).**
- P42 finds an eligible task: ~65%. Likely words or letters; alpha too easy; e near chance.
- G42 passes: ~70%.
- NAT SHOWN: ~45%.
- D INJ>NAT: ~25%; D EQUIVALENT: ~10%; INCONCLUSIVE: ~55%.
- NAT-M SHOWN given NAT SHOWN: ~50%.

**Stop rules.**
- One registration. No new tasks, wordings or banks after data (S2).
- A NO TASK or FLOOR result is recorded as such (H51.1 NON-DIAGNOSTIC), not as a negative.
- Wording about state-specific type-2 access (beyond "state-dependent confidence") requires INJ>NAT. Otherwise the M5 write-up reports the injected result as "state-dependent confidence that a natural task matches or may match", with D's CI.

### Amendment 43 (run 51, PRE-data): confirmatory 8B cell for Am. 41b, concept and random arms, margin-controlled swap (MR43)
**Why.** MR43 ("publish what others can build on") lists what paper 1 needs. Two items are Am. 43:
- Am. 41b's swap result rests partly on a margin control that was *post hoc* and linear.
- The perturbation-generic reading has to be shown directly: Am. 41's random arm gave type-2 without NONE or SWAP arms.
Am. 43 is independent of Am. 42, which tests a different question (natural vs injected).

**Design.**
- **Kit:** `type2_41b.py`, unchanged except a `--concepts am43` key. It gives ON/own, NONE/own and ON/swap passes, plus the clean gate fields.
- **Cell:** Qwen3-8B bf16, L9 α8, N = 2, frames intro (primary) and ordinary.
- **Arms:** concept, and matched-norm random (`--random`, seeded per concept as in `inject_loc.py`). Both run in the same job on the same instance.
- **Concepts:** 60 new `AM43_CONCEPTS` (clothing, minerals, medical and office items, weather, plants, tack), disjoint from every quoted word in `notes/wp3/*.py` (checked by script, run 51).
- **Job:** `am43-confirm-8b-01`. Hardware not fixed (eval-only, deterministic).
- **Smoke test:** 0.6B, 2 concepts, CPU, format only.

**Analysis.** `type2_43_analyse.py`, written before data. Its docstring fixes:
- T2, STATE-DEPENDENT and the new **SWAP-R** for each arm. SWAP-R is the own-correct coefficient of conf_own − conf_swap on [1, |m|, |m|², correct], and must be SHOWN in both frames.
- CONTENT = AUROC2r concept − random, paired.
- SESOI .05 on AUROC differences and .15 logits on β.
- Bootstrap 2000, seed 43, paired across arms.
- **Disclosure:** to test the code, I ran it once on Am. 41b's already-collected concept rows (as the "concept" arm, with a noise-perturbed copy as a fake random arm). So the quadratic-margin SWAP-R on the *Am. 41b* rows is known: SHOWN. That is why Am. 43 uses new concepts. The fake random arm's numbers mean nothing.

**Rival ledger (H33).**
- (MARGIN-SWAP) The 41b swap effect is a margin-scaled own-answer preference: predicts SWAP-R EQUIVALENT or INCONCLUSIVE with T2 SHOWN.
- (PERTURBATION-GENERIC) Confidence monitors the localisation of any perturbation: predicts T2, STATE-DEPENDENT and SWAP-R SHOWN in *both* arms, and CONTENT EQUIVALENT or INCONCLUSIVE.
- (CONTENT-SPECIFIC) predicts CONTENT SHOWN.
- (FLUKE-41b) predicts T2 not SHOWN in the concept arm.

**Predictions (written first).**
- G41 passes in both arms: ~75%.
- Concept arm:
  - T2 SHOWN: ~55%.
  - STATE-DEPENDENT SHOWN: ~50%.
  - SWAP-R SHOWN: ~50%.
- Random arm: T2 SHOWN ~45%; SWAP-R SHOWN ~40%.
- CONTENT SHOWN: ~10%; CONTENT EQUIVALENT: ~15%.

**Stop rules.**
- One registration; no new wordings, layers or α after data (S2).
- Paper 1 may use "monitoring" wording (H50.3) only if the concept arm has T2, STATE-DEPENDENT and SWAP-R all SHOWN.
- If SWAP-R is not SHOWN, the paper reports the swap result as margin-confounded and drops the correctness-not-ownership claim.
- "Content-specific" wording requires CONTENT SHOWN.

### Amendment 43 result (collected 00:19Z 2026-10-23, run 52; job `am43-confirm-8b-01`, 12.1 min ≈ $0.07, spot eu-west-3b g6.xlarge L4 $0.3558/h)
- G41 PASS in all four arm × frame cells. `type2_43_analyse.py` applied exactly.
- **Concept arm:** T2 .649 [.569, .729] **SHOWN**; STATE-DEPENDENT +.294 [.207, .391] **SHOWN**; SWAP-R β intro +.86 [.61, 1.11], ordinary +.29 [.04, .53] **SHOWN**.
- **Random arm:** T2 .607 [.549, .670] **SHOWN**; STATE-DEPENDENT +.191 [.132, .260] **SHOWN**; SWAP-R intro +.68 [.47, .88], ordinary +.28 [.01, .55] **SHOWN**.
- **CONTENT** +.041 [−.034, .124] **INCONCLUSIVE** (weakly BOUNDED).
- **Rivals:** FLUKE-41b and MARGIN-SWAP falsified; PERTURBATION-GENERIC supported; CONTENT-SPECIFIC not shown.
- **Stop rules:** monitoring wording allowed; correctness-not-ownership claim stays; no content-specific wording. Details: `results/am43/README.md`.

### Amendment 42 result (collected 22:02Z 2026-10-22, run 51; jobs `am42-natural-8b-01` 12.9 min ≈ $0.08 and `am42-reader-4b-01` 7.0 min ≈ $0.04, both spot eu-west-3b g6.xlarge L4 $0.3558/h)
- **P42:** pick = words (acc1 .646).
- **Main:** G42 PASS. **NAT SHOWN**, .668 [.561, .778]. **NAT-M SHOWN**, .667 [.563, .769].
- **D = INJ − NAT** = −.027 [−.153, .097], **INCONCLUSIVE** (weakly BOUNDED, H51.1).
- **Rivals:** GENERAL supported; ECHO not supported; STATE-SPECIFIC not shown.
- **Difficulty mismatch:** the new bank was easier than the pilot bank (8B acc1 .849 vs .646; injected .671).
- **Per the stop rule:** no "state-specific" wording. Details: `results/am42/README.md`.

### Amendment 44 (run 53, PRE-data): confirmatory test of the state-dependent part of confidence, after the paper-1 review's answer-identity confound (R53)
**Why.** Paper 1 (`EXfgvby_s1tnLjmd`) was rejected: |m| discards which answer was chosen, and answer identity predicts correctness (choosing sentence 1 is almost always right). R53 (`results/r53/README.md`, EXPLORATORY) confirmed and sharpened it on the existing rows:
- Leave-one-concept-out, the ON confidence adds **nothing** to [answer, |m|, |m|², answer×|m|] in any of 8 cell × frame combinations (Δ log-loss +.003 to +.009). The pooled within-stratum AUROC is .534 [.488, .580]. The T2 "beyond first-order evidence" claim is withdrawn.
- But the transcript-only (NONE) confidence enters with a *negative* sign, and ON with a near-equal positive one. The difference **d = conf_own − conf_none_own** (same transcript and answer, state re-applied vs not) adds held-out information in all 6 cells with a NONE arm (Δ AUROC +.07 to +.20; within-stratum AUROC .67–.82, CIs above .5).
That pattern was found post hoc after several tests (forking), so it needs fresh data before any wording uses it.

**Design.** Identical to Am. 43 except the concept list.
- **Kit:** `type2_41b.py`, unchanged except a `--concepts am44` key.
- **Cell:** Qwen3-8B bf16, L9 α8, N = 2, frames intro (primary) and ordinary; concept arm and matched-norm random arm, same job.
- **Concepts:** 60 new `AM44_CONCEPTS` (buildings, birds, weapons, toys, astronomy, writing tools...), disjoint from every quoted word in `notes/wp3/*.py` (checked by script, run 53).
- **Job:** `am44-statecheck-8b-01`. Hardware not fixed (eval-only).
- **Smoke test:** 0.6B, CPU, 2 concepts, format only.

**Analysis.** `type2_44_analyse.py`, written before data; its docstring fixes D44 (within-stratum AUROC of d beyond a quadratic margin), CV44 (leave-one-concept-out Δ log-loss of base + d vs base, stratified permutation p, 1000 perms) and ON44 (descriptive). SESOI .05 on AUROC. **Disclosure:** I ran it once on the Am. 43 rows to test the code; those values are R53c's and known.

**Rival ledger (H33).**
- (FIRST-ORDER-ONLY) All correctness information in the confidence report is answer identity + margin: D44 and CV44 not SHOWN in either arm.
- (STATE-CHECK) The state-induced change in confidence evaluates the named answer against the live state: D44 and CV44 SHOWN in both arms (perturbation-generic, as Am. 43).
- (FORK-FLUKE) R53c is a forking artifact: not SHOWN in the concept arm.
- (RE-READOUT, not separable here) d is a second first-order localisation readout taken at confidence time. Both STATE-CHECK and RE-READOUT predict the same data; the paper must say so.

**Predictions (written first).**
- G41 passes in both arms: ~80%.
- Concept arm, intro: D44 SHOWN ~65%; CV44 SHOWN ~60%; ON44 Δ log-loss ≥ 0 (no increment) ~70%.
- Random arm, intro: STATE-CHECK SHOWN ~55%.
- Ordinary frame, concept arm: STATE-CHECK SHOWN ~50%.

**Stop rules.**
- One registration; no new wordings, layers, α or covariates after data.
- "Monitoring" wording is withdrawn whatever the outcome (the review's point 2 and R53).
- A revised paper may say "the state-induced change in confidence carries correctness information beyond the answer's identity and margin" only if STATE-CHECK is SHOWN in the concept arm, intro frame; it must name RE-READOUT as unexcluded.
- If not SHOWN, the revision reports the negative: the confidence report adds nothing beyond first-order evidence once answer identity is controlled; only the swap result (SWAP-A, R53b) stays.

### Amendment 44 result (collected 02:51Z 2026-10-23, run 53; job `am44-statecheck-8b-01`, 11.8 min ≈ $0.07, spot eu-west-3b g6.xlarge L4 $0.3547/h)
- G41 PASS in all four arm × frame cells. `type2_44_analyse.py` applied exactly. Intro: choosing sentence 1 right 14/14 (concept), 13/13 (random).
- **Concept, intro (primary):** D44 .711 [.629, .780] **SHOWN**; CV44 Δ log-loss −.089, p < .001 **SHOWN** → **STATE-CHECK SHOWN**. ON44 +.006 (p .44): raw confidence adds nothing, as predicted.
- **Concept, ordinary:** D44 .835 [.725, .930]; CV44 −.074, p < .001 → SHOWN. ON44 +.009.
- **Random, intro:** D44 .735 [.661, .807]; CV44 −.120, p < .001 → SHOWN. ON44 −.000 (p .13).
- **Random, ordinary:** D44 .683 [.555, .815]; CV44 −.040, p < .001 → SHOWN. ON44 +.002.
- **Rivals:** FIRST-ORDER-ONLY and FORK-FLUKE falsified; STATE-CHECK supported, perturbation-generic (random arm too). RE-READOUT not excluded (by design).
- **Stop rules:** a revision may say "the state-induced change in confidence carries correctness information beyond the answer's identity and margin", naming RE-READOUT as unexcluded; no "monitoring" wording; the raw report's T2 claim stays withdrawn.
