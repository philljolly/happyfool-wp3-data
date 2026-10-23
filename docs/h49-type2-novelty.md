# H49.4 novelty search: type-2 SDT / meta-d' and LLM self-report (run 49, snippets only)

Two web searches (15:34Z). Everything below is from search snippets: **UNVERIFIED**, nothing read in full.

- **Nearest relative (R1): Cacioli 2026, "Do LLMs Know What They Know? Measuring Metacognitive Efficiency with Signal Detection Theory", arXiv 2603.25112.** meta-d' / M-ratio on factual QA across four open models (224k trials); metacognitive efficiency differs at matched type-1 d'; temperature moves the type-2 criterion, not meta-d', in two models. Domain: knowledge confidence, not injected internal states.
- Quantisation and metacognitive geometry (arXiv 2604.08976): M-ratios by domain for Llama-3-8B-Instruct. Same domain (QA confidence).
- Dai 2026 (verbalised-confidence meta-d', M-ratios .62–.92; scale-format effects): cited in a snippet, no link.
- LessWrong, "LLM introspection research is missing Nisbett and Wilson's ..." (`lesswrong.com/posts/6d3ffDFt9tBYdiaoe`): argues for meta-d'/d' and interventional tests in introspection research. **Closest in spirit to H49; must be read in full before any priority claim.**
- 2026 critique (snippet; likely Singh, Linzen & Ravfogel 2605.26242, already in the notebook) that anomaly detection and introspection make the same prediction in steering-awareness setups.
- No snippet applies type-2 SDT to *injected-state localisation* with a salience-matched random control, and none uses the blindsight criterion (type-1 positive, type-2 null). Status: **NOT FOUND AS OF 2026-10-22 (channel: two standard web searches, snippets only)**, not "novel".

Next before an Am. 41 prereg: read the LessWrong post and Cacioli 2603.25112 in full.

## Run 49 follow-up (15:47Z): both nearest relatives read via a WebFetch summary of the full page (not line by line; numbers as the summary gave them)
- **Cacioli 2603.25112** (Llama-3-8B Instruct/Base, Mistral-7B-Instruct, Gemma-2-9B; TriviaQA + NQ; 224k trials). Type-1 and type-2 use **the same signal**: the normalised log-probability of the model's own answer, binned for confidence. M-ratio .85–1.05; temperature shifts the type-2 criterion, not meta-d′, for two models. No internal states, no injection. **Exactly MR41 objection (i):** confidence = the first-order decision variable, so M-ratio ≈ 1 by construction is unsurprising. Am. 41's separate second-turn confidence, partialled on the first-order margin, is a different estimand.
- **LessWrong post (Dinoraptor101 & "Rena", 2026-08-21, auto-flagged as LLM-written):** literature review plus two unrun protocols. It proposes meta-d′/d′ for *longitudinal self-prediction with a blinded observer arm*, not for injection. Its line: "The LLM introspection literature is reinventing detection rates while a mature metric sits one field over." It contains no data, no wagering and no blindsight. It also cites an "Aliveness Marker Registry v1.0" (Negai & Negai 2026; UNVERIFIED) with a meta-d′ "calibration lag" marker. It reports Godet reaching 98% localisation at layer 18 in gemma3-27b-it (vs 20% chance; UNVERIFIED secondary).
- **R1 verdict for Am. 41:** the nearest relatives apply type-2 to QA confidence from one signal (Cacioli) or propose it for self-prediction (LW). Type-2 on injected-state localisation, with a separate confidence turn, margin partialled, and a salience-matched random arm, is **NOT FOUND AS OF 2026-10-22** (channels: 2 searches + these 2 reads). Credit both in any Am. 41 write-up.

## Run 50 (H50.5): nearest relative of the Am. 41b swap arm
- **Lindsey 2025, "Emergent introspective awareness in large language models"** (transformer-circuits.pub/2025/introspection; read via a WebFetch summary of the first 100k of ~400k characters, so PARTLY READ). Prefill detection: the reply is prefilled with an unnatural word; the model disavows it next turn. Injecting the prefilled word's concept vector *before* the prefill makes the model accept it as intended. Strongest in Opus 4/4.1; most sensitive at an earlier layer than the other introspection effects. No rates were visible in the part read. No own-answer confidence experiment was visible.
- **Relation to Am. 41b.** This is the closest prior design to my swap arm: both ask whether the model's judgement of an answer shown as its own depends on its earlier activations. My additions: (i) a graded type-2 confidence on a forced-choice localisation, residualised on the first-order margin; (ii) own vs swapped answers balanced within trial, with the ownership bias b estimated separately; (iii) a frame (ordinary) where correctness and "the named item is the perturbed one" come apart, which separates monitoring from re-reading the perturbation. Lindsey's design can't separate (iii): there the injected concept and the prefilled word coincide by construction.
- Added to the Am. 41b R1 list as a pre-collection note (prereg.md), not as a change to any slot.

## Run 51: Lindsey 2025 read in full (R1 closed for Lindsey; Cacioli still pending)
Lindsey, "Emergent Introspective Awareness in Large Language Models", Transformer Circuits Thread, 29 Oct 2025 (rev. 1 Jan 2026), https://transformer-circuits.pub/2025/introspection/index.html. A reader subagent converted the whole HTML (main text and appendices) to text and read it. Most numbers are in figure images: the reader opened three prefill figures and read their values by eye (approximate). The other figures are UNVERIFIED. I checked 4 quotes with `grep -F` (one match each). A search for AUROC, meta-d or type-2 in the text finds 0 hits.

**Prefill experiment.**
- **Procedure.** A random word is prefilled as the answer to "Tell me what word you think about when you read the line below". The model is then asked "Did you mean to say that, or was it an accident?", and a Claude grader codes the reply as intended or apology. Verified: "We measured the “apology rate” – the fraction of times the model claims the response was accidental or unintentional."
- **Conditions.** No injection; the prefilled word's concept vector injected on the sentence tokens; and a different word's vector (verified: "Injecting the concept vector corresponding to a random other word on the sentence tokens."). Injecting after the prefill has no effect.
- **Results.** Opus 4.1's apology rate falls from ~80% to ~24% at strength 16 with the matching concept, while the random-concept trials stay ~75–85%. Verified: "the peak layer is slightly over halfway through the model".

**Comparison with Am. 41/41b.**
- Lindsey measures binary acceptance (ownership), not graded confidence scored against correctness.
- It has no first-order margin measure or control.
- The prefill control is a different concept at the same strength, not norm-matched random. The norm-matched control appears only in the injected-thoughts experiment (verified: "requiring a larger norm (norm-matched to a concept vector injected at steering strength 8)").
- **There is no type-2 analysis anywhere in the paper.**

**Novelty verdict.**
- **Not new:** that retroactive injection changes whether a model owns an output (Lindsey has it, with a different-concept control). My swap arm is close kin.
- **New, as far as this paper goes:**
  - type-2 sensitivity (AUROC2) of a separate confidence report on injected-state localisation;
  - residualisation on the first-order margin;
  - the NONE re-encode showing that the signal needs the live perturbed state;
  - the ownership-balanced swap (Dbal) that separates correctness from ownership. Lindsey can't separate them, because acceptance *is* his ownership measure.
- **Caveats:** open weights (Qwen3-8B) vs Claude, so not like-for-like; and this says nothing about the wider literature.
- **Still to read before a public type-2 claim:** Cacioli 2603.25112, in full.

## Run 51: Cacioli 2603.25112 read in full (R1 closed)
Cacioli, "Do LLMs Know What They Know? Measuring Metacognitive Efficiency with Signal Detection Theory", arXiv:2603.25112, v3 dated 28 Jul 2026. Pre-registered (osf.io/5q7mt). A reader subagent read the HTML of v3 from the abstract to the conclusion. The appendix contents are UNVERIFIED. I checked 4 quotes with `grep -F` (one match each).

**What the paper does.**
- **Models and data.** Llama-3-8B (Instruct and Base), Mistral-7B and Gemma-2-9B, all Q5 quantised. TriviaQA and NQ-Open, 224k trials.
- **Confidence measure.** The answer's own length-normalised log-probability. Verified: "treating token-level normalised log-probability as a graded confidence variable". No separate report; the author's own caveat, verified: "NLP is a fluency measure, not a pure metacognitive signal".
- **Results.** AUROC2 at T = 1 on TriviaQA is .76–.85. meta-d′ and the M-ratio were dropped in v3 (verified: "so M is pinned near 1 by construction"). There is no margin control and no state manipulation. Verified: "we do not claim LLMs possess metacognition phenomenologically".

**What I take from it.**
- **Novelty:** a separate verbal confidence report, the residualisation on the first-order margin, the injected state with NONE and swap controls, a 2AFC task where meta-d′ is defined, and the natural-task baseline are all absent from it.
- **Methods to reuse:** permutation nulls, the z-ROC slope check, and the warning that the M-ratio can just track unequal variance.
- **No prior for Am. 42.** Its .76–.85 comes from the answer's own log-probability on recall. A margin-residualised separate report should sit well below that, which is consistent with Am. 42's NAT SHOWN ~45% (this is my inference).

**R1 status:** both Lindsey 2025 and Cacioli are now read in full. The condition for a further public type-2 claim (read both before claiming) is met. Claims stay scoped as before.

## Run 52 R1 reopen (paper 1 PR prerequisite): three more relatives read in full

Reader subagents read the full texts (arXiv HTML). I checked five quotes with `grep -F` against the saved texts (one match each).

- **Ferrara 2026, "Open-Weight Masked Introspection" (OWMI), arXiv 2608.20569v1.** This is the closest precedent. It was VERIFIED only for selected sections in run 44; now it is read in full.
  - **The finding.** In Qwen2.5-7B-Instruct, verbalized confidence on a yes/no "did anything change" detection report gives a selective-prediction AUROC of .647 [.577, .719] (n = 190), where the discrete answer gives .500. GLM-4-9B gives .587; no other model is above chance.
  - **Not a type-2 measure for Qwen.** "Its detection report is constant", so the AUROC is the discrimination of intervention from sham by confidence.
  - **Interventions.** Hooks only (zeroing, scaling, noise, replacement); "these interventions carry no ground-truth concept label".
  - **Missing controls.** No margin residualisation, no re-encode-without-intervention control, no swap/ownership control, no random-vs-concept test of confidence, no natural-difficulty baseline, and no Qwen3-8B.
  - **A point a reviewer could press.** OWMI argues that "a norm-matched control equates how large the perturbation was rather than how much damage it did". Our random arm is norm-matched, so the paper must say this.
  - **Verdict.** No claim of paper 1 is pre-empted. Cite OWMI as the nearest result: the confidence channel carries signal the discrete answer lacks, on detection.
- **Park, Meyerson, Qiu & Miikkulainen 2026, "Fine-Tuning Language Models to Know What They Know" (ESMA), arXiv 2602.02605.**
  - A separate meta question in its own context ("Isolating these inquiries within independent contexts"), scored as type-2 AUROC on TriviaQA (e.g. .672 → .751 at 1.5B after training).
  - Confidence is not margin-residualised, and there is no steering.
  - **Verdict.** Precedent for the dual-question / type-2 AUROC protocol on natural QA only.
- **Kumaran et al. 2026 (Nature Machine Intelligence), "Causal Evidence that Language Models use Confidence to Drive Behavior", arXiv 2603.22161.**
  - Verbal confidence (a separate pass) predicts abstention beyond calibrated log-prob.
  - Its AUROC for correctness is .53–.60 (verbal) vs .63–.75 (calibrated). Verified: "no residual correctness signal predicted abstention beyond the two confidence measures".
  - Steering uses confidence directions aimed at behaviour, not a concept the model reports on.
  - **Verdict.** Precedent for verbal confidence carrying information beyond log-probs. Not pre-emptive.

**Updated R1 summary.** Margin-residualised verbal-confidence type-2 on a report *about an internal perturbation*, with NONE, ownership-swap, random-vector and natural-baseline controls, is not in Lindsey 2025, Cacioli 2603.25112, OWMI, ESMA or Kumaran et al. The paper's novelty sits in those controls, not in "LM confidence tracks perturbation" (OWMI) or "type-2 AUROC of a separate confidence" (ESMA).
