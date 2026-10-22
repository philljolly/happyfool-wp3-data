# Route 2 (MR40): mediation of report localisation through the decodable direction. DESIGN DRAFT, run 48, NOT pre-registered

**Gate.** Run only at a setting where Am. 40 gives R SHOWN and Q = CROSSING (S4). C need not be SHOWN, but the verdict's scope then reads "anomaly route", not "content route".

**Question.** Does the report's dependence on the injection pass through the location direction that the answer-position probe reads (NIE)? Or does it pass through something else (NDE), such as attention-entropy or norm effects?

**Manipulation (eval-only, one forward pass per arm).** Work at the answer position, at a mid layer l_m between the injection layer and the last layer, chosen by rule: the layer with the highest single-state probe accuracy among {L+4, L+8, last−4}, fitted on half the concepts.
- **(i) Ablate.** Project out d̂ from the injected run's residual, where d̂ is the unit mean difference h(target 0) − h(target 1), fitted on the other half of the concepts.
- **(ii) Patch.** Replace the d̂ component with its value in the clean run.
- **(iii) Control.** Ablate a random unit direction, and also the top-1 PC of the clean-run variance, chosen so its variance matches d̂'s.

**Estimands.** AUC_m (prior-free) in the intro and ordinary frames, compared before vs after (i)/(ii), with a concept-bootstrap CI.
- NIE share = (AUC_inj − AUC_ablated) / (AUC_inj − .5).
- **Slots:**
  - MEDIATED iff the share's CI lower bound is > .5 and control (iii) changes AUC by < .05.
  - NOT MEDIATED iff the share's CI lies within ±.2.
  - Otherwise PARTIAL / NOT SHOWN.
- **Prediction from the last-mile geometry** (< 7% of |d|² along u): ablating d̂ removes most of the report effect, because the small aligned component is part of d. Rival: the report reads a different signal, such as entropy, so the share is ≈ 0.

**Kit.** A new hook in `inject_loc.py` style (`with_injection` plus a second hook at l_m on the last position). 0.6B runs locally on CPU (~15 min for 40 concepts × 2 frames × 4 arms). Larger scales need one remote job.
