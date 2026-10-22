# K design (MR37 M2): content-conditional choice. Draft, run 45; NOT pre-registered

**Why.** Q (does the answer change when the question's polarity is reversed?) can be passed by a detector plus a switch on the wording (H42; holiday 36). K asks whether the report depends on *what* was injected. Both sentences carry an injection at matched norm, and the question names one category. A pointer, or any rule blind to category, is at chance by construction.

**Kit.** `k_probe.py` + `k_analyse.py` (both smoke-tested on 0.6B, run 45; mechanics only).
- Pairs: AUDIT36B animals[:30] × foods[30:60], two test-bank sentences per pair, and both assignments.
- Arms: both injected; animal only; food only. The food vector is rescaled to the animal vector's norm.
- Frames: k_animal, k_food, neutral.
- Prime-only arm (H43): the no-injection logits of each frame. Every pick is bias-corrected by them.
- Saved hidden states: both-arm neutral frame, mid layer and final layer.

**Planned estimands** (fix in the pre-registration before any 4B data).
- **K** = P(pick animal sentence | k_animal, both) − P(pick animal sentence | k_food, both). Concept-pair bootstrap.
  - A content-conditional reader gives K → +1.
  - A pointer or a category-blind rule gives K ≈ 0.
  - A wrong-signed reader gives K < 0. Per H39 that counts as access, not loss.
- **Salience asymmetry (H43a)** = P(pick animal | neutral, both) − .5, and the same from the single arms.
  - If |asymmetry| ≥ .15, K is reported stratified by assignment. A category-salience account gets its own row in the rival ledger.
- **Feasibility probe**: does a leave-pair-out logistic probe on (h_both_neutral − hnone_neutral) read where the animal is?
  - H22/D4: if the untrained probe can't read it (≤ .60), K on any adapter is uninterpretable. The next step is then a K-trained positive control, not a K test.
- **Interference**: in the single-injection arms, compare the k_animal pick of the injected sentence when the injection is the animal vs the food (this is C under a category question).

**Order.**
1. Am. 38 (M1 gate).
2. Untrained K feasibility (U only, L24 α2 and L20 α1, float32): one remote job, cheap.
3. Only if M1 passes and the probe reads location: K on M, with a K-trained control.

**Caps.** A K pass = "content-conditional access to an injected state". It does not show self-access; that needs H23's own-vs-foreign arm.
