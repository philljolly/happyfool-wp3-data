# R49: H49.1/H49.2 SDT split of the run-48 0.6B preliminary cells (exploratory, post hoc)

`sdt49.py results/r49/sdt49_p06.json ...` on `results/r48/p06/slim/*` (Qwen3-0.6B, old concepts, n = 60 rows/cell, concept bootstrap 2000). Signal = injection in sentence 1. d' > 0 = the answer picks the injected sentence. c > 0 = bias to answer "2". d'_probe = 2 z(probe accuracy at the answer position, from `rung_prelim.json`).

Key cell, L12 α8:

| arm | frame | raw d' | raw c | prior-free d' | prior-free c | d'_AUC |
|---|---|---|---|---|---|---|
| concept | intro | .34 [.00, .66] | 1.90 [1.68, 2.14] | .37 [.06, .74] | −.34 | .58 [.14, 1.06] |
| concept | ordinary | −.70 [−1.22, −.34] | .91 | −.29 [−.59, .00] | −.04 | −.71 [−1.07, −.42] |
| random | intro | .00 (all answers "2") | 2.14 (floor) | .72 [.41, 1.15] | .09 | 1.16 [.79, 1.71] |
| random | ordinary | −.86 [−1.16, −.34] | 1.53 | −.23 [−.59, .12] | .08 | −.37 [−.78, .02] |

Reading:
1. **The run-48 crossing is a sensitivity sign flip, not a criterion shift (H49.1 passes for the concept arm).** d'_AUC goes from +.58 (intro, picks injected) to −.71 (ordinary, picks clean), both CIs off 0. The prior-free c stays near 0 in both frames.
2. **The model's actual answers are criterion-dominated.** Raw c is +0.9 to +2.1 in every cell: the raw answer is nearly always "2", and in several intro cells *always* "2" (raw d' = 0 by construction). So the localisation lives in the graded logit margin; it reaches the emitted answer only after removing a large position prior. "The report localises" must be stated as "the answer's logit margin localises; the emitted answer is bias-bound" (Azzopardi & Cowey's point, in an LM).
3. **ρ (H49.2, exploratory):** concept L12 α8 ρ = .23 (d'_AUC .58 vs d'_probe 2.56); random L12 α8 ρ = .52; L7 α8 concept .03. The report uses a quarter of the linearly decodable signal at the best concept cell.
4. Random's raw ordinary d' (−.86) exceeds concept's (−.70) in the same direction: the ordinary-frame "correct" picks partly reflect an avoid-the-perturbed bias that a random vector also triggers (consistent with the run-48 "question-sensitive anomaly reading").

Not a registered result. Am. 40 slots stand as written; these numbers are added beside them per H49.
