# Am. 40 results: controlled rung map across scale (run 49)

Jobs: `am40-rung-small4b-01` (0.6B, 1.7B, 4B; fp32), `am40-rung8b-a-01` and `am40-rung8b-b-01` (8B; bf16). All spot, g6.xlarge L4, eu-west-3b. Full rows (with hidden states) are on the `wp3-results/<job-id>` branches. `slim/` holds every row without hidden states (3 MB). `rung40_<scale>.json` is `rung40.py` applied exactly as pre-registered. `summary.txt` is `summ40.py` over them.

**Deviation D40.1:** the collector dropped all 12 4B random-arm files (size limit). They were re-run unchanged as `am40-rung4b-random-02` (spot, 14.3 min, $0.08; status and provenance in `results/am40-rung4b-random-02/`) and scored at 16:01Z. `rung40_4b.json` is the full 4B analysis; `rung40_4b_conceptonly.json` was the interim R/Q read (identical R and Q).

**Instrument gates.** Every concept file passes the mass gate (mean log index mass ≈ 0 in all frames, equal to no-injection). Answer-position probe ≥ .75 at some setting at every scale, so no "state not decodable" floor.
- **8B caveat:** 8B is bf16, so logits come quantised (steps of .125–.25). Many m values tie (ties count .5 in AUC), and the impact medians are coarse (e.g. .25 vs .25), which makes C's impact matching coarse at 8B.

## Registered slots (intro primary; AUC_m point [95% CI])

| scale | setting (depth) | R | Q | C (intro) | concept intro / ordinary | matched random intro / ordinary | probe (intro) |
|---|---|---|---|---|---|---|---|
| 0.6B | L12 α8 (.43) | SHOWN | **CROSSING** | NOT SHOWN +.05 [−.09, .21] | .76 / .67 | .71 / .54 | .94 |
| 0.6B | L19 α2, α8 (.68) | NOT SHOWN | POLARITY INVERSION* | NOT SHOWN / EXCLUDED | .34–.36 / .71–.81 | same | .96–1.00 |
| 1.7B | L12 α8 (.43) | SHOWN | **SALIENCE PULL** | **SHOWN +.24 [.13, .36]** | .89 / .06 | .65 / .60 | .93 |
| 1.7B | L12 α2 | NOT SHOWN | SALIENCE PULL | SHOWN +.06 [.00, .14] | .57 / .34 | .51 / .42 | .88 |
| 4B | L9 α8 (.25) | SHOWN | **CROSSING** | NOT SHOWN −.00 [−.08, .07] | .62 / .68 | .63 / .70 | .65 |
| 4B | L16 α8 (.44) | SHOWN | UNDETERMINED | NOT SHOWN −.01 [−.13, .13] (neutral +.22 [.11, .34]) | .77 / .57 [.47, .68] | .78 / .68 | .90 |
| 4B | L24 α8 (.67) | EXCLUDED | SALIENCE PULL (tiny) | EXCLUDED | .55 / .46 | – | .96 |
| 8B | L9 α8 (.25) | SHOWN | **CROSSING** | **SHOWN +.08 [.02, .14]** | .92 / .91 | .85 / .84 (2α: 1.00 / 1.00) | .90 |
| 8B | L16 α2, α8 (.44) | NOT SHOWN | UNDETERMINED | NOT SHOWN, **−.35** | .31–.41 / .41–.97 | .66–.76 / .74–.93 | .80–.97 |
| 8B | L24 α2, α8 (.67) | EXCLUDED | POLARITY INVERSION* | EXCLUDED | .46–.48 / .53 | same | .88–.96 |

All other settings: R EXCLUDED or NOT SHOWN, Q UNDETERMINED (see `summary.txt`).
\* Descriptively this is **question-blind avoidance**, not an inversion. The model picks the clean sentence under both questions, by a small margin at 8B L24 and a large one at 0.6B L19. The random arm does the same.

## Per scale

- **Highest rung:**
  - 0.6B: Q (CROSSING), C not shown.
  - 1.7B: C SHOWN, but Q is a SALIENCE PULL (a content-sensitive, question-blind pointer).
  - 4B: Q (CROSSING at L9 α8); C NOT SHOWN (the matched random crosses equally).
  - 8B: **CROSSING + C SHOWN** at L9 α8.
- **R-boundary depth** (deepest depth with R SHOWN): 0.6B .43, 1.7B .43, 4B .44, **8B .25**.

## Rival ledger (H33), as registered

- **SAL** (forbids CROSSING): **falsified** at 0.6B, 4B and 8B. It fits 1.7B.
- **CONT** (CROSSING and C SHOWN somewhere; forbids SALIENCE PULL at every R-SHOWN setting):
  - Its positive prediction is **met at 8B L9 α8**.
  - At 1.7B the only R-SHOWN setting is a SALIENCE PULL, which CONT forbids at that scale.
- **POL:** the slot fires at 0.6B L19 and 8B L24 only with R not shown. Descriptively these are avoidance, not inversion. **Not supported.**
- **LM** (probe ≥ .9 with R EXCLUDED at depth .67): **holds** at 4B (.96, EXCLUDED) and at 8B (.96, EXCLUDED). Not forbidden-pattern-violated.
- **SCALE** (the R-boundary is non-decreasing in scale; forbids R-boundary(8B) < R-boundary(1.7B)): **falsified**, .25 < .43. At 8B, mid-depth (L16) concept injection is *avoided* (intro .31–.41) while a matched random vector is localised (.66–.76). That is C = −.35, a content-specific, wrong-signed response, the 8B analogue of the 4B L20 avoidance (Am. 31–35).

## Predictions (written first) vs outcome

| prediction | outcome |
|---|---|
| 4B R SHOWN at L16 α8, ~75% | yes (.77) |
| 4B Q = SALIENCE PULL there, ~55%; CROSSING ~20% | no: UNDETERMINED (ordinary .57 [.47, .68]) |
| 4B C SHOWN ~25% | no (NOT SHOWN at both R-SHOWN settings) |
| 8B R-boundary deeper than 4B, ~35% | no: shallower (.25) |
| 8B CROSSING somewhere, ~35% | **yes** |
| 0.6B no R SHOWN, ~60% | no: R SHOWN at L12 α8 |

## Exploratory, post hoc (flagged; not registered)

1. **Impact-interpolated C** (random AUC interpolated on log impact between α and 2α, at the concept's impact; point only):
   - 8B L9 α8: +.056 (registered C +.078). It shrinks but stays positive. 8B impact is coarsely quantised.
   - 1.7B L12 α8: +.25. Here the concept's impact (8.1) exceeds both random arms' (6.1, 7.5), so this is an extrapolation. Random at 2α is *worse* (.65) than at α, so more impact doesn't buy localisation there.
   - 0.6B L12 α8: +.08.
   - 8B L16: −.31 / −.37.
2. **H49 SDT split** (`sdt49_am40.json`):
   - **8B L9 α8:** the *emitted* answers cross. Raw d′ is +1.37 under intro and −1.10 under ordinary, with c +1.28 / −0.12. So at 8B the crossing is in the answers themselves, not only in the logit margin. Matched random: +.69 / −1.04.
   - **0.6B L12 α8:** raw intro d′ = 0 (every intro answer is "2", c 1.79). The crossing lives only in the margin (d′_AUC +.98 / −.63).
   - **4B L9 α8:** raw intro d′ = 0, ordinary −.63. Margin +.45 / −.66.
   - **1.7B L12 α8:** raw d′ +1.38 / +1.01. Both questions pick the injected sentence, which is the pointer.
3. **ρ = d′_AUC(report, intro) / d′_probe** at the R-SHOWN concept cells: 0.6B L12 α8 .32; 1.7B L12 α8 .60; 4B L9 α8 .58, L16 α8 .40; 8B L9 α8 .79. At the CROSSING cells, the share of the decodable signal that the report uses rises with scale (.32 → .58 → .79). This is one cell per scale, uses probe accuracy (not AUC), and is not registered.

## Reading (scoped)

- In untrained Qwen3 models, a **question-sensitive** localisation report (CROSSING) exists at 0.6B, 4B and 8B, at shallow-to-mid injection with a large dose.
- At 8B it also beats an impact-matched random vector (C +.08). That margin is small, partly an impact-matching artefact (interpolated +.06), and a random vector at 2α localises perfectly.
- So the strongest supportable statement is **at the C rung, borderline, at one 8B setting**: content-specificity beyond salience is shown by the registered rule and is fragile under exploratory impact interpolation.
- 1.7B's report is content-sensitive but question-blind, a natural pointer.
- Deeper injections are decodable at the answer position (probe ≥ .9) and unreported at every scale (last mile).
- Depth does not grow with scale. At 8B mid-depth, concept injections are actively avoided.
- 4B: the crossing is matched by the random arm (C −.00), so 4B's highest rung is Q, as at 0.6B. The type-2 question (MR41, Am. 41) is next.
