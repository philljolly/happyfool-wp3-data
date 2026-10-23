# Am. 43 results: confirmatory 8B type-2 cell, concept and random arms (run 52)

**Job `am43-confirm-8b-01`.** Spot, eu-west-3 (eu-west-3b), g6.xlarge L4, $0.3558/h. Started 22:00:20Z, finished 22:12:24Z 2026-10-22 (12.1 billed min, ≈ $0.07). Collected 00:19Z 2026-10-23 (run 52). Qwen3-8B bf16, L9 α8, N = 2, 60 new `AM43_CONCEPTS`. Rows: `t43_8b_concept.json`, `t43_8b_random.json` (also on `wp3-results/am43-confirm-8b-01`). `type2_43_analyse.py` applied exactly (`analysis.json`). Values are [point, CI low, CI high, kept resamples].

## Gate G41: PASS in all four arm × frame cells
concept intro 78/42 correct/incorrect, ordinary 91/29; random intro 69/51, ordinary 86/34; clean conf SD .74/.75.

## Registered slots

| slot | concept arm | random arm |
|---|---|---|
| **T2** (intro AUROC2r) | .649 [.569, .729] **SHOWN** | .607 [.549, .670] **SHOWN** |
| **STATE-DEPENDENT** (X = ON − NONE, intro) | +.294 [.207, .391] **SHOWN** | +.191 [.132, .260] **SHOWN** |
| **SWAP-R** (β on own-correct, quadratic margin; both frames) | intro +.86 [.61, 1.11]; ordinary +.29 [.04, .53] **SHOWN** | intro +.68 [.47, .88]; ordinary +.28 [.01, .55] **SHOWN** |
| **CONTENT** (concept − random AUROC2r, intro, paired) | +.041 [−.034, .124] **INCONCLUSIVE** | |

Also (not slots): acc1 concept intro .650, ordinary .758; random intro .575, ordinary .717. Ordinary-frame own-answer AUROC2r concept .447 [.340, .558], random .503 [.407, .594] (low power with ~30 incorrect, as in Am. 41b).

## Predictions (written first)

| prediction | prior | outcome |
|---|---|---|
| G41 passes in both arms | ~75% | yes |
| concept T2 SHOWN | ~55% | yes |
| concept STATE-DEPENDENT SHOWN | ~50% | yes |
| concept SWAP-R SHOWN | ~50% | yes |
| random T2 SHOWN | ~45% | yes |
| random SWAP-R SHOWN | ~40% | yes |
| CONTENT SHOWN | ~10% | no |
| CONTENT EQUIVALENT | ~15% | no (INCONCLUSIVE) |

## Rival ledger
- **FLUKE-41b:** falsified (concept T2 SHOWN on a third concept set).
- **MARGIN-SWAP:** falsified. The swap effect survives a pre-registered quadratic margin control in both frames and both arms.
- **PERTURBATION-GENERIC:** all its predictions met (T2, STATE-DEPENDENT and SWAP-R SHOWN in both arms; CONTENT not SHOWN).
- **CONTENT-SPECIFIC:** not shown. CONTENT is INCONCLUSIVE (H51: weakly BOUNDED; the CI excludes concept exceeding random by more than .124 and random exceeding concept by more than .034; it does not reach the ±.05 equivalence bound).

## Stop rules applied
- "Monitoring" wording (H50.3) is **allowed** for paper 1: the concept arm has T2, STATE-DEPENDENT and SWAP-R all SHOWN.
- The correctness-not-ownership claim **stays** (SWAP-R SHOWN).
- **No "content-specific" wording** (CONTENT not SHOWN).
- With Am. 42 (D INCONCLUSIVE) and H52 (holiday 41): equal AUROCs across injected / random / natural error sources are "comparable", not "the same detector"; transfer is untested.

## Reading (scoped)
At Qwen3-8B, L9 α8, one wording: the model's verbal confidence in a shown localisation answer tracks that answer's correctness beyond a quadratic first-order-margin control, needs the perturbed state to be present in the pass, and follows correctness rather than ownership. All of this holds as much for a matched-norm random vector as for a concept vector. What is monitored is whether a localisation of *a perturbation* was right, not which concept was injected. Third independent concept set (Am. 41, 41b, 43).
