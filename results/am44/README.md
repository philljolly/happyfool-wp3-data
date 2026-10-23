# Am. 44 result (run 53): the state-dependent part of confidence, confirmatory

Job `am44-statecheck-8b-01` (Qwen3-8B bf16, L9 α8, 60 new concepts, concept + random arms; spot eu-west-3b g6.xlarge L4, 11.8 min ≈ $0.07). Scored with `type2_44_analyse.py` (pre-data) → `analysis.json`. d = conf_own − conf_none_own.

| Arm, frame | D44 (within "sentence 2", d beyond quadratic margin) | CV44 Δ log-loss (LOCO; perm p) | ON44 (raw conf) | STATE-CHECK |
|---|---|---|---|---|
| concept, intro (primary) | .711 [.629, .780] | −.089 (< .001) | +.006 (.44) | **SHOWN** |
| concept, ordinary | .835 [.725, .930] | −.074 (< .001) | +.009 | SHOWN |
| random, intro | .735 [.661, .807] | −.120 (< .001) | −.000 (.13) | SHOWN |
| random, ordinary | .683 [.555, .815] | −.040 (< .001) | +.002 | SHOWN |

All registered predictions came true (priors 50–70%). Reading: the raw confidence report carries no correctness information beyond the chosen answer and margin, but the change the live state makes to confidence on a fixed transcript does, for random directions as much as concepts. Not excluded: RE-READOUT (the confidence query re-localises the still-present perturbation rather than judging the first answer).
