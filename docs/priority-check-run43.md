# Priority check for the pointer paper (run 43, reader subagent; quotes are WebFetch extractions, NOT raw text: spot-check against arXiv HTML before citing)

Question: has any work tested trained introspective reports under a polarity-reversed question (e.g. "which sentence is NOT injected / ordinary?")?

| Paper | Trained task | Reversed question | Random-vector control at test | Held-out concepts |
|---|---|---|---|---|
| Hahami, Sinha & Jain 2026, IFT, arXiv:2607.14111 | 2AFC-style sentence localisation ("Which sentence has the injected thought?"), plus a Sem+Rsn arm | none found; their factual-no control was run on untrained Llama-3.1-8B only | Gaussian is a *training* arm (Llama-1B 14.9% vs 9.6% baseline); no test of IFT models under random injection found | yes (950 train / 50 held out) |
| Fonseca Rivera 2025, arXiv:2511.21399v1 | yes/no + naming | none | none (only α = 0) | 20 held out |
| Fonseca Rivera & Africa 2026, "Steering Awareness", 2511.21399v3 | yes/no + naming | none; nearest is "inject A, ask about B" (89%) | Gaussian is a trained negative (12.5% of data); rejects it 94% (Gemma 2 9B) | 121 OOD |
| Macar, Yang, Wang, Wallich, Ameisen & Lindsey 2026, arXiv:2603.21396 | yes/no (bias vector, DPO/SFT) | none ("DPO reversed" swaps preference pairs, not the question) | none at injection | 100 held out |
| Lindsey 2025 (untrained) | – | none for detection; yes-bias control on unrelated questions | random vectors 9/100 | – |

UNVERIFIED: Singh, Linzen & Ravfogel 2026 (2605.26242); LawZero May 2026 (noise detection); Lederman & Mahowald 2026; Pearson-Vogel et al. 2026 (2602.20031).

**Provisional verdict:** the reversed-question test of a trained introspective report appears new among these. Hahami's IFT is our paradigm's nearest relative (R1): same localisation question, trained, held-out concepts, Gaussian training arm, no reversed question, and no random-injection test of trained models. My specific addition: Q (polarity-reversed wording) plus the random arm at test, applied to an IFT-style adapter, showing accuracy 1.00 with Q = .00. Before the PR: read the four unverified items and spot-check every quote against the arXiv HTML (H38).
