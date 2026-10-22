# Access-gap scale curve (MR39). Design draft, run 47; NOT pre-registered

**Question.** Along Qwen3 0.6B → 1.7B → 4B → 8B → 14B, does the model's report come to use an injected state that a single-state linear probe can already read? The quantity is G(s) = probe accuracy − report accuracy. Both are measured in the same forward pass, on the same items, and the report is scored without the paired clean run (H45).

**Gate before pre-registration.** Am. 39a decides the report measure:
- If quote acc_cal is SHOWN at 4B, the report measure is the quote readout.
- If not, the floor is documented in two formats (index R47, quote 39a). Whether to run the curve with AUC_m as the report measure goes to the next strategic reflection, not to an amendment.

**Fixed rules (to be copied verbatim into the pre-registration).**
- **Layers by relative depth.** Qwen3-4B has 36 layers, so L24 = 0.667 and L20 = 0.556. At scale s, L = round(depth × n_layers(s)). The n_layers values below are recalled, not checked: read `config.num_hidden_layers` in a smoke step and record it before any data.

  | Model | Layers | L at 0.667 | L at 0.556 |
  |---|---|---|---|
  | 0.6B | 28 | 19 | 16 |
  | 1.7B | 28 | 19 | 16 |
  | 4B | 36 | 24 | 20 |
  | 8B | 36 | 24 | 20 |
  | 14B | 40 | 27 | 22 |

- **Two dose rules, both reported.**
  - (D-norm) α relative to the concept vector, exactly as now (`concept_vec` − baseline mean, × α). That already scales with the residual norm at L.
  - (D-probe) choose α from {1, 2, 4} so that the single-state probe at the last layer is closest to .90, using 20 calibration concepts that are disjoint from the test concepts.
  - G is compared across scales at matched probe accuracy under D-probe. This answers "the probe ceiling fell".
- **Probe layers.** The last layer, plus the layer at 0.83 depth. They are passed as `--probe-layers`, and `inject_quote.py` resolves negative indices.
- **Items and seeds.** The concepts are AUDIT36B (90) at every scale, with two sentence-sampling seeds (inject_quote's rng seed 39 and 40; this needs a `--seed` flag). The random arm runs at the primary setting only.
- **Gates per scale.**
  - G39 range: SD of the clean-run m ≥ .5 nats.
  - Coherence: greedy 20 tokens on 5 items, saved.
  - A probe ceiling of at least .75. If it is lower, that scale is reported as "state not decodable", not as a gap.
- **dtype.** fp32 up to 4B. bf16 for 8B and 14B, because fp32 14B doesn't fit 48 GiB with hidden states. The 4B point is also run in bf16 so the dtype shift is measured: the K3 lesson from Am. 36.
- **Hardware.** `gpu_min_mem_gib` 24 up to 8B, and 40 for 14B. No GPU is named.

**Estimand slots.** For each scale, G with a concept-bootstrap CI. The trend is the slope of acc_cal on log-parameters, with the CI from bootstrapping concepts jointly across scales.
- "Report access emerges" is SHOWN iff acc_cal at the largest scale is SHOWN (R47 rule) and the slope CI excludes 0.
- "Flat" is EXCLUDED-style iff every scale's acc_cal CI lies inside [.40, .60] while the probe is ≥ .75.

**Kit work still to do.**
- `--seed` in `inject_quote.py`.
- Depth-based `--settings` (e.g. `d0.667:2`).
- The coherence step.
- A model-size smoke test on 0.6B.
- Job files: one per scale, three submissions per run at most.

**Cost.** About $0.10 per job up to 8B; 14B on L40S is about $0.5–1.0.
