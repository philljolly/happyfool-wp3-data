# Amendment 22: readings listed before the analysis (policies H15, H17a, H19b, H20)

Written run 27, ~00:20Z 2026-10-20. Untrained eval done (accuracy only seen: A 0.30, B 0.15). C, A, B, D evals running. **No probe output read.**

For each pattern `implant_analyse.py` can print, the readings I will consider (not only the registered verdict):

| Pattern | Readings to keep open |
|---|---|
| M fails (A or B < 0.8 on own items) | under-training at lr 2e-4 / 5 epochs; MC format not reached from declarative training (reversal/format gap); retry at 5× lr per Am. 22, else documented floor |
| Δ_self > 0, SB_imp > 0 | self-route; OR self probe has more headroom than others (check 22b headroom first); OR answer-entropy confound (mediation slope) |
| Δ_self ≈ Δ_world ≈ agents > 0 | world-model update P6; fluency-consistent ascription (H20, normative given encyclopedic training); C-GA |
| agents > world/book, E ≈ 0 | C-GA; similarity gradient P8 if ordered |
| PI > 0 | fluency-consistent other-ascription (H20), not "projection failure"; projection proper only via 22e D |
| PI ≈ 0, Δ_self > 0 | self-marking of fine-tuned content (surprising; replicate at 0.6B before any claim) |
| SC < 0 (22f) | self uses an answer-accessibility cue beyond familiarity that others don't |
| everything ≈ 0 | saturation (headroom); trained facts not consulted at probe time; probes read the question form only |
| C's content-free shift ≈ Δ_self | any LoRA shifts hedging; report as content-free |
| Δ(B−D) ≥ Δ | familiarity/register (P3/P7), not knowledge |
| y_self on non-implanted items < untrained | Gekhman-type overconfidence spill-over |

Scope (H19a): whatever comes out is "at 1.7B, LoRA r16, 20 facts per arm, encyclopedic installation, binary sure/unsure log-odds probes".
