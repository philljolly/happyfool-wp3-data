# type2_41.py smoke test (run 49, instrument check only; not Am. 41 data)
Qwen3-0.6B, L12 α8, the first 3 *old* `inject_detect.CONCEPTS` (bread, violin, cathedral), none of them Am. 40 concepts. Local CPU, 171 s.
- Format: log mass on A/B ≈ 0 (≈ all probability mass on "A"/"B") in both frames, injected and clean.
- conf = logit(B) − logit(A) is near-constant: −.59 to −.64 (intro) and −1.11 to −1.27 (ordinary), with the clean values the same. The model leans "guessing" and barely varies across items.
- **So at 0.6B a format gate on item-level variance of clean confidence will probably fail.** The pre-registration needs that gate stated numerically, plus an 8B-first design (the CROSSING + C cell).
