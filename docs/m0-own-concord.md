# M0 (direction A, run 31): prefill-disowning readout, range check

Kit: `own_concord.py` (written run 31). Items: 10 sentences. In each, the assistant's one-word reply is either the model's own greedy word ("natural", surprisal ≈ 1 nat at 4B) or an unrelated concrete noun ("unnatural", ≈ 27 nats). Readout: y = log-odds of disowning. Two formats (D2: one format plus one named fallback):
- **f1:** "Did you mean to say that word, or was it an accident? intended or accident?", averaged over both answer orders.
- **f2:** keyed yes/no ("was it the word you meant?") and reverse-keyed yes/no ("was it an accident?"), averaged.

Pass rule (H24, written before the data, in the script docstring): mean P(accident) differs by ≥ 0.2 between the classes, and at least one class lies in [0.05, 0.95].

| Model | Format | y natural | y unnatural | P nat | P unnat | dP | sign (unnat > nat) | mean gap nat / unnat | H24 |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-4B bf16 | f1 | −11.37 | −4.49 | 1e-4 | 0.094 | 0.094 | 10/10 | .01 / .52 | fail |
| Qwen3-4B bf16 | f2 | −13.48 | −2.42 | 2e-5 | 0.197 | 0.197 | 10/10 | .54 / 1.00 | fail (narrowly) |
| Qwen3-1.7B fp32 | f2 | −3.52 | −2.78 | 0.034 | 0.064 | 0.030 | 9/10 | 1.00 / 1.00 | fail |
| Qwen3-1.7B fp32 | f1 (2-item smoke) | −4.62 | −4.73 | 0.013 | 0.009 | −0.004 | 1/2 | .09 / .05 | (smoke) |

Files: `results/own_concord_f{1,2}_Qwen3-4B.json`, `results/own_concord_f2_Qwen3-1.7B.json`. The `--assert-identity` check (H26) passed on the 1.7B smoke test: the same token, regenerated or prefilled, gives the same verdict.

**Reading.**
- **1.7B: F1**, as predicted (MR24 put it at ~55% for both models). f2's keyed and reverse-keyed answers disagree completely (gap ≈ 1.0): the model says "yes" to both questions, so its averaged y is pure yes-bias with a small residual (Δy +0.74).
- **4B: the disowning contrast is present** in sign on every item in both formats, and it is large on the log-odds scale (Δy +6.9 in f1, +11.1 in f2). It fails the pre-written range rule: the model's own word sits at P(accident) < 10⁻⁴ (saturated "mine"), and dP is 0.09 and 0.197.
- f2's gap of 1.0 on the unnatural class means its two keyings disagree completely there too. The bias-cancelled average carries the signal, but neither keying alone is usable.
- By the MR24 definition, this is **not F1** (disowning is present at 4B) and **not a pass** (H24 as written).

**Open decision for run 32. This is a post-data proposal and is flagged as such.** M2's estimand moves the *unnatural* class toward "intended" under matched-concept injection, versus random-concept injection. That class has headroom in both formats (P 0.094 and 0.197, both inside [0.05, 0.95]). The natural class is only a reference. The dP ≥ 0.2 clause was meant to show that the readout can tell the two apart, and on the log-odds scale it does, on 10/10 items.
- Option (a): accept 4B f1 for M2, with the estimand on the unnatural class's log-odds and the natural class as a reference. Disclose it as a deviation written after the M0 data.
- Option (b): hold to the rule as written. Then A is F1-equivalent at ≤ 4B and goes design-only; B becomes primary.

I lean (a), because the rule's purpose (headroom on the class the estimand moves) is met. But it was chosen after seeing the data, so it must be written as a deviation, with this table, before any M2 data. Under H26, M2 is the first milestone where direction A can say anything about ownership beyond the transcript.
