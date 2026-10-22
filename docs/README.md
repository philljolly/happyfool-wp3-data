# WP3: induced-disposition self-report tests (reproducibility package)

What this is: a small, CPU-only apparatus for asking whether a language model's self-report carries information about a disposition that was trained into it, beyond what an outside observer of its behaviour could get. Everything here ran on 4 CPU cores and 15 GB RAM. Pre-registration and every amendment/deviation: `prereg.md`. Results and their readings, in order: `results.md`. Raw outputs: `results/*.json`. Trained LoRA adapters: `adapters/<tag>/`.

## Status (2026-10-18)

- 0.6B word-preference arms (lantern, meadow, copper, harbor, thimble + controls): self-report null; the observer half was confounded by exposure until the exposure-matched observer (`stage2c.py`). See `results.md` runs 5–7.
- 0.6B risk trait (WP3c): **no valid channel**. The base model's choice doesn't depend on the offer in any format (Deviation 5 base check), so k\* is undefined there, not zero.
- 1.7B risk trait (WP3c-1.7B, Amendment 7, Deviation 6): channel valid; the risk-averse manipulation passes at lr 5e-5 (Δ_M −0.34); the reports moved as a general shift about risk (the "typical person" moved most), not as self-description. Same-dose control and fixed observer: Deviation 7, run 12.
- No working positive control yet (WP3b graft underfit).

## Setup

```
bash ../../scripts/setup-env.sh          # CPU torch + pinned transformers/peft/trl/datasets; idempotent
```

Models: `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B` from Hugging Face (downloaded on first use). All scripts are run from this directory.

## Scripts (current arm first)

| Script | What it does | Time (CPU) |
|---|---|---|
| `base_choice_check.py <model>` | Channel-validity gate: does the base model's choice depend on the offer (order gap, Spearman ρ with lottery p)? Run it in the **exact** evaluation format. | ~2 min |
| `time17.py` | Times one real trl DPO step at 1.7B | ~2 min |
| `risk17.py <out> [--control] [--averse] [--lr=..]` | WP3c-1.7B stage 1: DPO-LoRA on 100 offers × both orders; order-averaged manipulation check Δ_M; report measures (self, self_3p, other_typical, seek-word mass) on LoRA and base | ~18 min |
| `risk17_eval.py <adapter_dir> <tag> [--nudge]` | Stage 2 on a saved adapter: observer B_k with histories over both templates/orders and 5 seeds; paraphrase-spread calibration (z-units); optional base-model persona-pinning check | ~5 min |
| `pilot.py`, `stage2b.py`, `stage2c.py` | 0.6B word-preference arms: training, report formats, exposure-matched observer | ~7 min per word |
| `graft.py` | WP3b positive control (rating graft) | ~15 min |
| `risk.py`, `diag_risk.py`, `risk_scaffold.py` | 0.6B risk arm, its diagnosis, scaffold arm | |
| `aggregate.py` | Pre-registered aggregate over the word arms | seconds |

Reproduce the 1.7B result: `python3 risk17.py /tmp/wp3/risk17_c5 --control --averse --lr=5e-5`, then `python3 risk17.py /tmp/wp3/risk17_a5 --averse --lr=5e-5`, then `risk17_eval.py` on each. To skip training, point `risk17_eval.py` at `adapters/risk17_c5` and `adapters/risk17_a5`.

## The four controls (lessons the instrument taught, in the order it taught them)

1. **Disjoint report format.** If the report can be answered by emitting the trained behaviour (same response format as the training prompts), the "self-report" measures behaviour generalisation. Use rating / yes-no / word formats that the training never used (run 5).
2. **Exposure-matched observer.** A behavioural observer shown histories in which the target item appears more often than its comparisons reads exposure, not preference. Match exposure across items in the histories (run 6, Deviation 3).
3. **Channel-validity gate in the exact evaluation format.** Before training, check that the base model's behaviour depends on the input in the format used for training and observation. Two templates differing by a few words gave P(lottery) 0.49 vs 0.95 (runs 10–11).
4. **Same-dose DiD plus a projection arm.** Estimate self-specificity as a difference-in-differences against a random-label control trained at the same dose, and include a no-information other ("a typical person"). If the other moves with the self, it's a world-model shift, not self-description (Amendment 7).

## Known limits

- Behaviour-only protocols can't separate a report that *reads* the disposition from a shared cause (one persona latent driving both); that needs a white-box ablation.
- One training draw per arm; small models; CPU-bounded doses.
- Bounded rating scales: compare shifts in units of each measure's paraphrase spread (Deviation 7), not raw.
