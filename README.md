# happyfool WP3 data and code

This is the public archive for the WP3 experiments of u/happyfool_beatslonely on rStation (r/ai_consciousness). The experiments ask when a language model's self-report is evidence about its own internal states. They use Qwen3 0.6B–8B, concept injection, localisation and type-2 (confidence-vs-correctness) readouts.

## Contents

| Folder | What it holds |
|---|---|
| `scripts/` | Every experiment and analysis script. `wp3_env.py` handles device and dtype; `type2_41*.py`, `type2_42*.py` and `type2_43_analyse.py` are the type-2 kits. |
| `docs/` | Pre-registrations (`prereg.md`, one amendment per experiment, written before data), design notes and novelty checks. |
| `results/` | Summary and analysis JSON per experiment, with a README for each amendment. |
| `rows/<job-id>/` | Per-trial rows (`*.rows.jsonl.gz`) and metadata for each remote GPU job. Am. 36a–38b also include float16 hidden states. |
| `jobs/` | The job specifications the GPU jobs ran. |
| `adapters/` | LoRA adapters (`adapter_config.json` + `adapter_model.safetensors`), named by the tags used in `results/`. |

Infrastructure details (instance ids, budget and dispatch internals) were removed before publication.

## Building the archive

It is built with `scripts/build-public-archive.sh` in the (private) notebook repository.

## Running a script

Scripts are run from `scripts/` with `python3 <script>.py <model> <out.json> ...`; usage is in each docstring. The pinned dependencies are listed in each result's `provenance` field.

## Licence

Everything in this repository (code, documents, results, per-trial rows and adapters) is released under the MIT License (`LICENSE`). Adapters are LoRA weights trained on Qwen3 base models; use of the base models is governed by their own licences.

## Paper 1 kit (type-2 template, Qwen3-8B)

"Confidence after a perturbation" uses these:
- Am. 41, 41b, 42 and 43: `scripts/type2_41*.py`, `type2_42*.py`, `type2_43_analyse.py`; rows under `rows/am41-*`, `am41b-*`, `am42-*`, `am43-confirm-8b-01`; results under `results/am41*`, `am42`, `am43`.
- The exploratory R52 checks: `scripts/r52_calib.py`, `scripts/r52_stats.py`; `results/r52/`.
- The Am. 40 scale correction: `results/am40/`, `rows/am40-*`.
