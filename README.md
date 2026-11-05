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
| `adapters/` | One folder per LoRA adapter tag (the tags used in `results/`): `adapter_config.json` and `manifest.json` (sizes, sha256, base model, and where the weights are). The weights themselves are on the Hugging Face Hub, see "Adapters" below; `adapters/MANIFEST.json` indexes every tag. |

Infrastructure details (instance ids, budget and dispatch internals) were removed before publication.

## Adapters

The LoRA weights live in the Hub model repo [philljolly/happyfool-wp3-adapters](https://huggingface.co/philljolly/happyfool-wp3-adapters), one folder per tag with the same `adapter_config.json`, the `adapter_model.safetensors` and a copy of the manifest. They left this repository on 2026-11-05 (git is a poor home for 857 MB of weights); the earlier commits still hold the 21 that were published here before.

To use one:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = "Qwen/Qwen3-1.7B"  # adapters/<tag>/manifest.json: base_model
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base), "philljolly/happyfool-wp3-adapters", subfolder="risk17_c")
```

Verify a download against `adapters/<tag>/manifest.json`: `sha256sum adapter_model.safetensors` must equal `files.adapter_model.safetensors.sha256`.

## Building the archive

It is built with `scripts/build-public-archive.sh` in the (private) notebook repository.

## Running a script

Scripts are run from `scripts/` with `python3 <script>.py <model> <out.json> ...`; usage is in each docstring. The pinned dependencies are listed in each result's `provenance` field.

## Licence

Everything in this repository (code, documents, results, per-trial rows and adapter manifests) and the adapters on the Hub are released under the MIT License (`LICENSE`). Adapters are LoRA weights trained on Qwen3 base models; use of the base models is governed by their own licences.

## Paper 1 kit (type-2 template, Qwen3-8B)

"Confidence after a perturbation" uses these:
- Am. 41, 41b, 42 and 43: `scripts/type2_41*.py`, `type2_42*.py`, `type2_43_analyse.py`; rows under `rows/am41-*`, `am41b-*`, `am42-*`, `am43-confirm-8b-01`; results under `results/am41*`, `am42`, `am43`.
- The exploratory R52 checks: `scripts/r52_calib.py`, `scripts/r52_stats.py`; `results/r52/`.
- Am. 44 (confirmatory, state-induced confidence change d): `scripts/type2_44_analyse.py`; rows under `rows/am44-statecheck-8b-01`; results under `results/am44`. Concepts: `AM44_CONCEPTS` in `scripts/inject_loc.py`.
- The exploratory R53 checks (the reviewer's answer-identity control, held out): `scripts/r53_answer.py`, `r53_diff.py`, `r53_swap.py`; `results/r53/`.
- The exploratory R54 checks: `scripts/r54_dose.py` (d is absorbed by the stateless clean-pass prior for the chosen answer; the "dose" label in its first version was wrong, see `results/r54/README.md`), `scripts/r54_coherence.py` (two-consumer coherence pilot), and the omitted-cause red team under `results/r54/redteam/` (`REPORT.md` and its scripts). Revision plan for paper 1: `docs/paper1-revision-plan.md`.
- The Am. 40 scale correction: `results/am40/`, `rows/am40-*`.
