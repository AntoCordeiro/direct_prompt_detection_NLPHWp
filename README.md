# Detecting Direct Prompt Injections

Comparative study of lightweight binary detectors (benign vs. prompt injection / jailbreak) that weighs **detection quality**, **over-defense** (false positives on benign prompts that contain trigger words) and **latency**. Everything runs locally on a single 16 GB GPU (developed on an RTX 4070 Ti Super, Windows 11).

| Group | Systems |
|---|---|
| Baselines | keyword/regex heuristic; TF-IDF n-gram + logistic regression |
| Methodology 1 (encoders) | `meta-llama/Prompt-Guard-86M` (two score variants), `protectai/deberta-v3-base-prompt-injection-v2`, and our own fully fine-tuned `microsoft/deberta-v3-base` |
| Methodology 2 (small LLMs) | `meta-llama/Llama-Guard-3-1B` zero/few-shot (custom prompt-injection category); `Qwen2.5-1.5B/3B-Instruct` zero/few-shot prompting and QLoRA fine-tuning (4-bit NF4 + sequence-classification head) |

Data: [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections), [HackAPrompt](https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset) (attacks only), [NotInject](https://huggingface.co/datasets/leolee99/NotInject) (benign, trigger-word-rich), plus Alpaca-cleaned as generic benign text.
Metrics: accuracy, precision, recall, F1, FPR on benign prompts, AUROC, TPR@FPR, bootstrap CIs, McNemar tests, latency (batch 1 / 32), VRAM.

**Results and discussion: [REPORT.md](REPORT.md)** (Part 2 tests whether the benchmarks are fair: level split, out-of-distribution training, style probe, trigger-word hard negatives) · full tables/plots: [analysis.ipynb](analysis.ipynb) · `results/` (metrics, tables, figures).

## Setup
```bash
python -m venv .venv                       # Python 3.11 or 3.12
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m bitsandbytes       # should print SUCCESS (needed for QLoRA)
```
(On Linux/macOS use `.venv/bin/` instead of `.venv/Scripts/`.)

### Hugging Face access (required)
Some of the data and models are **gated**: you must have a free Hugging Face account, request access once, and log in with a token. Nothing secret is stored in the repo; the token stays in your Hugging Face cache.

1. **Create an account** at <https://huggingface.co/join> and a **read token** at <https://huggingface.co/settings/tokens>.
2. **Request access** to each gated repo, logged in with that account, by opening the page and accepting the terms (approval can take from a few minutes to a few days):

   | Resource | Type | Link |
   |---|---|---|
   | HackAPrompt | dataset | <https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset> |
   | Prompt-Guard-86M | model | <https://huggingface.co/meta-llama/Prompt-Guard-86M> |
   | Llama-Guard-3-1B | model | <https://huggingface.co/meta-llama/Llama-Guard-3-1B> |

   The other resources (Deepset, NotInject, Alpaca-cleaned, ProtectAI DeBERTa, DeBERTa-v3-base, Qwen2.5) are open.
3. **Log in** with the token: `.venv/Scripts/hf auth login` (or set the `HF_TOKEN` environment variable).
4. **Build the data folder**:
   ```bash
   python setup_data.py                  # checks your access, downloads the datasets, builds data/
   python setup_data.py --check          # only checks login and access (also for the models)
   python setup_data.py --with-models    # additionally pre-downloads all model weights (~15 GB)
   ```
   If access is still missing, the script lists exactly which repo to request. If you see `GATED`, the request has not been approved yet.

## Reproduce
```bash
python setup_data.py                              # check access + build all splits (main and v2) -> data/ (same as: python data.py; python data.py --extra)
python run_all.py --models everything --seeds 0 1 2   # all models + v2 protocols; resumable; predictions -> results/predictions/, summary -> results/metrics.csv
python probe.py                                   # style/source probe -> results/style_probe.csv
python ablate.py                                  # QLoRA hyper-parameter ablation -> results/ablation.csv
python make_notebook.py && jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
```
A full run takes roughly 1-1.5 h on the reference GPU (QLoRA 1.5B ~1.5 min, 3B ~7 min per seed). Run GPU jobs one at a time so latency numbers stay clean.

`data/`, `checkpoints/` and `results/predictions/` are not committed (dataset licences / size); `analysis.ipynb` needs `results/predictions/`, so run `run_all.py` once before re-executing it. The committed notebook already contains its outputs.

## Layout
| File | Purpose |
|---|---|
| `config.py`, `common.py` | paths, seeds, model ids; shared `Detector` interface (every model returns P(malicious)) |
| `setup_data.py` | checks Hugging Face access, downloads the datasets and builds `data/` (calls `data.py`) |
| `data.py` | builds the benchmarks, hard-negative splits and leakage checks |
| `metrics.py` | metrics, bootstrap CIs, threshold selection, McNemar, latency protocol |
| `baselines.py`, `encoders.py`, `llama_guard.py`, `qwen_prompt.py`, `qlora.py`, `finetune_encoder.py` | the models |
| `run_all.py`, `ablate.py` | CLIs |
| `make_notebook.py` → `analysis.ipynb` | reporting notebook |
| `PLAN.md` | the improvement plan this implementation follows; `NLP_HWp.ipynb` is the original Colab notebook (archived) |

## Notes
- The hard-negative (`+HN`) rows train on 40% of NotInject and are evaluated with those rows removed; they are an ablation, not a like-for-like comparison with the zero-shot detectors.
- transformers 5.x: load trainable models with `dtype=torch.float32`; DeBERTa's tokenizer needs `protobuf`.
- Dataset and model licences apply to the data/weights you download; this repository contains only code and aggregate results.
