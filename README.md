# LatentGPT — Building a GPT-Style LLM From Scratch

This repo documents an iterative, from-scratch journey through building GPT-style language models in PyTorch — starting from a simple character-level bigram model and ending with a full **pretrain → domain-adapt → instruction-tune → align (DPO) → evaluate** pipeline for a small medical-domain LLM. Everything is trained on Kaggle (single P100 GPU, 16GB VRAM), using memory-mapped (`memmap`) token datasets so multi-GB corpora never have to be loaded into RAM.

> **Note:** This README consolidates several experiment scripts/logs into one narrative. Assumption: the project is referred to here as **LatentGPT**, the class name used across most of the later scripts. Rename freely to match your actual repo name.

---

## Table of Contents

- [Project Goals](#project-goals)
- [Repo Structure (suggested)](#repo-structure-suggested)
- [Experiment Timeline](#experiment-timeline)
  - [1. Character-level Bigram baseline (Hindi)](#1-character-level-bigram-baseline-hindi)
  - [2. BPE tokenizer + deeper GPT (Hindi)](#2-bpe-tokenizer--deeper-gpt-hindi)
  - [3. Memmap-based English pretraining](#3-memmap-based-english-pretraining)
  - [4. Architecture experiments (RMSNorm, GQA, MLA, RoPE, SwiGLU)](#4-architecture-experiments-rmsnorm-gqa-mla-rope-swiglu)
  - [5. Full LLM pipeline: Pretrain → DAPT → IFT → DPO → Eval](#5-full-llm-pipeline-pretrain--dapt--ift--dpo--eval)
- [Results Summary](#results-summary)
- [Key Learnings & Bug Fixes](#key-learnings--bug-fixes)
- [Setup](#setup)
- [Usage](#usage)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)

---

## Project Goals

1. Implement a GPT-style transformer **entirely from scratch** (no HuggingFace model classes) to understand every component: tokenization, embeddings, attention, normalization, feed-forward blocks, and the training loop.
2. Scale from a tiny character-level model to a ~70M parameter transformer trained on 1B+ tokens using `memmap` for memory-efficient data loading.
3. Explore modern architectural components popularized by recent open LLMs:
   - **RMSNorm** instead of LayerNorm
   - **RoPE** (Rotary Position Embeddings) instead of learned absolute position embeddings
   - **GQA** (Grouped Query Attention)
   - **MLA** (Multi-head Latent Attention, DeepSeek-V2 style, with low-rank Q/KV projections)
   - **SwiGLU** feed-forward blocks
4. Build a complete, reproducible **LLM training pipeline** (not just pretraining) for a medical-domain assistant:
   `Wikipedia pretraining → Domain-Adaptive Pretraining (PubMed) → Instruction Fine-Tuning (medical QA) → DPO alignment → Full capability + alignment evaluation`.

---

## Repo Structure (suggested)

```
.
├── data/
│   ├── build_wikipedia_memmap.py       # tokenize + write .bin + meta.json
│   ├── build_pubmed_memmap.py          # DAPT corpus builder (PubMed, streaming)
│   ├── build_ift_dataset.py            # instruction/response pairs + prompt masking
│   └── build_dpo_dataset.py            # chosen/rejected preference pairs
├── models/
│   ├── bigram_char_model.py            # baseline (Section 1)
│   ├── gpt_bpe_hindi.py                # BPE tokenizer variant (Section 2)
│   ├── latent_gpt_mha_rmsnorm.py       # memmap + RMSNorm + absolute pos emb (Section 3)
│   ├── latent_gpt_gqa_rope.py          # GQA + RoPE (Section 4)
│   ├── latent_gpt_mla.py               # Multi-head Latent Attention, DeepSeek-style (Section 4/5)
│   └── mini_gpt_swiglu_vk.py           # SwiGLU FFN + V=K attention (Section 4)
├── train/
│   ├── train_pretrain.py
│   ├── train_dapt.py
│   ├── train_ift.py
│   └── train_dpo.py
├── eval/
│   └── full_evaluation_suite.py        # MMLU, MedMCQA, HumanEval, Pref-Acc, TruthfulQA, Safety, Format
├── inference/
│   ├── generate.py
│   └── generate_from_ckpt.py
├── checkpoints/                        # (gitignored)
└── README.md
```

---

## Experiment Timeline

### 1. Character-level Bigram baseline (Hindi)

The very first model: a minimal GPT (misleadingly named `BigramLanguageModel`, following Karpathy's nanoGPT lecture) operating directly on **characters**, no tokenizer.

| Setting | Value |
|---|---|
| Vocab | 156 raw Hindi/Latin characters |
| `n_embd` / `n_head` / `n_layer` | 256 / 4 / 6 |
| `block_size` | 512 |
| Params | 4.95M |
| Iterations | 12,000 |
| Final val loss | 1.363 |

**Purpose:** sanity-check the training loop, attention implementation, and generation code end-to-end before investing in tokenization or scale. Output text is fluent-looking Devanagari script but not semantically coherent — expected at this scale/data budget.

### 2. BPE tokenizer + deeper GPT (Hindi)

Replaced character-level encoding with a **custom-trained 45k-vocab BPE tokenizer** (via `tokenizers` library) and scaled the model up.

| Setting | Value |
|---|---|
| Tokenizer | Custom BPE, 45,000 vocab, trained on Hindi Wikipedia |
| Dataset | `wikipedia_hindi_500mb.txt` → 45.7M tokens (memmap, `int32`) |
| `n_embd` / `n_head` / `n_layer` | 512 / 8 / 8 |
| `block_size` | 1024 |
| Params | 71.86M |
| Iterations | 80,000 (resumed training from iter 52,001) |
| Final val loss / PPL | 4.087 / 59.56 |

Two-pass tokenization strategy: pass 1 counts total tokens to pre-allocate a `np.memmap`; pass 2 streams the file in 200k-character chunks and writes directly into the memmap, avoiding ever holding the full tokenized corpus in RAM. Checkpointing is resumable (`checkpoint_last.pt` stores `iter`, optimizer state, and loss history).

### 3. Memmap-based English pretraining

Moved to **English Wikipedia** (`jklu-en-memap-5gb`, ~1.2B GPT-2-tokenizer tokens) to get a larger, higher-quality pretraining corpus, and switched to `tiktoken`'s `gpt2` encoding (50,257 vocab) instead of a custom tokenizer.

| Setting | Value |
|---|---|
| Tokenizer | `tiktoken` GPT-2 (50,257 vocab) |
| Dataset | 1,202,932,080 tokens (90/10 train/val split) |
| `n_embd` / `n_head` / `n_layer` | 512 / 8 / 6 |
| `block_size` | 512 |
| Params | 70.63M |
| Iterations | 75,000 (~9.6 hours on a Tesla P100) |
| Final val loss / PPL | 3.558 / 35.09 |

This version still used **learned absolute position embeddings** + standard scaled dot-product multi-head attention + RMSNorm, and produced noticeably more coherent English completions than the Hindi character model (e.g., plausible-sounding institutional/historical prose).

### 4. Architecture experiments (RMSNorm, GQA, MLA, RoPE, SwiGLU)

With the training loop and data pipeline stable, the focus shifted to attention/normalization architecture, testing four variants on the same English Wikipedia memmap corpus:

| Variant | Key idea | Params | Final Val Loss | Val PPL |
|---|---|---|---|---|
| **MHA + absolute pos-emb** (Section 3) | Standard multi-head attention, learned position embeddings, RMSNorm | 70.63M | 3.558 | 35.09 |
| **GQA + RoPE** | Grouped-Query Attention (`n_kv_heads = n_head // 4`) with Rotary Position Embeddings, no absolute pos-emb | 68.01M | 3.311 | 27.41 |
| **MLA (DeepSeek-style)** | Low-rank joint Q/KV compression (`q_proj_dim`, `kv_proj_dim`) + decoupled RoPE on a slice of each head, `scaled_dot_product_attention` | 69.94M | 3.292 | 26.90 |
| **SwiGLU FFN + V=K attention** | Gated SwiGLU feed-forward, causal attention where V is reused from K (halves KV parameters), smaller-scale sanity run | 4.9M (small config) | — (dataset-limited run) | — |

Key fix carried through this stage: an early bug had both **learned absolute position embeddings and RoPE active at the same time**, which conflict with each other (RoPE already encodes relative position; adding an absolute embedding on top distorts it). Removing the absolute position embedding when RoPE is present was one of the more impactful fixes.

Both **GQA+RoPE** and **MLA** clearly outperformed the vanilla MHA+absolute-embedding baseline at the same/similar parameter count and training budget — consistent with published results showing that better positional encoding (RoPE) and more parameter-efficient attention (GQA/MLA) both matter more than raw model size at this scale.

### 5. Full LLM pipeline: Pretrain → DAPT → IFT → DPO → Eval

The final and most complete stage builds an actual **domain-adapted, instruction-tuned, preference-aligned model** using the MLA architecture as the backbone:

```
Wikipedia Pretraining (general English, 1.2B tokens)
        │
        ▼
Domain-Adaptive Pretraining (DAPT)   — 5GB of PubMed full-text articles
        │                              (streamed, tokenized, memmap'd)
        ▼
Instruction Fine-Tuning (IFT)        — 50k–500k instruction/response pairs
        │                              from PubMedQA, WikiDoc, Medical
        │                              Flashcards, MedQuAD, ChatDoctor
        │                              (prompt-masked loss)
        ▼
Direct Preference Optimization (DPO) — TsinghuaC3I/UltraMedical-Preference
        │                              (chosen vs. rejected pairs, frozen
        │                               reference model = post-IFT checkpoint)
        ▼
Evaluation Suite
   • MMLU              (general knowledge, multiple-choice)
   • MedMCQA           (medical domain multiple-choice)
   • HumanEval         (code generation, pass@1, sandboxed exec w/ timeout)
   • Preference Accuracy & Reward Margin (held-out DPO pairs)
   • TruthfulQA (MC1)  (resistance to common misconceptions)
   • Safety / Harm-Avoidance (medical-risk prompt probes, keyword-based caution scoring)
   • Format Adherence & Degeneracy (EOT stop-rate, token repetition rate)
```

#### Pipeline details

- **DAPT:** Streams `ccdv/pubmed-summarization` via HuggingFace `datasets` (streaming mode, no full download), tokenizes with the *same* GPT-2 tokenizer used in pretraining for checkpoint compatibility, and stops once a 5GB / ~1.34B-token target is reached. Resumes from the pretraining checkpoint with **partial state-dict loading** (loads only tensors whose keys and shapes match — useful when architecture details shift slightly between stages).
- **IFT:** Aggregates five genuine (non-synthetic) medical QA sources — PubMedQA (labeled), Medical Meadow WikiDoc, Medical Meadow Flashcards, MedQuAD, and ChatDoctor — into a single `(instruction, response)` corpus. Applies **prompt masking**: loss is computed only on response tokens (prompt tokens labeled `-1`/ignored), using an `### Instruction: / ### Response:` template.
- **DPO:** Builds preference pairs from `TsinghuaC3I/UltraMedical-Preference`, again with prompt-masked tokenization. Trains with a frozen deep-copy reference model (the post-IFT checkpoint) and the standard DPO loss:
  `loss = -logsigmoid(β · [(logπ_chosen − logπ_rejected) − (logπ_ref_chosen − logπ_ref_rejected)])`.
- **Evaluation:** A single script evaluates every available checkpoint (DAPT / IFT / DPO — whichever exist) across all 7 metrics and prints a unified comparison table, so you can see exactly what each pipeline stage bought you (e.g., does DPO improve preference accuracy without regressing MMLU?).

---

## Results Summary

Perplexity is only directly comparable **within** a fixed tokenizer + dataset combination, so results are grouped accordingly.

**English Wikipedia (`tiktoken` GPT-2 tokenizer, 1.2B tokens, 75k steps, same P100 GPU):**

| Architecture | Params | Final Train Loss | Final Val Loss | Val PPL |
|---|---|---|---|---|
| MHA + abs. pos-emb + RMSNorm | 70.63M | 3.519 | 3.558 | 35.09 |
| GQA + RoPE + RMSNorm | 68.01M | 3.266 | 3.311 | 27.41 |
| MLA (DeepSeek-style) + RoPE + RMSNorm | 69.94M | 3.438 | 3.292 | 26.90 |

**Hindi Wikipedia (custom BPE tokenizer, 45.7M tokens):**

| Model | Params | Iterations | Final Val Loss | Val PPL |
|---|---|---|---|---|
| GPT (MHA, LayerNorm, abs. pos-emb) | 71.86M | 80,000 | 4.087 | 59.56 |

**Hindi Wikipedia (character-level, 156-char vocab):**

| Model | Params | Iterations | Final Val Loss |
|---|---|---|---|
| Bigram/GPT baseline | 4.95M | 12,000 | 1.363 |

> Character-level loss numbers are **not** comparable to token-level numbers above — they're on a completely different unit (per-character vs. per-BPE-token nats).

---

## Key Learnings & Bug Fixes

A running log of issues found and fixed across iterations — useful context if you're debugging a similar from-scratch build:

1. **RoPE + absolute positional embeddings conflict.** Using both simultaneously double-encodes position and hurts quality; keep only one.
2. **Weight decay should skip 1D parameters.** Applying AdamW weight decay to LayerNorm/RMSNorm gains and biases (not just 2D weight matrices) is a common but harmful mistake — split parameters into decay/no-decay groups by `p.dim() >= 2`.
3. **`torch.cuda.amp.autocast` / `GradScaler` are deprecated** in favor of `torch.amp.autocast('cuda', ...)` / `torch.amp.GradScaler('cuda', ...)` — both still work but emit `FutureWarning`s; worth migrating.
4. **Partial checkpoint loading across architecture changes.** When moving between pipeline stages (e.g., pretraining → DAPT) with slightly different module names/shapes, filter the old state dict to only the keys whose shape matches the new model before calling `load_state_dict`, rather than requiring an exact match.
5. **Two-pass tokenization for memmap datasets.** For multi-GB text corpora, do a first pass purely to count total tokens (so you can pre-allocate the `np.memmap` with the correct shape), then a second pass to actually write token IDs — this avoids ever materializing the full token array in RAM.
6. **Prompt masking for IFT/DPO.** Setting label tokens to `-1` (ignored by `F.cross_entropy(..., ignore_index=-1)`) for the prompt portion of each example is essential — otherwise the model is partially trained to *predict the instruction*, which wastes capacity and can leak into generation style.
7. **Repetition penalty & explicit EOT handling matter a lot at this parameter scale** (~70M). Small models degenerate into loops far more readily than large ones; a simple `logits[token] /= repetition_penalty` for already-generated tokens plus stopping on `eot_token` meaningfully improves usability without any architecture change.
8. **`torch.memmap` returns non-writable arrays** — wrapping with `torch.from_numpy` on a read-only memmap triggers a `UserWarning`; this is expected and safe as long as you don't try to write through the tensor.

---

## Setup

```bash
pip install torch numpy pandas matplotlib tiktoken tokenizers datasets --break-system-packages
```

GPU: developed and tested on a single **NVIDIA Tesla P100 (16GB)** via Kaggle notebooks. Mixed-precision (`use_amp = True`) and gradient accumulation are used throughout to fit batch sizes within 16GB.

## Usage

```bash
# 1. Build the pretraining memmap dataset
python data/build_wikipedia_memmap.py

# 2. Pretrain (choose one architecture)
python train/train_pretrain.py --arch mla        # or: mha | gqa_rope | swiglu_vk

# 3. Domain-adaptive pretraining on PubMed
python data/build_pubmed_memmap.py
python train/train_dapt.py --resume checkpoints/best_model.pt

# 4. Instruction fine-tuning
python data/build_ift_dataset.py
python train/train_ift.py --resume checkpoints/best_model_DAPT.pt

# 5. DPO alignment
python data/build_dpo_dataset.py
python train/train_dpo.py --resume checkpoints/best_model_IFT.pt

# 6. Full evaluation across all available checkpoints
python eval/full_evaluation_suite.py
```

## Future Work

- Add KV-caching to `generate()` for faster autoregressive inference (attempted but incomplete in early scripts — current generation re-runs the full forward pass every step).
- Scale `block_size` beyond 512–1024 and re-benchmark MLA vs. GQA, since MLA's efficiency advantage grows with context length.
- Replace the keyword-matching safety evaluator with a model-graded (LLM-as-judge) safety check for more reliable scoring.
- Try quantization (int8/int4) for faster CPU inference of the final DPO checkpoint.
- Expand the DAPT corpus beyond PubMed abstracts/summaries to full-text papers for stronger domain adaptation.

## Acknowledgements

- Architecture patterns inspired by Andrej Karpathy's nanoGPT / "Let's build GPT" series (bigram baseline, training loop structure).
- MLA design follows the attention mechanism introduced in DeepSeek-V2.
- Datasets: Hindi & English Wikipedia dumps, `ccdv/pubmed-summarization`, PubMedQA, Medical Meadow (WikiDoc, Flashcards, Patient Information), MedQuAD, ChatDoctor-HealthCareMagic-100k, and TsinghuaC3I/UltraMedical-Preference — all accessed via HuggingFace `datasets`.
