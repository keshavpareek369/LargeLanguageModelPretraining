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
   - **GQA** (Grouped Query Attention)
   - **MHA** (Multi-Head Attention)
   - **MLA** (Multi-head Latent Attention, DeepSeek-V2 style, with low-rank Q/KV projections)
   - **RMSNorm** instead of LayerNorm
   - **RoPE** (Rotary Position Embeddings) instead of learned absolute position embeddings
   - **SwiGLU** feed-forward blocks
5. Build a complete, reproducible **LLM training pipeline** (not just pretraining) for a medical-domain assistant:
   `Wikipedia pretraining → Domain-Adaptive Pretraining (PubMed) → Instruction Fine-Tuning (medical QA) → DPO alignment → Full capability + alignment evaluation`.

---

## Architectures Explained
1. **End-to-End Alignment Pipeline — `MLA-with_DAPT_IFT_DPO_Evaluation_70M.ipynb`** Full LLM pipeline: Pretrain → DAPT → IFT → DPO → Eval
   - **Base Model Architecture:** Multi-head Latent Attention (MLA) backbone (`n_embd=512`, `n_head=8`, `n_layer=6`, `block_size=512`) featuring low-rank Q/KV compression, decoupled RoPE, and RMSNorm.
   - **Stage 1: Domain-Adaptive Pretraining (DAPT)**
     - **Dataset:** PubMed corpus (`ccdv/pubmed-summarization`), streamed and tokenized using GPT-2 encoding into a 5GB (~1.34B tokens) memory-mapped (`np.memmap`) binary dataset.
     - **Objective:** Continues self-supervised next-token prediction starting from the general Wikipedia-pretrained checkpoint to adapt vocabulary and latent representations to medical/scientific literature.
   - **Stage 2: Instruction Fine-Tuning (IFT)**
     - **Datasets:** Aggregates 5 real-world medical QA sources: PubMedQA (`pubmed_qa`), Medical Meadow WikiDoc (`medalpaca/medical_meadow_wikidoc`), Medical Meadow Flashcards, Medical Meadow Patient Info, and ChatDoctor (`lavita/ChatDoctor-HealthCareMagic-100k`).
     - **Loss Mechanics:** Implements **Prompt Masking**. Prompts are formatted as `### Instruction:\n{instruction}\n\n### Response:\n{response}`. Target label tokens corresponding to the instruction are assigned `IGNORE_INDEX` (`-1`), ensuring `F.cross_entropy` loss is calculated **strictly on response tokens**, preventing model capacity waste on prompt prediction.
   - **Stage 3: Direct Preference Optimization (DPO)**
     - **Dataset:** Preference pairs from `TsinghuaC3I/UltraMedical-Preference` (`chosen` vs. `rejected` medical responses).
     - **Loss Mechanics:** Trains the active policy model against a **frozen deep-copy reference model** (initialized from the post-IFT checkpoint). Maximizes the reward margin using the standard DPO loss:
       $$\mathcal{L}_{\text{DPO}} = -\log \sigma \left( \beta \cdot \left[ \left(\log \pi_\theta(y_w|x) - \log \pi_{\text{ref}}(y_w|x)\right) - \left(\log \pi_\theta(y_l|x) - \log \pi_{\text{ref}}(y_l|x)\right) \right] \right)$$
       with prompt masking applied to evaluate per-sequence log probabilities strictly over response tokens.
   - **Stage 4: Comprehensive Evaluation Suite (Capability + Alignment)**
     - **Capability Metrics:**
       - **MMLU (`cais/mmlu`):** Multiple-choice general knowledge across 57 subjects (likelihood scoring).
       - **MedMCQA (`openlifescienceai/medmcqa`):** Domain-specific medical entrance exam QA (likelihood scoring).
       - **HumanEval (`openai_humaneval`):** Python coding capability evaluated via functional execution (`pass@1`).
     - **Alignment Metrics:**
       - **Preference Accuracy & Reward Margin:** Evaluated on held-out test pairs from `UltraMedical-Preference`.
       - **TruthfulQA (`truthful_qa`):** Measuring resistance to common misconceptions and hallucinations.
       - **Safety Check:** Evaluated on medical-domain risk prompts for appropriate disclaimers and refusal rates.
       - **Format Adherence & Degeneracy:** Measures End-Of-Text stopping rates and n-gram repetition percentages.

2. **Multi-Head Attention (MHA) — `MHA_70M.ipynb`**
   - **Components:** MHA (scaled dot-product, causal mask), RMSNorm (pre-norm), learned absolute positional embeddings, GELU FFN (4× expansion), AdamW with cosine LR decay, mixed-precision (AMP), gradient accumulation, memmap data loading.
   - **Benefits:** Provides the standard baseline — each head independently learns Q/K/V projections giving maximum representational flexibility. Simple to implement and debug, making it the reference point for comparing all other variants.

3. **Grouped-Query Attention (GQA) — `GQA_68M.ipynb`**
   - **Components:** GQA (`n_kv_heads = n_head // 4`, KV heads repeated to match query heads), RoPE (rotary positional embeddings, precomputed cos/sin buffers), RMSNorm (pre-norm), GELU FFN (4× expansion), **no learned positional embeddings**, memmap data loading.
   - **Benefits:** Reduces the KV cache size by 4× compared to MHA (fewer KV heads) while preserving per-query expressivity. RoPE replaces absolute positional embeddings, encoding relative position directly into the attention scores — this generalizes better and avoids the RoPE + absolute embedding conflict found in early experiments.

4. **Multi-head Latent Attention (MLA) — `MLA_70M.ipynb`**
   - **Components:** MLA with low-rank Q compression (`W_dq`, `W_uq`, `q_layernorm`) and low-rank KV compression (`W_dkv`, `W_ukv`, `kv_layernorm`), decoupled RoPE (applied only to a slice of each head: `qk_rope_dim`), RMSNorm (pre-norm), GELU FFN (4× expansion), **no learned positional embeddings**, memmap data loading.
   - **Benefits:** Aggressively compresses the KV cache by projecting K and V through a shared low-rank bottleneck before up-projecting — the stored latent is much smaller than full KV pairs. The decoupled RoPE design applies rotary encoding to only half the head dimension, leaving the other half as "nope" (non-positional) features that can carry content-only information. Best validation loss (3.29) among all English variants.

5. **K=V Attention — `K=V_attention.ipynb`**
   - **Components:** Causal attention with V reused from K (only `q_proj` and `k_proj`, no `v_proj`), SwiGLU FFN (gated activation: `SiLU(x₁) * x₂`, 4× expansion), RMSNorm (pre-norm), learned positional embeddings (`nn.Parameter`), memmap data loading.
   - **Benefits:** Eliminates the entire V projection — halving KV parameter count and memory. SwiGLU replaces GELU in the feed-forward block, providing a gated nonlinearity that has been shown to improve training efficiency in modern LLMs (PaLM, LLaMA). This notebook serves as an architectural probe to test these two ideas at small scale.

6. **Character-level Hindi — `Hindi-char-llm-5m.ipynb`**
   - **Components:** MHA (per-head Q/K/V with causal mask), LayerNorm (pre-norm), learned absolute positional embeddings, GELU FFN (4× expansion), raw character vocabulary (156 chars).
   - **Benefits:** Zero-dependency baseline — no tokenizer needed. Validates the full training pipeline end-to-end (attention, loss, generation) before investing in tokenization or scale. Uses LayerNorm (not RMSNorm), following the original nanoGPT design.

7. **BPE Hindi — `Hindi-bytepair-LLM-72m.ipynb`**
   - **Components:** MHA (per-head Q/K/V with causal mask), LayerNorm (pre-norm), learned absolute positional embeddings, GELU FFN (4× expansion), custom BPE tokenizer (45k vocab, trained on Hindi Wikipedia), two-pass memmap tokenization, resumable checkpointing.
   - **Benefits:** BPE compresses Hindi text ~10× vs characters, giving the model far more context per sequence. The two-pass memmap strategy (count tokens → pre-allocate → write) handles multi-GB corpora without loading everything into RAM. Resumable training from checkpoint enables long runs on time-limited platforms like Kaggle.

    
| Variant | Key idea | Params | Final Val Loss | Val PPL | training time | RAM | GPU | Total Tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **MHA_70M** | Standard multi-head attention | ~70.6M | 3.5580 | 35.09 | 9.61h | - | Tesla P100-PCIE-16GB | ~1.2B |
| **GQA_68M** | Grouped-Query Attention | 68.0M | 3.3109 | 27.41 | - | - | - | ~1.2B |
| **MLA_70M** | DeepSeek-style Q/KV compression | 69.9M | 3.2922 | 26.90 | - | - | - | ~1.2B |
| **K=V_attention** | Shared Key and Value vectors | - | - | - | - | - | - | - |
| **Hindi-bytepair-LLM** | Custom BPE tokenizer for Hindi | 71.86M | 4.0870 | 59.56 | - | - | Tesla P100-PCIE-16GB | ~45.6M |
| **Hindi-char-llm** | Character-level GPT for Hindi | 4.9M | 1.3630 | - | - | - | - | - |
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
