# ❄️ Neural Book Search & Retrieval with Snowflake Arctic Embed (v1.5)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)
[![Snowflake Arctic](https://img.shields.io/badge/Snowflake-Arctic--Embed--M--v1.5-blue?logo=snowflake)](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A state-of-the-art dense book retrieval engine fine-tuned on **64,552 Goodreads books**, powered by **`Snowflake/snowflake-arctic-embed-m-v1.5`** (109M parameters, 512 context window, 768 dimensions with Matryoshka MRL support). Features contrastive learning, duplicate-positive masking, and an **8-layer false-negative protection shield** for iterative hard negative mining.

---

## 📌 Table of Contents
- [Architecture & Overview](#architecture--overview)
- [Why Snowflake Arctic Embed (v1.5)?](#why-snowflake-arctic-embed-v15)
- [The 8-Layer False-Negative Protection Engine](#the-8-layer-false-negative-protection-engine)
- [Directory Structure](#directory-structure)
- [Step-by-Step Execution Guide](#step-by-step-execution-guide)
  - [Step 1: Zero-Shot Baseline Benchmark](#step-1-zero-shot-baseline-benchmark)
  - [Step 2: Contrastive Fine-Tuning (Phase 1)](#step-2-contrastive-fine-tuning-phase-1)
  - [Step 3: Iterative Hard Negative Mining](#step-3-iterative-hard-negative-mining)
  - [Step 4: Final Benchmarking & Progression](#step-4-final-benchmarking--progression)
- [Hardware & Environment](#hardware--environment)
- [Author](#author)

---

## 🚀 Architecture & Overview

Traditional keyword search fails when readers search with abstract tropes, emotional vibes, or plot descriptions (*"dark academia boarding school mystery with secret society"*). 

This project trains a high-precision dense vector retriever to project queries and complex book descriptions into a shared 768-dimensional semantic space:

```
[ User Search Query ]
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ Query Instruction Prompt:                                   │
│ "Represent this sentence for searching relevant passages: " │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Snowflake Arctic Embed (v1.5) Encoder (109M Params)         │
│ - 512 Context Window                                        │
│ - Normalized 768-dimensional dense vector                   │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 2D Block-Chunked Vector Dot-Product (GPU)                   │
│ - Evaluates against all 64,552 catalog book embeddings      │
│ - Outputs Top-K ranked books (Recall@1, 5, 10, 50)          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🌟 Why Snowflake Arctic Embed (v1.5)?

| Feature | `bge-small-en-v1.5` (Previous) | `snowflake-arctic-embed-m-v1.5` (This Project) |
| :--- | :---: | :---: |
| **Model Architecture** | BERT (33.4M Parameters) | Arctic Transformer (**109M Parameters** — $3.3\times$ capacity) |
| **Context Window** | 256 tokens (truncated blurbs) | **512 tokens** (full synopsis, series details, and shelf tags) |
| **Embedding Size** | Fixed 384 dimensions | **768 dimensions** with native **Matryoshka (MRL)** slicing |
| **Retrieval Quality** | Strong baseline | **#1 on MTEB Retrieval Leaderboard** for its size class |
| **Hardware Fit** | ~2.5 GB VRAM | **~3.2 GB VRAM** (Optimized for 6GB RTX 4050 Laptop GPUs) |

---

## 🛡️ The 8-Layer False-Negative Protection Engine

A critical trap in hard negative mining is **False-Negative Poisoning**: naively treating a retrieved book as a negative when it is actually an unindexed edition, adaptation, or sequel in the same series.

Before any book is accepted as a training negative, it must pass through an 8-stage gatekeeper:

```
Candidate Book Retrieved by Model
               │
   [1. Exact & Cross-Query ID Ban] ────────────► Reject if target or shared positive
               │ Pass
   [2. Cosine Ceiling (> 0.88)] ───────────────► Reject if semantically near-identical
               │ Pass
   [3. Normalized Full Title Match] ───────────► Reject if identical text
               │ Pass
   [4. Stripped Subtitle / Edition Match] ─────► Reject if "Dune: Deluxe" == "Dune"
               │ Pass
   [5. Significant Word Overlap (>= 65%)] ─────► Reject if shared non-stopword tokens
               │ Pass
   [6. Regex Series / Universe Matching] ──────► Reject if from same series (e.g. Percy Jackson)
               │ Pass
   [7. Goodreads Author Shelf Overlap] ────────► Reject if matching author shelf tag
               │ Pass
   [8. Frequency Capping (Max 30)] ────────────► Reject if book is over-sampled as distractor
               │ Pass
         ACCEPTED AS HARD NEGATIVE
```

---

## 📂 Directory Structure

```
neural_book_search_arctic/
├── .gitignore                      # Configured for GitHub 100MB file limit
├── README.md                       # Complete documentation
├── requirements.txt                # Exact dependency specifications
├── data/                           # Work-clustered splits & catalog database
│   ├── books_metadata.sqlite3      # 64,552 Goodreads books (title, authors, shelves, desc)
│   ├── train.jsonl                 # Work-clustered training queries
│   ├── val.jsonl                   # Validation queries
│   └── test.jsonl                  # Benchmark test queries
├── src/
│   ├── __init__.py
│   ├── config.py                   # Central hyperparameters, device, and paths
│   ├── data_loader.py              # SQLite loader, formatting & dataset generator
│   ├── filters.py                  # 8-layer false-negative protection shield
│   └── metrics.py                  # GPU 2D block-chunked Recall@K evaluator
├── scripts/
│   ├── 01_baseline_benchmark.py    # Step 1: Pretrained Arctic baseline evaluation
│   ├── 02_train_phase1.py          # Step 2: Phase 1 contrastive fine-tuning (MNRL)
│   └── 03_iterative_mining.py      # Step 3: Iterative hard negative mining engine
└── results/                        # Generated benchmark metrics JSONs
```

---

## 🛠️ Step-by-Step Execution Guide

### Step 1: Zero-Shot Baseline Benchmark
Evaluates the untouched pretrained `Snowflake/snowflake-arctic-embed-m-v1.5` over the full 64,552 catalog to establish our baseline:
```bash
python scripts/01_baseline_benchmark.py
```
*Outputs: `results/baseline_metrics.json`*

### Step 2: Contrastive Fine-Tuning (Phase 1)
Trains all 109M parameters using **Multiple Negatives Ranking Loss (MNRL)** with in-batch negatives, duplicate-positive masking, linear warmup + cosine decay, and atomic double-buffered checkpointing:
```bash
python scripts/02_train_phase1.py
```
*Outputs: `arctic_output/arctic_phase1/best_model` and `results/phase1_metrics.json`*

### Step 3: Iterative Hard Negative Mining
Re-encodes the catalog with the Phase 1 model, mines its specific false-positive confusions using the 8-layer shield, and retrains:
```bash
python scripts/03_iterative_mining.py
```

---

## 💻 Hardware & Environment

* **Target Hardware**: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM) / Intel i7-13645HX (20 Threads).
* **VRAM Optimizations**:
  * FP16 Mixed Precision (`torch.cuda.amp.autocast`).
  * Gradient Checkpointing (`encoder.gradient_checkpointing_enable()`).
  * Micro-batch size 16 with Gradient Accumulation 2 (Effective Batch Size = 32).
  * 2D Block-Chunking for full-catalog dot products.

---

## 👤 Author
**Nizam** (`nansari.0134@gmail.com`)  
GitHub: [https://github.com/nansari-0134/](https://github.com/nansari-0134/)
