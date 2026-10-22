# 📋 Project Context & Living Execution Tracker

## 1. Project Overview & Objectives
* **Project Name**: ArcticDense: Neural Book Retriever (Snowflake Arctic Embed v1.5)
* **Directory**: `neural_book_search_arctic/`
* **Author / Git Identity**: Nizam (`nansari.0134@gmail.com`) | GitHub: [https://github.com/nansari-0134/](https://github.com/nansari-0134/)
* **Primary Target Model**: `Snowflake/snowflake-arctic-embed-m-v1.5`
  * Architecture: Arctic Transformer (108.9M Parameters)
  * Context Window: 512 tokens
  * Embedding Dimension: 768-d (with native Matryoshka MRL support)
  * Pretrained Retrieval Prompt: `"Represent this sentence for searching relevant passages: "`
* **Primary Catalog**: 64,552 Goodreads books (structured by Title, Authors, Genres, Description)
* **Final Deliverables**:
  1. Fine-tuned high-performance book retriever outperforming baseline.
  2. Multi-cycle iterative hard negative mining pipeline.
  3. Model published to Hugging Face Hub with complete metadata & model card.
  4. Comprehensive MkDocs technical documentation site ready for GitHub Pages.

---

## 2. Hardware Environment
* **OS**: Windows
* **CPU**: Intel Core i7-13645HX (20 Threads)
* **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)
* **Python Environment**: `..\.venv_gpu` (PyTorch 2.6.0+cu124, CUDA active, FP16 enabled)
* **Memory Optimizations**:
  * FP16 mixed-precision inference & training (`torch.cuda.amp.autocast`)
  * PyTorch Gradient Checkpointing (`encoder.gradient_checkpointing_enable()`)
  * Micro-batch size: 16 | Gradient Accumulation: 2 (Effective Batch Size = 32)
  * 2D Block-Chunked evaluation matrix multiplication

---

## 3. Step-by-Step Execution Plan & Status

| Phase | Milestone / Task | Status | Output Artifact | Notes |
| :---: | :--- | :---: | :--- | :--- |
| **0** | **Environment & Repo Setup** | ✅ **COMPLETE** | `neural_book_search_arctic/` | Git initialized, `.gitignore`, clean architecture |
| **0b** | **Documentation & Tracker Setup** | ✅ **COMPLETE** | `mkdocs.yml`, `docs/`, `PROJECT_CONTEXT.md` | MkDocs Material + living documentation |
| **1** | **Zero-Shot Baseline Benchmark** | ✅ **COMPLETE** | `results/baseline_metrics.json` | Test: R@1: 21.67%, R@10: 55.40%, R@50: 69.57% |
| **2** | **Phase 1 Contrastive Fine-Tuning** | ✅ **COMPLETE** | `arctic_output/arctic_phase1/best_model` | Test: R@1: 29.16%, R@10: 64.67%, R@50: 78.20% |
| **3a** | **Iterative Negative Mining (Cycle 1)**| ✅ **COMPLETE** | `data/train_arctic_v2.jsonl` | 199,808 pairs mined; 1.8M false negatives shielded |
| **3b** | **Fine-Tuning Cycle 1** | ✅ **COMPLETE** | `arctic_output/arctic_v2/best_model` | Test: R@1: 35.60%, R@10: 73.96%, R@50: 84.66% (+18.6% R@10) |
| **3c** | **Iterative Negative Mining (Cycle 2)**| 🚀 **IN PROGRESS** | `data/train_arctic_v3.jsonl` | Mining 199,808 pairs with Cycle 1 model + 8-layer shield |
| **3d** | **Fine-Tuning Cycle 2** | ⏳ **PENDING** | `arctic_output/arctic_v3/best_model` | Progressive model evolution |
| **4** | **Final Benchmarking & Comparison** | ⏳ **PENDING** | `results/final_comparison.json` | Side-by-side progression table |
| **5** | **Hugging Face Hub Model Publication**| ⏳ **PENDING** | Hugging Face Repository | Automatic upload script with full model card |
| **6** | **MkDocs Build & GitHub Pages Prep** | ⏳ **PENDING** | `site/` | Complete static HTML documentation site |

---

## 4. Key Engineering Guardrails
1. **False-Negative Defense Shield**:
   - Skips ranks 0–1 (unindexed editions / sequels danger zone).
   - Cosine ceiling: $\le 0.88$ (blocks near-duplicate plots).
   - Normalization & subtitle stripping.
   - Word overlap threshold: $\ge 65\%$ shared non-stopwords rejected.
   - Regex series & universe matching.
   - Author shelf matching from Goodreads popular shelves.
   - Negative frequency capping: Max 30 occurrences per book across training dataset.
2. **Duplicate-Positive Loss Masking**:
   - Mask off-diagonal identical positive IDs to $-\infty$ in micro-batch similarity logits to avoid penalizing correct matches.
3. **Crash-Proof Checkpointing**:
   - Atomic double-buffered checkpointing (`resume_state.pt` and `.prev`) written to disk every 5 minutes.
4. **Git Protection**:
   - Strictly keeps files $> 100\text{ MB}$ out of Git commits.
