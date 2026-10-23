# 🚀 Jina Reranker v2 Fine-Tuning Monitor (Task-2240)

* **Model Architecture**: `jinaai/jina-reranker-v2-base-multilingual` (278M parameters, XLM-RoBERTa backbone)
* **Target Output**: [`arctic_output/jina_reranker/best_model`](file:///c:/Users/Shilpa/OneDrive/Documents/Book%20Semantic%20Search/neural_book_search_arctic/arctic_output/jina_reranker/best_model)
* **Dataset**: 29,922 training pairs (10,000 queries + mined hard negatives from Champion Arctic retriever)
* **Hardware Profile**: NVIDIA GeForce RTX 4050 6GB Laptop GPU with Native BFloat16 (`bf16=True`)
* **Effective Batch Size**: 8 (micro-batch) × 4 (grad accum) = 32
* **Current Progress**: Step 936 / 936 (**100% COMPLETE** in 16.48 minutes)
* **Champion Model Export**: [`arctic_output/jina_reranker/best_model`](file:///c:/Users/Shilpa/OneDrive/Documents/Book%20Semantic%20Search/neural_book_search_arctic/arctic_output/jina_reranker/best_model)
* **Raw Live Log**: [`training/task_2240_jina_training.log`](file:///c:/Users/Shilpa/OneDrive/Documents/Book%20Semantic%20Search/neural_book_search_arctic/training/task_2240_jina_training.log)

---

## 📈 Optimization & Evaluation Milestones

| Training Step | Recorded Loss | Grad Norm | Learning Rate | Status / Milestone |
| :---: | :---: | :---: | :---: | :--- |
| **Step 50** | `0.5977` | 3.719 | $7.82 \times 10^{-6}$ | Warmup progression |
| **Step 100** | `0.5580` | 3.188 | $1.49 \times 10^{-5}$ | Peak learning rate reached |
| **Step 150** | `0.5533` | 3.812 | $1.40 \times 10^{-5}$ | Steady contrastive optimization |
| **Step 200** | `0.5497` | 3.016 | $1.31 \times 10^{-5}$ | Loss continues monotonic decline |
| **Step 233 (Eval 1)** | **`eval_loss: 0.4493`** | — | — | 🌟 **Checkpoint 1 saved** |
| **Step 466 (Eval 2)** | **`eval_loss: 0.4321`** | — | — | 🌟 **Checkpoint 2 saved** |
| **Step 700 (Eval 3)** | **`eval_loss: 0.4267`** | — | — | 🌟 **Checkpoint 3 saved** |
| **Step 936 (Final Eval)** | **`eval_loss: 0.4258`** | — | — | 👑 **Champion Checkpoint Selected & Saved** |

---

## 💻 Hardware Telemetry
* **GPU VRAM Utilization**: ~2,710 MiB / 6,144 MiB (44% utilization)
* **Throughput**: ~1.01 steps/second
* **Attention Engine**: PyTorch Native SDPA
