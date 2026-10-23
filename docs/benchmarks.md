# 📈 Benchmark Progression & Empirical Results

This document tracks empirical retrieval metrics across all training stages on the **64,552 book catalog** evaluated over **32,170 test queries** and **32,340 validation queries** (100% work-clustered, zero edition contamination).

---

## 📊 Comprehensive Results Progression

```mermaid
graph LR
    subgraph S1 ["Stage 1: Pretrained Baseline"]
        B1["R@1: 21.67%"]
        B5["R@5: 48.41%"]
        B10["R@10: 55.40%"]
        B50["R@50: 69.57%"]
    end
    subgraph S2 ["Stage 2: In-Batch Training"]
        P1["R@1: 29.16% (+7.49%)"]
        P5["R@5: 58.29% (+9.88%)"]
        P10["R@10: 64.67% (+9.27%)"]
        P50["R@50: 78.20% (+8.63%)"]
    end
    subgraph S3 ["Stage 3: Targeted Hard Negatives"]
        C1["R@1: 35.60% (+13.93%)"]
        C5["R@5: 68.09% (+19.68%)"]
        C10["R@10: 73.96% (+18.56%)"]
        C50["R@50: 84.66% (+15.09%)"]
    end
    subgraph S4 ["Stage 4: Progressive Refinement"]
        C2_1["R@1: 35.48% (+13.81%)"]
        C2_5["R@5: 68.50% (+20.09%)"]
        C2_10["R@10: 74.70% (+19.30%)"]
        C2_50["R@50: 85.51% (+15.94%)"]
    end
    S1 --> S2 --> S3 --> S4
```

---

## 🏆 Full-Catalog Evaluation Comparison (64,552 Books)

All evaluations search across the **entire 64,552-book catalog** in exact rank order using GPU 2D block-chunking:

| Evaluation Split | Model Stage | Recall@1 | Recall@5 | Recall@10 | Recall@50 | Delta vs Baseline | Relative Gain |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Test Set**<br>*(32,170 queries)* | **Stage 1: Pretrained Baseline** | 21.67% | 48.41% | 55.40% | 69.57% | *Starting Point* | — |
| | **Stage 2: In-Batch Training** | 29.16% | 58.29% | 64.67% | 78.20% | :rocket: **+7.49% R@1, +9.27% R@10** | +16.7% |
| | **Stage 3: Targeted Hard Negatives** | 35.60% | 68.09% | 73.96% | 84.66% | 🌟 **+13.93% R@1, +18.56% R@10** | +33.5% |
| | **Stage 4: Progressive Refinement (Champion)** | **35.48%** | **68.50%** | **74.70%** | **85.51%** | 👑 **+13.81% R@1, +19.30% R@10** | **+34.8%** |
| **Validation Set**<br>*(32,340 queries)* | **Stage 1: Pretrained Baseline** | 22.24% | 49.02% | 55.43% | 69.41% | *Starting Point* | — |
| | **Stage 2: In-Batch Training** | 29.86% | 59.55% | 65.70% | 78.36% | :rocket: **+7.62% R@1, +10.27% R@10** | +18.5% |
| | **Stage 3: Targeted Hard Negatives** | 36.01% | 68.78% | 74.41% | 84.84% | 🌟 **+13.77% R@1, +18.98% R@10** | +34.2% |
| | **Stage 4: Progressive Refinement (Champion)** | **36.39%** | **69.51%** | **75.04%** | **85.65%** | 👑 **+14.15% R@1, +19.61% R@10** | **+35.4%** |

---

## ⚡ Stage 2 Training Telemetry (In-Batch Contrastive Training)

Executed on a single consumer laptop GPU (**NVIDIA GeForce RTX 4050 6GB VRAM**):

| Hyperparameter / Metric | Observed Value | Description |
| :--- | :--- | :--- |
| **Dataset Size** | **258,250 pairs** | 51,650 training books $\times$ 5 synthetic queries |
| **Total Steps** | **16,141 steps** | 1 full epoch with batch size 16 and grad accum 2 (eff. batch 32) |
| **Execution Time** | **761.2 minutes (~12.7 hours)** | Average step latency: ~2.4 seconds |
| **Loss Trajectory** | **`1.7508` $\rightarrow$ `0.6073`** | Stable monotonic decline without spikes |
| **Peak Learning Rate** | **`1.50e-05`** | Linear warmup (807 steps) + Cosine decay to `0.0` |
| **VRAM Footprint** | **5,879 MiB / 6,141 MiB** | 95.7% VRAM utilization with FP16 + Gradient Checkpointing |
| **Thermal Profile** | **74°C sustained** | Stable power draw: 73W / 80W max |

### Loss Milestones During Stage 2 Training

| Training Progress | Optimizer Step | Recorded MNRL Loss | Learning Rate | Elapsed Time |
| :---: | :---: | :---: | :---: | :---: |
| **0.3%** | Step 50 / 16,141 | **1.7508** | $4.65 \times 10^{-7}$ | 1.9 min |
| **12.4%** | Step 2,000 / 16,141 | **1.2842** | $1.48 \times 10^{-5}$ | 94.2 min |
| **24.8%** | Step 4,000 / 16,141 | **1.0519** | $1.39 \times 10^{-5}$ | 188.5 min |
| **49.6%** | Step 8,000 / 16,141 | **0.8234** | $9.82 \times 10^{-6}$ | 377.0 min |
| **74.3%** | Step 12,000 / 16,141 | **0.6841** | $4.31 \times 10^{-6}$ | 565.8 min |
| **100.0%** | Step 16,141 / 16,141 | **0.6073** | $0.00$ | 761.2 min (12.7h) |

---

## ⚡ Stage 3 Training Telemetry (Targeted Hard-Negative Training)

Introduced **Top-5 actively mined hard negatives per query** (199,808 adversarial pairs) with sub-batch negative chunking and non-reentrant gradient checkpointing:

| Hyperparameter / Metric | Observed Value | Description |
| :--- | :--- | :--- |
| **Dataset Size** | **199,808 pairs** | 5 adversarial hard negatives mined from 64,552 books |
| **Micro-Batch / Grad Accum** | **Batch 4 / Accum 8** | Effective batch size: 32 (exact gradient preservation) |
| **Total Micro-Steps** | **49,952 micro-steps** | 6,244 full optimizer steps |
| **Negative Chunking** | **Chunk Size: 12** | Prevents WDDM PCIe paging thrash |
| **Execution Time** | **416.7 minutes (~6.9 hours)** | Average step latency: **0.49 seconds** (vs 26.3s unoptimized) |
| **VRAM Footprint** | **2,835 – 3,429 MiB** | Stable 55% utilization on RTX 4050 6GB |
| **Thermal & Power** | **74–77°C sustained** | Stable power draw: 73W – 76W |
| **Final Loss** | **`1.4070` $\rightarrow$ `0.5841`** | Strong convergence on adversarial pairs |

---

## ⚡ Stage 4 Training Telemetry (Progressive Refinement - Champion)

Refined the retriever on **179,319 progressive adversarial pairs** (3 fresh negatives + 2 retained negatives) warm-started from Stage 3 weights:

| Hyperparameter / Metric | Observed Value | Description |
| :--- | :--- | :--- |
| **Base Model Checkpoint** | **`arctic_v2/best_model`** | Warm-starts directly from Stage 3 model |
| **Dataset Size** | **179,319 pairs** | Filtered from Stage 3 confusions with 8-layer shield |
| **Micro-Batch / Grad Accum** | **Batch 4 / Accum 8** | Effective batch size: 32 (exact gradient preservation) |
| **Total Micro-Steps** | **44,830 micro-steps** | 5,604 full optimizer steps |
| **Learning Rate** | **`1.20e-05`** | Cosine decay with linear warmup (560 steps) |
| **Execution Time** | **374.8 minutes (~6.2 hours)** | Average step latency: **0.50 seconds** |
| **VRAM Footprint** | **2,850 – 3,450 MiB** | Stable 55% utilization on RTX 4050 6GB |
| **Thermal & Power** | **72–76°C sustained** | Stable power draw: 70W – 74W |
| **Loss Trajectory** | **`1.1224` $\rightarrow$ `0.5312`** | High transfer efficiency from Stage 3 weights |

---

## 🎯 Evaluation Protocol & Recall Math

For any query $q$, let $\text{Rank}(q, d^+)$ be the 1-indexed position of the ground-truth book $d^+$ when sorting all $N = 64,552$ catalog books by descending cosine similarity:

$$\text{Recall@K} = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}\left(\text{Rank}(q, d^+) \le K\right)$$

* **Recall@1**: Measures exact first-try precision. A jump from **21.67% to 35.48%** (+63.7% relative gain) means more than 1 in every 3 user trope queries returns the target book as the very first result.
* **Recall@10**: Represents "Page 1" retrieval performance. Reaching **74.70%** (Validation: **75.04%**) ensures three out of four semantic queries place the ground truth book within the top 10 search results.
* **Recall@50**: Represents candidate generation recall. At **85.51%** (Validation: **85.65%**), downstream rerankers or interactive exploration modes receive an exceptionally rich candidate pool.
