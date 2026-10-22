# 📈 Benchmark Progression

This document tracks empirical performance across training cycles on the **64,552 book catalog**.

---

## 📊 Progression Table

| Model Stage | Recall@1 | Recall@5 | Recall@10 | Recall@50 | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **BGE-Small Baseline (Prior Reference)** | 11.52% | 32.80% | 38.84% | 53.22% | Untrained BGE-Small zero-shot |
| **BGE-Small Fine-Tuned (Prior Reference)**| 21.27% | 53.85% | 60.45% | 74.68% | Phase 1 fine-tuning result |
| **Snowflake Arctic Baseline (Step 1)** | **21.67%** | **48.41%** | **55.40%** | **69.57%** | Untrained zero-shot evaluation (Val: R@10=55.43%) |
| **Snowflake Arctic Phase 1 (Step 2)** | **29.16%** | **58.29%** | **64.67%** | **78.20%** | Full catalog test benchmark (+7.49% R@1, +9.27% R@10) |
| **Snowflake Arctic Cycle 1 (Step 3a)** | *Pending* | *Pending* | *Pending* | *Pending* | Hard negative mining Cycle 1 |
| **Snowflake Arctic Cycle 2 (Step 3b)** | *Pending* | *Pending* | *Pending* | *Pending* | Hard negative mining Cycle 2 |

---

## 🎯 Evaluation Protocol
* **Catalog Size**: 64,552 distinct books
* **Evaluation Queries**: 32,170 test queries (work-clustered, 0% edition overlap with train)
* **Metric**: Exact ground-truth rank position counted via GPU 2D block-chunking:

$$\text{Recall@K} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\text{Rank}_i < K)$$
