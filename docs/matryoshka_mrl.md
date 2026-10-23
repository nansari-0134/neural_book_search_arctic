# 🪆 Matryoshka Representation Learning (MRL) & Vector Compression

**Matryoshka Representation Learning (MRL)** is a core capability native to **Snowflake Arctic Embed (`snowflake-arctic-embed-m-v1.5`)**. It allows high-dimensional dense embeddings to be dynamically sliced into smaller sub-vectors at search time without requiring model retraining, fine-tuning, or decompression steps.

---

## 1. The Core Intuition: Russian Nesting Dolls

In traditional embedding models (such as BERT, RoBERTa, or standard sentence-transformers), vector coordinates are homogenous and entangled. Every dimension from $1$ to $768$ carries an arbitrary fraction of the semantic representation. If you slice a traditional 768-d vector down to 256-d, the geometric alignment collapses and retrieval accuracy drops to near zero.

**MRL fundamentally alters how the transformer structures its latent space:**

* Like a set of **Russian Matryoshka nesting dolls**, where smaller, fully functional dolls are contained within larger dolls, an MRL embedding contains smaller, complete, and fully functional sub-vectors nested directly inside the full embedding.
* The transformer is trained to prioritize and pack the **most critical, high-level semantic concepts** (overarching genre, primary plot tropes, emotional tone) into the earliest dimensions (the first 64 and 128 coordinates).
* The subsequent dimensions (129 through 768) provide progressively finer semantic granularity, stylistic subtleties, and edge-case disambiguation.

```mermaid
graph LR
    subgraph Full ["Full Latent Space (768-d)"]
        D64["[1:64] Coarse Semantics<br>Genre, Setting, Tone"]
        D128["[65:128] Subgenres & Tropes"]
        D256["[129:256] Plot Mechanics & Character Roles"]
        D512["[257:512] Granular Thematic Relationships"]
        D768["[513:768] Micro-Context & Disambiguation"]
    end
    D64 --> D128 --> D256 --> D512 --> D768
```

---

## 2. Mathematical Foundation of MRL Training

During contrastive pre-training and fine-tuning, standard bi-encoders optimize contrastive loss (such as Multiple Negatives Ranking Loss) strictly at the maximum vector dimension $d = 768$:

$$\mathcal{L}_{\text{standard}} = \mathcal{L}\left(\mathbf{q}_{1:d}, \mathbf{d}_{1:d}\right)$$

In contrast, **Matryoshka Representation Learning** evaluates the contrastive objective simultaneously across a pre-defined set of nested sub-vector granularities $\mathcal{M} = \{64, 128, 256, 512, 768\}$:

$$\mathcal{L}_{\text{MRL}} = \sum_{m \in \mathcal{M}} w_m \cdot \mathcal{L}\left(\frac{\mathbf{q}_{1:m}}{\|\mathbf{q}_{1:m}\|_2}, \frac{\mathbf{d}_{1:m}}{\|\mathbf{d}_{1:m}\|_2}\right)$$

Where:
* $m \in \mathcal{M}$ represents the target sub-dimension cut.
* $\mathbf{q}_{1:m}$ and $\mathbf{d}_{1:m}$ are the sliced query and document representations.
* $\|\cdot\|_2$ denotes $L_2$ normalization onto the unit hypersphere $\mathbb{S}^{m-1}$.
* $w_m$ is the loss weighting factor (typically $w_m = 1.0$ for uniform gradient contribution).

By backpropagating gradients from multiple sub-vector dimensions through the shared encoder weights, the model forces its initial hidden states to capture maximal mutual information with the document passage.

---

## 3. Traditional Fixed Embeddings vs. Matryoshka Embeddings

| Feature | Traditional Dense Bi-Encoder | Snowflake Arctic Embed (MRL Native) |
| :--- | :--- | :--- |
| **Vector Geometry** | Uniformly entangled representation | **Hierarchically nested representations** |
| **Dimensional Flexibility** | Fixed (always $768$ coordinates) | **Dynamic** ($64$, $128$, $256$, $512$, or $768$) |
| **Slicing Impact** | ❌ Catastrophic accuracy collapse | ✅ **Retains $>99\%$ Recall@10 at 256-d** |
| **Inference Compression Cost** | Requires distillation or separate models | **Zero cost** (2-line array slicing) |
| **Storage & RAM Savings** | 0% (Must store all 768 float values) | **Up to 12x reduction** (from 99MB to 8MB) |

---

## 4. Runtime Vector Truncation & Normalization

At inference time, compressing pre-computed catalog embeddings or online search queries requires **zero neural compute**. It is performed via simple vector array operations:

```python
import numpy as np

def compress_matryoshka_vector(vector: np.ndarray, target_dim: int = 256) -> np.ndarray:
    """
    Truncates a 768-d Snowflake Arctic embedding to a target Matryoshka dimension
    and re-projects it onto the unit hypersphere.
    """
    # 1. Coordinate slicing (zero-copy memory view)
    sub_vec = vector[:target_dim]
    
    # 2. L2 Re-normalization
    norm = np.linalg.norm(sub_vec)
    if norm > 0:
        sub_vec = sub_vec / norm
        
    return sub_vec
```

---

## 5. Empirical Performance on the 64,552 Goodreads Book Catalog

Our production application evaluates the Champion Arctic retriever across the entire **64,552-book catalog** over **1,000 test queries**. The table below documents the exact empirical trade-offs between vector dimensions, retrieval precision, memory footprint, and compute complexity:

| Dimension | Recall@1 | Recall@5 | Recall@10 | Recall@50 | MRR@10 | Recall@10 Retained | RAM Footprint (FP16) | Dot-Product FLOPs | Speedup Factor |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **768-d** *(Full)* | **49.80%** | **66.50%** | **71.90%** | **83.30%** | **0.5726** | **100.0%** *(Baseline)* | **94.6 MiB** | $99.2 \times 10^6$ ops | *Baseline (1.0x)* |
| **512-d** | 49.00% | 65.80% | 70.70% | 82.90% | 0.5624 | **98.33%** | **63.0 MiB** *(33% less)* | $66.1 \times 10^6$ ops | **1.5x faster** |
| **256-d** *(Sweet Spot)* | **48.70%** | **65.60%** | **71.40%** | **83.40%** | **0.5597** | **99.30%** | **31.5 MiB** *(3x less)* | $33.1 \times 10^6$ ops | **3.0x faster** |
| **128-d** | 40.90% | 56.10% | 63.20% | 76.20% | 0.4791 | **87.90%** | **15.8 MiB** *(6x less)* | $16.5 \times 10^6$ ops | **6.0x faster** |
| **64-d** *(Edge)* | 27.10% | 40.50% | 46.80% | 60.00% | 0.3290 | **65.09%** | **7.9 MiB** *(12x less)*| $8.3 \times 10^6$ ops | **12.0x faster** |

### Key Observation: The 256-d "Sweet Spot"
By reducing vector dimension from **768 down to 256**:
* Memory and disk storage dropped by **$3\times$** (from 94.6 MiB to 31.5 MiB).
* Dot-product FLOPs dropped by **$3\times$** (from 99.2M to 33.1M operations per query).
* The search engine still preserves **99.30% of full Recall@10** (71.40% vs 71.90%) and **97.75% of MRR@10** (0.5597 vs 0.5726).
* Even Recall@50 is virtually identical (**83.40% vs 83.30%**), meaning candidate pools fed to downstream rerankers suffer zero candidate degradation.

---

## 6. Interactive MRL Dimension Switching in the Finished Web App

In our production web interface ([`app/`](file:///c:/Users/Shilpa/OneDrive/Documents/Book%20Semantic%20Search/neural_book_search_arctic/app)), users can interactively benchmark MRL dimensional truncation in real time:

* **Interactive Dimension Selector**: The UI provides glassmorphic pills for `768-d`, `512-d`, `256-d`, `128-d`, and `64-d`.
* **Zero Latency Penalty**: Clicking different dimension chips dynamically slices the pre-warmed GPU tensor and executes query scoring instantly.
* **Telemetry Verification**: The live telemetry bar reports the active Matryoshka dimension alongside query execution latency (e.g. `128-d Matryoshka Vector • 6.2 ms`).
