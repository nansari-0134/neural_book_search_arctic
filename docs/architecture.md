# 🏛️ System Architecture

This section covers the mathematical foundations and design decisions behind the dense bi-encoder architecture.

---

## 1. Bi-Encoder Dense Retrieval

In a bi-encoder architecture, queries and documents are mapped into the same continuous vector space independently:

$$\mathbf{q} = \text{Encoder}(Q) \in \mathbb{R}^d, \quad \mathbf{d} = \text{Encoder}(D) \in \mathbb{R}^d$$

Both vectors are $L_2$-normalized such that $\|\mathbf{q}\|_2 = \|\mathbf{d}\|_2 = 1$. The semantic relevance between a query and a document is calculated using cosine similarity, which simplifies to a dot product:

$$\text{Sim}(Q, D) = \mathbf{q} \cdot \mathbf{d} = \sum_{i=1}^{d} q_i d_i$$

### Why Bi-Encoder for Stage 1 Retrieval?
* **Offline Pre-computation**: All 64,552 catalog book descriptions are encoded offline once and cached.
* **Low Latency Query Search**: At search time, only the user query is encoded (~3ms). The search is reduced to a matrix-vector dot product against pre-computed vectors ($< 2\text{ms}$).
* **Total Latency**: $< 5\text{ms}$ over 64k books on an NVIDIA RTX 4050 GPU.

---

## 2. Asymmetric Instruction Tuning

Unlike generic symmetric sentence similarity models, `Snowflake/snowflake-arctic-embed-m-v1.5` uses asymmetric instruction prompts to distinguish query intent from document semantics:

* **Query Representation**:
  ```text
  Represent this sentence for searching relevant passages: {user_query}
  ```
* **Document Representation**:
  ```text
  {catalog_book_text}
  ```

This prompt guides the attention mechanism to position queries into the appropriate retrieval subspace.

---

## 3. Matryoshka Representation Learning (MRL)

Snowflake Arctic natively supports Matryoshka Representation Learning. During training, sub-vector slices are penalized alongside the full vector:

```
Full Vector (768-d):    [========================================] -> Highest Precision
Truncated (512-d):      [==============================]
Truncated (256-d):      [================] -> 3x Faster Dot Product, 3x Less RAM
Truncated (128-d):      [========]
```

This allows users to truncate embeddings to 256 dimensions with negligible ($< 1\%$) drops in Recall@10.
