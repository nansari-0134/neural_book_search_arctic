# 🏛️ System Architecture

This document details the mathematical foundations, neural bi-encoder design, asymmetric instruction tuning, and the production application architecture of **ArcticDense**.

---

## 1. Bi-Encoder Dense Retrieval

In dense neural retrieval, queries and documents are independently mapped into the same continuous latent vector space $\mathbb{R}^d$:

$$\mathbf{q} = \text{Encoder}_{\theta}(Q) \in \mathbb{R}^d, \quad \mathbf{d} = \text{Encoder}_{\theta}(D) \in \mathbb{R}^d$$

Both representations are $L_2$-normalized onto the unit hypersphere such that $\|\mathbf{q}\|_2 = \|\mathbf{d}\|_2 = 1$. The semantic relevance score between query $Q$ and catalog book $D$ is computed via cosine similarity, which simplifies to an inner dot product:

$$\text{Sim}(Q, D) = \mathbf{q} \cdot \mathbf{d} = \sum_{i=1}^{d} q_i d_i$$

```mermaid
graph TD
    subgraph Offline_Indexing ["Offline Indexing (Once)"]
        D1["64,552 Goodreads Book Descriptions"] --> EncD["Document Encoder (Arctic v1.5)"]
        EncD --> NormD["L2 Normalization"]
        NormD --> EmbCache["Embeddings Matrix (64552 x 768) (~99 MB)"]
    end

    subgraph Online_Query ["Online Search (< 15 ms)"]
        Q["User Search Query"] --> Prompt["Asymmetric Instruction Prefix"]
        Prompt --> EncQ["Query Encoder (Arctic v1.5)"]
        EncQ --> NormQ["L2 Normalization"]
        NormQ --> Dot["Vector Dot Product (q · D^T)"]
        EmbCache --> Dot
        Dot --> TopK["Top-K Candidate Ranking"]
        TopK --> Enrich["SQLite Metadata Enrichment"]
        Enrich --> UI["Web Application Interface"]
    end
```

### Why Bi-Encoder for Catalog Retrieval?
* **Offline Pre-computation**: All 64,552 catalog book descriptions are encoded offline once and cached.
* **Blazing Fast Query Search**: At search time, only the user query is encoded (~3ms on GPU). The search is reduced to a matrix-vector dot product against pre-computed vectors ($< 2\text{ms}$).
* **High Candidate Recall**: Filters the entire 64,552-book catalog down to the top 50 candidates in $< 15\text{ms}$ with **85.51% Recall@50**.

---

## 2. Two-Stage Retrieval Paradigm: Bi-Encoder + Cross-Encoder

In modern neural information retrieval, a single model cannot optimize both sub-15ms latency across tens of thousands of items and deep token-to-token semantic understanding. ArcticDense adopts the **Two-Stage Cascade Architecture**:

```mermaid
graph LR
    Catalog["64,552 Catalog Books"] --> Stage1["Stage 1: Arctic Bi-Encoder<br>(MIPS Cosine Search)"]
    Stage1 -->|15 ms (Recall@50: 85.51%)| Candidates["Top-50 Candidate Books"]
    Candidates --> Stage2["Stage 2: Cross-Encoder Reranker<br>(Joint Multi-Head Attention)"]
    Stage2 -->|45 ms (Target Recall@1: 55-65%+)| PrecisionRank["Top-10 Precision Ranked Books"]
```

### Architectural Comparison

| Dimension | Stage 1: Arctic Bi-Encoder | Stage 2: Cross-Encoder Reranker |
| :--- | :--- | :--- |
| **Input Format** | $E(q)$ and $E(d)$ independently | `[CLS] Query [SEP] Book Description [SEP]` |
| **Attention Mechanism** | Intra-sequence only (no cross-attention) | **Full All-to-All Cross-Attention** across all tokens |
| **Scoring Function** | Dot Product $\mathbf{q} \cdot \mathbf{d}$ | Multi-layer Perceptron head: $\sigma(W h_{\text{CLS}} + b)$ |
| **Search Scope** | Entire Catalog (**64,552 books**) | Top Candidates (**50 books**) |
| **Primary Objective** | **High Candidate Recall** (85.51% Recall@50) | **High Top-1 Precision** & Nuance Disambiguation |
| **Execution Latency** | **~12–15 ms** on GPU | **~40–50 ms** on GPU |

---

## 3. Asymmetric Instruction Tuning


Unlike symmetric sentence-similarity transformers, `Snowflake/snowflake-arctic-embed-m-v1.5` employs asymmetric instruction conditioning. Queries and documents occupy different distribution spaces; an explicit prompt aligns the query's latent representation with document passages:

* **Query Representation**:
  ```text
  Represent this sentence for searching relevant passages: {user_query}
  ```
* **Document Representation**:
  ```text
  Title: {title} | Genres: {genres} | Description: {description}
  ```

This prompt guides the cross-attention mechanism to position queries into the appropriate retrieval subspace, improving zero-shot transfer and domain adaptation.

---

## 4. Matryoshka Representation Learning (MRL)

Snowflake Arctic natively incorporates **Matryoshka Representation Learning**. During contrastive pre-training and fine-tuning, loss gradients are applied across multiple nested sub-vector dimensions simultaneously:

```mermaid
graph LR
    subgraph MRL ["Nested Sub-Vectors"]
        V768["768-d: Full Precision (Baseline: 100% capacity)"]
        V512["512-d: 1.5x Storage Compression"]
        V256["256-d: 3x Storage Compression & Speedup"]
        V128["128-d: 6x Storage Compression"]
        V64["64-d: Ultra-light Edge Deployment"]
    end
    V768 -.-> V512 -.-> V256 -.-> V128 -.-> V64
```

### Mathematical Formulation
For any embedding $\mathbf{z} \in \mathbb{R}^d$ and a target Matryoshka dimension $m \le d$:
1. Slice the vector to the first $m$ coordinates: $\mathbf{z}_{1:m}$
2. Re-normalize to unit length:
   $$\hat{\mathbf{z}}_{1:m} = \frac{\mathbf{z}_{1:m}}{\|\mathbf{z}_{1:m}\|_2}$$

In empirical benchmarks, truncating vectors from **768-d down to 256-d** reduces memory usage and dot-product compute by $3\times$ while retaining over **99% of Recall@10**.

---

## 5. Finished Product Web Application Architecture

The complete system includes a full-stack, production-ready web application (`app/`) allowing users to interact with the fine-tuned champion model:

```mermaid
graph TD
    Client["Web Browser / Client UI"] <-->|HTTP / REST JSON| FastAPI["FastAPI Application (app/app.py)"]
    
    subgraph Backend ["Inference & Search Engine (app/search_engine.py)"]
        FastAPI --> SearchEngine["BookSearchEngine Singleton"]
        SearchEngine --> Model["Champion Model (Snowflake Arctic v1.5)"]
        SearchEngine --> MRL["MRL Truncation & L2 Normalizer"]
        SearchEngine --> DotProd["Matrix Similarity Engine (NumPy / Torch)"]
        DotProd --> DB[("SQLite Database (books_metadata.sqlite3)")]
    end

    subgraph Frontend ["Glassmorphic Frontend (Vanilla HTML/CSS/JS)"]
        Client --> SearchBar["Live Debounced Search Bar"]
        Client --> MRLChips["MRL Dimension Selector (64-d to 768-d)"]
        Client --> GenreFilter["Dynamic Genre & Shelf Filter"]
        Client --> BookCards["Responsive Result Cards with Similarity Scores"]
        Client --> Modal["Interactive Book Synopsis Modal"]
    end
```

The application is entirely self-contained, requiring zero external search infrastructure (such as Elasticsearch or Pinecone), running directly against the local SQLite database and pre-computed embedding cache.
