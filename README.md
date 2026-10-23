# ❄️ ArcticDense: Two-Stage Cascade Neural Book Search Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)
[![Snowflake Arctic](https://img.shields.io/badge/Snowflake-Arctic--Embed--M--v1.5-blue?logo=snowflake)](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5)
[![Jina Reranker v2](https://img.shields.io/badge/Jina-Reranker--v2--278M-blueviolet)](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **ArcticDense** is a production-grade, Two-Stage Cascade neural book retrieval and search platform fine-tuned across **64,552 Goodreads books**. It combines a **109M parameter Snowflake Arctic Embed Bi-Encoder** (MRL 64D–768D), an **Elasticsearch BM25 lexical engine** with Reciprocal Rank Fusion (RRF), a **278M parameter Jina Reranker v2 Cross-Encoder**, and **Redis sub-millisecond RAM caching**.

---

## 📌 Table of Contents
1. [⚡ Quick Start (< 2 Minutes)](#-quick-start--2-minutes)
2. [🌐 REST API Reference](#-rest-api-reference)
3. [🏛️ System Architecture & Workflow](#️-system-architecture--workflow)
4. [🔍 Step-by-Step Query Execution Lifecycle](#-step-by-step-query-execution-lifecycle)
5. [🏆 Empirical Benchmarks & Performance Progression](#-empirical-benchmarks--performance-progression)
6. [📐 Evaluation Metrics Guide (MRR, mAP, NDCG, Recall)](#-evaluation-metrics-guide)
7. [🧠 Core Techniques & Why They Were Used](#-core-techniques--why-they-were-used)
8. [🤗 Hugging Face Model Upload Guide](#-hugging-face-model-upload-guide)
9. [📂 Repository Structure](#-repository-structure)
10. [👤 Author & License](#-author--license)

---

## ⚡ Quick Start (< 2 Minutes)

You can launch the complete search platform using either Docker (recommended) or a local Python environment.

### Option A: 1-Command Docker Launch (Recommended)

```bash
# Clone the repository
git clone https://github.com/nansari-0134/neural_book_search_arctic.git
cd neural_book_search_arctic

# Launch with NVIDIA GPU acceleration (CUDA)
docker compose up -d

# OR launch in CPU fallback mode
docker compose -f docker-compose.cpu.yml up -d
```

Open your browser at **`http://localhost:8080`** for the web application, or visit **`http://localhost:8080/docs`** for interactive Swagger API documentation.

---

### Option B: Local Python Environment

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the FastAPI server
uvicorn app.app:app --host 0.0.0.0 --port 8080
```

---

## 🌐 REST API Reference

ArcticDense provides high-performance RESTful JSON endpoints for search, system statistics, and health monitoring.

### 1. `GET /api/search` or `POST /api/search`

Executes semantic, lexical, or hybrid search with optional Matryoshka dimension truncation and Stage 2 Cross-Encoder reranking.

#### Request Parameters (Query String or JSON Body)

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `q` | `string` | *Required* | Natural language search query (e.g. `"cyberpunk detective AI conspiracy"`) |
| `dim` | `int` | `768` | Matryoshka MRL dimension (`64`, `128`, `256`, `512`, `768`) |
| `top_k` | `int` | `10` | Number of ranked books to return (1 to 50) |
| `rerank` | `bool` | `false` | Enable Stage 2 Cross-Encoder precision reranking over Top 50 candidates |
| `mode` | `string` | `"hybrid"` | Retrieval mode: `"hybrid"` (Dense + BM25 via RRF), `"dense"` (Arctic only), or `"lexical"` (BM25 only) |
| `genre` | `string` | `null` | Optional genre / shelf filter (e.g. `"science-fiction"`, `"fantasy"`, `"mystery"`) |

#### Example: `cURL` GET Request
```bash
curl -X GET "http://localhost:8080/api/search?q=cyberpunk+detective&dim=768&top_k=3&rerank=true&mode=hybrid"
```

#### Example: `POST` with JSON Body
```bash
curl -X POST "http://localhost:8080/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "q": "atmospheric gothic boarding school mystery",
    "dim": 256,
    "top_k": 3,
    "rerank": true,
    "mode": "hybrid",
    "genre": "mystery"
  }'
```

#### Example: Python `requests`
```python
import requests

response = requests.post(
    "http://localhost:8080/api/search",
    json={
        "q": "space exploration alien artifact mystery",
        "dim": 768,
        "top_k": 5,
        "rerank": True,
        "mode": "hybrid"
    }
)
results = response.json()
print(f"Retrieved {len(results['results'])} books in {results['latency_ms']}ms (Cache Hit: {results['cache_hit']})")
for b in results["results"]:
    print(f"- {b['title']} (Score: {b['score']:.4f})")
```

#### Sample JSON Response
```json
{
  "query": "cyberpunk detective",
  "mode": "hybrid",
  "latency_ms": 178.4,
  "bi_encoder_latency_ms": 11.2,
  "rerank_latency_ms": 167.2,
  "reranked": true,
  "reranker_name": "Jina Contrastive Reranker v2 (InfoNCE 278M)",
  "mrl_dim": 768,
  "genre_filter": null,
  "cache_hit": false,
  "total_catalog": 64552,
  "results": [
    {
      "book_id": "22328",
      "title": "Neuromancer (Sprawl #1)",
      "score": 0.8842,
      "score_pct": "88.4%",
      "bi_encoder_score": 0.8124,
      "genres": ["science-fiction", "cyberpunk", "fiction"],
      "snippet": "Case was the sharpest data-thief in the matrix...",
      "full_description": "Case was the sharpest data-thief in the matrix until he crossed the wrong people...",
      "reranked": true
    }
  ]
}
```

---

### 2. `GET /api/health`
Returns immediate container and index health:
```bash
curl http://localhost:8080/api/health
```
```json
{
  "status": "ready",
  "ready": true,
  "catalog_indexed": 64552
}
```

---

### 3. `GET /api/stats`
Returns system status, active hardware device, MRL dimensions, and cache metrics:
```bash
curl http://localhost:8080/api/stats
```
```json
{
  "status": "ready",
  "model_name": "Snowflake Arctic Embed (v1.5) - Fine-Tuned Champion",
  "catalog_size": 64552,
  "device": "cuda",
  "fp16_enabled": true,
  "supported_mrl_dimensions": [64, 128, 256, 512, 768],
  "redis_cache": {
    "enabled": true,
    "connected": true,
    "hits": 142,
    "misses": 18,
    "hit_ratio_percent": 88.75
  },
  "elasticsearch": {
    "enabled": true,
    "connected": true,
    "document_count": 64552
  }
}
```

---

## 🏛️ System Architecture & Workflow

ArcticDense uses a **Two-Stage Retrieval Cascade** with hybrid lexical-semantic fusion and sub-millisecond RAM caching:

```mermaid
flowchart TD
    Client["Client / Web UI / REST API"] --> RedisCache{"1. Redis Cache Lookup<br>(sha256 key)"}
    
    RedisCache -- "Cache Hit (< 1ms)" --> FastReturn["Instant Response (< 1ms)"]
    
    RedisCache -- "Cache Miss" --> Stage1Parallel["2. Stage 1: Parallel Candidate Generation"]
    
    subgraph Stage1Parallel ["Stage 1: Multi-Modal Retrieval (Top 50 Candidates)"]
        Arctic["Snowflake Arctic Embed M v1.5 (109M)<br>Dense Vector Semantic Search<br>(Matryoshka MRL 64-768D)"]
        ES["Elasticsearch Cluster (BM25)<br>Multi-Field Lexical Search<br>(title^3.0, author^2.0, genres^1.5)"]
    end
    
    Arctic --> RRF["3. Reciprocal Rank Fusion (RRF)<br>RRF(d) = sum 1 / (60 + rank(d))"]
    ES --> RRF
    
    RRF --> Candidates["Top-50 Fused Candidates"]
    
    Candidates --> JinaRerank["4. Stage 2: Jina Reranker v2 (278M)<br>Joint Token-to-Token Cross-Attention<br>(InfoNCE Contrastive Trained)"]
    
    JinaRerank --> CascadeFusion["5. Hybrid Cascade Fusion<br>0.30 Dense + 0.70 Reranker Score"]
    
    CascadeFusion --> Metadata["6. SQLite Metadata Enrichment<br>(Titles, Shelves, Synopses, Ratings)"]
    
    Metadata --> RedisStore["7. Write to Redis Cache (TTL: 1 hour)"]
    RedisStore --> ClientResponse["Deliver Final Ranked JSON Response"]
```

---

## 🔍 Step-by-Step Query Execution Lifecycle

When a query arrives at the system, it traverses the following 7 steps:

1. **Step 1: In-Memory Cache Verification (`< 1ms`)**:
   The engine computes a deterministic SHA256 hash of `(query, dim, rerank, mode, genre)`. If present in Redis, the cached JSON response is returned immediately.
2. **Step 2: Multi-Modal Candidate Retrieval (`~10–12ms`)**:
   - **Dense Semantic Stream**: The query is prepended with the asymmetric retrieval prompt (`"Represent this sentence for searching relevant passages: "`) and encoded into a 768-d unit vector. A matrix dot-product is evaluated against the 64,552 catalog embeddings.
   - **Lexical BM25 Stream**: In parallel, Elasticsearch searches indexed text with field boosts (`title^3.0`, `author^2.0`, `genres^1.5`, `description^1.0`).
3. **Step 3: Reciprocal Rank Fusion (RRF)**:
   The two ranked candidate streams are merged using RRF ($k=60$):
   $$\text{RRF}(d) = \frac{1}{60 + \text{rank}_{\text{dense}}(d)} + \frac{1}{60 + \text{rank}_{\text{BM25}}(d)}$$
4. **Step 4: Cross-Encoder Precision Reranking (`~160–170ms`)**:
   The top 50 fused candidates are paired with the query as joint sequences (`[CLS] Query [SEP] Book Description [SEP]`) and evaluated by Jina Reranker v2 with full token-to-token cross-attention.
5. **Step 5: Hybrid Cascade Fusion**:
   Dense vector similarity and Cross-Encoder probabilities are blended:
   $$\text{Score}_{\text{final}} = 0.30 \cdot \text{Dense}_{\text{normalized}} + 0.70 \cdot \text{CrossEncoder}_{\text{prob}}$$
6. **Step 6: Metadata Enrichment**:
   Book IDs are matched against `books_metadata.sqlite3` to pull title, genres, popular shelves, full synopsis, and average rating.
7. **Step 7: Cache Population & Delivery**:
   The payload is written to Redis with a 1-hour TTL and delivered to the client.

---

## 🏆 Empirical Benchmarks & Performance Progression

All models were evaluated across the entire **64,552 Goodreads book catalog** over **32,170 test queries** and **32,340 validation queries** (100% work-clustered, zero edition leakage):

### Comprehensive Benchmark Table

| Model Stage | Recall@1 | Recall@5 | Recall@10 | Recall@50 | MRR@10 | mAP@10 | NDCG@10 | Mean Latency | Model Weights |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 1: Pretrained Baseline (Zero-Shot)** | 21.67% | 48.41% | 55.40% | 69.57% | 0.3082 | 0.3082 | 0.3642 | 11.2 ms | 438 MB |
| **Stage 2: In-Batch Contrastive Training** | 29.16% | 58.29% | 64.67% | 78.20% | 0.3951 | 0.3951 | 0.4538 | 11.2 ms | 438 MB |
| **Stage 3: Targeted Hard-Negative Training** | 35.60% | 68.09% | 73.96% | 84.66% | 0.4688 | 0.4688 | 0.5322 | 11.2 ms | 438 MB |
| **Stage 4: Progressive Refinement (Champion Retriever)** | **35.48%** | **68.50%** | **74.70%** | **85.51%** | **0.4725** | **0.4725** | **0.5369** | **11.2 ms** | **438 MB** |
| **Stage 7: Pointwise Jina Fine-Tuning (BCE)** | 48.10% | 65.80% | 72.40% | 83.10% | 0.5540 | 0.5540 | 0.6120 | 170.5 ms | 557 MB |
| **Stage 8: Contrastive InfoNCE Jina Fine-Tuning** | 49.00% | 66.40% | 73.80% | 83.10% | 0.5668 | 0.5668 | 0.6241 | 171.4 ms | 557 MB |
| **Two-Stage Cascade Fusion (0.30 Dense + 0.70 Jina)** | **52.50%** | **70.50%** | **75.00%** | **83.10%** | **0.6001** | **0.6001** | **0.6485** | **182.8 ms** | **995 MB** |

> **Net Gains Across Pipeline**:
> * **Recall@1**: **21.67% $\rightarrow$ 52.50% (+30.83% absolute / +142.3% relative)**
> * **Recall@10**: **55.40% $\rightarrow$ 75.00% (+19.60% absolute / +35.4% relative)**
> * **MRR@10**: **0.3082 $\rightarrow$ 0.6001 (+0.2919 / +94.7% relative)**

---

## 📐 Evaluation Metrics Guide

Information retrieval and recommendation systems require a multi-dimensional set of metrics to capture both breadth (recall) and ordering quality (precision and rank discount):

### 1. Recall@K (Recall at K)
* **What it measures**: The percentage of queries where the true relevant document is present somewhere in the top $K$ retrieved candidates.
* **Why it matters**: Stage 1 retriever quality is dictated by Recall@50. If the relevant book is not in the top 50, the Stage 2 reranker never sees it.

### 2. MRR (Mean Reciprocal Rank / MRR@10)
* **What it measures**: How close to rank 1 the first relevant document appears:
  $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
* **Why it matters**: If a correct book is at rank 1, reciprocal rank is $1.0$; at rank 2, it is $0.5$; at rank 10, it is $0.1$. MRR directly reflects user satisfaction because users rarely look past the first 3 results.

### 3. mAP (Mean Average Precision / MAP@K)
* **What it measures**: The mean of average precision scores across all queries:
  $$\text{MAP} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \text{AP}(q)$$
* **Relationship to MRR**: In single-ground-truth evaluations (where each query has 1 target book), **mAP@K is mathematically equivalent to MRR@K**. When queries have multiple relevant items, mAP evaluates how well all items are clustered near the top.

### 4. NDCG@K (Normalized Discounted Cumulative Gain)
* **What it measures**: Evaluates ranking quality by applying a logarithmic discount factor based on position:
  $$\text{DCG}@K = \sum_{i=1}^K \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}, \quad \text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$
* **Why it matters**: NDCG is the gold standard across Google, Bing, and Amazon Search because placing a relevant result at rank 2 instead of rank 1 receives a gentle penalty, whereas dropping it to rank 10 receives a severe penalty.

### 5. Latency & Compression Retention (MRL)
* Measures the millisecond response time and accuracy retained when compressing vectors from 768-d down to 256-d (retaining **99.30% of Recall@10** at 3x compression).

---

## 🧠 Core Techniques & Why They Were Used

1. **Asymmetric Instruction Tuning**:
   Queries and book passages have different length and style distributions. Prepending `"Represent this sentence for searching relevant passages: "` guides the encoder's self-attention to project queries into document passage space.
2. **Matryoshka Representation Learning (MRL)**:
   Trained with loss gradients evaluated at sub-dimensions $\{64, 128, 256, 512, 768\}$. This packs high-level semantics into the first 256 dimensions, enabling **3x RAM reduction and 3x dot-product speedup** with $< 1\%$ drop in Recall@10.
3. **8-Layer False-Negative Defense Shield**:
   Mining hard negatives from the model's confusion zone (Ranks 2–35) risks mining alternate editions, translations, sequels, or series siblings. The shield filters:
   - Same `work_id` or banned IDs
   - High cosine similarity ceiling ($> 0.88$)
   - Exact and stripped subtitle title matches
   - Content word overlap ($\ge 65\%$)
   - Series regex and author shelf tag matches
   - Negative frequency caps ($> 30$)
4. **VRAM Cliff Fix (Sub-Batch Negative Chunking)**:
   Encoding 16 queries + 16 positive books + 80 adversarial hard negatives (112 sequences $\times$ 512 tokens) caused 9.7 GB activation peaks, overflowing 6 GB VRAM GPUs into slow PCIe RAM paging (26.3s/step). Sub-batch chunking (`chunk_size=12`) eliminated thrashing and reduced latency to **0.49s/step (55x speedup)**.
5. **Listwise InfoNCE Contrastive Fine-Tuning**:
   Pits the positive book against 5 hard negatives in a shared softmax competition, directly optimizing the model to prevent ranking inversions.
6. **Reciprocal Rank Fusion (RRF)**:
   Robustly fuses lexical BM25 and dense vector candidate ranks without needing score normalization across different score scales.

---

## 🤗 Hugging Face Model Upload Guide

ArcticDense includes a dedicated upload script ([`tools/upload_to_hf.py`](file:///c:/Users/Shilpa/OneDrive/Documents/Book%20Semantic%20Search/neural_book_search_arctic/tools/upload_to_hf.py)) that validates model files, auto-generates rich model cards with performance tables, and uploads checkpoints to the Hugging Face Hub.

### Step 1: Authenticate with Hugging Face
```bash
pip install huggingface_hub
huggingface-cli login
```

### Step 2: Upload Fine-Tuned Models

```bash
# Upload the Stage 4 Champion Retriever:
python tools/upload_to_hf.py --model retriever --repo-id "YOUR_USERNAME/arctic-embed-m-goodreads"

# Upload the Jina Contrastive Reranker:
python tools/upload_to_hf.py --model reranker --repo-id "YOUR_USERNAME/jina-reranker-v2-goodreads"

# OR upload both models automatically under your username:
python tools/upload_to_hf.py --model both --prefix "YOUR_USERNAME"
```

Once uploaded, anyone can load and run your models with standard `sentence-transformers`:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("YOUR_USERNAME/arctic-embed-m-goodreads")
```

---

## 📂 Repository Structure

```text
neural_book_search_arctic/
├── app/                                 # PRODUCTION WEB APPLICATION & REST API
│   ├── app.py                           # FastAPI application (GET/POST /api/search, /health, /stats)
│   ├── search_engine.py                 # Cascade search engine (Dense + BM25 RRF + Jina + Redis)
│   ├── static/                          # CSS design system and JavaScript UI interactions
│   └── templates/index.html             # HTML5 glassmorphic UI interface
│
├── training/                            # REPRODUCIBLE FINE-TUNING PIPELINE
│   ├── 01_baseline_benchmark.py         # Stage 1: Zero-shot baseline evaluation
│   ├── 02_inbatch_training.py           # Stage 2: In-batch contrastive training (258k pairs)
│   ├── 03_mine_adversarial_negs.py      # Stage 3/4: Goldilocks mining & 8-layer defense shield
│   ├── 04_adversarial_training.py       # Stage 3/4: Targeted adversarial training & progressive refinement
│   ├── 05_evaluate_retriever.py         # Full catalog 2D block-chunked evaluator
│   ├── 11_train_jina_contrastive.py     # Stage 8: Listwise InfoNCE Jina fine-tuning
│   └── 12_train_jina_pointwise_bce.py   # Stage 7: Pointwise BCE Jina fine-tuning
│
├── src/                                 # REUSABLE CORE LIBRARY
│   ├── cache_service.py                 # Redis sub-millisecond query & vector caching service
│   ├── elastic_search_service.py        # Elasticsearch BM25 lexical search & faceted filter service
│   ├── config.py                        # Hardware auto-detection, paths, and hyperparameters
│   ├── data_loader.py                   # In-memory datasets, SQLite readers, string normalizers
│   ├── filters.py                       # 8-layer false-negative protection shield implementation
│   └── metrics.py                       # GPU 2D block-chunked Recall@K & MRR evaluator
│
├── tools/                               # UTILITIES & VERIFICATION
│   ├── upload_to_hf.py                  # Automated Hugging Face model uploader with model cards
│   ├── verify_container.py              # Automated REST API & container verification script
│   └── index_elasticsearch.py           # Bulk streaming script indexing 64,552 books into Elasticsearch
│
├── results/                             # VERIFIED EMPIRICAL BENCHMARKS (JSON)
│   ├── final_comparison.json            # Master comparative benchmark across all stages
│   ├── benchmark_jina_contrastive.json  # Two-Stage Cascade benchmark metrics
│   └── mrl_dimension_benchmark.json     # Matryoshka dimension tradeoff evaluation
│
├── Dockerfile                           # Production Debian container with CUDA & CPU fallback
├── docker-compose.yml                   # 3-Tier Compose (Search + Redis + Elasticsearch)
├── docker-compose.cpu.yml               # CPU companion compose file
├── DOCKER.md                            # Complete Docker & cloud production deployment guide
├── requirements.txt                     # Production dependencies
└── README.md                            # Main project showcase
```

---

## 👤 Author & License

* **Author**: Nizam (`nansari.0134@gmail.com`)
* **GitHub**: [https://github.com/nansari-0134/](https://github.com/nansari-0134/)
* **Repository**: [https://github.com/nansari-0134/neural_book_search_arctic](https://github.com/nansari-0134/neural_book_search_arctic)
* **License**: MIT License
