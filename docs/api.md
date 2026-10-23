# 🌐 REST API Reference

The **ArcticDense** backend exposes a high-performance RESTful API built on **FastAPI** and **Uvicorn**, designed for sub-15ms semantic search, lexical BM25 matching, and two-stage cross-encoder reranking.

Interactive API documentation is also available directly in the browser when running the server:
* **Interactive Swagger UI**: [`http://localhost:8080/docs`](http://localhost:8080/docs)
* **ReDoc Specification**: [`http://localhost:8080/redoc`](http://localhost:8080/redoc)

---

## 1. Endpoints Overview

| Method | Endpoint | Description | Cacheable |
| :--- | :--- | :--- | :---: |
| `GET` | `/api/search` | Search via URL query parameters | ✅ Yes (Redis) |
| `POST` | `/api/search` | Search via structured JSON request body | ✅ Yes (Redis) |
| `GET` | `/api/health` | Service and index readiness probe | ❌ No |
| `GET` | `/api/stats` | System telemetry, hardware device, and cache stats | ❌ No |

---

## 2. Search Endpoint (`GET /api/search` & `POST /api/search`)

Executes neural, lexical, or hybrid search across **64,552 Goodreads books**.

### Request Parameters

| Parameter | Type | Required | Default | Description |
| :--- | :---: | :---: | :---: | :--- |
| `q` | `string` | **Yes** | — | Natural language query (e.g. `"dystopian hacker rebellion"`) |
| `dim` | `integer` | No | `768` | Matryoshka dimension truncation (`64`, `128`, `256`, `512`, `768`) |
| `top_k` | `integer` | No | `10` | Number of ranked books to return (range: `1` to `50`) |
| `rerank` | `boolean` | No | `false` | Enable Stage 2 Jina Reranker v2 Cross-Encoder reranking |
| `mode` | `string` | No | `"hybrid"` | Search mode: `"hybrid"` (Dense + BM25 via RRF), `"dense"`, or `"lexical"` |
| `genre` | `string` | No | `null` | Optional genre / shelf filter (e.g. `"science-fiction"`, `"fantasy"`) |

---

### Request Examples

=== "cURL (GET)"
    ```bash
    curl -X GET "http://localhost:8080/api/search?q=cyberpunk+detective&dim=768&top_k=3&rerank=true&mode=hybrid"
    ```

=== "cURL (POST JSON)"
    ```bash
    curl -X POST "http://localhost:8080/api/search" \
      -H "Content-Type: application/json" \
      -d '{
        "q": "cyberpunk detective artificial intelligence",
        "dim": 256,
        "top_k": 3,
        "rerank": true,
        "mode": "hybrid",
        "genre": "science-fiction"
      }'
    ```

=== "Python (requests)"
    ```python
    import requests

    response = requests.post(
        "http://localhost:8080/api/search",
        json={
            "q": "atmospheric gothic boarding school mystery",
            "dim": 768,
            "top_k": 5,
            "rerank": True,
            "mode": "hybrid"
        }
    )

    data = response.json()
    print(f"Latency: {data['latency_ms']}ms (Cache Hit: {data['cache_hit']})")
    for book in data["results"]:
        print(f"[{book['score']:.4f}] {book['title']}")
    ```

=== "JavaScript (Fetch)"
    ```javascript
    const response = await fetch("http://localhost:8080/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        q: "space opera alien artifact",
        dim: 768,
        top_k: 5,
        rerank: true,
        mode: "hybrid"
      })
    });

    const data = await response.json();
    console.log(`Retrieved ${data.results.length} books in ${data.latency_ms}ms`);
    ```

---

### Response Schema & Field Glossary

```json
{
  "query": "cyberpunk detective",
  "mode": "hybrid",
  "latency_ms": 178.42,
  "bi_encoder_latency_ms": 11.20,
  "rerank_latency_ms": 167.22,
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
      "genres": [
        "science-fiction",
        "cyberpunk",
        "fiction"
      ],
      "snippet": "Case was the sharpest data-thief in the matrix until he crossed the wrong people...",
      "full_description": "Case was the sharpest data-thief in the matrix until he crossed the wrong people and they crippled his nervous system...",
      "reranked": true
    }
  ]
}
```

| Field | Type | Description |
| :--- | :---: | :--- |
| `query` | `string` | The submitted query string |
| `mode` | `string` | Retrieval pipeline executed (`"hybrid"`, `"dense"`, or `"lexical"`) |
| `latency_ms` | `float` | Total end-to-end response latency in milliseconds |
| `bi_encoder_latency_ms` | `float` | Latency incurred by Stage 1 vector encoding and matrix dot-product |
| `rerank_latency_ms` | `float` | Latency incurred by Stage 2 Cross-Encoder reranking |
| `reranked` | `boolean` | `true` if Stage 2 Jina Reranker was applied |
| `reranker_name` | `string` | Name and parameter count of the active reranker |
| `cache_hit` | `boolean` | `true` if response was served directly from Redis RAM in < 1ms |
| `total_catalog` | `integer` | Total number of indexed books (64,552) |
| `results[].score` | `float` | Normalized relevance score (0.0 to 1.0) |
| `results[].bi_encoder_score` | `float` | Raw Stage 1 cosine similarity score |
| `results[].snippet` | `string` | Truncated description snippet suitable for card previews |
| `results[].full_description`| `string` | Full Goodreads book blurb |

---

## 3. Health Endpoint (`GET /api/health`)

Used by Docker container health checks and Kubernetes readiness probes.

### Example Request
```bash
curl http://localhost:8080/api/health
```

### Healthy Response (`200 OK`)
```json
{
  "status": "ready",
  "ready": true,
  "catalog_indexed": 64552
}
```

### Initializing Response (`503 Service Unavailable`)
```json
{
  "status": "indexing",
  "ready": false,
  "message": "Engine is currently encoding/loading catalog embeddings in background..."
}
```

---

## 4. Telemetry Endpoint (`GET /api/stats`)

Returns hardware device info, active MRL dimensions, and cache metrics.

### Example Request
```bash
curl http://localhost:8080/api/stats
```

### Response
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
    "redis_url": "redis://redis:6379/0",
    "hits": 142,
    "misses": 18,
    "hit_ratio_percent": 88.75
  },
  "elasticsearch": {
    "enabled": true,
    "connected": true,
    "cluster_url": "http://elasticsearch:9200",
    "index_name": "arctic_books",
    "document_count": 64552
  },
  "top_genres": [
    "fiction", "fantasy", "romance", "young-adult", "mystery",
    "historical-fiction", "science-fiction", "thriller", "non-fiction",
    "classics", "paranormal", "horror", "history", "memoir"
  ]
}
```
