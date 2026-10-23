# ArcticDense Web Application & Inference Engine

An interactive, production-ready neural book search engine powered by the fine-tuned **Snowflake Arctic Embed (v1.5)** champion model.

---

## 🚀 Features

- **Blazing Fast Vector Search**: Executes semantic retrieval across **64,552 Goodreads books** in `< 15 ms`.
- **Matryoshka Representation Learning (MRL)**: Dynamically truncates embedding vectors into **64-d, 128-d, 256-d, 512-d, or 768-d** representations with L2 re-normalization on the fly.
- **Glassmorphic Modern Dark-Mode UI**: Built with pure Vanilla HTML5/CSS/JavaScript with responsive cards, micro-animations, latency telemetry, and book synopsis modals.
- **RESTful API**: Fast and light FastAPI endpoints for direct integration with search clients or mobile apps.

---

## 🛠️ Quickstart

### 1. Launch the Interactive Web Application

```bash
# Start with Uvicorn dev server on port 8000
python app/app.py
```

Open your browser at:
👉 **`http://127.0.0.1:8000`**

### 2. Run CLI Semantic Search

You can also run instant search directly from your command line:

```bash
# Basic natural query search
python app/search_engine.py --query "dystopian rebellion where memories are traded as currency"

# Test Matryoshka dimension truncation (e.g. 256 dimensions)
python app/search_engine.py --query "ancient cosmic horror in deep space" --dim 256 --top-k 5

# Filter by genre
python app/search_engine.py --query "time travel paradox" --genre "science-fiction"
```

---

## 📡 API Reference

### `GET /api/search`

Executes dense neural retrieval against the 64,552 book catalog.

**Parameters**:
- `q` (*string*, required): Natural language search query or theme.
- `dim` (*integer*, optional, default: `768`): Matryoshka representation dimension (`64`, `128`, `256`, `512`, `768`).
- `genre` (*string*, optional): Filter by Goodreads shelf category (e.g. `fantasy`, `mystery`, `science-fiction`).
- `top_k` (*integer*, optional, default: `10`): Number of ranked books to return.

**Response**:
```json
{
  "query": "cozy murder mystery in an English coastal village",
  "latency_ms": 11.42,
  "mrl_dim": 768,
  "genre_filter": null,
  "total_catalog": 64552,
  "results": [
    {
      "book_id": "421092",
      "title": "The Lighthouse Mystery",
      "score": 0.8241,
      "score_pct": "82.4%",
      "genres": ["mystery", "cozy-mystery", "fiction"],
      "snippet": "A quiet seaside village is disrupted by an eerie death at the historic lighthouse...",
      "full_description": "..."
    }
  ]
}
```

### `GET /api/stats`

Returns catalog indexing telemetry, active hardware acceleration device, and supported MRL dimensions.
