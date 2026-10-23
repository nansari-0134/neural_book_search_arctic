"""
ArcticDense Web Application
===========================
Interactive, high-performance semantic search UI for 64,552 Goodreads books.
Features:
- Sub-15ms vector search with Snowflake Arctic Embed (v1.5) Champion Model
- Matryoshka Representation Learning (MRL) dimension slider (64, 128, 256, 512, 768)
- Real-time latency tracking and similarity score visualization
- Glassmorphic modern dark-mode UI with smooth micro-animations
"""

import sys
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, Query
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.search_engine import BookSearchEngine

app = FastAPI(
    title="ArcticDense Book Search Engine",
    description="Neural Book Retrieval with Fine-Tuned Snowflake Arctic Embed",
    version="1.0.0"
)

# Mount static and templates
app_dir = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(app_dir / "static")), name="static")
templates = Jinja2Templates(directory=str(app_dir / "templates"))

# Global search engine singleton (lazy loaded on first request if needed)
_engine: Optional[BookSearchEngine] = None

def get_engine() -> BookSearchEngine:
    global _engine
    if _engine is None:
        _engine = BookSearchEngine(lazy_load=False)
    return _engine

@app.on_event("startup")
async def startup_event():
    # Warm up engine in background on startup
    try:
        get_engine()
    except Exception as e:
        print(f"[Warning] Engine startup initialization deferred: {e}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/search")
async def search_endpoint(
    q: str = Query(..., description="Natural language search query"),
    dim: int = Query(768, ge=64, le=768, description="Matryoshka representation dimension"),
    genre: Optional[str] = Query(None, description="Optional genre / shelf filter"),
    top_k: int = Query(10, ge=1, le=50, description="Number of results to return")
):
    if not q or not q.strip():
        return JSONResponse({"error": "Query cannot be empty"}, status_code=400)

    engine = get_engine()
    results = engine.search(query=q.strip(), top_k=top_k, mrl_dim=dim, genre_filter=genre)
    return results

@app.get("/api/stats")
async def stats_endpoint():
    engine = get_engine()
    return {
        "model_name": "Snowflake Arctic Embed (v1.5) - Fine-Tuned Champion",
        "catalog_size": len(engine.catalog_ids),
        "device": engine.device,
        "fp16_enabled": engine.use_fp16,
        "supported_mrl_dimensions": [64, 128, 256, 512, 768],
        "top_genres": [
            "fiction", "fantasy", "romance", "young-adult", "mystery",
            "historical-fiction", "science-fiction", "thriller", "non-fiction",
            "classics", "paranormal", "horror", "history", "memoir"
        ]
    }

@app.get("/api/health")
async def health_endpoint():
    engine = get_engine()
    return {
        "status": "healthy" if engine.is_ready else "initializing",
        "catalog_indexed": len(engine.catalog_ids),
        "ready": engine.is_ready
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.app:app", host="127.0.0.1", port=8000, reload=True)
