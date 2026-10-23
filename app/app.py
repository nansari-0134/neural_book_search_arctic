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

import threading

# Global search engine singleton (loaded in background thread on startup)
_engine: Optional[BookSearchEngine] = None
_init_thread: Optional[threading.Thread] = None
_init_error: Optional[str] = None

def _background_init():
    global _engine, _init_error
    try:
        print("[Startup] Initializing BookSearchEngine in background...", flush=True)
        eng = BookSearchEngine(lazy_load=False)
        _engine = eng
        print("[Startup] BookSearchEngine initialized and ready for queries!", flush=True)
    except Exception as e:
        _init_error = str(e)
        print(f"[Startup Error] Engine initialization failed: {e}", flush=True)

@app.on_event("startup")
async def startup_event():
    global _init_thread
    _init_thread = threading.Thread(target=_background_init, daemon=True)
    _init_thread.start()

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
    global _engine, _init_error
    if not q or not q.strip():
        return JSONResponse({"error": "Query cannot be empty"}, status_code=400)

    if _init_error:
        return JSONResponse({"error": f"Search engine initialization error: {_init_error}"}, status_code=500)

    if _engine is None or not _engine.is_ready:
        return JSONResponse(
            {"error": "Catalog embeddings are currently loading/indexing. Please wait a moment..."},
            status_code=503
        )

    results = _engine.search(query=q.strip(), top_k=top_k, mrl_dim=dim, genre_filter=genre)
    return results
@app.get("/api/stats")
async def stats_endpoint():
    global _engine
    if _engine is None or not _engine.is_ready:
        return {
            "status": "initializing",
            "model_name": "Snowflake Arctic Embed (v1.5) - Fine-Tuned Champion",
            "catalog_size": 64552,
            "message": "Engine is loading catalog embeddings into memory..."
        }
    return {
        "status": "ready",
        "model_name": "Snowflake Arctic Embed (v1.5) - Fine-Tuned Champion",
        "catalog_size": len(_engine.catalog_ids),
        "device": _engine.device,
        "fp16_enabled": _engine.use_fp16,
        "supported_mrl_dimensions": [64, 128, 256, 512, 768],
        "top_genres": [
            "fiction", "fantasy", "romance", "young-adult", "mystery",
            "historical-fiction", "science-fiction", "thriller", "non-fiction",
            "classics", "paranormal", "horror", "history", "memoir"
        ]
    }

@app.get("/api/health")
async def health_endpoint():
    global _engine, _init_error
    if _init_error:
        return {"status": "error", "ready": False, "error": _init_error}
    if _engine is not None and _engine.is_ready:
        return {
            "status": "ready",
            "ready": True,
            "catalog_indexed": len(_engine.catalog_ids)
        }
    return {
        "status": "indexing",
        "ready": False,
        "message": "Engine is currently encoding/loading catalog embeddings in background..."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.app:app", host="127.0.0.1", port=8000, reload=False)
