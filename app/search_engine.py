"""
Neural Book Search Engine
=========================
Core inference engine powered by fine-tuned Snowflake Arctic Embed (v1.5).
Supports:
1. Fast cosine similarity search across 64,552 catalog books.
2. Matryoshka Representation Learning (MRL) dimension truncation (64, 128, 256, 512, 768).
3. Metadata enrichment from SQLite (title, synopsis, popular shelves).
4. Genre filtering and sub-15ms query execution.
"""

import sys
import os
import json
import time
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    MODEL_NAME, DB_FILE, OUTPUT_DIR, MAX_SEQ_LENGTH,
    QUERY_INSTRUCTION, DEVICE, USE_FP16
)

# Priority order for model checkpoint loading
CHAMPION_MODEL_PATH = OUTPUT_DIR / "arctic_v3" / "best_model"
FALLBACK_MODEL_PATH = OUTPUT_DIR / "arctic_v2" / "best_model"

CACHE_DIR = PROJECT_ROOT / "app" / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_EMBEDDINGS = CACHE_DIR / "catalog_embeddings_fp16.npy"
CACHE_BOOK_IDS = CACHE_DIR / "catalog_book_ids.json"


class BookSearchEngine:
    def __init__(self, model_path: Optional[str] = None, lazy_load: bool = False):
        self.db_path = DB_FILE
        self.device = DEVICE
        self.use_fp16 = USE_FP16 and (self.device == "cuda")
        
        # Determine model path
        if model_path:
            self.model_path = Path(model_path)
        elif CHAMPION_MODEL_PATH.exists():
            self.model_path = CHAMPION_MODEL_PATH
        elif FALLBACK_MODEL_PATH.exists():
            self.model_path = FALLBACK_MODEL_PATH
        else:
            self.model_path = Path(MODEL_NAME)
            
        self.model: Optional[SentenceTransformer] = None
        self.catalog_ids: List[str] = []
        self.catalog_embeddings: Optional[np.ndarray] = None
        self.id_to_idx: Dict[str, int] = {}
        self.book_metadata: Dict[str, Dict[str, Any]] = {}
        self.is_ready = False

        if not lazy_load:
            self.initialize()

    def initialize(self):
        """Loads model, catalog metadata, and pre-computed embeddings."""
        print(f"[Search Engine] Initializing with model: {self.model_path}")
        t0 = time.time()
        
        # 1. Load SQLite metadata
        self._load_metadata()
        
        # 2. Load model
        print(f"[Search Engine] Loading model onto {self.device} (FP16: {self.use_fp16})...")
        self.model = SentenceTransformer(str(self.model_path), device=self.device)
        self.model.max_seq_length = MAX_SEQ_LENGTH
        if self.use_fp16:
            self.model.half()
        self.model.eval()

        # 3. Load or compute catalog embeddings
        self._load_or_compute_embeddings()
        
        self.is_ready = True
        print(f"[Search Engine] Ready! Initialized {len(self.catalog_ids):,} books in {time.time() - t0:.2f}s.")

    def _load_metadata(self):
        """Loads book metadata into memory for fast enrichment."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found at {self.db_path}")

        print(f"[Search Engine] Loading metadata from {self.db_path.name}...")
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        query = "SELECT book_id, title, description, popular_shelves FROM books"
        for row in cur.execute(query):
            bid = str(row[0])
            title = row[1] or "Untitled"
            desc = (row[2] or "").strip()
            shelves_raw = row[3] or "[]"
            
            try:
                shelves = json.loads(shelves_raw)
            except Exception:
                shelves = []

            # Clean genres (filter generic shelf tags)
            clean_shelves = [
                s for s in shelves 
                if s not in {"to-read", "currently-reading", "favorites", "books-i-own", "owned", "default"}
            ]

            self.book_metadata[bid] = {
                "book_id": bid,
                "title": title,
                "description": desc,
                "genres": clean_shelves[:6],
                "all_shelves": set(s.lower() for s in shelves)
            }
            
        conn.close()
        self.catalog_ids = sorted(list(self.book_metadata.keys()))
        self.id_to_idx = {bid: idx for idx, bid in enumerate(self.catalog_ids)}

    def _load_or_compute_embeddings(self):
        """Loads cached embeddings or computes them across 64,552 books."""
        if CACHE_EMBEDDINGS.exists() and CACHE_BOOK_IDS.exists():
            print(f"[Search Engine] Loading cached embeddings from {CACHE_EMBEDDINGS.name}...")
            with open(CACHE_BOOK_IDS, "r", encoding="utf-8") as f:
                cached_ids = json.load(f)
            if cached_ids == self.catalog_ids:
                self.catalog_embeddings = np.load(CACHE_EMBEDDINGS)
                print(f"[Search Engine] Loaded {self.catalog_embeddings.shape[0]:,} embeddings ({self.catalog_embeddings.shape[1]}-d).")
                return

        print(f"[Search Engine] Encoding {len(self.catalog_ids):,} catalog books on {self.device}...")
        texts = []
        for bid in self.catalog_ids:
            meta = self.book_metadata[bid]
            genres_str = ", ".join(meta["genres"]) if meta["genres"] else "General"
            desc_snip = meta["description"][:900]
            text = f"Title: {meta['title']} | Genres: {genres_str} | Description: {desc_snip}"
            texts.append(text)

        batch_size = 128 if self.device == "cuda" else 32
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True
        ).astype("float32")

        self.catalog_embeddings = embeddings
        try:
            print(f"[Search Engine] Saving embeddings cache to {CACHE_EMBEDDINGS}...")
            np.save(CACHE_EMBEDDINGS, embeddings.astype("float16"))
            with open(CACHE_BOOK_IDS, "w", encoding="utf-8") as f:
                json.dump(self.catalog_ids, f)
        except Exception as e:
            print(f"[Search Engine] Note: Could not cache embeddings to disk: {e}")

    def search(
        self,
        query: str,
        top_k: int = 10,
        mrl_dim: int = 768,
        genre_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes semantic search with Matryoshka dimension truncation & optional genre filtering.
        """
        if not self.is_ready:
            self.initialize()

        t0 = time.perf_counter()
        query_text = f"{QUERY_INSTRUCTION}{query.strip()}"

        # 1. Encode query
        q_emb = self.model.encode(
            query_text,
            convert_to_numpy=True,
            normalize_embeddings=False
        ).astype("float32")

        # 2. Apply Matryoshka Truncation & L2 Re-normalization
        dim = min(mrl_dim, self.catalog_embeddings.shape[1])
        q_vec = q_emb[:dim]
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        doc_matrix = self.catalog_embeddings[:, :dim]
        if dim < self.catalog_embeddings.shape[1]:
            # Re-normalize document slices for lower dimension
            doc_norms = np.linalg.norm(doc_matrix, axis=1, keepdims=True)
            doc_norms[doc_norms == 0] = 1e-12
            doc_matrix = doc_matrix / doc_norms

        # 3. Compute cosine similarities
        scores = doc_matrix @ q_vec

        # 4. Optional genre filtering
        if genre_filter:
            target_genre = genre_filter.strip().lower()
            filtered_indices = []
            for idx, bid in enumerate(self.catalog_ids):
                if target_genre in self.book_metadata[bid]["all_shelves"]:
                    filtered_indices.append(idx)
            if filtered_indices:
                filtered_indices = np.array(filtered_indices)
                sub_scores = scores[filtered_indices]
                sub_top_k = min(top_k, len(sub_scores))
                top_sub_idx = np.argpartition(-sub_scores, sub_top_k - 1)[:sub_top_k]
                sorted_sub_idx = top_sub_idx[np.argsort(-sub_scores[top_sub_idx])]
                best_indices = filtered_indices[sorted_sub_idx]
            else:
                best_indices = []
        else:
            top_k_clamp = min(top_k, len(scores))
            candidate_indices = np.argpartition(-scores, top_k_clamp - 1)[:top_k_clamp]
            best_indices = candidate_indices[np.argsort(-scores[candidate_indices])]

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # 5. Enrich results
        results = []
        for idx in best_indices:
            bid = self.catalog_ids[idx]
            meta = self.book_metadata[bid]
            score = float(scores[idx])
            desc = meta["description"]
            snippet = (desc[:240] + "...") if len(desc) > 240 else desc
            
            results.append({
                "book_id": bid,
                "title": meta["title"],
                "score": round(score, 4),
                "score_pct": f"{max(0.0, score) * 100:.1f}%",
                "genres": meta["genres"],
                "snippet": snippet,
                "full_description": desc,
            })

        return {
            "query": query,
            "latency_ms": round(elapsed_ms, 2),
            "mrl_dim": dim,
            "genre_filter": genre_filter,
            "total_catalog": len(self.catalog_ids),
            "results": results
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Neural Book Search Engine CLI")
    parser.add_argument("--query", "-q", type=str, default="A lone astronaut stranded on a red planet", help="Natural language query")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--dim", "-d", type=int, default=768, choices=[64, 128, 256, 512, 768], help="Matryoshka MRL dimension")
    parser.add_argument("--genre", "-g", type=str, default=None, help="Filter by genre / shelf")
    args = parser.parse_args()

    engine = BookSearchEngine()
    res = engine.search(query=args.query, top_k=args.top_k, mrl_dim=args.dim, genre_filter=args.genre)

    print("\n" + "=" * 75)
    print(f"  QUERY: \"{res['query']}\"")
    print(f"  Latency: {res['latency_ms']} ms | Matryoshka Dim: {res['mrl_dim']}-d | Total Catalog: {res['total_catalog']:,}")
    if res['genre_filter']:
        print(f"  Filter: {res['genre_filter']}")
    print("=" * 75)

    for i, item in enumerate(res["results"], 1):
        print(f"\n[{i}] {item['title']} (Score: {item['score']:.4f} | {item['score_pct']})")
        if item['genres']:
            print(f"    Genres: {', '.join(item['genres'])}")
        print(f"    {item['snippet']}")
    print("\n" + "=" * 75)


if __name__ == "__main__":
    main()
