"""
Evaluation Engine: 2D Block-Chunked GPU Recall@K Evaluation
"""
import time
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.config import DEVICE, QUERY_INSTRUCTION

@torch.no_grad()
def evaluate_recall(
    model: SentenceTransformer,
    split_path: Path,
    split_name: str,
    meta: Dict,
    all_book_ids: List[str],
    cached_catalog_emb: Optional[np.ndarray] = None,
    limit: Optional[int] = None
) -> Dict[str, float]:
    """
    Computes exact Recall@1, 5, 10, 50 across the full 64k catalog
    using 2D block-chunking to stay safely within GPU VRAM limits.
    """
    print(f"\n[Evaluation] Running {split_name} ({split_path.name})...")
    model.eval()

    rows = []
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break

    # 1. Obtain or compute catalog embeddings
    if cached_catalog_emb is not None:
        catalog_emb = cached_catalog_emb
    else:
        print(f"  Encoding {len(all_book_ids):,} evaluation catalog books...")
        catalog_texts = [meta["book_texts"].get(bid, "Untitled") for bid in all_book_ids]
        catalog_emb = model.encode(
            catalog_texts,
            batch_size=128 if DEVICE == "cuda" else 32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        ).astype("float32")

    id_index = {bid: i for i, bid in enumerate(all_book_ids)}
    ranks = []

    # 2. Block-chunked evaluation
    q_chunk = 128
    c_chunk = 8192
    t0 = time.time()

    for start in range(0, len(rows), q_chunk):
        batch = rows[start:start + q_chunk]
        q_texts = [QUERY_INSTRUCTION + r["query"] for r in batch]
        query_emb = model.encode(
            q_texts,
            batch_size=len(batch),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype("float32")

        target = np.asarray([id_index[str(r["positive_id"])] for r in batch], dtype=np.int64)
        positive_scores = np.sum(query_emb * catalog_emb[target], axis=1)
        greater = np.zeros(len(batch), dtype=np.int32)

        for cstart in range(0, len(all_book_ids), c_chunk):
            scores = query_emb @ catalog_emb[cstart:cstart + c_chunk].T
            greater += np.count_nonzero(scores > positive_scores[:, None], axis=1)
            del scores
        ranks.extend(greater.tolist())

    elapsed = time.time() - t0
    metrics = {f"R@{k}": float(np.mean(np.asarray(ranks) < k)) for k in (1, 5, 10, 50)}
    print(f"  {split_name} Results ({len(rows):,} queries vs {len(all_book_ids):,} books in {elapsed:.1f}s):")
    for k, v in metrics.items():
        print(f"    {k}: {v*100:5.2f}%")
    return metrics
