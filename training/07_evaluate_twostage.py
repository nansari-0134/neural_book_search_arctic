"""
Evaluation Script: Two-Stage Cascade Benchmark (Bi-Encoder + Cross-Encoder)
===========================================================================
Measures exact Recall@K and MRR@10 before and after Cross-Encoder reranking
across the full 64,552 book catalog.

Stage 1: Arctic Dense Retriever (filters 64,552 books -> Top 50 candidates)
Stage 2: Cross-Encoder Precision Reranker (re-ranks Top 50 candidates)
"""

import sys
import os
import json
import time
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    MODEL_NAME, DB_FILE, OUTPUT_DIR, MAX_SEQ_LENGTH,
    QUERY_INSTRUCTION, DEVICE, USE_FP16
)
from src.data_loader import load_book_metadata

ARCTIC_MODEL_PATH = OUTPUT_DIR / "arctic_v3" / "best_model"
CROSS_ENCODER_PATH = OUTPUT_DIR / "cross_encoder" / "best_model"

CACHE_EMBEDDINGS = PROJECT_ROOT / "app" / ".cache" / "catalog_embeddings_fp16.npy"
CACHE_BOOK_IDS = PROJECT_ROOT / "app" / ".cache" / "catalog_book_ids.json"
TEST_DATA_FILE = PROJECT_ROOT / "data" / "test.jsonl"
RESULTS_FILE = PROJECT_ROOT / "results" / "benchmark_twostage.json"

EVAL_SAMPLE_SIZE = 1000
TOP_K_CANDIDATES = 50

def run_twostage_benchmark():
    print("=" * 75)
    print("[BENCHMARK] TWO-STAGE RETRIEVAL CASCADE (ARCTIC + CROSS-ENCODER)")
    print(f"Catalog Size: 64,552 Books | Top Candidates Mined: {TOP_K_CANDIDATES}")
    print(f"Evaluated Test Queries: {EVAL_SAMPLE_SIZE:,}")
    print(f"Device: {DEVICE}")
    print("=" * 75)


    # 1. Load book metadata
    meta = load_book_metadata(DB_FILE)
    book_texts = meta["book_texts"]

    # 2. Load pre-computed catalog embeddings & IDs
    print(f"\n[Embeddings] Loading catalog embeddings from {CACHE_EMBEDDINGS.name}...")
    catalog_embeddings = np.load(CACHE_EMBEDDINGS)
    with open(CACHE_BOOK_IDS, "r", encoding="utf-8") as f:
        catalog_ids = json.load(f)
    print(f"[Embeddings] Loaded {len(catalog_ids):,} book embeddings {catalog_embeddings.shape}.")

    # Convert embeddings to torch tensor on GPU for fast matrix multiplication
    catalog_tensor = torch.from_numpy(catalog_embeddings).to(DEVICE)
    if catalog_tensor.dtype != torch.float16 and USE_FP16 and DEVICE == "cuda":
        catalog_tensor = catalog_tensor.half()

    # 3. Load Stage 1 Arctic Retriever
    print(f"\n[Models] Loading Stage 1 Arctic Retriever: {ARCTIC_MODEL_PATH.name}...")
    bi_encoder = SentenceTransformer(str(ARCTIC_MODEL_PATH), device=DEVICE)
    if USE_FP16 and DEVICE == "cuda":
        bi_encoder.half()

    # 4. Load Stage 2 Cross-Encoder Reranker
    ce_path = str(CROSS_ENCODER_PATH) if CROSS_ENCODER_PATH.exists() else "cross-encoder/ms-marco-MiniLM-L-6-v2"
    print(f"[Models] Loading Stage 2 Cross-Encoder Reranker: {Path(ce_path).name}...")
    cross_encoder = CrossEncoder(ce_path, max_length=256, device=DEVICE)

    # 5. Load Test Queries
    print(f"\n[Data] Reading {EVAL_SAMPLE_SIZE} test queries from {TEST_DATA_FILE.name}...")
    test_rows = []
    with open(TEST_DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if len(test_rows) >= EVAL_SAMPLE_SIZE:
                break
            test_rows.append(json.loads(line))

    # Metrics trackers
    # Stage 1 (Bi-Encoder alone)
    b_r1, b_r5, b_r10, b_r50 = 0, 0, 0, 0
    b_mrr10 = 0.0

    # Stage 2 (Bi-Encoder + Cross-Encoder)
    ce_r1, ce_r5, ce_r10 = 0, 0, 0
    ce_mrr10 = 0.0

    s1_latencies = []
    s2_latencies = []

    print("\n[Benchmark] Running two-stage retrieval evaluation...")
    t0_all = time.time()

    for idx, row in enumerate(test_rows):
        query = row["query"]
        pos_id = str(row["positive_id"])

        # Stage 1: Encode query with instruction prefix & search
        t0_s1 = time.time()
        q_text = f"{QUERY_INSTRUCTION}{query}"
        q_emb = bi_encoder.encode([q_text], convert_to_tensor=True, normalize_embeddings=True)
        if q_emb.dtype != catalog_tensor.dtype:
            q_emb = q_emb.to(catalog_tensor.dtype)

        # Cosine dot product across 64,552 catalog vectors
        scores = torch.matmul(catalog_tensor, q_emb.squeeze(0))
        top_k_indices = torch.topk(scores, k=TOP_K_CANDIDATES).indices.cpu().numpy()
        top_k_ids = [catalog_ids[i] for i in top_k_indices]
        s1_latencies.append((time.time() - t0_s1) * 1000)

        # Compute Stage 1 Metrics
        if pos_id in top_k_ids:
            rank = top_k_ids.index(pos_id) + 1
            if rank <= 1:
                b_r1 += 1
            if rank <= 5:
                b_r5 += 1
            if rank <= 10:
                b_r10 += 1
                b_mrr10 += 1.0 / rank
            if rank <= 50:
                b_r50 += 1

        # Stage 2: Cross-Encoder Reranking
        t0_s2 = time.time()
        # Prepare pairs: (query, candidate_passage)
        candidate_texts = [book_texts.get(cid, "Untitled") for cid in top_k_ids]
        ce_pairs = [(query, text) for text in candidate_texts]

        # Predict relevance scores
        ce_scores = cross_encoder.predict(ce_pairs, batch_size=TOP_K_CANDIDATES, show_progress_bar=False)
        
        # Sort candidates by descending cross-encoder score
        reranked_order = np.argsort(ce_scores)[::-1]
        reranked_ids = [top_k_ids[i] for i in reranked_order]
        s2_latencies.append((time.time() - t0_s2) * 1000)

        # Compute Stage 2 Metrics
        if pos_id in reranked_ids:
            rerank_pos = reranked_ids.index(pos_id) + 1
            if rerank_pos <= 1:
                ce_r1 += 1
            if rerank_pos <= 5:
                ce_r5 += 1
            if rerank_pos <= 10:
                ce_r10 += 1
                ce_mrr10 += 1.0 / rerank_pos

        if (idx + 1) % 200 == 0 or (idx + 1) == EVAL_SAMPLE_SIZE:
            print(f"  Processed {idx + 1}/{EVAL_SAMPLE_SIZE} queries | Current S1 R@1: {b_r1/(idx+1)*100:.2f}% | Current Rerank R@1: {ce_r1/(idx+1)*100:.2f}%")

    total_eval_time = time.time() - t0_all
    N = len(test_rows)

    # Compute final metrics
    s1_metrics = {
        "recall_1": (b_r1 / N) * 100,
        "recall_5": (b_r5 / N) * 100,
        "recall_10": (b_r10 / N) * 100,
        "recall_50": (b_r50 / N) * 100,
        "mrr_10": b_mrr10 / N,
        "mean_latency_ms": float(np.mean(s1_latencies))
    }

    s2_metrics = {
        "recall_1": (ce_r1 / N) * 100,
        "recall_5": (ce_r5 / N) * 100,
        "recall_10": (ce_r10 / N) * 100,
        "mrr_10": ce_mrr10 / N,
        "mean_latency_ms": float(np.mean(s2_latencies)),
        "total_latency_ms": float(np.mean(s1_latencies) + np.mean(s2_latencies))
    }

    print("\n" + "=" * 75)
    print("[RESULTS] TWO-STAGE CASCADE BENCHMARK RESULTS")
    print("=" * 75)
    print(f"{'Metric':<15} | {'Stage 1 (Arctic Bi-Encoder)':<28} | {'Stage 2 (Cross-Encoder Rerank)':<30} | {'Delta':<10}")
    print("-" * 90)
    print(f"{'Recall@1':<15} | {s1_metrics['recall_1']:>26.2f}% | {s2_metrics['recall_1']:>28.2f}% | {s2_metrics['recall_1'] - s1_metrics['recall_1']:>+8.2f}%")
    print(f"{'Recall@5':<15} | {s1_metrics['recall_5']:>26.2f}% | {s2_metrics['recall_5']:>28.2f}% | {s2_metrics['recall_5'] - s1_metrics['recall_5']:>+8.2f}%")
    print(f"{'Recall@10':<15} | {s1_metrics['recall_10']:>26.2f}% | {s2_metrics['recall_10']:>28.2f}% | {s2_metrics['recall_10'] - s1_metrics['recall_10']:>+8.2f}%")
    print(f"{'Recall@50':<15} | {s1_metrics['recall_50']:>26.2f}% | {'N/A (Top 50 Reranked)':>28} | {'—':>8}")
    print(f"{'MRR@10':<15} | {s1_metrics['mrr_10']:>28.4f} | {s2_metrics['mrr_10']:>30.4f} | {s2_metrics['mrr_10'] - s1_metrics['mrr_10']:>+8.4f}")
    print(f"{'Mean Latency':<15} | {s1_metrics['mean_latency_ms']:>26.1f}ms | {s2_metrics['total_latency_ms']:>26.1f}ms (Total) | {'+'+str(round(s2_metrics['mean_latency_ms'],1))+'ms':>8}")
    print("=" * 75)

    # Save results
    PROJECT_ROOT.joinpath("results").mkdir(parents=True, exist_ok=True)
    results_payload = {
        "evaluation_queries": N,
        "catalog_size": len(catalog_ids),
        "stage1_bi_encoder": s1_metrics,
        "stage2_cross_encoder": s2_metrics,
        "relative_gain_r1": ((s2_metrics["recall_1"] - s1_metrics["recall_1"]) / s1_metrics["recall_1"]) * 100,
        "relative_gain_mrr10": ((s2_metrics["mrr_10"] - s1_metrics["mrr_10"]) / s1_metrics["mrr_10"]) * 100
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\n[Results] Saved benchmark summary to {RESULTS_FILE.name}")

if __name__ == "__main__":
    run_twostage_benchmark()
