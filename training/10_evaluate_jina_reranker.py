"""
Step 10: Evaluation & Benchmark of Fine-Tuned Jina Reranker v2
=============================================================
Evaluates the two-stage cascade with the Fine-Tuned Jina Reranker v2 (278M)
against the Arctic dense retriever and previous MiniLM reranker.

Full Catalog: 64,552 books
Test Queries: 1,000 trope/plot queries
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

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

# Apply Jina transformers 5.x patch
import src.jina_patch

from src.config import (
    DB_FILE, OUTPUT_DIR, RESULTS_DIR, QUERY_INSTRUCTION, DEVICE, USE_FP16
)
from src.data_loader import load_book_metadata

ARCTIC_MODEL_PATH = OUTPUT_DIR / "arctic_v3" / "best_model"
JINA_MODEL_PATH = OUTPUT_DIR / "jina_reranker" / "best_model"
MINILM_MODEL_PATH = OUTPUT_DIR / "cross_encoder" / "best_model"

CACHE_EMBEDDINGS = PROJECT_ROOT / "app" / ".cache" / "catalog_embeddings_fp16.npy"
CACHE_BOOK_IDS = PROJECT_ROOT / "app" / ".cache" / "catalog_book_ids.json"
TEST_DATA_FILE = PROJECT_ROOT / "data" / "test.jsonl"
OUT_JSON = RESULTS_DIR / "benchmark_jina_reranker.json"

TOP_K_CANDIDATES = 50

def run_jina_benchmark(sample_size: int = 1000):
    print("=" * 80)
    print("  TWO-STAGE BENCHMARK: ARCTIC RETRIEVER + JINA RERANKER V2 (278M)")
    print(f"  Catalog Size: 64,552 Books | Top Candidates Mined: {TOP_K_CANDIDATES}")
    print(f"  Test Query Sample: {sample_size:,} queries")
    print(f"  Device: {DEVICE}")
    print("=" * 80)

    # 1. Load book metadata
    print("\n[1/5] Loading book catalog metadata...")
    meta = load_book_metadata(DB_FILE)
    book_texts = meta["book_texts"]

    # 2. Load pre-computed catalog embeddings
    print(f"\n[2/5] Loading catalog embeddings from {CACHE_EMBEDDINGS.name}...")
    catalog_embeddings = np.load(CACHE_EMBEDDINGS)
    with open(CACHE_BOOK_IDS, "r", encoding="utf-8") as f:
        catalog_ids = json.load(f)
    print(f"      Loaded {len(catalog_ids):,} book embeddings.")

    catalog_tensor = torch.from_numpy(catalog_embeddings).to(DEVICE)
    if catalog_tensor.dtype != torch.float16 and USE_FP16 and DEVICE == "cuda":
        catalog_tensor = catalog_tensor.half()

    # 3. Load Stage 1 Retriever & Stage 2 Jina Reranker
    print(f"\n[3/5] Loading Stage 1 Arctic Retriever: {ARCTIC_MODEL_PATH.name}...")
    bi_encoder = SentenceTransformer(str(ARCTIC_MODEL_PATH), device=DEVICE)
    if USE_FP16 and DEVICE == "cuda":
        bi_encoder.half()

    jina_path = str(JINA_MODEL_PATH) if JINA_MODEL_PATH.exists() else "jinaai/jina-reranker-v2-base-multilingual"
    print(f"      Loading Stage 2 Jina Reranker v2: {Path(jina_path).name}...")
    jina_reranker = CrossEncoder(
        jina_path,
        max_length=256,
        trust_remote_code=True,
        device=DEVICE
    )

    # 4. Load Test Queries
    print(f"\n[4/5] Reading {sample_size:,} test queries from {TEST_DATA_FILE.name}...")
    test_rows = []
    with open(TEST_DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            test_rows.append(json.loads(line))
            if len(test_rows) >= sample_size:
                break

    N = len(test_rows)

    # Trackers
    # Stage 1 (Bi-Encoder alone)
    s1_r1, s1_r5, s1_r10, s1_r50 = 0, 0, 0, 0
    s1_mrr10 = 0.0

    # Stage 2 (Arctic + Jina Reranker)
    jina_r1, jina_r5, jina_r10 = 0, 0, 0
    jina_mrr10 = 0.0

    s1_latencies = []
    jina_latencies = []

    print(f"\n[5/5] Evaluating Two-Stage Retrieval across {N:,} queries...")
    t0_all = time.time()

    for idx, row in enumerate(test_rows):
        query = row["query"]
        pos_id = str(row["positive_id"])

        # Stage 1: Dense Retrieval across 64,552 books
        t0_s1 = time.time()
        q_text = f"{QUERY_INSTRUCTION}{query}"
        q_emb = bi_encoder.encode([q_text], convert_to_tensor=True, normalize_embeddings=True)
        if q_emb.dtype != catalog_tensor.dtype:
            q_emb = q_emb.to(catalog_tensor.dtype)

        scores = torch.matmul(catalog_tensor, q_emb.squeeze(0))
        top_k_indices = torch.topk(scores, k=TOP_K_CANDIDATES).indices.cpu().numpy()
        top_k_ids = [catalog_ids[i] for i in top_k_indices]
        s1_latencies.append((time.time() - t0_s1) * 1000)

        # Stage 1 Metrics
        if pos_id in top_k_ids:
            rank = top_k_ids.index(pos_id) + 1
            if rank <= 1:
                s1_r1 += 1
            if rank <= 5:
                s1_r5 += 1
            if rank <= 10:
                s1_r10 += 1
                s1_mrr10 += 1.0 / rank
            if rank <= 50:
                s1_r50 += 1

        # Stage 2: Jina Cross-Encoder Reranking
        t0_s2 = time.time()
        candidate_texts = [book_texts.get(cid, "Untitled") for cid in top_k_ids]
        jina_pairs = [(query, text) for text in candidate_texts]

        jina_scores = jina_reranker.predict(jina_pairs, batch_size=TOP_K_CANDIDATES, show_progress_bar=False)
        reranked_order = np.argsort(jina_scores)[::-1]
        reranked_ids = [top_k_ids[i] for i in reranked_order]
        jina_latencies.append((time.time() - t0_s2) * 1000)

        # Stage 2 Metrics
        if pos_id in reranked_ids:
            rpos = reranked_ids.index(pos_id) + 1
            if rpos <= 1:
                jina_r1 += 1
            if rpos <= 5:
                jina_r5 += 1
            if rpos <= 10:
                jina_r10 += 1
                jina_mrr10 += 1.0 / rpos

        if (idx + 1) % 200 == 0 or (idx + 1) == N:
            print(f"  Processed {idx + 1}/{N} queries | S1 R@1: {s1_r1/(idx+1)*100:.2f}% | Jina R@1: {jina_r1/(idx+1)*100:.2f}% | Jina R@10: {jina_r10/(idx+1)*100:.2f}%")

    total_eval_time = time.time() - t0_all

    s1_metrics = {
        "recall_1": round((s1_r1 / N) * 100, 2),
        "recall_5": round((s1_r5 / N) * 100, 2),
        "recall_10": round((s1_r10 / N) * 100, 2),
        "recall_50": round((s1_r50 / N) * 100, 2),
        "mrr_10": round(s1_mrr10 / N, 4),
        "mean_latency_ms": round(float(np.mean(s1_latencies)), 2)
    }

    jina_metrics = {
        "recall_1": round((jina_r1 / N) * 100, 2),
        "recall_5": round((jina_r5 / N) * 100, 2),
        "recall_10": round((jina_r10 / N) * 100, 2),
        "recall_50": round((s1_r50 / N) * 100, 2), # R@50 remains candidate pool upper bound
        "mrr_10": round(jina_mrr10 / N, 4),
        "mean_latency_ms": round(float(np.mean(jina_latencies)), 2)
    }

    print("\n" + "=" * 80)
    print("  TWO-STAGE RETRIEVAL BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Metric':<20} | {'Stage 1: Arctic Dense':<25} | {'Stage 2: Jina Reranker v2':<25}")
    print("-" * 80)
    print(f"{'Recall@1':<20} | {s1_metrics['recall_1']:>23.2f}% | {jina_metrics['recall_1']:>23.2f}%")
    print(f"{'Recall@5':<20} | {s1_metrics['recall_5']:>23.2f}% | {jina_metrics['recall_5']:>23.2f}%")
    print(f"{'Recall@10':<20} | {s1_metrics['recall_10']:>23.2f}% | {jina_metrics['recall_10']:>23.2f}%")
    print(f"{'Recall@50':<20} | {s1_metrics['recall_50']:>23.2f}% | {jina_metrics['recall_50']:>23.2f}%")
    print(f"{'MRR@10':<20} | {s1_metrics['mrr_10']:>24.4f} | {jina_metrics['mrr_10']:>24.4f}")
    print(f"{'Latency (ms)':<20} | {s1_metrics['mean_latency_ms']:>21.2f} ms | {jina_metrics['mean_latency_ms']:>21.2f} ms")
    print("=" * 80)

    # Save results payload
    payload = {
        "title": "Two-Stage Retrieval Benchmark (Arctic Dense + Jina Reranker v2)",
        "retriever": "Snowflake/snowflake-arctic-embed-m-v1.5 (Progressive Refinement)",
        "reranker": "jinaai/jina-reranker-v2-base-multilingual (Fine-Tuned)",
        "catalog_size": 64552,
        "sample_size": N,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage_1_retriever": s1_metrics,
        "stage_2_jina_reranker": jina_metrics
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[Saved] Detailed Jina reranker benchmark written to {OUT_JSON}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of test queries")
    args = parser.parse_args()
    run_jina_benchmark(args.sample_size)
