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
CONTRASTIVE_MODEL_PATH = OUTPUT_DIR / "jina_contrastive_reranker" / "best_model"
POINTWISE_MODEL_PATH = OUTPUT_DIR / "jina_reranker" / "best_model"

CACHE_EMBEDDINGS = PROJECT_ROOT / "app" / ".cache" / "catalog_embeddings_fp16.npy"
CACHE_BOOK_IDS = PROJECT_ROOT / "app" / ".cache" / "catalog_book_ids.json"
TEST_DATA_FILE = PROJECT_ROOT / "data" / "test.jsonl"
OUT_JSON = RESULTS_DIR / "benchmark_jina_contrastive.json"

TOP_K_CANDIDATES = 50

def run_jina_benchmark(sample_size: int = 1000, model_path_str: str = None):
    # Determine model path
    if model_path_str:
        jina_path = Path(model_path_str)
    elif CONTRASTIVE_MODEL_PATH.exists():
        jina_path = CONTRASTIVE_MODEL_PATH
    elif POINTWISE_MODEL_PATH.exists():
        jina_path = POINTWISE_MODEL_PATH
    else:
        jina_path = Path("jinaai/jina-reranker-v2-base-multilingual")

    model_display_name = "Jina Contrastive Reranker v2 (InfoNCE)" if "contrastive" in str(jina_path) else "Jina Reranker v2"

    print("=" * 85)
    print(f"  TWO-STAGE BENCHMARK: ARCTIC RETRIEVER + {model_display_name.upper()}")
    print(f"  Model Path:   {jina_path}")
    print(f"  Catalog Size: 64,552 Books | Top Candidates Mined: {TOP_K_CANDIDATES}")
    print(f"  Test Query Sample: {sample_size:,} queries")
    print(f"  Device: {DEVICE}")
    print("=" * 85)

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

    print(f"      Loading Stage 2 Reranker: {jina_path.name}...")
    jina_reranker = CrossEncoder(
        str(jina_path),
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

    # Stage 2: Pure Jina Reranker
    pure_r1, pure_r5, pure_r10 = 0, 0, 0
    pure_mrr10 = 0.0

    # Stage 2: Hybrid Score Fusion (0.30 Dense + 0.70 Reranker)
    hyb_r1, hyb_r5, hyb_r10 = 0, 0, 0
    hyb_mrr10 = 0.0

    s1_latencies = []
    rerank_latencies = []

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

        ce_logits = jina_reranker.predict(jina_pairs, batch_size=TOP_K_CANDIDATES, show_progress_bar=False)
        rerank_latencies.append((time.time() - t0_s2) * 1000)

        # 2a. Pure Reranker Ranking
        pure_order = np.argsort(ce_logits)[::-1]
        pure_ranked_ids = [top_k_ids[i] for i in pure_order]

        if pos_id in pure_ranked_ids:
            rpos = pure_ranked_ids.index(pos_id) + 1
            if rpos <= 1:
                pure_r1 += 1
            if rpos <= 5:
                pure_r5 += 1
            if rpos <= 10:
                pure_r10 += 1
                pure_mrr10 += 1.0 / rpos

        # 2b. Hybrid Score Fusion (0.30 Dense + 0.70 Reranker)
        bi_sub_scores = scores[top_k_indices].cpu().numpy()
        bi_norm = (bi_sub_scores - bi_sub_scores.min()) / (bi_sub_scores.max() - bi_sub_scores.min() + 1e-6)
        ce_probs = 1.0 / (1.0 + np.exp(-ce_logits))
        hybrid_scores = 0.30 * bi_norm + 0.70 * ce_probs
        hybrid_order = np.argsort(hybrid_scores)[::-1]
        hybrid_ranked_ids = [top_k_ids[i] for i in hybrid_order]

        if pos_id in hybrid_ranked_ids:
            hpos = hybrid_ranked_ids.index(pos_id) + 1
            if hpos <= 1:
                hyb_r1 += 1
            if hpos <= 5:
                hyb_r5 += 1
            if hpos <= 10:
                hyb_r10 += 1
                hyb_mrr10 += 1.0 / hpos

        if (idx + 1) % 200 == 0 or (idx + 1) == N:
            print(f"  Processed {idx + 1}/{N} queries | Dense R@10: {s1_r10/(idx+1)*100:.2f}% | Pure Jina R@10: {pure_r10/(idx+1)*100:.2f}% | Hybrid R@10: {hyb_r10/(idx+1)*100:.2f}%")

    total_eval_time = time.time() - t0_all

    s1_metrics = {
        "recall_1": round((s1_r1 / N) * 100, 2),
        "recall_5": round((s1_r5 / N) * 100, 2),
        "recall_10": round((s1_r10 / N) * 100, 2),
        "recall_50": round((s1_r50 / N) * 100, 2),
        "mrr_10": round(s1_mrr10 / N, 4),
        "mean_latency_ms": round(float(np.mean(s1_latencies)), 2)
    }

    pure_metrics = {
        "recall_1": round((pure_r1 / N) * 100, 2),
        "recall_5": round((pure_r5 / N) * 100, 2),
        "recall_10": round((pure_r10 / N) * 100, 2),
        "recall_50": round((s1_r50 / N) * 100, 2),
        "mrr_10": round(pure_mrr10 / N, 4),
        "mean_latency_ms": round(float(np.mean(rerank_latencies)), 2)
    }

    hybrid_metrics = {
        "recall_1": round((hyb_r1 / N) * 100, 2),
        "recall_5": round((hyb_r5 / N) * 100, 2),
        "recall_10": round((hyb_r10 / N) * 100, 2),
        "recall_50": round((s1_r50 / N) * 100, 2),
        "mrr_10": round(hyb_mrr10 / N, 4),
        "mean_latency_ms": round(float(np.mean(s1_latencies)) + float(np.mean(rerank_latencies)), 2)
    }

    print("\n" + "=" * 95)
    print("  TWO-STAGE RETRIEVAL BENCHMARK RESULTS (64,552 BOOKS)")
    print("=" * 95)
    print(f"{'Metric':<18} | {'Stage 1: Arctic Dense':<23} | {'Stage 2: Pure Jina':<23} | {'Stage 2: Hybrid Cascade':<23}")
    print("-" * 95)
    print(f"{'Recall@1':<18} | {s1_metrics['recall_1']:>21.2f}% | {pure_metrics['recall_1']:>21.2f}% | {hybrid_metrics['recall_1']:>21.2f}%")
    print(f"{'Recall@5':<18} | {s1_metrics['recall_5']:>21.2f}% | {pure_metrics['recall_5']:>21.2f}% | {hybrid_metrics['recall_5']:>21.2f}%")
    print(f"{'Recall@10':<18} | {s1_metrics['recall_10']:>21.2f}% | {pure_metrics['recall_10']:>21.2f}% | {hybrid_metrics['recall_10']:>21.2f}%")
    print(f"{'Recall@50':<18} | {s1_metrics['recall_50']:>21.2f}% | {pure_metrics['recall_50']:>21.2f}% | {hybrid_metrics['recall_50']:>21.2f}%")
    print(f"{'MRR@10':<18} | {s1_metrics['mrr_10']:>22.4f} | {pure_metrics['mrr_10']:>22.4f} | {hybrid_metrics['mrr_10']:>22.4f}")
    print(f"{'Latency (ms)':<18} | {s1_metrics['mean_latency_ms']:>19.2f} ms | {pure_metrics['mean_latency_ms']:>19.2f} ms | {hybrid_metrics['mean_latency_ms']:>19.2f} ms")
    print("=" * 95)

    # Save results payload
    payload = {
        "title": "Two-Stage Retrieval Benchmark (Arctic Dense + Jina Reranker v2)",
        "retriever": "Snowflake/snowflake-arctic-embed-m-v1.5 (Progressive Refinement)",
        "reranker": str(jina_path),
        "catalog_size": 64552,
        "sample_size": N,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage_1_retriever": s1_metrics,
        "stage_2_pure_reranker": pure_metrics,
        "stage_2_hybrid_cascade": hybrid_metrics
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[Saved] Detailed Jina reranker benchmark written to {OUT_JSON}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of test queries")
    parser.add_argument("--model-path", type=str, default=None, help="Path to reranker model checkpoint")
    args = parser.parse_args()
    run_jina_benchmark(args.sample_size, args.model_path)
