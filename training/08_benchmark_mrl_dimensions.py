"""
Step 8: Empirical Matryoshka Representation Learning (MRL) Benchmark
====================================================================
Evaluates the Champion Arctic retriever across different vector dimensions:
[768, 512, 256, 128, 64] against the full 64,552 book catalog.

Computes exact Recall@1, 5, 10, 50, MRR@10, MRR@full, latency, and retention rates.
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
from sentence_transformers import SentenceTransformer

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    OUTPUT_DIR, RESULTS_DIR, QUERY_INSTRUCTION, DEVICE, USE_FP16
)

ARCTIC_MODEL_PATH = OUTPUT_DIR / "arctic_v3" / "best_model"
CACHE_EMBEDDINGS = PROJECT_ROOT / "app" / ".cache" / "catalog_embeddings_fp16.npy"
CACHE_BOOK_IDS = PROJECT_ROOT / "app" / ".cache" / "catalog_book_ids.json"
TEST_DATA_FILE = PROJECT_ROOT / "data" / "test.jsonl"
OUT_JSON = RESULTS_DIR / "mrl_dimension_benchmark.json"

MRL_DIMENSIONS = [768, 512, 256, 128, 64]

def benchmark_dimensions(sample_size: int = 1000):
    print("=" * 80)
    print("  EMPIRICAL MATRYOSHKA DIMENSIONAL BENCHMARK (64,552 CATALOG BOOKS)")
    print(f"  Dimensions Evaluated: {MRL_DIMENSIONS}")
    print(f"  Test Query Sample:    {sample_size:,} queries")
    print(f"  Device:               {DEVICE} (FP16: {USE_FP16})")
    print("=" * 80)

    # 1. Load catalog embeddings and IDs
    print(f"\n[1/4] Loading catalog embeddings from {CACHE_EMBEDDINGS.name}...")
    catalog_np = np.load(CACHE_EMBEDDINGS)  # shape (64552, 768), float16 or float32
    with open(CACHE_BOOK_IDS, "r", encoding="utf-8") as f:
        catalog_ids = json.load(f)
    catalog_size = len(catalog_ids)
    print(f"      Loaded {catalog_size:,} catalog vectors of shape {catalog_np.shape}.")

    id_to_idx = {str(bid): i for i, bid in enumerate(catalog_ids)}

    # 2. Load test queries
    print(f"\n[2/4] Reading {sample_size:,} test queries from {TEST_DATA_FILE.name}...")
    test_rows = []
    with open(TEST_DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            pos_id = str(item.get("positive_id", ""))
            if pos_id in id_to_idx:
                test_rows.append(item)
                if len(test_rows) >= sample_size:
                    break

    num_eval = len(test_rows)
    print(f"      Valid evaluation pairs loaded: {num_eval:,}")

    # 3. Load model and encode test queries at 768-d
    print(f"\n[3/4] Loading Champion Arctic retriever from {ARCTIC_MODEL_PATH.name}...")
    model = SentenceTransformer(str(ARCTIC_MODEL_PATH), device=DEVICE)
    if USE_FP16 and DEVICE == "cuda":
        model.half()
    model.eval()

    print(f"      Pre-encoding {num_eval:,} query vectors (768-d) with instruction...")
    query_texts = [f"{QUERY_INSTRUCTION}{r['query']}" for r in test_rows]
    target_indices = np.array([id_to_idx[str(r["positive_id"])] for r in test_rows], dtype=np.int64)

    t_enc_start = time.time()
    query_emb_768 = model.encode(
        query_texts,
        batch_size=128 if DEVICE == "cuda" else 32,
        convert_to_numpy=True,
        normalize_embeddings=False, # We normalize per-dimension slice!
        show_progress_bar=False
    ).astype("float32")
    enc_duration = time.time() - t_enc_start
    print(f"      Queries encoded in {enc_duration:.2f}s ({enc_duration/num_eval*1000:.2f} ms/query).")

    # Convert catalog to float32 numpy for consistent slicing and normalization
    cat_np_f32 = catalog_np.astype("float32")

    # 4. Evaluate each MRL dimension
    print("\n[4/4] Evaluating Retrieval Performance Across MRL Dimensions...")
    results_by_dim = {}
    baseline_768_r10 = None
    baseline_768_mrr10 = None

    for dim in MRL_DIMENSIONS:
        print(f"\n--- Dimension: {dim}-d ---")
        # 1. Truncate & L2 re-normalize catalog embeddings
        cat_slice = cat_np_f32[:, :dim].copy()
        cat_norms = np.linalg.norm(cat_slice, axis=1, keepdims=True)
        cat_norms[cat_norms == 0] = 1e-12
        cat_slice = cat_slice / cat_norms

        # 2. Truncate & L2 re-normalize query embeddings
        q_slice = query_emb_768[:, :dim].copy()
        q_norms = np.linalg.norm(q_slice, axis=1, keepdims=True)
        q_norms[q_norms == 0] = 1e-12
        q_slice = q_slice / q_norms

        # 3. Transfer to torch on target device
        cat_tensor = torch.from_numpy(cat_slice).to(DEVICE)
        q_tensor = torch.from_numpy(q_slice).to(DEVICE)
        if USE_FP16 and DEVICE == "cuda":
            cat_tensor = cat_tensor.half()
            q_tensor = q_tensor.half()

        # 4. Compute ranking metrics in batches
        batch_size = 128
        ranks = []
        latencies = []

        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t_search_start = time.time()

        for b_start in range(0, num_eval, batch_size):
            b_end = min(b_start + batch_size, num_eval)
            q_b = q_tensor[b_start:b_end] # [B, dim]
            targets_b = target_indices[b_start:b_end]

            t0_b = time.time()
            # Cosine similarity matrix: [B, 64552]
            sims = torch.matmul(q_b, cat_tensor.T)
            torch.cuda.synchronize() if DEVICE == "cuda" else None
            latencies.append((time.time() - t0_b) / (b_end - b_start) * 1000)

            # Get target positive scores
            pos_scores = sims[torch.arange(len(targets_b)), targets_b] # [B]

            # Exact rank = count of items with score > positive_score
            higher_counts = torch.sum(sims > pos_scores.unsqueeze(1), dim=1).cpu().numpy()
            ranks.extend(higher_counts.tolist())

        total_search_time = time.time() - t_search_start
        ranks = np.array(ranks)

        # Compute recall at various thresholds (rank is 0-indexed count of items beating positive)
        r1 = float(np.mean(ranks < 1) * 100)
        r5 = float(np.mean(ranks < 5) * 100)
        r10 = float(np.mean(ranks < 10) * 100)
        r50 = float(np.mean(ranks < 50) * 100)

        # MRR@10: 1 / (rank + 1) for rank < 10, else 0
        reciprocal_ranks_10 = np.where(ranks < 10, 1.0 / (ranks + 1), 0.0)
        mrr10 = float(np.mean(reciprocal_ranks_10))

        # Full MRR
        reciprocal_ranks_full = 1.0 / (ranks + 1)
        mrr_full = float(np.mean(reciprocal_ranks_full))

        avg_latency_ms = float(np.mean(latencies))
        ram_mb = float((catalog_size * dim * 2) / (1024 * 1024))
        flops_per_search = catalog_size * dim * 2

        if dim == 768:
            baseline_768_r10 = r10
            baseline_768_mrr10 = mrr10
            retention_r10 = 100.0
            retention_mrr10 = 100.0
            speedup_ratio = 1.0
        else:
            retention_r10 = float((r10 / baseline_768_r10) * 100)
            retention_mrr10 = float((mrr10 / baseline_768_mrr10) * 100)
            speedup_ratio = float(768 / dim)

        results_by_dim[str(dim)] = {
            "dimension": dim,
            "recall_1": round(r1, 2),
            "recall_5": round(r5, 2),
            "recall_10": round(r10, 2),
            "recall_50": round(r50, 2),
            "mrr_10": round(mrr10, 4),
            "mrr_full": round(mrr_full, 4),
            "r10_retention_pct": round(retention_r10, 2),
            "mrr10_retention_pct": round(retention_mrr10, 2),
            "ram_fp16_mb": round(ram_mb, 2),
            "flops_per_query_m": round(flops_per_search / 1e6, 2),
            "speedup_theoretical": round(speedup_ratio, 1),
            "avg_matrix_latency_ms": round(avg_latency_ms, 3)
        }

        print(f"  R@1:  {r1:5.2f}% | R@5:  {r5:5.2f}% | R@10: {r10:5.2f}% | R@50: {r50:5.2f}%")
        print(f"  MRR@10: {mrr10:.4f} | R@10 Retained: {retention_r10:5.2f}% | RAM: {ram_mb:5.1f} MB | Latency: {avg_latency_ms:.3f} ms")

    # 5. Output Summary Table
    print("\n" + "=" * 95)
    print("  SUMMARY: EMPIRICAL MATRYOSHKA METRICS (CHAMPION ARCTIC MODEL)")
    print("=" * 95)
    header = f"{'Dim':>5} | {'R@1':>6} | {'R@5':>6} | {'R@10':>6} | {'R@50':>6} | {'MRR@10':>7} | {'R@10 Ret':>9} | {'RAM (FP16)':>10} | {'FLOPs Red':>9}"
    print(header)
    print("-" * 95)
    for dim_str, m in results_by_dim.items():
        row = f"{dim_str:>5} | {m['recall_1']:5.2f}% | {m['recall_5']:5.2f}% | {m['recall_10']:5.2f}% | {m['recall_50']:5.2f}% | {m['mrr_10']:7.4f} | {m['r10_retention_pct']:8.2f}% | {m['ram_fp16_mb']:8.1f} MB | {m['speedup_theoretical']:7.1f}x"
        print(row)
    print("=" * 95)

    # 6. Save payload
    payload = {
        "title": "Empirical Matryoshka Representation Learning (MRL) Benchmark",
        "model": "Snowflake/snowflake-arctic-embed-m-v1.5 (Progressive Refinement Champion)",
        "catalog_size": catalog_size,
        "sample_size": num_eval,
        "device": DEVICE,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dimensions": results_by_dim
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[Saved] Detailed MRL benchmark written to {OUT_JSON}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of test queries to evaluate")
    args = parser.parse_args()
    benchmark_dimensions(args.sample_size)
