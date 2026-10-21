"""
Step 3: Iterative Hard Negative Mining and Multi-Cycle Fine-Tuning
Mines actual false-positive confusions using the fine-tuned Snowflake Arctic model,
filters with the 8-layer protection shield, and trains progressive cycles.
"""
import sys
import os
import json
import time
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TRAIN_FILE, VAL_FILE, TEST_FILE, DB_FILE, OUTPUT_DIR, RESULTS_DIR,
    TOP_K_RETRIEVE, MIN_RANK, MAX_RANK, NEW_NEGS_TO_ADD, OLD_NEGS_TO_KEEP,
    MAX_NEG_FREQUENCY, EASY_PAIR_KEEP_RATIO, QUERY_INSTRUCTION, DEVICE, USE_FP16
)
from src.data_loader import load_book_metadata, norm_text
from src.filters import is_false_negative
from src.metrics import evaluate_recall

def encode_catalog(model: SentenceTransformer, book_meta: dict, all_book_ids: list) -> np.ndarray:
    """Encodes catalog books into normalized embeddings on GPU."""
    print(f"[Catalog] Encoding {len(all_book_ids):,} books on {DEVICE}...")
    t0 = time.time()
    texts = [book_meta["book_texts"][bid] for bid in all_book_ids]

    with torch.no_grad():
        embeddings = model.encode(
            texts,
            batch_size=128 if DEVICE == "cuda" else 32,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype("float32")

    elapsed = time.time() - t0
    print(f"[Catalog] Encoded {len(all_book_ids):,} books in {elapsed:.1f}s.")
    return embeddings

def mine_dataset(
    model: SentenceTransformer,
    catalog_embeddings: np.ndarray,
    all_book_ids: list,
    meta: dict,
    input_file: Path,
    output_file: Path,
    cycle_num: int,
):
    print(f"\n=======================================================")
    print(f"  CYCLE {cycle_num}: HARD NEGATIVE MINING (ARCTIC)")
    print(f"  Input:  {input_file.name}")
    print(f"  Output: {output_file.name}")
    print(f"=======================================================")

    query_positives = defaultdict(set)
    train_rows = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                train_rows.append(row)
                query_positives[norm_text(row["query"])].add(str(row["positive_id"]))

    catalog_tensor = torch.from_numpy(catalog_embeddings).to(DEVICE)
    if USE_FP16 and DEVICE == "cuda":
        catalog_tensor = catalog_tensor.half()

    neg_frequency = Counter()
    output_rows = []
    blocked_counts = Counter()
    skipped_easy = 0

    batch_size = 128
    total = len(train_rows)
    t0 = time.time()

    for b_start in tqdm(range(0, total, batch_size), desc=f"Mining Cycle {cycle_num}"):
        b_end = min(b_start + batch_size, total)
        batch = train_rows[b_start:b_end]

        active_items = []
        for r in batch:
            pos_rank = r.get("positive_rank", -1)
            if pos_rank == 0 and random.random() > EASY_PAIR_KEEP_RATIO:
                skipped_easy += 1
                continue
            active_items.append(r)

        if not active_items:
            continue

        q_texts = [QUERY_INSTRUCTION + r["query"] for r in active_items]
        with torch.no_grad():
            q_embs = model.encode(
                q_texts,
                batch_size=len(q_texts),
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_tensor=True
            )
            if USE_FP16 and DEVICE == "cuda":
                q_embs = q_embs.half()

            sim_matrix = torch.matmul(q_embs, catalog_tensor.T)
            top_scores, top_indices = torch.topk(sim_matrix, k=TOP_K_RETRIEVE, dim=1)
            top_scores = top_scores.cpu().numpy()
            top_indices = top_indices.cpu().numpy()

        for i, row in enumerate(active_items):
            pos_id = str(row["positive_id"])
            pos_title = meta["title_of"].get(pos_id, "")
            norm_q = norm_text(row["query"])

            banned_ids = {pos_id}
            banned_ids.update(query_positives.get(norm_q, set()))

            cand_indices = top_indices[i]
            cand_scores = top_scores[i]

            new_negs = []
            for rank, (c_idx, score) in enumerate(zip(cand_indices, cand_scores)):
                if rank < MIN_RANK:
                    continue
                if rank > MAX_RANK:
                    break

                cand_id = all_book_ids[c_idx]

                # 8-Layer False Negative Check
                is_false, reason = is_false_negative(pos_id, cand_id, score, meta, banned_ids)
                if is_false:
                    blocked_counts[reason] += 1
                    continue

                if neg_frequency[cand_id] >= MAX_NEG_FREQUENCY:
                    blocked_counts["frequency_cap"] += 1
                    continue

                new_negs.append(cand_id)
                if len(new_negs) >= NEW_NEGS_TO_ADD:
                    break

            for nid in new_negs:
                neg_frequency[nid] += 1

            # Retain prior negatives, filtering any that now match strict checks
            old_negs = []
            for oid in [str(x) for x in row.get("negative_ids", [])]:
                if oid == pos_id or oid in new_negs:
                    continue
                is_false, _ = is_false_negative(pos_id, oid, 0.0, meta, banned_ids)
                if is_false:
                    continue
                old_negs.append(oid)
                if len(old_negs) >= OLD_NEGS_TO_KEEP:
                    break

            combined_negs = new_negs + old_negs
            if not combined_negs:
                combined_negs = [str(x) for x in row.get("negative_ids", [])[:3]]

            combined_titles = [meta["title_of"].get(nid, "Untitled") for nid in combined_negs]

            output_rows.append({
                "query": row["query"],
                "positive_id": pos_id,
                "positive_title": pos_title,
                "positive_rank": row.get("positive_rank", -1),
                "negative_ids": combined_negs,
                "negative_titles": combined_titles,
                "mined_arctic_count": len(new_negs),
                "retained_old_count": len(old_negs),
            })

    elapsed = time.time() - t0
    print(f"\n[Mining Complete] {len(output_rows):,} rows in {elapsed/60:.1f}m.")
    print(f"  Downsampled easy pairs: {skipped_easy:,} ({skipped_easy/total*100:.1f}%)")
    print(f"  False-negative blocks breakdown:")
    for reason, count in blocked_counts.most_common():
        print(f"    - {reason:25}: {count:,}")

    with open(output_file, "w", encoding="utf-8") as f:
        for r in output_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(output_rows):,} pairs to {output_file.name}")

if __name__ == "__main__":
    print("Iterative mining script loaded. Ready for pipeline invocation.")
