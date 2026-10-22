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
    DATA_DIR, TRAIN_FILE, VAL_FILE, TEST_FILE, DB_FILE, OUTPUT_DIR, RESULTS_DIR,
    TOP_K_RETRIEVE, MIN_RANK, MAX_RANK, NEW_NEGS_TO_ADD, OLD_NEGS_TO_KEEP,
    MAX_NEG_FREQUENCY, EASY_PAIR_KEEP_RATIO, QUERY_INSTRUCTION, DEVICE, USE_FP16
)
from src.data_loader import load_book_metadata, norm_text
from src.filters import is_false_negative
from src.metrics import evaluate_recall

def encode_catalog(model: SentenceTransformer, book_meta: dict, all_book_ids: list) -> np.ndarray:
    """Encodes catalog books into normalized embeddings on GPU."""
    print(f"[Catalog] Encoding {len(all_book_ids):,} books on {DEVICE}...", flush=True)
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
    print(f"[Catalog] Encoded {len(all_book_ids):,} books in {elapsed:.1f}s.", flush=True)
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
    print(f"\n=======================================================", flush=True)
    print(f"  CYCLE {cycle_num}: HARD NEGATIVE MINING (ARCTIC)", flush=True)
    print(f"  Input:  {input_file.name}", flush=True)
    print(f"  Output: {output_file.name}", flush=True)
    print(f"=======================================================", flush=True)

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

            cand_id_list = [all_book_ids[c] for c in cand_indices]
            pos_rank_new = cand_id_list.index(pos_id) if pos_id in cand_id_list else -1

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
            # Ensure exactly 5 negatives by backfilling from prior negatives if needed
            if len(combined_negs) < 5:
                for oid in [str(x) for x in row.get("negative_ids", [])]:
                    if oid not in combined_negs and oid != pos_id:
                        combined_negs.append(oid)
                    if len(combined_negs) >= 5:
                        break

            combined_titles = [meta["title_of"].get(nid, "Untitled") for nid in combined_negs]

            output_rows.append({
                "query": row["query"],
                "positive_id": pos_id,
                "positive_title": pos_title,
                "positive_rank": pos_rank_new,
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

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Mine hard negatives using fine-tuned Arctic model")
    parser.add_argument("--model-dir", default=str(OUTPUT_DIR / "arctic_phase1" / "best_model"), help="Path to fine-tuned model")
    parser.add_argument("--input-file", default=str(TRAIN_FILE), help="Input JSONL file")
    parser.add_argument("--output-file", default=str(DATA_DIR / "train_arctic_v2.jsonl"), help="Output JSONL file")
    parser.add_argument("--cycle", type=int, default=1, help="Mining cycle number")
    args = parser.parse_args()

    model_path = Path(args.model_dir)
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    print("=" * 70, flush=True)
    print(f"  STEP 3: HARD NEGATIVE MINING (CYCLE {args.cycle})", flush=True)
    print(f"  Model:   {model_path}", flush=True)
    print(f"  Input:   {input_path.name}", flush=True)
    print(f"  Output:  {output_path.name}", flush=True)
    print(f"  Device:  {DEVICE} (FP16: {USE_FP16})", flush=True)
    print("=" * 70, flush=True)

    # 1. Load book metadata
    meta = load_book_metadata(DB_FILE)
    all_book_ids = sorted(list(meta["book_texts"].keys()))
    print(f"Total catalog books: {len(all_book_ids):,}", flush=True)

    # 2. Load fine-tuned Arctic model
    print(f"\nLoading fine-tuned model from {model_path}...", flush=True)
    model = SentenceTransformer(str(model_path), device=DEVICE)
    if USE_FP16 and DEVICE == "cuda":
        model.half()
    model.eval()

    # 3. Pre-encode catalog
    catalog_emb = encode_catalog(model, meta, all_book_ids)

    # 4. Mine hard negatives
    mine_dataset(
        model=model,
        catalog_embeddings=catalog_emb,
        all_book_ids=all_book_ids,
        meta=meta,
        input_file=input_path,
        output_file=output_path,
        cycle_num=args.cycle
    )

    print("\n" + "=" * 70, flush=True)
    print(f"  CYCLE {args.cycle} MINING COMPLETED SUCCESSFULLY!", flush=True)
    print(f"  Output Dataset: {output_path}", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()

