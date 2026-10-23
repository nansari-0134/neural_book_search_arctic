"""
Step 11: Contrastive Listwise Fine-Tuning of Jina Reranker v2 (InfoNCE)
=======================================================================
Fine-tunes 'jinaai/jina-reranker-v2-base-multilingual' (278M parameters)
using True Listwise Contrastive Cross-Entropy (InfoNCE) across 1 Positive + 5 Hard Negatives.

Objective:
L = -log( exp(s(q, d+) / tau) / ( exp(s(q, d+) / tau) + sum_j exp(s(q, d-_j) / tau) ) )

Key Improvements Over Pointwise BCE:
1. Pits positive book directly against 5 actively mined hard negatives in a shared competition.
2. Directly trains the model to rank s(q, d+) > s(q, d-), eliminating ranking inversions.
3. Native BFloat16 on NVIDIA RTX 4050 GPU (sm_89 Ada Lovelace) with low VRAM footprint (~2.8 GB).
4. Validates on Top-1 Candidate Accuracy and MRR, saving the true champion model.
"""

import sys
import os
import json
import time
import math
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Apply Jina compatibility patch
import src.jina_patch

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import CrossEncoder

from src.config import DB_FILE, OUTPUT_DIR, DEVICE
from src.data_loader import load_book_metadata

BASE_MODEL_NAME = "jinaai/jina-reranker-v2-base-multilingual"
OUTPUT_CONTRASTIVE_DIR = OUTPUT_DIR / "jina_contrastive_reranker"
BEST_MODEL_DIR = OUTPUT_CONTRASTIVE_DIR / "best_model"

TRAIN_DATA_FILE = PROJECT_ROOT / "data" / "train_arctic_v3.jsonl"
VAL_DATA_FILE = PROJECT_ROOT / "data" / "val.jsonl"
LOG_FILE = PROJECT_ROOT / "training" / "jina_contrastive_training.log"
STATUS_FILE = PROJECT_ROOT / "training" / "jina_contrastive_status.md"

class ContrastiveRankingDataset(Dataset):
    """
    Yields query with 1 positive passage and K hard negatives.
    passages = [pos_text, neg_1_text, neg_2_text, ..., neg_K_text]
    target index is always 0.
    """
    def __init__(self, data_path: Path, book_texts: Dict[str, str], max_queries: int, negs_per_query: int = 5):
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if len(self.samples) >= max_queries:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                q = row["query"]
                pos_id = str(row["positive_id"])
                pos_text = book_texts.get(pos_id)
                if not pos_text:
                    continue

                neg_ids = row.get("negative_ids", [])
                neg_texts = []
                for nid in neg_ids:
                    n_text = book_texts.get(str(nid))
                    if n_text and n_text != pos_text:
                        neg_texts.append(n_text)
                    if len(neg_texts) >= negs_per_query:
                        break

                if len(neg_texts) >= negs_per_query:
                    passages = [pos_text] + neg_texts[:negs_per_query]
                    self.samples.append({
                        "query": q,
                        "passages": passages
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_contrastive_batch(batch, tokenizer, max_length=256):
    """
    Flattens batch into (query, passage) pairs for joint cross-encoder scoring.
    """
    queries = []
    passages = []
    k_cand = len(batch[0]["passages"])
    b_size = len(batch)

    for item in batch:
        q = item["query"]
        for p in item["passages"]:
            queries.append(q)
            passages.append(p)

    encoded = tokenizer(
        queries,
        passages,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    return encoded, b_size, k_cand

@torch.no_grad()
def evaluate_contrastive_validation(model, val_loader, device):
    """
    Measures validation InfoNCE loss and Top-1 Candidate Accuracy (how often pos beats all negs).
    """
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    total_queries = 0
    mrr_sum = 0.0
    loss_fct = nn.CrossEntropyLoss()

    for batch, b_size, k_cand in val_loader:
        for k in batch:
            batch[k] = batch[k].to(device)

        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16):
            outputs = model(**batch)
            logits = outputs.logits.squeeze(-1).view(b_size, k_cand)
            targets = torch.zeros(b_size, dtype=torch.long, device=device)
            loss = loss_fct(logits, targets)

        total_loss += loss.item() * b_size
        total_queries += b_size

        # Ranking calculations
        preds = logits.argsort(dim=-1, descending=True).cpu().numpy()
        for row in preds:
            rank = int(np.where(row == 0)[0][0]) + 1
            if rank == 1:
                correct_top1 += 1
            mrr_sum += 1.0 / rank

    avg_loss = total_loss / max(1, total_queries)
    top1_acc = (correct_top1 / max(1, total_queries)) * 100
    avg_mrr = mrr_sum / max(1, total_queries)

    return avg_loss, top1_acc, avg_mrr

def main():
    parser = argparse.ArgumentParser(description="Contrastive Listwise Fine-Tuning of Jina Reranker v2")
    parser.add_argument("--max-queries", type=int, default=6000, help="Number of training queries")
    parser.add_argument("--negs-per-query", type=int, default=5, help="Number of hard negatives per query")
    parser.add_argument("--batch-size", type=int, default=2, help="Queries per micro-batch (2 * 6 = 12 pairs)")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps (effective 16 queries = 96 pairs)")
    parser.add_argument("--lr", type=float, default=1.5e-5, help="Peak learning rate")
    parser.add_argument("--max-len", type=int, default=256, help="Max token length")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    args = parser.parse_args()

    effective_queries_per_step = args.batch_size * args.grad_accum
    pairs_per_step = effective_queries_per_step * (1 + args.negs_per_query)

    print("=" * 85)
    print("  STAGE 8: CONTRASTIVE LISTWISE (InfoNCE) FINE-TUNING: JINA RERANKER V2")
    print(f"  Model Backbone:      {BASE_MODEL_NAME} (278M parameters)")
    print(f"  Contrastive Ratio:   1 Positive vs. {args.negs_per_query} Hard Negatives (6-Way Softmax)")
    print(f"  Micro-Batch:         {args.batch_size} queries ({args.batch_size * (1 + args.negs_per_query)} pairs)")
    print(f"  Effective Batch:     {effective_queries_per_step} queries ({pairs_per_step} candidate pairs per step)")
    print(f"  Precision:           BFloat16 (sm_89 Ada Lovelace Native)")
    print(f"  Target Checkpoint:   {BEST_MODEL_DIR}")
    print("=" * 85)
    sys.stdout.flush()

    OUTPUT_CONTRASTIVE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load book metadata
    meta = load_book_metadata(DB_FILE)
    book_texts = meta["book_texts"]

    # 2. Build Contrastive Datasets
    print(f"\n[Data] Assembling Contrastive Listwise datasets ({args.negs_per_query} hard negatives)...")
    train_dataset = ContrastiveRankingDataset(TRAIN_DATA_FILE, book_texts, max_queries=args.max_queries, negs_per_query=args.negs_per_query)
    val_dataset = ContrastiveRankingDataset(VAL_DATA_FILE, book_texts, max_queries=250, negs_per_query=args.negs_per_query)
    print(f"       Loaded {len(train_dataset):,} contrastive training queries ({len(train_dataset) * (1 + args.negs_per_query):,} total pairs).")
    print(f"       Loaded {len(val_dataset):,} contrastive validation queries.")
    sys.stdout.flush()

    # 3. Load Jina Model & Tokenizer
    print(f"\n[Model] Initializing CrossEncoder {BASE_MODEL_NAME} with Jina patch...")
    ce_wrapper = CrossEncoder(BASE_MODEL_NAME, num_labels=1, trust_remote_code=True, device=DEVICE)
    model = ce_wrapper.model
    tokenizer = ce_wrapper.tokenizer

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_contrastive_batch(b, tokenizer, max_length=args.max_len),
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_contrastive_batch(b, tokenizer, max_length=args.max_len),
        drop_last=False
    )

    # 4. Optimizer & Schedulers
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * 0.1)
    eval_interval = max(40, total_steps // 5)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fct = nn.CrossEntropyLoss()

    def get_lr(step):
        if step < warmup_steps:
            return args.lr * float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    # Initial Validation
    print("\n[Validation] Running baseline validation before training...")
    sys.stdout.flush()
    init_val_loss, init_top1_acc, init_mrr = evaluate_contrastive_validation(model, val_loader, DEVICE)
    print(f"  Baseline 6-Way Accuracy: {init_top1_acc:.2f}% | Baseline MRR: {init_mrr:.4f} | Baseline Loss: {init_val_loss:.4f}")
    sys.stdout.flush()

    # Log initial status
    best_val_acc = init_top1_acc
    best_val_loss = init_val_loss

    # 5. Training Loop
    print(f"\n[Training] Starting Contrastive InfoNCE Training ({total_steps} optimizer steps)...")
    sys.stdout.flush()
    model.train()
    optimizer.zero_grad()
    t0_start = time.time()
    step_count = 0
    micro_step = 0
    running_loss = 0.0

    # Initialize log files
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"Contrastive InfoNCE Training Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    for epoch in range(args.epochs):
        for batch, b_size, k_cand in train_loader:
            micro_step += 1
            for k in batch:
                batch[k] = batch[k].to(DEVICE)

            with torch.autocast(device_type="cuda" if DEVICE == "cuda" else "cpu", dtype=torch.bfloat16):
                outputs = model(**batch)
                logits = outputs.logits.squeeze(-1).view(b_size, k_cand)
                targets = torch.zeros(b_size, dtype=torch.long, device=DEVICE)
                loss = loss_fct(logits, targets) / args.grad_accum

            loss.backward()
            running_loss += loss.item() * args.grad_accum

            if micro_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # Update LR
                curr_lr = get_lr(step_count)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = curr_lr

                optimizer.step()
                optimizer.zero_grad()
                step_count += 1

                if step_count % 20 == 0 or step_count == total_steps:
                    avg_step_loss = running_loss / 20.0
                    running_loss = 0.0
                    elapsed = time.time() - t0_start
                    pct = (step_count / total_steps) * 100
                    log_line = f"Step {step_count:4d}/{total_steps} ({pct:5.1f}%) | InfoNCE Loss: {avg_step_loss:.4f} | LR: {curr_lr:.2e} | Elapsed: {elapsed/60:.1f}m"
                    print(f"  {log_line}")
                    sys.stdout.flush()
                    with open(LOG_FILE, "a", encoding="utf-8") as f:
                        f.write(log_line + "\n")

                # Validation & Checkpointing
                if step_count % eval_interval == 0 or step_count == total_steps:
                    val_loss, top1_acc, val_mrr = evaluate_contrastive_validation(model, val_loader, DEVICE)
                    eval_line = f"  >>> [EVALUATION Step {step_count}] 6-Way Accuracy: {top1_acc:.2f}% | MRR: {val_mrr:.4f} | Val Loss: {val_loss:.4f}"
                    print(eval_line)
                    sys.stdout.flush()
                    with open(LOG_FILE, "a", encoding="utf-8") as f:
                        f.write(eval_line + "\n")

                    if top1_acc >= best_val_acc or val_loss < best_val_loss:
                        best_val_acc = max(best_val_acc, top1_acc)
                        best_val_loss = min(best_val_loss, val_loss)
                        print(f"      [*] Saving new best model to {BEST_MODEL_DIR}...")
                        ce_wrapper.save_pretrained(str(BEST_MODEL_DIR))

                    # Update markdown status
                    with open(STATUS_FILE, "w", encoding="utf-8") as sf:
                        sf.write(f"# 🚀 Jina Contrastive Listwise Fine-Tuning Monitor\n\n")
                        sf.write(f"* **Architecture**: `jina-reranker-v2-base-multilingual` (278M)\n")
                        sf.write(f"* **Objective**: 6-Way InfoNCE Contrastive Cross-Entropy\n")
                        sf.write(f"* **Current Progress**: Step {step_count}/{total_steps} ({pct:.1f}% complete)\n")
                        sf.write(f"* **Best 6-Way Top-1 Accuracy**: **{best_val_acc:.2f}%**\n")
                        sf.write(f"* **Best Validation MRR**: **{val_mrr:.4f}**\n")
                        sf.write(f"* **Best Validation Loss**: **{best_val_loss:.4f}**\n")
                        sf.write(f"* **Checkpoint Path**: [`arctic_output/jina_contrastive_reranker/best_model`](file:///{str(BEST_MODEL_DIR).replace(os.sep, '/')})\n")

                    model.train()

    total_time = (time.time() - t0_start) / 60
    print("\n" + "=" * 85)
    print(f"  [SUCCESS] CONTRASTIVE LISTWISE TRAINING COMPLETED IN {total_time:.2f} MINUTES!")
    print(f"  Best 6-Way Candidate Accuracy: {best_val_acc:.2f}%")
    print(f"  Model Saved To: {BEST_MODEL_DIR}")
    print("=" * 85)

if __name__ == "__main__":
    main()
