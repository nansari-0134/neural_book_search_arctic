"""
Step 12: High-Performance Pointwise BCE Fine-Tuning of Jina Reranker v2
========================================================================
Fine-tunes 'jinaai/jina-reranker-v2-base-multilingual' (278M parameters)
using Pointwise Binary Cross-Entropy (BCEWithLogitsLoss) on curated positive (1.0)
and adversarial hard negative (0.0) pairs.

Key Features:
- Native PyTorch loop with nn.BCEWithLogitsLoss()
- Dynamic batch padding in collate_fn for high throughput
- Native BFloat16 mixed precision on Ada Lovelace RTX 4050 GPU
- Evaluates validation BCE loss, candidate discrimination accuracy, and MRR@10
- Exports champion checkpoint to 'arctic_output/jina_bce_reranker/best_model'
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
from collections import defaultdict

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
OUTPUT_BCE_DIR = OUTPUT_DIR / "jina_bce_reranker"
BEST_MODEL_DIR = OUTPUT_BCE_DIR / "best_model"

TRAIN_DATA_FILE = PROJECT_ROOT / "data" / "train_arctic_v3.jsonl"
VAL_DATA_FILE = PROJECT_ROOT / "data" / "val.jsonl"
LOG_FILE = PROJECT_ROOT / "training" / "jina_pointwise_bce.log"
STATUS_FILE = PROJECT_ROOT / "training" / "jina_pointwise_bce_status.md"


class PointwisePairDataset(Dataset):
    """Yields individual (query, passage, label) tuples."""
    def __init__(self, data_path: Path, book_texts: Dict[str, str], max_queries: int, negs_per_query: int = 3):
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            count = 0
            for line in f:
                if count >= max_queries:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                q = row["query"]
                pos_id = str(row["positive_id"])
                pos_text = book_texts.get(pos_id)
                if not pos_text:
                    continue

                # Positive pair (label: 1.0)
                self.samples.append((q, pos_text, 1.0))

                # Hard negative pairs (label: 0.0)
                neg_ids = row.get("negative_ids", [])
                neg_count = 0
                for nid in neg_ids:
                    n_text = book_texts.get(str(nid))
                    if n_text and n_text != pos_text:
                        self.samples.append((q, n_text, 0.0))
                        neg_count += 1
                        if neg_count >= negs_per_query:
                            break

                count += 1

        random.seed(42)
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class GroupedValDataset:
    """Holds validation queries grouped into (pos_text, [neg_texts]) for MRR & ranking metrics."""
    def __init__(self, data_path: Path, book_texts: Dict[str, str], max_queries: int = 300, negs_per_query: int = 4):
        self.queries = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if len(self.queries) >= max_queries:
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
                    self.queries.append({
                        "query": q,
                        "pos_text": pos_text,
                        "neg_texts": neg_texts
                    })


def make_collate_fn(tokenizer, max_length: int = 256):
    """Dynamic batch padding to the longest sequence in the micro-batch."""
    def collate_fn(batch):
        queries = [b[0] for b in batch]
        passages = [b[1] for b in batch]
        labels = torch.tensor([b[2] for b in batch], dtype=torch.float)

        encoded = tokenizer(
            queries,
            passages,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        return encoded, labels
    return collate_fn


@torch.no_grad()
def evaluate_pointwise_validation(model, tokenizer, val_group: GroupedValDataset, device: str, max_length: int = 256):
    """Evaluates validation BCE loss, binary accuracy, Top-1 accuracy, and MRR@10."""
    model.eval()
    bce_loss_fct = nn.BCEWithLogitsLoss()
    total_bce = 0.0
    total_pairs = 0
    correct_binary = 0

    correct_top1 = 0
    mrr_sum = 0.0
    total_queries = len(val_group.queries)

    for item in val_group.queries:
        q = item["query"]
        candidates = [item["pos_text"]] + item["neg_texts"]
        labels = torch.tensor([1.0] + [0.0] * len(item["neg_texts"]), dtype=torch.float, device=device)

        queries = [q] * len(candidates)
        encoded = tokenizer(
            queries,
            candidates,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        for k in encoded:
            encoded[k] = encoded[k].to(device)

        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16):
            outputs = model(**encoded)
            logits = outputs.logits.squeeze(-1)
            loss = bce_loss_fct(logits, labels)

        total_bce += loss.item() * len(candidates)
        total_pairs += len(candidates)

        # Binary accuracy (threshold > 0.0 logit => sigmoid > 0.5)
        preds_binary = (logits > 0.0).float()
        correct_binary += (preds_binary == labels).sum().item()

        # Ranking calculations (relative score of pos at index 0 vs negs)
        scores_np = logits.float().cpu().numpy()
        ranked_indices = np.argsort(scores_np)[::-1]
        rank = int(np.where(ranked_indices == 0)[0][0]) + 1
        if rank == 1:
            correct_top1 += 1
        mrr_sum += 1.0 / rank

    avg_bce = total_bce / max(1, total_pairs)
    bin_acc = (correct_binary / max(1, total_pairs)) * 100
    top1_acc = (correct_top1 / max(1, total_queries)) * 100
    avg_mrr = mrr_sum / max(1, total_queries)

    return avg_bce, bin_acc, top1_acc, avg_mrr


def main():
    parser = argparse.ArgumentParser(description="Pointwise BCE Fine-Tuning of Jina Reranker v2")
    parser.add_argument("--max-queries", type=int, default=8000, help="Number of training queries (8k queries * 4 pairs = ~32k pairs)")
    parser.add_argument("--negs-per-query", type=int, default=3, help="Adversarial hard negatives per query")
    parser.add_argument("--batch-size", type=int, default=8, help="Pairs per micro-batch")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (effective batch: 32 pairs)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Peak learning rate")
    parser.add_argument("--max-len", type=int, default=256, help="Max sequence length")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    args = parser.parse_args()

    effective_batch = args.batch_size * args.grad_accum

    print("=" * 85)
    print("  STAGE 7: POINTWISE BCE FINE-TUNING: JINA RERANKER V2 (278M)")
    print(f"  Model Backbone:      {BASE_MODEL_NAME} (278M parameters)")
    print(f"  Loss Function:       Pointwise BCEWithLogitsLoss() [Binary 1.0 vs 0.0]")
    print(f"  Micro-Batch:         {args.batch_size} pairs | Effective Batch: {effective_batch} pairs")
    print(f"  Precision:           Native BFloat16 (sm_89 Ada Lovelace Native)")
    print(f"  Target Checkpoint:   {BEST_MODEL_DIR}")
    print("=" * 85)
    sys.stdout.flush()

    OUTPUT_BCE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load book metadata
    meta = load_book_metadata(DB_FILE)
    book_texts = meta["book_texts"]

    # 2. Build Datasets
    print(f"\n[Data] Assembling Pointwise BCE pairs ({args.negs_per_query} hard negatives per query)...")
    train_dataset = PointwisePairDataset(TRAIN_DATA_FILE, book_texts, max_queries=args.max_queries, negs_per_query=args.negs_per_query)
    val_grouped = GroupedValDataset(VAL_DATA_FILE, book_texts, max_queries=300, negs_per_query=args.negs_per_query)
    print(f"       Loaded {len(train_dataset):,} pointwise training pairs ({len(train_dataset) // (1 + args.negs_per_query):,} queries).")
    print(f"       Loaded {len(val_group := val_grouped.queries):,} grouped validation queries ({len(val_group) * (1 + args.negs_per_query):,} pairs).")
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
        collate_fn=make_collate_fn(tokenizer, max_length=args.max_len),
        drop_last=True
    )

    # 4. Optimizer & Schedulers
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * 0.1)
    eval_interval = max(50, total_steps // 5)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fct = nn.BCEWithLogitsLoss()

    def get_lr(step):
        if step < warmup_steps:
            return args.lr * float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    # Baseline Zero-Shot Validation
    print("\n[Validation] Running baseline validation before Pointwise BCE training...")
    sys.stdout.flush()
    init_loss, init_bin_acc, init_top1_acc, init_mrr = evaluate_pointwise_validation(model, tokenizer, val_grouped, DEVICE, args.max_len)
    print(f"  Baseline BCE Loss: {init_loss:.4f} | Binary Acc: {init_bin_acc:.2f}% | Top-1 Acc: {init_top1_acc:.2f}% | MRR@10: {init_mrr:.4f}")
    sys.stdout.flush()

    best_mrr = init_mrr
    best_loss = init_loss

    # 5. Training Loop
    print(f"\n[Training] Starting Pointwise BCE Training ({total_steps} optimizer steps across {len(train_dataset):,} pairs)...")
    sys.stdout.flush()
    model.train()
    optimizer.zero_grad()
    t0_start = time.time()
    step_count = 0
    micro_step = 0
    running_loss = 0.0

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"Pointwise BCE Training Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    for epoch in range(args.epochs):
        for batch, labels in train_loader:
            micro_step += 1
            for k in batch:
                batch[k] = batch[k].to(DEVICE)
            labels = labels.to(DEVICE)

            with torch.autocast(device_type="cuda" if DEVICE == "cuda" else "cpu", dtype=torch.bfloat16):
                outputs = model(**batch)
                logits = outputs.logits.squeeze(-1)
                loss = loss_fct(logits, labels) / args.grad_accum

            loss.backward()
            running_loss += loss.item() * args.grad_accum

            if micro_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                curr_lr = get_lr(step_count)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = curr_lr

                optimizer.step()
                optimizer.zero_grad()
                step_count += 1

                if step_count % 25 == 0 or step_count == total_steps:
                    avg_step_loss = running_loss / 25.0
                    running_loss = 0.0
                    elapsed = time.time() - t0_start
                    pct = (step_count / total_steps) * 100
                    log_line = f"Step {step_count:4d}/{total_steps} ({pct:5.1f}%) | BCE Loss: {avg_step_loss:.4f} | LR: {curr_lr:.2e} | Elapsed: {elapsed/60:.1f}m"
                    print(f"  {log_line}")
                    sys.stdout.flush()
                    with open(LOG_FILE, "a", encoding="utf-8") as f:
                        f.write(log_line + "\n")

                # Validation & Checkpointing
                if step_count % eval_interval == 0 or step_count == total_steps:
                    v_loss, v_bin_acc, v_top1, v_mrr = evaluate_pointwise_validation(model, tokenizer, val_grouped, DEVICE, args.max_len)
                    eval_line = f"  >>> [EVALUATION Step {step_count}] BCE Loss: {v_loss:.4f} | Binary Acc: {v_bin_acc:.2f}% | Top-1 Acc: {v_top1:.2f}% | MRR@10: {v_mrr:.4f}"
                    print(eval_line)
                    sys.stdout.flush()
                    with open(LOG_FILE, "a", encoding="utf-8") as f:
                        f.write(eval_line + "\n")

                    if v_mrr >= best_mrr or v_loss < best_loss:
                        best_mrr = max(best_mrr, v_mrr)
                        best_loss = min(best_loss, v_loss)
                        print(f"      [*] Saving new best Pointwise BCE model to {BEST_MODEL_DIR}...")
                        ce_wrapper.save_pretrained(str(BEST_MODEL_DIR))

                    # Update markdown status
                    with open(STATUS_FILE, "w", encoding="utf-8") as sf:
                        sf.write(f"# 📊 Jina Pointwise BCE Fine-Tuning Monitor\n\n")
                        sf.write(f"* **Architecture**: `jina-reranker-v2-base-multilingual` (278M)\n")
                        sf.write(f"* **Objective**: Pointwise `BCEWithLogitsLoss()` on 1.0 vs 0.0 pairs\n")
                        sf.write(f"* **Current Progress**: Step {step_count}/{total_steps} ({pct:.1f}% complete)\n")
                        sf.write(f"* **Best Validation BCE Loss**: **{best_loss:.4f}**\n")
                        sf.write(f"* **Best Validation MRR@10**: **{best_mrr:.4f}**\n")
                        sf.write(f"* **Best Top-1 Accuracy**: **{v_top1:.2f}%**\n")
                        sf.write(f"* **Best Binary Classification Accuracy**: **{v_bin_acc:.2f}%**\n")
                        sf.write(f"* **Checkpoint Path**: [`arctic_output/jina_bce_reranker/best_model`](file:///{str(BEST_MODEL_DIR).replace(os.sep, '/')})\n")

                    model.train()

    total_time = (time.time() - t0_start) / 60
    print("\n" + "=" * 85)
    print(f"  [SUCCESS] POINTWISE BCE FINE-TUNING COMPLETED IN {total_time:.2f} MINUTES!")
    print(f"  Best Validation MRR@10: {best_mrr:.4f} | Best BCE Loss: {best_loss:.4f}")
    print(f"  Model Saved To: {BEST_MODEL_DIR}")
    print("=" * 85)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
