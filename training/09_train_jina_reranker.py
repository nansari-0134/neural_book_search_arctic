"""
Step 9: Fine-Tuning Jina Reranker v2 (278M Multilingual Cross-Encoder)
=====================================================================
Fine-tunes 'jinaai/jina-reranker-v2-base-multilingual' on query-document pairs
using hard negatives mined from the champion Arctic dense retriever.

Key Features:
- 278M parameter multilingual transformer architecture
- Full token-to-token cross-attention for trope/genre disambiguation
- BFloat16 (BF16) mixed precision on NVIDIA RTX 4050 (sm_89 Ada Lovelace)
- Gradient accumulation (effective batch size: 32)
- Evaluates on work-clustered validation set and saves champion checkpoint
"""

import sys
import os
import json
import time
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

# Apply Jina transformers 5.x patch
import src.jina_patch

import torch
from datasets import Dataset
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments

from src.config import DB_FILE, OUTPUT_DIR, DEVICE
from src.data_loader import load_book_metadata

BASE_MODEL_NAME = "jinaai/jina-reranker-v2-base-multilingual"
OUTPUT_JINA_DIR = OUTPUT_DIR / "jina_reranker"
BEST_MODEL_DIR = OUTPUT_JINA_DIR / "best_model"

TRAIN_DATA_FILE = PROJECT_ROOT / "data" / "train_arctic_v3.jsonl"
VAL_DATA_FILE = PROJECT_ROOT / "data" / "val.jsonl"

def prepare_dataset(
    data_path: Path,
    book_texts: Dict[str, str],
    max_queries: int,
    negs_per_query: int = 2
) -> Dataset:
    print(f"\n[Data] Reading pairs from {data_path.name}...")
    t0 = time.time()
    queries = []
    passages = []
    labels = []

    with open(data_path, "r", encoding="utf-8") as f:
        count = 0
        for line in f:
            if count >= max_queries:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            query = row["query"]
            pos_id = str(row["positive_id"])
            pos_text = book_texts.get(pos_id)
            if not pos_text:
                continue

            # Positive pair (label = 1.0)
            queries.append(query)
            passages.append(pos_text)
            labels.append(1.0)

            # Hard negative pairs (label = 0.0)
            neg_ids = row.get("negative_ids", [])[:negs_per_query]
            for nid in neg_ids:
                neg_text = book_texts.get(str(nid))
                if neg_text:
                    queries.append(query)
                    passages.append(neg_text)
                    labels.append(0.0)

            count += 1

    elapsed = time.time() - t0
    print(f"[Data] Loaded {len(queries):,} total pairs across {count:,} queries in {elapsed:.2f}s.")

    # Shuffle training pairs
    combined = list(zip(queries, passages, labels))
    random.seed(42)
    random.shuffle(combined)
    q_shuff, p_shuff, l_shuff = zip(*combined)

    return Dataset.from_dict({
        "sentence1": list(q_shuff),
        "sentence2": list(p_shuff),
        "label": list(l_shuff)
    })

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Jina Reranker v2 on Arctic hard negatives")
    parser.add_argument("--max-queries", type=int, default=15000, help="Max queries to train on")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Per device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1.5e-5, help="Learning rate")
    parser.add_argument("--max-len", type=int, default=256, help="Max sequence length")
    args = parser.parse_args()

    print("=" * 80)
    print("  STAGE 7: FINE-TUNING JINA RERANKER V2 (278M MULTILINGUAL)")
    print(f"  Base Backbone: {BASE_MODEL_NAME}")
    print(f"  Target Checkpoint: {BEST_MODEL_DIR}")
    print(f"  Device: {DEVICE} (BF16 enabled on CUDA)")
    print(f"  Batch Size: {args.batch_size} * Grad Accum: {args.grad_accum} = {args.batch_size * args.grad_accum} eff. batch")
    print(f"  Max Queries: {args.max_queries:,} | Epochs: {args.epochs}")
    print("=" * 80)

    OUTPUT_JINA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load book metadata for passage text
    print("\n[1/4] Loading book catalog metadata...")
    meta = load_book_metadata(DB_FILE)
    book_texts = meta["book_texts"]
    print(f"      Loaded {len(book_texts):,} book passages.")

    # 2. Build Datasets
    print("\n[2/4] Preparing training and validation datasets...")
    train_dataset = prepare_dataset(
        TRAIN_DATA_FILE, book_texts, max_queries=args.max_queries, negs_per_query=2
    )
    val_dataset = prepare_dataset(
        VAL_DATA_FILE, book_texts, max_queries=1000, negs_per_query=2
    )

    # 3. Load Jina Reranker v2 model
    print(f"\n[3/4] Initializing CrossEncoder: {BASE_MODEL_NAME}...")
    model = CrossEncoder(
        BASE_MODEL_NAME,
        num_labels=1,
        max_length=args.max_len,
        trust_remote_code=True,
        device=DEVICE
    )

    # 4. Training Arguments
    total_samples = len(train_dataset)
    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = total_samples // effective_batch
    eval_freq = max(100, steps_per_epoch // 4)

    training_args = CrossEncoderTrainingArguments(
        output_dir=str(OUTPUT_JINA_DIR / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=16,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=eval_freq,
        save_strategy="steps",
        save_steps=eval_freq,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        bf16=True if DEVICE == "cuda" else False,
        fp16=False,
        report_to="none"
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # 5. Train model
    print(f"\n[4/4] Starting training loop ({steps_per_epoch} total steps)...")
    t0_train = time.time()
    trainer.train()
    duration_min = (time.time() - t0_train) / 60

    # 6. Save fine-tuned champion model
    print(f"\n[Save] Exporting best fine-tuned Jina Reranker to {BEST_MODEL_DIR}...")
    model.save_pretrained(str(BEST_MODEL_DIR))

    # Save summary metadata
    meta_info = {
        "model_name": BASE_MODEL_NAME,
        "parameters": "278M",
        "precision": "bfloat16",
        "dataset_queries": args.max_queries,
        "dataset_pairs": total_samples,
        "training_duration_min": round(duration_min, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(OUTPUT_JINA_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)

    print("\n" + "=" * 80)
    print(f"  [SUCCESS] JINA RERANKER V2 FINE-TUNING COMPLETED IN {duration_min:.2f} MINUTES!")
    print(f"  Model saved to: {BEST_MODEL_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
