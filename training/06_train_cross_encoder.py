"""
Training Script: Two-Stage Precision Reranker (Cross-Encoder)
============================================================
Fine-tunes a cross-encoder model (cross-encoder/ms-marco-MiniLM-L-6-v2)
on (query, document) pairs using the hard negatives mined from the champion
Arctic dense retriever.

Key Features:
- Joint cross-attention across [CLS] Query [SEP] Book Description [SEP]
- Binary relevance scoring head with BCE loss
- Modern CrossEncoderTrainer from Sentence Transformers v3+
- Mixed Precision (FP16) on NVIDIA RTX 4050 GPU
"""

import sys
import os
import json
import time
import random
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
from datasets import Dataset
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DB_FILE, OUTPUT_DIR, DEVICE, USE_FP16
from src.data_loader import load_book_metadata

# Hyperparameters
BASE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OUTPUT_CROSS_ENCODER_DIR = OUTPUT_DIR / "cross_encoder"
BEST_MODEL_DIR = OUTPUT_CROSS_ENCODER_DIR / "best_model"

TRAIN_DATA_FILE = PROJECT_ROOT / "data" / "train_arctic_v3.jsonl"
VAL_DATA_FILE = PROJECT_ROOT / "data" / "val.jsonl"

MAX_SEQ_LENGTH = 256
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
NUM_EPOCHS = 1
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
MAX_TRAIN_QUERIES = 40000
NEGATIVES_PER_QUERY = 3
MAX_VAL_QUERIES = 1000

def prepare_dataset(
    data_path: Path,
    book_texts: Dict[str, str],
    max_queries: int,
    negs_per_query: int = 3
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
            row = json.loads(line)
            query = row["query"]
            pos_id = str(row["positive_id"])
            pos_text = book_texts.get(pos_id)
            if not pos_text:
                continue

            # Positive sample (label = 1.0)
            queries.append(query)
            passages.append(pos_text)
            labels.append(1.0)

            # Hard negative samples (label = 0.0)
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
    print("=" * 70)
    print("[TRAINING] STAGE 5: TWO-STAGE CROSS-ENCODER RERANKER")
    print(f"Base Backbone: {BASE_MODEL_NAME}")
    print(f"Target Output: {BEST_MODEL_DIR}")
    print(f"Device: {DEVICE} | FP16: {USE_FP16}")
    print("=" * 70)

    OUTPUT_CROSS_ENCODER_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load book metadata for passage reconstruction
    meta = load_book_metadata(DB_FILE)
    book_texts = meta["book_texts"]

    # 2. Build HuggingFace Datasets
    train_dataset = prepare_dataset(
        TRAIN_DATA_FILE, book_texts, max_queries=MAX_TRAIN_QUERIES, negs_per_query=NEGATIVES_PER_QUERY
    )
    eval_dataset = prepare_dataset(
        VAL_DATA_FILE, book_texts, max_queries=MAX_VAL_QUERIES, negs_per_query=2
    )

    # 3. Initialize Cross-Encoder
    print(f"\n[Model] Initializing Cross-Encoder: {BASE_MODEL_NAME}...")
    model = CrossEncoder(
        BASE_MODEL_NAME,
        num_labels=1,
        max_length=MAX_SEQ_LENGTH,
        device=DEVICE
    )

    # 4. Training Arguments
    training_args = CrossEncoderTrainingArguments(
        output_dir=str(OUTPUT_CROSS_ENCODER_DIR / "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=200,
        fp16=USE_FP16 and (DEVICE == "cuda"),
        report_to="none"
    )

    # 5. Initialize Trainer
    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    # 6. Execute Training
    print("\n[Training] Starting Cross-Encoder training loop...")
    t0_train = time.time()
    trainer.train()
    duration_min = (time.time() - t0_train) / 60

    # 7. Save Champion Model Checkpoint
    print(f"\n[Model] Saving best Cross-Encoder to {BEST_MODEL_DIR}...")
    model.save_pretrained(str(BEST_MODEL_DIR))

    print("\n" + "=" * 70)
    print(f"[SUCCESS] CROSS-ENCODER TRAINING COMPLETED IN {duration_min:.2f} MINUTES!")
    print(f"Model saved to: {BEST_MODEL_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
