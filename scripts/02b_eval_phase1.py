"""
Evaluation of Phase 1 Fine-Tuned Model (arctic_phase1/best_model)
Evaluates on both validation and test sets across all 64,552 catalog books.
"""
import sys
import json
import time
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    VAL_FILE, TEST_FILE, DB_FILE, OUTPUT_DIR, RESULTS_DIR,
    MAX_SEQ_LENGTH, DEVICE, USE_FP16
)
from src.data_loader import load_book_metadata
from src.metrics import evaluate_recall

def main():
    model_dir = OUTPUT_DIR / "arctic_phase1" / "best_model"
    print("=" * 70)
    print("  EVALUATING PHASE 1 FINE-TUNED MODEL")
    print(f"  Model Path: {model_dir}")
    print(f"  Device:     {DEVICE} (FP16: {USE_FP16})")
    print("=" * 70)

    # 1. Load book metadata
    meta = load_book_metadata(DB_FILE)
    all_book_ids = sorted(list(meta["book_texts"].keys()))
    print(f"Total catalog books: {len(all_book_ids):,}")

    # 2. Load fine-tuned model
    print(f"\nLoading saved fine-tuned model from {model_dir}...")
    model = SentenceTransformer(str(model_dir), device=DEVICE)
    model.max_seq_length = MAX_SEQ_LENGTH
    if DEVICE == "cuda" and USE_FP16:
        model.half()
    model.eval()

    # 3. Pre-encode catalog once
    print(f"\n[Catalog] Pre-encoding {len(all_book_ids):,} catalog books on {DEVICE}...")
    catalog_texts = [meta["book_texts"].get(bid, "Untitled") for bid in all_book_ids]
    catalog_emb = model.encode(
        catalog_texts,
        batch_size=128 if DEVICE == "cuda" else 32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype("float32")

    # 4. Evaluate Validation and Test
    val_metrics = evaluate_recall(
        model=model,
        split_path=VAL_FILE,
        split_name="Phase 1 Validation",
        meta=meta,
        all_book_ids=all_book_ids,
        cached_catalog_emb=catalog_emb
    )

    test_metrics = evaluate_recall(
        model=model,
        split_path=TEST_FILE,
        split_name="Phase 1 Test",
        meta=meta,
        all_book_ids=all_book_ids,
        cached_catalog_emb=catalog_emb
    )

    # 5. Save results to disk
    results_payload = {
        "model": "arctic_phase1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "validation": val_metrics,
        "test": test_metrics
    }

    out_file = RESULTS_DIR / "phase1_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\n[Done] Phase 1 metrics saved successfully to {out_file}")

if __name__ == "__main__":
    main()
