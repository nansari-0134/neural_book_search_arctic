"""
Step 1: Baseline Evaluation of Untouched Snowflake Arctic Embed (v1.5)
Establishes the starting benchmark before any fine-tuning.
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
    MODEL_NAME, VAL_FILE, TEST_FILE, DB_FILE, RESULTS_DIR,
    MAX_SEQ_LENGTH, DEVICE, USE_FP16
)
from src.data_loader import load_book_metadata
from src.metrics import evaluate_recall

def main():
    print("=" * 70)
    print("  STEP 1: ZERO-SHOT BASELINE EVALUATION")
    print(f"  Model:  {MODEL_NAME}")
    print(f"  Device: {DEVICE} (FP16: {USE_FP16})")
    print("=" * 70)

    # 1. Load book metadata
    meta = load_book_metadata(DB_FILE)
    all_book_ids = sorted(list(meta["book_texts"].keys()))
    print(f"Total catalog books: {len(all_book_ids):,}")

    # 2. Load model
    print(f"\nLoading pretrained model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    model.max_seq_length = MAX_SEQ_LENGTH
    if DEVICE == "cuda" and USE_FP16:
        model.half()
    model.eval()

    # 3. Pre-encode catalog once and reuse for both validation and test
    print(f"\n[Catalog] Pre-encoding {len(all_book_ids):,} catalog books on {DEVICE}...")
    catalog_texts = [meta["book_texts"].get(bid, "Untitled") for bid in all_book_ids]
    catalog_emb = model.encode(
        catalog_texts,
        batch_size=128 if DEVICE == "cuda" else 32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype("float32")

    # 4. Compute baseline on Validation and Test
    val_metrics = evaluate_recall(
        model=model,
        split_path=VAL_FILE,
        split_name="Baseline Validation",
        meta=meta,
        all_book_ids=all_book_ids,
        cached_catalog_emb=catalog_emb
    )

    test_metrics = evaluate_recall(
        model=model,
        split_path=TEST_FILE,
        split_name="Baseline Test",
        meta=meta,
        all_book_ids=all_book_ids,
        cached_catalog_emb=catalog_emb
    )

    # 4. Save results to disk
    baseline_payload = {
        "model": MODEL_NAME,
        "max_seq_length": MAX_SEQ_LENGTH,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "validation": val_metrics,
        "test": test_metrics
    }

    out_file = RESULTS_DIR / "baseline_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(baseline_payload, f, indent=2)

    print("\n" + "=" * 70)
    print(f"  BASELINE EVALUATION COMPLETE -> {out_file.name}")
    print("=" * 70)
    print("Summary:")
    print(f"  Validation R@10: {val_metrics['R@10']*100:.2f}% | R@1: {val_metrics['R@1']*100:.2f}% | R@50: {val_metrics['R@50']*100:.2f}%")
    print(f"  Test       R@10: {test_metrics['R@10']*100:.2f}% | R@1: {test_metrics['R@1']*100:.2f}% | R@50: {test_metrics['R@50']*100:.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
