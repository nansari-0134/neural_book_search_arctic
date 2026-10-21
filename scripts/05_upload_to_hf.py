"""
Hugging Face Hub Model Upload Script
Publishes fine-tuned Snowflake Arctic retriever to Hugging Face with an automated model card.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR, OUTPUT_DIR

def generate_model_card(repo_id: str, results: dict = None) -> str:
    """Generates an aesthetic Hugging Face Model Card (README.md)."""
    metrics_block = ""
    if results and "test" in results:
        t = results["test"]
        metrics_block = f"""
### 📊 Benchmark Results (64,552 Goodreads Book Catalog)

| Metric | Score |
| :--- | :--- |
| **Recall@1** | **{t.get('R@1', 0)*100:.2f}%** |
| **Recall@5** | **{t.get('R@5', 0)*100:.2f}%** |
| **Recall@10** | **{t.get('R@10', 0)*100:.2f}%** |
| **Recall@50** | **{t.get('R@50', 0)*100:.2f}%** |
"""

    return f"""---
language:
- en
license: mit
library_name: sentence-transformers
pipeline_tag: feature-extraction
tags:
- sentence-transformers
- sentence-similarity
- retrieval
- semantic-search
- books
- recommendation
- snowflake-arctic
- matryoshka
---

# ❄️ {repo_id.split('/')[-1]}

Fine-tuned **`Snowflake/snowflake-arctic-embed-m-v1.5`** on a catalog of **64,552 Goodreads books** for high-precision book semantic search, trope-based retrieval, and neural recommendation.

Features full contrastive fine-tuning with **Multiple Negatives Ranking Loss (MNRL)**, duplicate-positive loss masking, and **iterative hard negative mining** with an 8-layer false-negative protection shield.

{metrics_block}

## 🚀 Quick Usage

```python
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("{repo_id}")

# 1. Catalog Books formatting
books = [
    "Title: The Hunger Games | Genres: young-adult, dystopian, survival | Description: In the ruins of North America...",
    "Title: Dune | Genres: science-fiction, space-opera | Description: Set on the desert planet Arrakis..."
]
book_embeddings = model.encode(books, normalize_embeddings=True)

# 2. Asymmetric Query Encoding (uses built-in prompt)
queries = ["story about young people fighting in a government arena survival game"]
query_embeddings = model.encode(
    queries,
    prompt_name="query",
    normalize_embeddings=True
)

# 3. Fast Dot-Product Similarity
similarities = query_embeddings @ book_embeddings.T
print("Top Match Index:", similarities.argmax())
```

## 📐 Matryoshka Representation Learning (MRL) Support
This model supports flexible vector dimensionalities (768, 512, 256, 128) via Matryoshka truncation:
```python
# Truncate embeddings to 256 dimensions for 3x faster vector search
embs_256 = model.encode(books, normalize_embeddings=True)[:, :256]
# Re-normalize after truncation
import numpy as np
embs_256 = embs_256 / np.linalg.norm(embs_256, axis=1, keepdims=True)
```

## 👤 Author
Developed and fine-tuned by **Nizam** (`nansari.0134@gmail.com`).
"""

def main():
    parser = argparse.ArgumentParser(description="Upload fine-tuned model to Hugging Face Hub")
    parser.add_argument("--repo-id", required=True, help="HF Repo ID (e.g. your_username/snowflake-arctic-book-retriever)")
    parser.add_argument("--model-dir", default=None, help="Path to best_model directory (defaults to newest in arctic_output/)")
    parser.add_argument("--token", default=None, help="Hugging Face Write Token (or set HF_TOKEN env var)")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("Error: Hugging Face token required! Provide --token or set HF_TOKEN environment variable.")
        sys.exit(1)

    # Determine model directory
    if args.model_dir:
        model_path = Path(args.model_dir)
    else:
        # Find latest best_model directory
        candidates = list(OUTPUT_DIR.glob("**/best_model"))
        if not candidates:
            print(f"Error: No trained model found in {OUTPUT_DIR}. Train a model first!")
            sys.exit(1)
        model_path = candidates[-1]

    print("=" * 70)
    print("  HUGGING FACE HUB MODEL UPLOAD")
    print(f"  Target Repo: {args.repo_id}")
    print(f"  Source Dir:  {model_path}")
    print("=" * 70)

    # Load metrics if available
    latest_metrics = None
    metric_files = sorted(list(RESULTS_DIR.glob("*_metrics.json")))
    if metric_files:
        with open(metric_files[-1], "r", encoding="utf-8") as f:
            latest_metrics = json.load(f)

    # Write Model Card
    readme_path = model_path / "README.md"
    readme_content = generate_model_card(args.repo_id, latest_metrics)
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"Generated Model Card at {readme_path.name}")

    api = HfApi()
    print(f"Creating repository on Hugging Face (if not exists): {args.repo_id}...")
    create_repo(
        repo_id=args.repo_id,
        token=token,
        private=args.private,
        exist_ok=True,
        repo_type="model"
    )

    print(f"Uploading files from {model_path}...")
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=args.repo_id,
        repo_type="model",
        token=token,
        commit_message="Publish fine-tuned Snowflake Arctic book retriever",
        ignore_patterns=["*.pt", "*.tmp", "*.prev", "__pycache__/*"]
    )

    print("\n" + "=" * 70)
    print(f"  MODEL SUCCESSFULLY PUBLISHED TO HUGGING FACE!")
    print(f"  URL: https://huggingface.co/{args.repo_id}")
    print("=" * 70)

if __name__ == "__main__":
    main()
