"""
Hugging Face Hub Model Upload Utility
=====================================
Automates packaging, generating model card metadata, and uploading fine-tuned
ArcticDense models to the Hugging Face Hub:
1. Retriever Bi-Encoder: Snowflake Arctic Embed M v1.5 (Progressive Refinement Champion)
2. Reranker Cross-Encoder: Jina Reranker v2 (InfoNCE Contrastive Champion)

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login  (or set HF_TOKEN environment variable)

Usage:
    # Upload Retriever:
    python tools/upload_to_hf.py --model retriever --repo-id "YOUR_USERNAME/arctic-embed-m-goodreads"

    # Upload Reranker:
    python tools/upload_to_hf.py --model reranker --repo-id "YOUR_USERNAME/jina-reranker-v2-goodreads"

    # Upload Both:
    python tools/upload_to_hf.py --model both --prefix "YOUR_USERNAME"
"""

import sys
import os
import argparse
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR = PROJECT_ROOT / "arctic_output"

try:
    from huggingface_hub import HfApi, create_repo, whoami
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


RETRIEVER_CHECKPOINT = OUTPUT_DIR / "arctic_v3" / "best_model"
RERANKER_CHECKPOINT = OUTPUT_DIR / "jina_contrastive_reranker" / "best_model"


def generate_retriever_model_card(repo_id: str) -> str:
    return f"""---
language:
- en
license: apache-2.0
library_name: sentence-transformers
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- mteb
- retrieval
- matryoshka
- snowflake-arctic
pipeline_tag: feature-extraction
---

# ❄️ ArcticDense: Snowflake Arctic Embed M v1.5 (Goodreads Champion)

This model is a fine-tuned version of [`Snowflake/snowflake-arctic-embed-m-v1.5`](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5) (109M parameters, 768 dimensions) specialized for **book thematic retrieval and semantic search across 64,552 Goodreads books**.

It was developed through **Progressive Refinement Contrastive Fine-Tuning** using an **8-layer false-negative protection shield** to eliminate edition leakage, series contamination, and label noise.

## 📊 Benchmark Progression (64,552-Book Catalog, 32,170 Test Queries)

| Training Stage | Recall@1 | Recall@5 | Recall@10 | Recall@50 | MRR@10 | NDCG@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pretrained Baseline (Zero-Shot)** | 21.67% | 48.41% | 55.40% | 69.57% | 0.3082 | 0.3642 |
| **In-Batch Contrastive Training** | 29.16% | 58.29% | 64.67% | 78.20% | 0.3951 | 0.4538 |
| **Targeted Hard-Negative Training** | 35.60% | 68.09% | 73.96% | 84.66% | 0.4688 | 0.5322 |
| **Progressive Refinement (Champion)** | **35.48%** | **68.50%** | **74.70%** | **85.51%** | **0.4725** | **0.5369** |
| **Net Gain vs Baseline** | **+13.81%** | **+20.09%** | **+19.30%** | **+15.94%** | **+0.1643** | **+0.1727** |

## 🪆 Matryoshka Representation Learning (MRL) Truncation

The model natively supports dynamic dimension truncation without loss of semantic coherence:
* **768-d**: Baseline full precision (Recall@10: 71.9%, MRR@10: 0.5726)
* **256-d (Sweet Spot)**: 3x storage compression & speedup while **retaining 99.3% Recall@10** (Recall@10: 71.4%, MRR@10: 0.5597)
* **128-d**: 6x storage compression (Recall@10: 63.2%, MRR@10: 0.4791)
* **64-d**: 12x storage compression for ultra-low memory edge deployments

## 🚀 Quickstart & Usage

### Asymmetric Query Instruction
To search, always prefix queries with Snowflake Arctic's retrieval instruction:
`"Represent this sentence for searching relevant passages: "`

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model
model = SentenceTransformer("{repo_id}")

# 1. Asymmetric query
query = "Represent this sentence for searching relevant passages: atmospheric gothic boarding school mystery"
q_emb = model.encode(query, normalize_embeddings=True)

# 2. Document description
passage = "Title: The Secret History | Genres: dark-academia, mystery | Description: Under the influence of their charismatic classics professor..."
p_emb = model.encode(passage, normalize_embeddings=True)

# 3. Fast Cosine Similarity
similarity = np.dot(q_emb, p_emb)
print(f"Similarity Score: {{similarity:.4f}}")
```

### Matryoshka Vector Truncation (256-d)
```python
# Truncate to 256 coordinates and re-normalize onto unit sphere
q_256 = q_emb[:256] / np.linalg.norm(q_emb[:256])
p_256 = p_emb[:256] / np.linalg.norm(p_emb[:256])
sim_256 = np.dot(q_256, p_256)
print(f"256-d Similarity: {{sim_256:.4f}}")
```
"""


def generate_reranker_model_card(repo_id: str) -> str:
    return f"""---
language:
- en
license: apache-2.0
library_name: sentence-transformers
tags:
- cross-encoder
- reranker
- jina-reranker
- sentence-transformers
- retrieval
pipeline_tag: text-classification
---

# ⚡ ArcticDense: Jina Reranker v2 (InfoNCE Contrastive Champion)

This model is a fine-tuned Cross-Encoder reranker based on [`jinaai/jina-reranker-v2-base-multilingual`](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual) (278M parameters) trained with **Listwise InfoNCE Contrastive Cross-Entropy** over actively mined hard negatives.

## 📊 Two-Stage Cascade Retrieval Performance

Evaluated on 1,000 queries searching across the full 64,552 Goodreads book catalog:

| Pipeline Stage | Recall@1 | Recall@5 | Recall@10 | MRR@10 | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Stage 1: Arctic Bi-Encoder Standalone** | 49.00% | 66.40% | 71.80% | 0.5681 | 11.5 ms |
| **Stage 2: Contrastive Hybrid Cascade (0.30 Dense + 0.70 Reranker)** | **52.50%** | **70.50%** | **75.00%** | **0.6001** | 182.8 ms |
| **Improvement** | **+3.50%** | **+4.10%** | **+3.20%** | **+0.0320** | — |

## 🚀 Usage with Sentence-Transformers

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("{repo_id}", max_length=256, trust_remote_code=True)

query = "cyberpunk detective investigation in dystopian megacity"
documents = [
    "Title: Neuromancer | Genres: sci-fi, cyberpunk | Description: Case was the sharpest data-thief in the matrix...",
    "Title: Pride and Prejudice | Genres: romance, classics | Description: A story of manners, marriage, and society..."
]

pairs = [[query, doc] for doc in documents]
scores = model.predict(pairs)
print("Reranked Scores:", scores)
```
"""


def upload_single_model(model_path: Path, repo_id: str, card_generator):
    print(f"\n[*] Preparing upload for: {model_path.name}")
    print(f"    Target Hugging Face Repo: {repo_id}")

    if not model_path.exists():
        print(f"[Error] Model directory not found: {model_path}")
        return False

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    # Verify authentication
    try:
        user_info = whoami(token=token)
        print(f"[*] Authenticated as: {user_info.get('name')} ({user_info.get('fullname')})")
    except Exception as e:
        print(f"[Error] Not authenticated with Hugging Face: {e}")
        print("        Please run: huggingface-cli login or set the HF_TOKEN environment variable.")
        return False

    # Create repo if it doesn't exist
    print(f"[*] Ensuring repository '{repo_id}' exists...")
    try:
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)
        print("    [SUCCESS] Repository verified.")
    except Exception as e:
        print(f"    [Error] Failed to create repo: {e}")
        return False

    # Write model card
    readme_path = model_path / "README.md"
    print(f"[*] Writing metadata-rich README.md model card...")
    readme_content = card_generator(repo_id)
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    # Upload folder
    print(f"[*] Uploading model artifacts from {model_path} to {repo_id}...")
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload {model_path.name} fine-tuned weights and model card"
    )

    print("=" * 65)
    print(f"[SUCCESS] Model uploaded successfully!")
    print(f"HF URL: https://huggingface.co/{repo_id}")
    print("=" * 65)
    return True


def main():
    parser = argparse.ArgumentParser(description="Upload ArcticDense models to Hugging Face Hub")
    parser.add_argument("--model", choices=["retriever", "reranker", "both"], required=True, help="Which model to upload")
    parser.add_argument("--repo-id", help="Full repository ID (e.g. username/model-name)")
    parser.add_argument("--prefix", help="Hugging Face username/org for --model both (e.g. 'nansari-0134')")
    args = parser.parse_args()

    if not HF_HUB_AVAILABLE:
        print("[Error] huggingface_hub is not installed. Please run: pip install huggingface_hub")
        sys.exit(1)

    if args.model == "retriever":
        if not args.repo_id:
            print("[Error] Please specify --repo-id (e.g., --repo-id YOUR_USER/arctic-embed-m-goodreads)")
            sys.exit(1)
        upload_single_model(RETRIEVER_CHECKPOINT, args.repo_id, generate_retriever_model_card)

    elif args.model == "reranker":
        if not args.repo_id:
            print("[Error] Please specify --repo-id (e.g., --repo-id YOUR_USER/jina-reranker-v2-goodreads)")
            sys.exit(1)
        upload_single_model(RERANKER_CHECKPOINT, args.repo_id, generate_reranker_model_card)

    elif args.model == "both":
        prefix = args.prefix
        if not prefix:
            user = whoami(token=os.environ.get("HF_TOKEN")).get("name")
            if not user:
                print("[Error] Please provide --prefix (e.g. --prefix YOUR_USER)")
                sys.exit(1)
            prefix = user

        repo_retriever = f"{prefix}/arctic-embed-m-goodreads"
        repo_reranker = f"{prefix}/jina-reranker-v2-goodreads"

        print(f"[*] Uploading both models under prefix: {prefix}")
        upload_single_model(RETRIEVER_CHECKPOINT, repo_retriever, generate_retriever_model_card)
        upload_single_model(RERANKER_CHECKPOINT, repo_reranker, generate_reranker_model_card)


if __name__ == "__main__":
    main()
