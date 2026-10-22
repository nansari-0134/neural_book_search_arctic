# 🤗 Hugging Face Deployment

This guide outlines publishing fine-tuned models to the Hugging Face Hub with automated metadata and model cards.

---

## 1. Setup Your Token
Generate a write token from [Hugging Face Settings > Access Tokens](https://huggingface.co/settings/tokens) and set it in your environment:
```powershell
$env:HF_TOKEN = "your_hugging_face_write_token"
```

---

## 2. Publishing the Model
Run the automated upload script from the project root:
```bash
python scripts/05_upload_to_hf.py --repo-id your_username/arcticdense-book-retriever
```

### Automated Actions:
* Checks the newest trained checkpoint in `arctic_output/`.
* Automatically loads final evaluation metrics from `results/` and formats an aesthetic Model Card.
* Tags repository with appropriate pipeline tags (`sentence-similarity`, `feature-extraction`).
* Uploads model safetensors, tokenizer, and config while safely ignoring local checkpoint caches (`.pt`, `.tmp`, `.prev`).

---

## 3. Inferencing from the Published Model

Once published, anyone can load and use your fine-tuned model via standard `sentence-transformers`:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("your_username/arcticdense-book-retriever")

# Encode catalog book
book_emb = model.encode("Title: Dune | Genres: sci-fi | Description: Desert planet...")

# Encode query using asymmetric prompt
query_emb = model.encode(
    "epic sci-fi about sand worms and spice trade",
    prompt_name="query"
)

similarity = query_emb @ book_emb.T
print(f"Cosine Similarity: {similarity:.4f}")
```
