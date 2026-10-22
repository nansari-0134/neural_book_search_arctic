# ⛏️ Iterative Hard Negative Mining

This section covers why static negatives fail, the iterative mining cycle, and the 8-layer defense engine.

---

## 1. Why Static Negatives Fail

* **Random Negatives**: Too easy. Distinguishing a romance novel from a physics paper produces negligible gradients.
* **Static BM25 Negatives**: Only capture keyword overlap, not deep semantic confusion.
* **The Solution (Iterative Mining)**: Use the model's *own embedding space* to find the books it mistakenly ranks highly (ranks 2 to 35). As the model improves, its confusion patterns shift, requiring progressive mining cycles.

---

## 2. The 8-Layer False-Negative Protection Engine

If a mined candidate is actually a valid alternative answer or sequel, penalizing it poisons the training gradients. Every candidate book must pass our 8-layer filter:

```
[Candidate Book]
       │
       ▼
1. Banned ID Check ─────────────► Blocks target and cross-query positives
       │ Pass
2. Cosine Ceiling (> 0.88) ─────► Blocks near-duplicate semantics
       │ Pass
3. Normalized Title Match ──────► Blocks identical titles
       │ Pass
4. Base Title Subtitle Stripping ► Blocks edition subtitles (e.g. "Dune: Deluxe")
       │ Pass
5. Significant Word Overlap ────► Blocks >= 65% shared content words
       │ Pass
6. Series Regex Match ──────────► Blocks sibling books in same series (e.g. Harry Potter #2)
       │ Pass
7. Author Shelf Tag Overlap ────► Blocks books by the same author
       │ Pass
8. Negative Frequency Cap ──────► Caps any book from appearing >30 times
       │ Pass
 [ACCEPTED AS HARD NEGATIVE]
```

---

## 3. Cycle 1 Empirical Mining Results

Cycle 1 mining executed on the Phase 1 fine-tuned model against the full 64,552 book catalog:

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Catalog Scope** | **64,552 books** | Pre-encoded into 768-d vectors on GPU (FP16) |
| **Input Queries** | **258,250 queries** | Full training pool |
| **Downsampled Easy Pairs** | **58,442 (22.6%)** | Queries already ranked at #0 downsampled by 65% to concentrate compute |
| **Final Mined Dataset** | **199,808 pairs** | Saved to `data/train_arctic_v2.jsonl` (95.2 MB) |
| **Mining Speed** | **1.6 minutes** | High-throughput GPU matrix dot-product + top-k retrieval |

### 🛡️ Defense Shield Block Breakdown

The 8-layer shield successfully blocked **over 1.8 million toxic candidates** from poisoning the negative pool:

| Layer Filter | Blocked Count | Purpose |
| :--- | :---: | :--- |
| **`shared_author_shelf`** | **1,006,554** | Prevented penalizing books written by the exact same author |
| **`frequency_cap`** | **724,352** | Prevented popular books from over-saturating negatives ($>30$ cap) |
| **`banned_id`** | **45,084** | Blocked cross-query identical book references |
| **`title_word_overlap`** | **9,980** | Blocked candidates sharing $\ge 65\%$ non-stopwords with target |
| **`series_match`** | **5,372** | Blocked direct series siblings (e.g. Sequels, Prequels) |
| **`identical_title`** | **5,177** | Blocked exact title matches |
| **`base_title_match`** | **4,715** | Blocked alternate edition subtitles and translations |
| **`series_in_cand_title`** | **2,056** | Blocked series title mentions in candidate book name |
| **`cand_series_in_pos_title`** | **1,939** | Blocked target series title mentions in candidate name |
| **Total Shielded Blocks** | **1,805,229** | **Guaranteed clean, high-precision contrastive gradients** |

