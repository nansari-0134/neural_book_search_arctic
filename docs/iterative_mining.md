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

---

## 4. Cycle 2 Empirical Mining Results

Cycle 2 mining was executed against the **Cycle 1 fine-tuned model** (`arctic_output/arctic_v2/best_model`, R@10 = 73.96%) across the full 64,552 book catalog:

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Catalog Scope** | **64,552 books** | Pre-encoded with Cycle 1 weights into 768-d vectors on GPU (FP16) |
| **Input Queries** | **199,808 queries** | Adversarial pool from Cycle 1 |
| **Downsampled Easy Pairs** | **20,489 (10.3%)** | Pairs solved at rank #0 downsampled to focus on tough confusions |
| **Final Mined Dataset** | **179,319 pairs** | Saved to `data/train_arctic_v3.jsonl` (90.0 MB) |
| **Negative Composition** | **3 Fresh + 2 Retained = 5** | 3 ultra-hard negatives from Cycle 2 + 2 retained from Cycle 1 |
| **Mining Speed** | **1.5 minutes** | High-throughput GPU matrix dot-product + top-k retrieval |

### 🛡️ Cycle 2 Defense Shield Block Breakdown

The 8-layer shield successfully blocked **1,442,395 false-negative candidates** from entering Cycle 2:

| Layer Filter | Blocked Count | Purpose |
| :--- | :---: | :--- |
| **`shared_author_shelf`** | **1,346,710** | Prevented penalizing books written by the exact same author |
| **`banned_id`** | **40,430** | Blocked cross-query identical book references |
| **`frequency_cap`** | **20,569** | Prevented popular books from over-saturating negatives ($>30$ cap) |
| **`title_word_overlap`** | **11,827** | Blocked candidates sharing $\ge 65\%$ non-stopwords with target |
| **`series_match`** | **6,221** | Blocked direct series siblings (e.g. Sequels, Prequels) |
| **`identical_title`** | **5,862** | Blocked exact title matches |
| **`base_title_match`** | **5,479** | Blocked alternate edition subtitles and translations |
| **`series_in_cand_title`** | **2,731** | Blocked series title mentions in candidate book name |
| **`cand_series_in_pos_title`** | **2,566** | Blocked target series title mentions in candidate name |
| **Total Shielded Blocks** | **1,442,395** | **Zero gradient contamination on valid works** |


