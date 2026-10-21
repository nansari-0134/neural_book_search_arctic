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
