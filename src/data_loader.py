"""
Data Loading and Metadata Processing Engine
"""
import json
import sqlite3
import time
import re
from pathlib import Path
from typing import Dict, List, Set, Any
from datasets import Dataset

from src.config import DB_FILE, QUERY_INSTRUCTION

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "for", "to", "with",
    "about", "is", "are", "that", "this", "i", "my", "me", "book", "books",
    "novel", "novels", "story", "stories", "series", "vol", "volume", "part"
}

def norm_text(text: str) -> str:
    """Normalize whitespace and lowercases text."""
    return re.sub(r"\s+", " ", (text or "").lower().strip())

def extract_base_title(title: str) -> str:
    """Strips subtitles, edition notes, parentheses, brackets, and colons."""
    t = re.sub(r"\(.*?\)", "", title)
    t = re.sub(r"\[.*?\]", "", t)
    t = re.split(r"[:\-–—;]", t)[0]
    return norm_text(t)

def get_significant_words(title: str) -> Set[str]:
    """Returns content words of length >= 3 not in stopwords."""
    words = re.findall(r"[a-z0-9']+", norm_text(title))
    return {w for w in words if len(w) >= 3 and w not in STOPWORDS}

def extract_series_name(title: str) -> str:
    """Extracts series identifier from title string using regex heuristics."""
    m = re.search(r"\(([^,#\)]+)(?:,|\s+#|\s+Book|\s+Vol|\s+Part|\s+Novella|\s+Series|\s+Trilogy)", title, re.IGNORECASE)
    if m:
        s = norm_text(m.group(1))
        if len(s) >= 3 and s not in STOPWORDS:
            return s
    m = re.search(r":\s+A\s+([^\:\(\)]+?)\s+(?:Novel|Series|Novella|Trilogy|Chronicles)", title, re.IGNORECASE)
    if m:
        s = norm_text(m.group(1))
        if len(s) >= 3 and s not in STOPWORDS:
            return s
    return ""

def load_book_metadata(db_path: Path = DB_FILE) -> Dict[str, Any]:
    """Loads all book records into memory for instant lookup and false-negative checks."""
    print(f"\n[Metadata] Loading book catalog from {db_path.name}...")
    t0 = time.time()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    book_texts = {}
    title_of = {}
    base_title_of = {}
    sig_words_of = {}
    series_of = {}
    author_tags_of = {}

    query = "SELECT book_id, title, authors, description, popular_shelves FROM books"
    for row in cur.execute(query):
        bid = str(row[0])
        title = row[1] or ""
        authors_raw = row[2] or "[]"
        desc = (row[3] or "").strip()
        shelves_raw = row[4] or "[]"

        # Format exact book text matching retrieval structure
        parts = []
        if title.strip():
            parts.append(f"Title: {title.strip()}")
        try:
            shelf_list = json.loads(shelves_raw)
            if shelf_list:
                parts.append(f"Genres: {', '.join(shelf_list)}")
        except Exception:
            shelf_list = []
        if desc:
            parts.append(f"Description: {desc[:900]}")
        book_texts[bid] = " | ".join(parts) if parts else "Untitled"

        norm_t = norm_text(title)
        title_of[bid] = norm_t
        base_title_of[bid] = extract_base_title(title)
        sig_words_of[bid] = get_significant_words(title)
        series_of[bid] = extract_series_name(title)

        ath_tags = set()
        for s in shelf_list:
            if "-" in s and not any(k in s for k in ["read", "book", "fiction", "own", "ya", "lit", "club", "grade", "level", "award", "series", "audio", "kindle"]):
                ath_tags.add(s.lower())
        author_tags_of[bid] = ath_tags

    conn.close()
    elapsed = time.time() - t0
    print(f"[Metadata] Loaded {len(book_texts):,} books in {elapsed:.2f}s.")
    return {
        "book_texts": book_texts,
        "title_of": title_of,
        "base_title_of": base_title_of,
        "sig_words_of": sig_words_of,
        "series_of": series_of,
        "author_tags_of": author_tags_of,
    }

def read_jsonl(path: Path, limit: int = None) -> List[Dict[str, Any]]:
    """Reads rows from a jsonl file."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows
