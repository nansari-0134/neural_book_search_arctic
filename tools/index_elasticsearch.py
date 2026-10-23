"""
Elasticsearch Catalog Indexing Script
=====================================
Indexes all 64,552 Goodreads books from SQLite into Elasticsearch
with BM25 text analysis and faceted fields.

Usage:
    python tools/index_elasticsearch.py [--url http://localhost:9200]
"""

import sys
import json
import sqlite3
import argparse
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DB_FILE
from src.elastic_search_service import ElasticBookIndex, ELASTICSEARCH_AVAILABLE

if not ELASTICSEARCH_AVAILABLE:
    print("[Error] elasticsearch python package is not installed. Run: pip install elasticsearch")
    sys.exit(1)

try:
    from elasticsearch.helpers import bulk
except Exception:
    bulk = None


def load_books_from_sqlite(db_path: Path):
    print(f"[*] Reading books metadata from {db_path.name}...")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT book_id, title, author, description, popular_shelves, average_rating, ratings_count
        FROM books
    """)
    rows = cursor.fetchall()
    conn.close()
    print(f"[*] Loaded {len(rows):,} books from database.")
    return rows


def generate_es_actions(rows, index_name: str):
    for r in rows:
        shelves_raw = r["popular_shelves"]
        genres = []
        if shelves_raw:
            try:
                shelves = json.loads(shelves_raw)
                if isinstance(shelves, list):
                    genres = [s.strip().lower() for s in shelves if isinstance(s, str)][:10]
            except Exception:
                genres = []

        doc = {
            "_index": index_name,
            "_id": str(r["book_id"]),
            "_source": {
                "book_id": str(r["book_id"]),
                "title": r["title"] or "Untitled",
                "author": r["author"] or "Unknown",
                "description": r["description"] or "",
                "genres": genres,
                "popular_shelves": genres,
                "average_rating": float(r["average_rating"] or 0.0),
                "ratings_count": int(r["ratings_count"] or 0)
            }
        }
        yield doc


def main():
    parser = argparse.ArgumentParser(description="Index catalog into Elasticsearch")
    parser.add_argument("--url", default="http://localhost:9200", help="Elasticsearch cluster URL")
    parser.add_argument("--batch-size", type=int, default=1000, help="Bulk index batch size")
    args = parser.parse_args()

    es_service = ElasticBookIndex(es_url=args.url)
    if not es_service.is_connected or not es_service.client:
        print(f"[Error] Could not connect to Elasticsearch cluster at {args.url}. Is it running?")
        sys.exit(1)

    print("[*] Ensuring index exists...")
    es_service.create_index()

    rows = load_books_from_sqlite(DB_FILE)
    actions = generate_es_actions(rows, es_service.INDEX_NAME)

    print(f"[*] Bulk indexing {len(rows):,} books into '{es_service.INDEX_NAME}'...")
    success_count, failed_items = bulk(
        es_service.client,
        actions,
        chunk_size=args.batch_size,
        stats_only=True
    )

    print("=" * 60)
    print(f"[SUCCESS] Successfully indexed {success_count:,} books into Elasticsearch!")
    if failed_items:
        print(f"[WARNING] Failed to index {failed_items} items.")
    print("=" * 60)


if __name__ == "__main__":
    main()
