"""
Interactive Database Inspection Tool for books_metadata.sqlite3
Usage:
    python tools/inspect_db.py --summary
    python tools/inspect_db.py --search "Dune"
    python tools/inspect_db.py --random 3
"""
import sqlite3
import json
import argparse
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "books_metadata.sqlite3"

def get_connection():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
    return sqlite3.connect(DB_PATH)

def print_summary():
    conn = get_connection()
    cur = conn.cursor()
    
    total = cur.execute("SELECT COUNT(*) FROM books").fetchone()[0]
    with_desc = cur.execute("SELECT COUNT(*) FROM books WHERE description IS NOT NULL AND length(trim(description)) > 0").fetchone()[0]
    with_shelves = cur.execute("SELECT COUNT(*) FROM books WHERE popular_shelves IS NOT NULL AND popular_shelves != '[]'").fetchone()[0]
    with_authors = cur.execute("SELECT COUNT(*) FROM books WHERE authors IS NOT NULL AND authors != '[]'").fetchone()[0]
    
    print("=" * 65)
    print("  GOODREADS CATALOG DATABASE SUMMARY (books_metadata.sqlite3)")
    print("=" * 65)
    print(f"  Database Path:        {DB_PATH.resolve()}")
    print(f"  Database File Size:   {DB_PATH.stat().st_size / (1024**2):.1f} MB")
    print(f"  Total Catalog Books:  {total:,}")
    print(f"  Books with Synopses:  {with_desc:,} ({with_desc/total*100:.1f}%)")
    print(f"  Books with Shelves:   {with_shelves:,} ({with_shelves/total*100:.1f}%)")
    print(f"  Books with Authors:   {with_authors:,} ({with_authors/total*100:.1f}%)")
    print("=" * 65)
    conn.close()

def search_books(keyword: str, limit: int = 5):
    conn = get_connection()
    cur = conn.cursor()
    
    query = """
    SELECT book_id, title, authors, popular_shelves, description 
    FROM books 
    WHERE title LIKE ? OR description LIKE ?
    LIMIT ?
    """
    param = f"%{keyword}%"
    rows = cur.execute(query, (param, param, limit)).fetchall()
    
    print(f"\n[Search Results for '{keyword}'] (Found {len(rows)} books):")
    for idx, row in enumerate(rows, 1):
        bid, title, authors, shelves, desc = row
        try:
            auth_list = json.loads(authors)
        except Exception:
            auth_list = [authors]
        try:
            shelf_list = json.loads(shelves)
        except Exception:
            shelf_list = []
            
        print(f"\n[{idx}] Book ID: {bid}")
        safe_title = title.encode("ascii", errors="replace").decode("ascii")
        safe_desc = desc[:200].encode("ascii", errors="replace").decode("ascii")
        print(f"    Title:       {safe_title}")
        print(f"    Authors:     {', '.join(auth_list)}")
        print(f"    Genres:      {', '.join(shelf_list[:8])}")
        print(f"    Description: {safe_desc}...")
    conn.close()

def show_random(count: int = 3):
    conn = get_connection()
    cur = conn.cursor()
    rows = cur.execute(f"SELECT book_id, title, authors, popular_shelves, description FROM books ORDER BY RANDOM() LIMIT {count}").fetchall()
    print(f"\n[Sample of {count} Random Catalog Books]:")
    for idx, row in enumerate(rows, 1):
        bid, title, authors, shelves, desc = row
        safe_title = title.encode("ascii", errors="replace").decode("ascii")
        safe_desc = desc[:250].encode("ascii", errors="replace").decode("ascii")
        print(f"\n--- Book {idx} [ID: {bid}] ---")
        print(f"Title:       {safe_title}")
        print(f"Authors:     {authors}")
        print(f"Genres:      {shelves}")
        print(f"Description: {safe_desc}...")
    conn.close()

def main():
    parser = argparse.ArgumentParser(description="Inspect books_metadata.sqlite3")
    parser.add_argument("--summary", action="store_true", help="Print catalog summary statistics")
    parser.add_argument("--search", type=str, help="Search books by title or keyword")
    parser.add_argument("--random", type=int, default=0, help="Print N random sample books")
    args = parser.parse_args()

    if args.search:
        search_books(args.search)
    elif args.random > 0:
        show_random(args.random)
    else:
        print_summary()

if __name__ == "__main__":
    main()
