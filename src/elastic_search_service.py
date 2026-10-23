"""
Elasticsearch Lexical & BM25 Integration for ArcticDense
========================================================
Provides BM25 keyword matching, exact title/author lookups, and faceted
filtering across the 64,552 Goodreads book catalog.
Used in Hybrid Cascade Search:
    Hybrid Score = alpha * Dense(Arctic) + (1 - alpha) * BM25(Elasticsearch)
"""

import os
import time
from typing import List, Dict, Any, Optional

try:
    from elasticsearch import Elasticsearch, helpers
    ELASTICSEARCH_AVAILABLE = True
except Exception as e:
    ELASTICSEARCH_AVAILABLE = False


class ElasticBookIndex:
    INDEX_NAME = "arctic_books"

    def __init__(self, es_url: Optional[str] = None):
        self.es_url = es_url or os.environ.get("ELASTICSEARCH_URL", "http://127.0.0.1:9200")
        self.client: Optional["Elasticsearch"] = None
        self.is_connected: bool = False
        self._initialize()

    def _initialize(self):
        """Attempts connection to Elasticsearch cluster."""
        if not ELASTICSEARCH_AVAILABLE:
            print("[ElasticSearch] elasticsearch-py library not installed. Lexical retrieval disabled.")
            return

        try:
            self.client = Elasticsearch(
                self.es_url,
                request_timeout=2.0,
                max_retries=1,
                retry_on_timeout=False
            )
            if self.client.ping():
                self.is_connected = True
                print(f"[ElasticSearch] Connected successfully to Elasticsearch cluster at {self.es_url}")
            else:
                self.is_connected = False
                print(f"[ElasticSearch] Ping failed for {self.es_url}. Operating in dense-only mode.")
        except Exception as e:
            self.is_connected = False
            self.client = None
            print(f"[ElasticSearch] Cluster unavailable ({e}). Operating in dense-only mode.")

    def create_index(self) -> bool:
        """Creates the arctic_books index with optimized BM25 text mappings."""
        if not self.is_connected or not self.client:
            return False

        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "book_text_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stop", "snowball"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "book_id": {"type": "keyword"},
                    "title": {
                        "type": "text",
                        "analyzer": "book_text_analyzer",
                        "fields": {"keyword": {"type": "keyword"}}
                    },
                    "author": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}}
                    },
                    "description": {
                        "type": "text",
                        "analyzer": "book_text_analyzer"
                    },
                    "genres": {"type": "keyword"},
                    "popular_shelves": {"type": "keyword"},
                    "average_rating": {"type": "float"},
                    "ratings_count": {"type": "integer"}
                }
            }
        }

        try:
            if not self.client.indices.exists(index=self.INDEX_NAME):
                self.client.indices.create(index=self.INDEX_NAME, body=mapping)
                print(f"[ElasticSearch] Created index '{self.INDEX_NAME}'.")
            return True
        except Exception as e:
            print(f"[ElasticSearch] Index creation error: {e}")
            return False

    def search_bm25(
        self,
        query: str,
        top_k: int = 50,
        genre_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes BM25 search with field boosting:
        - title: 3.0x
        - author: 2.0x
        - genres: 1.5x
        - description: 1.0x
        """
        if not self.is_connected or not self.client:
            return []

        should_clause = [
            {"multi_match": {
                "query": query,
                "fields": [
                    "title^3.0",
                    "title.keyword^4.0",
                    "author^2.0",
                    "genres^1.5",
                    "popular_shelves^1.2",
                    "description^1.0"
                ],
                "type": "best_fields",
                "fuzziness": "AUTO"
            }}
        ]

        bool_query: Dict[str, Any] = {"must": should_clause}
        if genre_filter and genre_filter.strip():
            clean_genre = genre_filter.strip().lower()
            bool_query["filter"] = [
                {"terms": {"genres": [clean_genre]}}
            ]

        try:
            resp = self.client.search(
                index=self.INDEX_NAME,
                query={"bool": bool_query},
                size=top_k
            )
            hits = resp.get("hits", {}).get("hits", [])
            max_score = resp.get("hits", {}).get("max_score", 1.0) or 1.0

            results = []
            for hit in hits:
                src = hit.get("_source", {})
                score = hit.get("_score", 0.0)
                norm_score = float(score) / float(max_score) if max_score > 0 else 0.0
                results.append({
                    "book_id": src.get("book_id"),
                    "title": src.get("title"),
                    "author": src.get("author", "Unknown"),
                    "description": src.get("description", ""),
                    "genres": src.get("genres", []),
                    "average_rating": src.get("average_rating", 0.0),
                    "bm25_score": round(float(score), 4),
                    "norm_bm25_score": round(norm_score, 4)
                })
            return results
        except Exception as e:
            print(f"[ElasticSearch] Search error: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Returns index status, document count, and cluster connection status."""
        if not self.is_connected or not self.client:
            return {
                "enabled": False,
                "connected": False,
                "cluster_url": self.es_url,
                "document_count": 0
            }

        try:
            count = self.client.count(index=self.INDEX_NAME).get("count", 0)
            return {
                "enabled": True,
                "connected": True,
                "cluster_url": self.es_url,
                "index_name": self.INDEX_NAME,
                "document_count": count
            }
        except Exception:
            return {
                "enabled": True,
                "connected": False,
                "cluster_url": self.es_url,
                "document_count": 0
            }
