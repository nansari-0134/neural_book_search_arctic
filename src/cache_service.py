"""
Redis Caching Service for ArcticDense Search Engine
===================================================
Provides high-speed caching for:
1. Query search results (sub-millisecond repeat queries)
2. Normalized query embedding vectors
3. Graceful fallback when Redis is offline (no-op bypass)
"""

import os
import json
import hashlib
from typing import Optional, Dict, Any
import numpy as np

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class RedisCacheService:
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
        self.client: Optional["redis.Redis"] = None
        self.is_connected: bool = False
        self.hits: int = 0
        self.misses: int = 0
        self._initialize()

    def _initialize(self):
        """Attempts connection to Redis with short socket timeout."""
        if not REDIS_AVAILABLE:
            print("[RedisCache] redis-py library not installed. Caching disabled.")
            return

        try:
            try:
                self.client = redis.from_url(
                    self.redis_url,
                    protocol=2,
                    socket_timeout=1.0,
                    socket_connect_timeout=1.0,
                    decode_responses=False
                )
                self.client.ping()
            except Exception:
                self.client = redis.from_url(
                    self.redis_url,
                    socket_timeout=1.0,
                    socket_connect_timeout=1.0,
                    decode_responses=False
                )
                self.client.ping()
            self.is_connected = True
            print(f"[RedisCache] Connected to Redis at {self.redis_url}")
        except Exception as e:
            self.is_connected = False
            self.client = None
            print(f"[RedisCache] Redis not available ({e}). Operating in bypass mode.")

    def _hash_key(self, prefix: str, **kwargs) -> str:
        """Generates deterministic sha256 cache key."""
        payload = json.dumps(kwargs, sort_keys=True)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return f"arctic:{prefix}:{digest}"

    def get_query_cache(
        self,
        query: str,
        dim: int,
        rerank: bool,
        mode: str = "dense",
        genre: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieves cached search response for identical query parameters."""
        if not self.is_connected or not self.client:
            return None

        key = self._hash_key("query", q=query.strip().lower(), dim=dim, rerank=rerank, mode=mode, genre=genre or "")
        try:
            cached_bytes = self.client.get(key)
            if cached_bytes:
                self.hits += 1
                data = json.loads(cached_bytes.decode("utf-8"))
                data["cache_hit"] = True
                return data
            self.misses += 1
        except Exception as e:
            print(f"[RedisCache] Read error: {e}")
        return None

    def set_query_cache(
        self,
        query: str,
        dim: int,
        rerank: bool,
        mode: str,
        genre: Optional[str],
        result: Dict[str, Any],
        ttl_seconds: int = 3600
    ) -> bool:
        """Stores search response in Redis with TTL expiration."""
        if not self.is_connected or not self.client:
            return False

        key = self._hash_key("query", q=query.strip().lower(), dim=dim, rerank=rerank, mode=mode, genre=genre or "")
        try:
            serialized = json.dumps(result).encode("utf-8")
            self.client.setex(key, ttl_seconds, serialized)
            return True
        except Exception as e:
            print(f"[RedisCache] Write error: {e}")
            return False

    def get_embedding(self, text: str, dim: int) -> Optional[np.ndarray]:
        """Retrieves cached query vector."""
        if not self.is_connected or not self.client:
            return None

        key = self._hash_key("emb", text=text.strip(), dim=dim)
        try:
            raw = self.client.get(key)
            if raw:
                return np.frombuffer(raw, dtype=np.float32)
        except Exception as e:
            print(f"[RedisCache] Vector read error: {e}")
        return None

    def set_embedding(self, text: str, dim: int, vector: np.ndarray, ttl_seconds: int = 86400) -> bool:
        """Stores normalized query vector in Redis."""
        if not self.is_connected or not self.client:
            return False

        key = self._hash_key("emb", text=text.strip(), dim=dim)
        try:
            arr = np.ascontiguousarray(vector, dtype=np.float32)
            self.client.setex(key, ttl_seconds, arr.tobytes())
            return True
        except Exception as e:
            print(f"[RedisCache] Vector write error: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Returns connection and telemetry stats."""
        total_requests = self.hits + self.misses
        hit_ratio = (self.hits / total_requests * 100) if total_requests > 0 else 0.0
        return {
            "enabled": self.is_connected,
            "connected": self.is_connected,
            "redis_url": self.redis_url if self.is_connected else None,
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio_percent": round(hit_ratio, 2)
        }
