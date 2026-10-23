"""
Container & API Verification Script
===================================
Tests health, stats, and search endpoints of a running Docker container or local server.

Usage:
    python tools/verify_container.py [--url http://localhost:8080]
"""

import sys
import time
import argparse
import urllib.request
import urllib.parse
import json

def check_endpoint(url: str, description: str, timeout: int = 10) -> dict:
    print(f"[*] Testing {description}: {url}")
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "DockerVerifier/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            latency = (time.time() - t0) * 1000
            data = json.loads(response.read().decode("utf-8"))
            print(f"    [SUCCESS] Status {response.status} in {latency:.1f}ms")
            return data
    except Exception as e:
        print(f"    [FAILED] Error: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(description="Verify ArcticDense Container Endpoints")
    parser.add_argument("--url", default="http://localhost:8080", help="Base URL of the server")
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    print("=" * 60)
    print(f"  ArcticDense Container Verification: {base_url}")
    print("=" * 60)

    # 1. Healthcheck
    health = check_endpoint(f"{base_url}/api/health", "Healthcheck Endpoint")
    if not health.get("ready"):
        print(f"[!] Warning: Engine reports not ready: {health}")
    else:
        print(f"    Catalog Indexed: {health.get('catalog_indexed', 'N/A'):,} books")

    # 2. Stats
    stats = check_endpoint(f"{base_url}/api/stats", "System Stats Endpoint")
    if stats:
        print(f"    Model:      {stats.get('model_name')}")
        print(f"    Device:     {stats.get('device')}")
        print(f"    FP16:       {stats.get('fp16_enabled')}")
        print(f"    Dimensions: {stats.get('supported_mrl_dimensions')}")

    # 3. Dense Retrieval Test
    query = "cyberpunk detective artificial intelligence"
    params = urllib.parse.urlencode({"q": query, "dim": 768, "top_k": 3, "rerank": "false"})
    search_url = f"{base_url}/api/search?{params}"
    dense_res = check_endpoint(search_url, f"Dense Retrieval Query ('{query}')")
    if dense_res and "results" in dense_res:
        print(f"    Retrieved {len(dense_res['results'])} books in {dense_res.get('total_latency_ms', 0):.1f}ms")
        for i, b in enumerate(dense_res["results"]):
            print(f"      {i+1}. {b.get('title')} (Sim: {b.get('similarity_score', 0):.3f})")

    # 4. Two-Stage Cascade Reranking Test
    params_rerank = urllib.parse.urlencode({"q": query, "dim": 768, "top_k": 3, "rerank": "true"})
    cascade_url = f"{base_url}/api/search?{params_rerank}"
    cascade_res = check_endpoint(cascade_url, "Two-Stage Cascade Rerank Query")
    if cascade_res and "results" in cascade_res:
        reranker = cascade_res.get("reranker", "None")
        print(f"    Reranked via: {reranker}")
        print(f"    Retrieved {len(cascade_res['results'])} books in {cascade_res.get('total_latency_ms', 0):.1f}ms")
        for i, b in enumerate(cascade_res["results"]):
            score = b.get('reranker_score') or b.get('similarity_score', 0)
            print(f"      {i+1}. {b.get('title')} (Score: {score:.3f})")

    print("=" * 60)
    print("Verification complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
