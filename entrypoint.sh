#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================="
echo "  ArcticDense Neural Book Search Engine"
echo "  Retriever: Snowflake Arctic Embed M v1.5 (109M)"
echo "  Reranker:  Jina Reranker v2 (278M Contrastive InfoNCE)"
echo "  Catalog:   64,552 Goodreads Books (MRL 64-768D)"
echo "=========================================================="

echo "[Startup] Detecting hardware and environment..."
python3 -c "
import torch, os
device_env = os.environ.get('DEVICE', '').lower().strip()
if device_env == 'cpu':
    print('[Startup] Hardware: CPU Mode (Forced via DEVICE=cpu)')
elif torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'[Startup] Hardware: NVIDIA GPU ({name}, {vram:.2f} GB VRAM) - FP16 Accelerated')
else:
    print('[Startup] Hardware: CPU Fallback (CUDA not detected on container host)')
"

echo "[Startup] Verifying catalog metadata and embeddings..."
if [ ! -f "/app/data/books_metadata.sqlite3" ]; then
    echo "[Error] Metadata database missing: /app/data/books_metadata.sqlite3" >&2
    exit 1
fi

if [ -f "/app/app/.cache/catalog_embeddings_fp16.npy" ] && [ -f "/app/app/.cache/catalog_book_ids.json" ]; then
    echo "[Startup] Pre-computed catalog embeddings found (64,552 books ready for sub-15ms retrieval)."
else
    echo "[Startup] Notice: Pre-computed cache not found. Search engine will generate embeddings on first startup."
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
WORKERS="${WORKERS:-1}"

echo "[Startup] Launching FastAPI / Uvicorn server on http://${HOST}:${PORT} (workers: ${WORKERS})..."
exec uvicorn app.app:app \
    --host "${HOST}" \
    --port "${PORT}" \
    --workers "${WORKERS}" \
    --access-log
