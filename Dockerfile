# ==============================================================================
# ArcticDense: Neural Book Search Engine (Snowflake Arctic Embed + Jina Reranker v2)
# Production Container Image with NVIDIA CUDA Acceleration & CPU Fallback
# ==============================================================================

FROM python:3.11-slim-bookworm AS base

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080 \
    HOST=0.0.0.0 \
    DEVICE=cuda \
    OMP_NUM_THREADS=8

# Install essential system dependencies (curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root application user for secure execution
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -s /bin/bash -m appuser

# Set working directory
WORKDIR /app

# Step 1: Install Python Dependencies (Cached Layer)
COPY requirements.txt .

# Install PyTorch with CUDA 12.1 runtime support (falls back to CPU if no GPU present)
RUN pip install --no-cache-dir torch>=2.2.0 --extra-index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

# Step 2: Copy Application Source Code
COPY src/ ./src/
COPY app/ ./app/

# Step 3: Copy Required Runtime Metadata Database
COPY data/books_metadata.sqlite3 ./data/books_metadata.sqlite3

# Step 4: Copy Fine-Tuned Model Weights
# - Bi-Encoder Retriever: Snowflake Arctic Embed M v1.5 (109M) Champion
# - Cross-Encoder Reranker: Jina Reranker v2 (278M Contrastive InfoNCE)
COPY arctic_output/arctic_v3/best_model ./arctic_output/arctic_v3/best_model
COPY arctic_output/jina_contrastive_reranker/best_model ./arctic_output/jina_contrastive_reranker/best_model

# Step 5: Copy Startup Script and Set Permissions
COPY entrypoint.sh ./entrypoint.sh

RUN mkdir -p /app/app/.cache /app/results && \
    chmod +x ./entrypoint.sh && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Container Healthcheck via FastAPI /api/health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

# Expose default HTTP application port
EXPOSE 8080

# Run entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
