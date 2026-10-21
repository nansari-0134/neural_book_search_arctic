"""
Central Configuration for Neural Book Search (Snowflake Arctic Embed)
"""
import os
import torch
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "arctic_output"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Files
DB_FILE = DATA_DIR / "books_metadata.sqlite3"
TRAIN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE = DATA_DIR / "val.jsonl"
TEST_FILE = DATA_DIR / "test.jsonl"

# Model Identification
MODEL_NAME = "Snowflake/snowflake-arctic-embed-m-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
MAX_SEQ_LENGTH = 512

# Hardware Configuration (Optimized for NVIDIA RTX 4050 6GB)
NUM_THREADS = os.cpu_count() or 8
torch.set_num_threads(NUM_THREADS)

if torch.cuda.is_available():
    DEVICE = "cuda"
    USE_FP16 = True
    GPU_NAME = torch.cuda.get_device_name(0)
    VRAM_GB = torch.cuda.get_device_properties(0).total_memory / (1024**3)
else:
    DEVICE = "cpu"
    USE_FP16 = False
    GPU_NAME = "CPU"
    VRAM_GB = 0.0

# Training Hyperparameters
BATCH_SIZE = 16          # 16 queries per micro-batch (light on 6GB VRAM)
GRAD_ACCUM_STEPS = 2     # Effective contrastive batch size = 32
LEARNING_RATE = 1.5e-5   # Ideal LR for 109M parameter Arctic model
WARMUP_RATIO = 0.1
TEMPERATURE = 0.05
WEIGHT_DECAY = 0.01
SEED = 42

# Mining Hyperparameters
TOP_K_RETRIEVE = 50
MIN_RANK = 2             # Bypass ranks 0-1 (highest likelihood of unindexed sequels/editions)
MAX_RANK = 35            # Model confusion window
MAX_COSINE_SIM = 0.88    # Strict semantic similarity ceiling
NEW_NEGS_TO_ADD = 3      # Freshly mined hard negatives
OLD_NEGS_TO_KEEP = 2     # Retained broad negatives
MAX_NEG_FREQUENCY = 30   # Cap negative usage across dataset
EASY_PAIR_KEEP_RATIO = 0.35
