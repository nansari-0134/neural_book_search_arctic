"""
Step 2: Phase 1 Fine-Tuning of Snowflake Arctic Embed (v1.5)
Contrastive Multiple Negatives Ranking Loss (MNRL) with Duplicate-Positive Masking
"""
import sys
import os
import json
import time
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import Dataset
from sentence_transformers import SentenceTransformer

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    MODEL_NAME, TRAIN_FILE, VAL_FILE, TEST_FILE, DB_FILE, OUTPUT_DIR, RESULTS_DIR,
    BATCH_SIZE, GRAD_ACCUM_STEPS, LEARNING_RATE, WARMUP_RATIO, TEMPERATURE,
    WEIGHT_DECAY, SEED, MAX_SEQ_LENGTH, DEVICE, USE_FP16, QUERY_INSTRUCTION
)
from src.data_loader import load_book_metadata
from src.metrics import evaluate_recall

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def main():
    print("=" * 70)
    print("  STEP 2: PHASE 1 FINE-TUNING (SNOWFLAKE ARCTIC EMBED)")
    print(f"  Model:   {MODEL_NAME}")
    print(f"  Device:  {DEVICE} (FP16: {USE_FP16})")
    print(f"  Output:  {OUTPUT_DIR}")
    print("=" * 70)

    # 1. Load book metadata
    meta = load_book_metadata(DB_FILE)
    all_book_ids = sorted(list(meta["book_texts"].keys()))

    phase1_dir = OUTPUT_DIR / "arctic_phase1"
    best_model_dir = phase1_dir / "best_model"
    cache_dir = phase1_dir / "datasets_cache"
    phase1_dir.mkdir(parents=True, exist_ok=True)

    # 2. Build fast in-memory PyTorch training dataset (Bypasses Windows MAX_PATH limits)
    class BookSearchDataset(torch.utils.data.Dataset):
        def __init__(self, data_path, book_meta):
            print(f"Loading training rows into memory from {Path(data_path).name}...")
            self.rows = []
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.rows.append(json.loads(line))
            self.book_meta = book_meta
            print(f"Dataset ready: {len(self.rows):,} training pairs.")

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            row = self.rows[idx]
            neg_ids = row.get("negative_ids", [])
            neg_texts = [self.book_meta["book_texts"].get(nid, "Untitled") for nid in neg_ids]
            if not neg_texts:
                neg_texts = [self.book_meta["book_texts"].get(str(row["positive_id"]), "Untitled")]
            return {
                "anchor": QUERY_INSTRUCTION + row["query"],
                "positive": self.book_meta["book_texts"].get(str(row["positive_id"]), "Untitled"),
                "positive_id": str(row["positive_id"]),
                "negatives": neg_texts,
            }

    train_dataset = BookSearchDataset(TRAIN_FILE, meta)
    n_train = len(train_dataset)

    # 3. Load model with gradient checkpointing
    print(f"\nLoading model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    model.max_seq_length = MAX_SEQ_LENGTH
    encoder = model[0].auto_model
    encoder.gradient_checkpointing_enable()
    if hasattr(encoder.config, "use_cache"):
        encoder.config.use_cache = False

    # 4. Optimizer & Cosine Annealing Scheduler
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda" and USE_FP16))

    steps_per_epoch = math.ceil(n_train / BATCH_SIZE)
    total_opt_steps = math.ceil(steps_per_epoch / GRAD_ACCUM_STEPS) * 1  # 1 Epoch
    warmup_steps = max(1, int(total_opt_steps * WARMUP_RATIO))
    print(f"Schedule: {total_opt_steps:,} optimizer steps | {warmup_steps:,} warmup steps.")

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / warmup_steps
        progress = (current_step - warmup_steps) / max(1, total_opt_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def to_device(obj):
        if hasattr(obj, "to"):
            return obj.to(DEVICE)
        elif isinstance(obj, dict):
            return {k: to_device(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_device(v) for v in obj]
        return obj

    def embed_train(texts):
        preprocess_fn = getattr(model, "preprocess", getattr(model, "tokenize", None))
        features = preprocess_fn(texts)
        features = to_device(features)
        return model(features)["sentence_embedding"]

    def collate_rows(rows):
        return {
            "anchor": [r["anchor"] for r in rows],
            "positive": [r["positive"] for r in rows],
            "positive_id": [r["positive_id"] for r in rows],
            "negatives": [r["negatives"] for r in rows],
        }

    # 5. Dataloader & Checkpointing setup
    g = torch.Generator(); g.manual_seed(SEED)
    perm = torch.randperm(n_train, generator=g).tolist()
    batches = [perm[i:i + BATCH_SIZE] for i in range(0, n_train, BATCH_SIZE)]
    n_steps = len(batches)

    resume_file = phase1_dir / "resume_state.pt"
    resume_prev = resume_file.with_suffix(".pt.prev")

    def save_checkpoint(step, loss_val):
        payload = {
            "epoch": 0, "step": step, "total_loss": loss_val,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if (DEVICE == "cuda") else None,
            "scheduler": scheduler.state_dict(),
        }
        tmp = resume_file.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        if resume_file.exists():
            os.replace(resume_file, resume_prev)
        os.replace(tmp, resume_file)

    start_step = 0
    total_loss = 0.0
    for cand in (resume_file, resume_prev):
        if cand.exists():
            try:
                st = torch.load(cand, map_location=DEVICE, weights_only=False)
                if st.get("step", 0) < n_steps:
                    model.load_state_dict(st["model"])
                    optimizer.load_state_dict(st["optimizer"])
                    if st.get("scaler") and DEVICE == "cuda":
                        scaler.load_state_dict(st["scaler"])
                    if st.get("scheduler"):
                        scheduler.load_state_dict(st["scheduler"])
                    start_step = int(st["step"])
                    total_loss = float(st.get("total_loss", 0.0))
                    print(f"Resuming training from step {start_step:,}/{n_steps:,}")
                    break
            except Exception as e:
                print(f"Warning: could not read {cand} ({e})")

    loader = DataLoader(train_dataset, batch_sampler=batches[start_step:], num_workers=0, collate_fn=collate_rows)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = last_save = time.time()

    print("\nStarting Phase 1 Training Loop...")
    for step_idx, batch in enumerate(loader, start=start_step):
        group_start = (step_idx // GRAD_ACCUM_STEPS) * GRAD_ACCUM_STEPS
        group_size = min(GRAD_ACCUM_STEPS, n_steps - group_start)

        all_neg_texts = []
        for negs in batch["negatives"]:
            all_neg_texts.extend(negs)

        autocast_ctx = torch.autocast("cuda", dtype=torch.float16) if (DEVICE == "cuda" and USE_FP16) else torch.nullcontext()
        with autocast_ctx:
            q = embed_train(batch["anchor"])
            pos_emb = embed_train(batch["positive"])
            neg_emb = embed_train(all_neg_texts) if all_neg_texts else None
            B = q.size(0)

            candidates = torch.cat([pos_emb, neg_emb], dim=0) if neg_emb is not None else pos_emb
            logits = (q @ candidates.T) / TEMPERATURE
            labels = torch.arange(B, device=DEVICE)

            # Duplicate positive mask within micro-batch
            pos_ids = batch["positive_id"]
            if len(set(pos_ids)) < len(pos_ids):
                mask = torch.zeros_like(logits, dtype=torch.bool)
                for i, pid_i in enumerate(pos_ids):
                    for j, pid_j in enumerate(pos_ids):
                        if i != j and pid_i == pid_j:
                            mask[i, j] = True
                logits = logits.masked_fill(mask, float("-inf"))

            loss = F.cross_entropy(logits, labels)

        if DEVICE == "cuda" and USE_FP16:
            scaler.scale(loss / group_size).backward()
        else:
            (loss / group_size).backward()

        total_loss += loss.detach().item()

        at_boundary = (step_idx + 1) % GRAD_ACCUM_STEPS == 0 or (step_idx + 1) == n_steps
        if at_boundary:
            if DEVICE == "cuda" and USE_FP16:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        if (step_idx + 1) % 50 == 0 or (step_idx + 1) == n_steps:
            elapsed = time.time() - started
            steps_done = step_idx + 1 - start_step
            eta_sec = ((elapsed / max(1, steps_done)) * (n_steps - (step_idx + 1)))
            eta_str = f"{int(eta_sec // 3600)}h {int((eta_sec % 3600) // 60)}m" if eta_sec >= 3600 else f"{int(eta_sec // 60)}m {int(eta_sec % 60)}s"
            pct = (step_idx + 1) / n_steps * 100
            cur_lr = scheduler.get_last_lr()[0]
            print(f"Step {step_idx + 1:,}/{n_steps:,} [{pct:5.1f}%] | Loss: {total_loss / (step_idx + 1):.4f} | LR: {cur_lr:.2e} | Elapsed: {elapsed/60:.1f}m | ETA: {eta_str}", flush=True)

        del q, pos_emb, candidates, logits, labels, loss
        if neg_emb is not None:
            del neg_emb

        # Checkpoint every 5 mins
        if at_boundary and (step_idx + 1) < n_steps and (time.time() - last_save > 5 * 60):
            save_checkpoint(step_idx + 1, total_loss)
            last_save = time.time()

    # 6. Save model and run validation
    print(f"\nTraining completed. Saving fine-tuned model to {best_model_dir}...")
    model.save(str(best_model_dir))
    save_checkpoint(n_steps, total_loss)
    print("Model saved successfully.")

    # Validation
    val_metrics = evaluate_recall(model, VAL_FILE, "Phase 1 Validation", meta, all_book_ids)
    test_metrics = evaluate_recall(model, TEST_FILE, "Phase 1 Test", meta, all_book_ids)

    results_payload = {
        "model": "arctic_phase1",
        "validation": val_metrics,
        "test": test_metrics
    }
    with open(RESULTS_DIR / "phase1_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

if __name__ == "__main__":
    main()
