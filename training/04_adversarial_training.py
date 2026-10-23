"""
Step 4: Adversarial Hard-Negative Contrastive Fine-Tuning of Snowflake Arctic Embed (v1.5)
Trains contrastive retriever using mined adversarial hard negatives and 8-layer shield protection.
Employs sub-batch negative chunking and gradient checkpointing for 6GB VRAM stability.
"""
import sys
import os
import json
import time
import math
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_DIR, VAL_FILE, TEST_FILE, DB_FILE, OUTPUT_DIR, RESULTS_DIR,
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
    parser = argparse.ArgumentParser(description="Adversarial hard-negative fine-tuning of Snowflake Arctic retriever")
    parser.add_argument("--train-file", default=str(DATA_DIR / "train_arctic_v2.jsonl"), help="Training JSONL file with mined negatives")
    parser.add_argument("--base-model", default=str(OUTPUT_DIR / "arctic_phase1" / "best_model"), help="Base model checkpoint to resume/fine-tune from")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR / "arctic_v2"), help="Output directory for fine-tuned model and checkpoints")
    parser.add_argument("--stage", choices=["targeted_hard_negatives", "progressive_refinement"], default="targeted_hard_negatives", help="Fine-tuning stage")
    parser.add_argument("--stage-name", default="Targeted Hard-Negative Training", help="Human-readable stage title")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Micro-batch size (queries per micro-batch)")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps (effective batch size = batch_size * grad_accum)")
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    GRAD_ACCUM_STEPS = args.grad_accum
    EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRAD_ACCUM_STEPS

    train_path = Path(args.train_file)
    base_model_path = Path(args.base_model)
    stage_dir = Path(args.output_dir)
    best_model_dir = stage_dir / "best_model"
    stage_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print(f"  STEP 4: {args.stage_name.upper()} (SNOWFLAKE ARCTIC EMBED)", flush=True)
    print(f"  Base Model: {base_model_path}", flush=True)
    print(f"  Train Data: {train_path.name}", flush=True)
    print(f"  Output Dir: {stage_dir}", flush=True)
    print(f"  Device:     {DEVICE} (FP16: {USE_FP16})", flush=True)
    print(f"  LR:         {args.lr:.2e}", flush=True)
    print(f"  Micro-Batch:{BATCH_SIZE} | Grad Accum: {GRAD_ACCUM_STEPS} | Effective Batch: {EFFECTIVE_BATCH_SIZE}", flush=True)
    print("=" * 70, flush=True)

    # 1. Load book metadata
    meta = load_book_metadata(DB_FILE)
    all_book_ids = sorted(list(meta["book_texts"].keys()))

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

    train_dataset = BookSearchDataset(train_path, meta)
    n_train = len(train_dataset)

    # 3. Load model with gradient checkpointing
    print(f"\nLoading base model: {base_model_path}...")
    model = SentenceTransformer(str(base_model_path), device=DEVICE)
    model.max_seq_length = MAX_SEQ_LENGTH
    encoder = model[0].auto_model
    encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(encoder.config, "use_cache"):
        encoder.config.use_cache = False

    # 4. Optimizer & Cosine Annealing Scheduler
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda" and USE_FP16))

    steps_per_epoch = math.ceil(n_train / BATCH_SIZE)
    total_opt_steps = math.ceil(steps_per_epoch / GRAD_ACCUM_STEPS) * 1  # 1 Epoch
    warmup_steps = max(1, int(total_opt_steps * WARMUP_RATIO))
    print(f"Schedule: {total_opt_steps:,} optimizer steps | {warmup_steps:,} warmup steps.", flush=True)

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

    def embed_train(texts, chunk_size=12):
        if len(texts) <= chunk_size:
            preprocess_fn = getattr(model, "preprocess", getattr(model, "tokenize", None))
            features = preprocess_fn(texts)
            features = to_device(features)
            return model(features)["sentence_embedding"]
        chunks = []
        for i in range(0, len(texts), chunk_size):
            chunk_texts = texts[i:i + chunk_size]
            preprocess_fn = getattr(model, "preprocess", getattr(model, "tokenize", None))
            features = preprocess_fn(chunk_texts)
            features = to_device(features)
            chunks.append(model(features)["sentence_embedding"])
        return torch.cat(chunks, dim=0)

    def collate_rows(rows):
        return {
            "anchor": [r["anchor"] for r in rows],
            "positive": [r["positive"] for r in rows],
            "positive_id": [r["positive_id"] for r in rows],
            "negatives": [r["negatives"] for r in rows],
        }

    # 5. Dataloader & Checkpointing setup
    g = torch.Generator(); g.manual_seed(SEED + args.cycle)
    perm = torch.randperm(n_train, generator=g).tolist()
    batches = [perm[i:i + BATCH_SIZE] for i in range(0, n_train, BATCH_SIZE)]
    n_steps = len(batches)

    resume_file = cycle_dir / "resume_state.pt"
    resume_prev = resume_file.with_suffix(".pt.prev")

    def save_checkpoint(step, loss_val):
        payload = {
            "epoch": 0, "step": step, "total_loss": loss_val, "cycle": args.cycle,
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
                    print(f"Resuming Cycle {args.cycle} training from step {start_step:,}/{n_steps:,}")
                    break
            except Exception as e:
                print(f"Warning: could not read {cand} ({e})")

    loader = DataLoader(train_dataset, batch_sampler=batches[start_step:], num_workers=0, collate_fn=collate_rows)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = last_save = time.time()

    print(f"\nStarting Cycle {args.cycle} Training Loop...")
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
            neg_emb = embed_train(all_neg_texts, chunk_size=12) if all_neg_texts else None
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
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    # 6. Save fine-tuned model
    print(f"\nCycle {args.cycle} training completed. Saving model to {best_model_dir}...")
    model.save(str(best_model_dir))
    save_checkpoint(n_steps, total_loss)
    print("Model saved successfully.")

    # 7. Fast evaluation using pre-encoded catalog
    print(f"\n[Evaluation] Pre-encoding catalog of {len(all_book_ids):,} books on {DEVICE}...")
    model.eval()
    if DEVICE == "cuda" and USE_FP16:
        model.half()
    catalog_texts = [meta["book_texts"].get(bid, "Untitled") for bid in all_book_ids]
    catalog_emb = model.encode(
        catalog_texts,
        batch_size=128 if DEVICE == "cuda" else 32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype("float32")

    val_metrics = evaluate_recall(model, VAL_FILE, f"{args.stage_name} Validation", meta, all_book_ids, cached_catalog_emb=catalog_emb)
    test_metrics = evaluate_recall(model, TEST_FILE, f"{args.stage_name} Test", meta, all_book_ids, cached_catalog_emb=catalog_emb)

    results_payload = {
        "stage": args.stage,
        "description": args.stage_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "validation": val_metrics,
        "test": test_metrics
    }
    metrics_file = RESULTS_DIR / f"{args.stage}_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\n[Done] {args.stage_name} benchmark saved to {metrics_file.name}")

if __name__ == "__main__":
    main()
