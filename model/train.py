"""
Train the SeatAvailabilityDNN on the unified scraped dataset.

Usage:
    python -m model.train                       # default settings
    python -m model.train --epochs 30 --lr 3e-4
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from model.model import SeatAvailabilityDNN
from model.dataset import NEETDataset

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_DIR     = Path(__file__).parent.parent / "model" / "checkpoints"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH    = MODEL_DIR / "seat_dnn.pt"
META_PATH     = MODEL_DIR / "meta.json"


def train(
    epochs: int = 25,
    lr: float = 3e-4,
    batch_size: int = 512,
    val_split: float = 0.15,
    patience: int = 5,
):
    unified_path = PROCESSED_DIR / "unified.csv"
    if not unified_path.exists():
        raise FileNotFoundError(
            f"{unified_path} not found. Run `python -m scraper.scrape` first."
        )

    df = pd.read_csv(unified_path)
    print(f"Loaded {len(df)} rows from unified.csv")

    dataset = NEETDataset(df, samples_per_row=10)
    dataset.save_encoders()

    n_val   = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    model = SeatAvailabilityDNN(
        n_colleges  = dataset.n_colleges,
        n_states    = dataset.n_states,
        n_categories= dataset.n_categories,
        n_quotas    = dataset.n_quotas,
        n_genders   = dataset.n_genders,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5, verbose=True
    )

    best_val_loss = float("inf")
    no_improve    = 0

    for epoch in range(1, epochs + 1):
        # ── train ─────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            b = {k: v.to(device) for k, v in batch.items()}
            logits = model(
                b["log_rank"], b["year_norm"],
                b["college_id"], b["state_id"],
                b["category_id"], b["quota_id"], b["round_id"], b["gender_id"],
            )
            loss = criterion(logits, b["label"])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(b["label"])

        train_loss /= n_train

        # ── validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        correct  = 0
        with torch.no_grad():
            for batch in val_loader:
                b = {k: v.to(device) for k, v in batch.items()}
                logits = model(
                    b["log_rank"], b["year_norm"],
                    b["college_id"], b["state_id"],
                    b["category_id"], b["quota_id"], b["round_id"],
                )
                val_loss += criterion(logits, b["label"]).item() * len(b["label"])
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == b["label"]).sum().item()

        val_loss /= n_val
        val_acc   = correct / n_val * 100
        scheduler.step(val_loss)

        print(f"Epoch {epoch:3d}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
            # Save metadata for loading at inference
            meta = {
                "n_colleges":   dataset.n_colleges,
                "n_states":     dataset.n_states,
                "n_categories": dataset.n_categories,
                "n_quotas":     dataset.n_quotas,
                "n_genders":    dataset.n_genders,
            }
            with open(META_PATH, "w") as f:
                json.dump(meta, f, indent=2)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nBest val_loss: {best_val_loss:.4f}  → model saved to {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=25)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int,   default=512)
    parser.add_argument("--patience",   type=int,   default=5)
    args = parser.parse_args()
    train(args.epochs, args.lr, args.batch_size, patience=args.patience)
