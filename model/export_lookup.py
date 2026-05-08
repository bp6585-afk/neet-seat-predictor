"""
Export a DNN-calibrated lookup table.

For every row in unified.csv, binary-search the DNN to find the rank
at which it predicts exactly 50% probability.  Save the result as
data/processed/unified_dnn.csv — a drop-in replacement for unified.csv
that gives DNN-quality closing ranks without needing torch at runtime.

Usage:
    python -m model.export_lookup
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model.model import SeatAvailabilityDNN
from model.dataset import NEETDataset, ROUND_ORDER

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_DIR     = Path(__file__).parent.parent / "model" / "checkpoints"

MODEL_PATH   = MODEL_DIR / "seat_dnn.pt"
META_PATH    = MODEL_DIR / "meta.json"
ENCODER_PATH = PROCESSED_DIR / "encoders.json"
INPUT_CSV    = PROCESSED_DIR / "unified.csv"
OUTPUT_CSV   = PROCESSED_DIR / "unified_dnn.csv"


def _load_model():
    with open(META_PATH) as f:
        meta = json.load(f)
    encs = NEETDataset.load_encoders(ENCODER_PATH)
    device = torch.device(
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = SeatAvailabilityDNN(
        n_colleges   = meta["n_colleges"],
        n_states     = meta["n_states"],
        n_categories = meta["n_categories"],
        n_quotas     = meta["n_quotas"],
        n_genders    = meta.get("n_genders", 3),
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model.to(device), encs, device


def _batch_predict(model, device, encs, rows_df: pd.DataFrame, ranks: np.ndarray) -> np.ndarray:
    """
    Predict probability for each row in rows_df at the corresponding rank in ranks.
    rows_df and ranks must be the same length.
    Returns a 1-D float array of probabilities.
    """
    n = len(rows_df)
    log_ranks = np.log1p(ranks).astype(np.float32)
    yr_norm   = np.full(n, (2025 - 2020) / 5.0, dtype=np.float32)

    g_enc = encs.get("gender", None)

    c_ids  = encs["college"].transform(rows_df["college"].tolist())
    s_ids  = encs["state"].transform(rows_df["state"].tolist())
    ca_ids = encs["category"].transform(rows_df["category"].tolist())
    q_ids  = encs["quota"].transform(rows_df["quota"].tolist())
    r_ids  = [min(ROUND_ORDER.get(r, 4), 4) for r in rows_df["round"]]
    g_vals = rows_df["gender"].tolist() if "gender" in rows_df.columns else ["All"] * n
    g_ids  = g_enc.transform(g_vals) if g_enc else [0] * n

    with torch.no_grad():
        probs = model.predict_proba(
            torch.tensor(log_ranks).to(device),
            torch.tensor(yr_norm).to(device),
            torch.tensor(c_ids,  dtype=torch.long).to(device),
            torch.tensor(s_ids,  dtype=torch.long).to(device),
            torch.tensor(ca_ids, dtype=torch.long).to(device),
            torch.tensor(q_ids,  dtype=torch.long).to(device),
            torch.tensor(r_ids,  dtype=torch.long).to(device),
            torch.tensor(g_ids,  dtype=torch.long).to(device),
        ).cpu().numpy()
    return probs


def find_50pct_rank(model, device, encs, row_df: pd.DataFrame,
                    lo: int = 1, hi: int = 800_000, tol: int = 100) -> int:
    """
    Binary-search for the rank where the DNN predicts p ≈ 0.50 for this row.
    Returns the rank (integer).
    """
    # row_df is a single-row DataFrame; we replicate it across the binary search
    while hi - lo > tol:
        mid = (lo + hi) // 2
        p   = _batch_predict(model, device, encs, row_df, np.array([mid]))[0]
        if p > 0.5:
            lo = mid
        else:
            hi = mid
    return (lo + hi) // 2


def main():
    print("Loading model…")
    model, encs, device = _load_model()

    print(f"Reading {INPUT_CSV}…")
    df = pd.read_csv(INPUT_CSV)
    if "gender" not in df.columns:
        df["gender"] = "All"

    dnn_ranks = []
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        row_df = pd.DataFrame([row])
        raw_cr = int(row["closing_rank"])

        # Search window: ½× to 2× the raw closing rank, clamped to [1, 800000]
        lo = max(1,       raw_cr // 2)
        hi = min(800_000, raw_cr * 2)

        cutoff = find_50pct_rank(model, device, encs, row_df, lo=lo, hi=hi)
        dnn_ranks.append(cutoff)

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  {i+1}/{total}  last: {row['college'][:35]!r:37s}  "
                  f"raw={raw_cr:6,}  dnn={cutoff:6,}")

    df["closing_rank"] = dnn_ranks
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {total} rows → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
