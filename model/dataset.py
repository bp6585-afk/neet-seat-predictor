"""
Converts the scraped unified.csv into PyTorch-ready training samples.

Survival framing
────────────────
Each row in the CSV has a closing_rank.  We expand it into many
(rank_sample, label) pairs:
  label = 1  if rank_sample ≤ closing_rank  (seat available)
  label = 0  if rank_sample > closing_rank  (seat gone)

We sample rank values from the full observed rank distribution so the
model sees realistic rank frequencies.
"""

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
ENCODER_PATH  = PROCESSED_DIR / "encoders.json"
ROUND_ORDER   = {"Round 1": 1, "Round 2": 2, "Mop-up Round": 3, "Stray Round": 4}


class LabelEncoder:
    """Minimal string→int encoder that handles unseen labels with index 0."""

    def __init__(self):
        self.classes_: list[str] = []
        self._map: dict[str, int] = {}

    def fit(self, values):
        unique = sorted(set(str(v) for v in values if pd.notna(v)))
        self.classes_ = unique
        self._map = {v: i + 1 for i, v in enumerate(unique)}  # 0 reserved for UNK
        return self

    def transform(self, values):
        return [self._map.get(str(v), 0) for v in values]

    def to_dict(self):
        return {"classes": self.classes_}

    @classmethod
    def from_dict(cls, d):
        enc = cls()
        enc.classes_ = d["classes"]
        enc._map = {v: i + 1 for i, v in enumerate(enc.classes_)}
        return enc

    def __len__(self):
        return len(self.classes_)


class NEETDataset(Dataset):
    """
    Parameters
    ----------
    df          : unified DataFrame (already loaded)
    encoders    : dict of LabelEncoder, optional — if None, fit from df
    samples_per_row : how many rank samples to generate per closing-rank row
    """

    def __init__(
        self,
        df: pd.DataFrame,
        encoders=None,
        samples_per_row: int = 8,
        rng_seed: int = 42,
    ):
        rng = np.random.default_rng(rng_seed)

        df = df.copy()
        df["round_num"] = df["round"].map(ROUND_ORDER).fillna(4).astype(int)
        df["closing_rank"] = df["closing_rank"].astype(int)

        if "gender" not in df.columns:
            df["gender"] = "All"

        if encoders is None:
            self.enc_college  = LabelEncoder().fit(df["college"])
            self.enc_state    = LabelEncoder().fit(df["state"])
            self.enc_category = LabelEncoder().fit(df["category"])
            self.enc_quota    = LabelEncoder().fit(df["quota"])
            self.enc_gender   = LabelEncoder().fit(df["gender"])
        else:
            self.enc_college  = encoders["college"]
            self.enc_state    = encoders["state"]
            self.enc_category = encoders["category"]
            self.enc_quota    = encoders["quota"]
            self.enc_gender   = encoders.get("gender", LabelEncoder().fit(df["gender"]))

        self.n_colleges   = len(self.enc_college)
        self.n_states     = len(self.enc_state)
        self.n_categories = len(self.enc_category)
        self.n_quotas     = len(self.enc_quota)
        self.n_genders    = len(self.enc_gender)

        # ── Generate training samples ──────────────────────────────────────
        all_closing = df["closing_rank"].values
        rank_pool = np.concatenate([
            all_closing,
            rng.integers(1, all_closing.max() + 1, size=len(all_closing) * 3),
        ])

        rows = []
        for _, rec in df.iterrows():
            cr = int(rec["closing_rank"])
            # balanced: half below (positive), half above (negative)
            n_pos = samples_per_row // 2
            n_neg = samples_per_row - n_pos

            pos_ranks = rng.integers(max(1, cr - cr), cr + 1, size=n_pos)
            neg_ranks = rng.integers(cr + 1, min(cr * 4, 800_001), size=n_neg)

            for rank_val, label in [*zip(pos_ranks, [1] * n_pos),
                                     *zip(neg_ranks, [0] * n_neg)]:
                rows.append({
                    "log_rank":    np.log1p(rank_val),
                    "year_norm":   (int(rec["year"]) - 2020) / 5.0,
                    "college_id":  self.enc_college.transform([rec["college"]])[0],
                    "state_id":    self.enc_state.transform([rec["state"]])[0],
                    "category_id": self.enc_category.transform([rec["category"]])[0],
                    "quota_id":    self.enc_quota.transform([rec["quota"]])[0],
                    "round_id":    min(int(rec["round_num"]), 4),
                    "gender_id":   self.enc_gender.transform([rec.get("gender", "All")])[0],
                    "label":       float(label),
                })

        self._df = pd.DataFrame(rows)

    def save_encoders(self, path: Path = ENCODER_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "college":  self.enc_college.to_dict(),
            "state":    self.enc_state.to_dict(),
            "category": self.enc_category.to_dict(),
            "quota":    self.enc_quota.to_dict(),
            "gender":   self.enc_gender.to_dict(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Encoders saved → {path}")

    @staticmethod
    def load_encoders(path: Path = ENCODER_PATH) -> dict:
        with open(path) as f:
            data = json.load(f)
        return {k: LabelEncoder.from_dict(v) for k, v in data.items()}

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        row = self._df.iloc[idx]
        return {
            "log_rank":    torch.tensor(row["log_rank"],    dtype=torch.float32),
            "year_norm":   torch.tensor(row["year_norm"],   dtype=torch.float32),
            "college_id":  torch.tensor(row["college_id"],  dtype=torch.long),
            "state_id":    torch.tensor(row["state_id"],    dtype=torch.long),
            "category_id": torch.tensor(row["category_id"], dtype=torch.long),
            "quota_id":    torch.tensor(row["quota_id"],    dtype=torch.long),
            "round_id":    torch.tensor(row["round_id"],    dtype=torch.long),
            "gender_id":   torch.tensor(row["gender_id"],   dtype=torch.long),
            "label":       torch.tensor(row["label"],       dtype=torch.float32),
        }
