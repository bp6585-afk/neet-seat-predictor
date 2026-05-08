"""
Inference utilities — used by the Streamlit app.

Two modes:
1. DNN mode   — loads trained model + encoders, runs forward pass
2. Lookup mode — uses scraped CSV directly with a logistic buffer
                 (works before model is trained)
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    from model.model import SeatAvailabilityDNN
    from model.dataset import NEETDataset, LabelEncoder, ROUND_ORDER
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    # Minimal stubs so the rest of the file parses cleanly
    class LabelEncoder:
        pass
    ROUND_ORDER = {"Round 1": 1, "Round 2": 2, "Mop-up Round": 3, "Stray Round": 4}

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_DIR     = Path(__file__).parent.parent / "model" / "checkpoints"
ENCODER_PATH  = PROCESSED_DIR / "encoders.json"
MODEL_PATH    = MODEL_DIR / "seat_dnn.pt"
META_PATH     = MODEL_DIR / "meta.json"

ROUNDS = ["Round 1", "Round 2", "Mop-up Round", "Stray Round"]
ROUND_LABEL = {1: "R1", 2: "R2", 3: "Mop-up", 4: "Stray"}


def _logistic_chance(student_rank: int, closing_rank: int, buffer_frac: float = 0.12) -> float:
    """
    Smooth probability around the closing rank using a logistic curve.
    buffer_frac controls how steeply the curve drops — 0.12 → ~12% of rank range.
    """
    if closing_rank <= 0:
        return 0.0
    spread = max(closing_rank * buffer_frac, 500)
    z = (closing_rank - student_rank) / spread
    return float(1 / (1 + math.exp(-z)))


class Predictor:
    """
    Unified predictor that auto-detects DNN vs lookup mode.
    """

    def __init__(self):
        self._model  = None
        self._encs   = None
        self._device = None
        self._df     = None
        self._mode   = "none"
        self._load()

    def _load(self):
        dnn_lookup = PROCESSED_DIR / "unified_dnn.csv"
        unified    = PROCESSED_DIR / "unified.csv"
        if dnn_lookup.exists():
            self._df = pd.read_csv(dnn_lookup)
            self._mode = "lookup"
            print("Predictor: DNN-lookup mode (unified_dnn.csv)")
        elif unified.exists():
            self._df = pd.read_csv(unified)
            self._mode = "lookup"

        if _TORCH_AVAILABLE and MODEL_PATH.exists() and META_PATH.exists() and ENCODER_PATH.exists():
            try:
                with open(META_PATH) as f:
                    meta = json.load(f)
                self._encs = NEETDataset.load_encoders(ENCODER_PATH)
                self._device = torch.device(
                    "mps" if torch.backends.mps.is_available()
                    else "cuda" if torch.cuda.is_available() else "cpu"
                )
                m = SeatAvailabilityDNN(
                    n_colleges   = meta["n_colleges"],
                    n_states     = meta["n_states"],
                    n_categories = meta["n_categories"],
                    n_quotas     = meta["n_quotas"],
                    n_genders    = meta.get("n_genders", 3),
                )
                m.load_state_dict(torch.load(MODEL_PATH, map_location=self._device))
                m.eval()
                self._model = m.to(self._device)
                self._mode  = "dnn"
                print("Predictor: DNN mode")
            except Exception as e:
                print(f"Predictor: DNN load failed ({e}), falling back to lookup")
        else:
            if self._mode == "lookup":
                print("Predictor: Lookup mode (model not trained yet)")

    @property
    def mode(self):
        return self._mode

    @property
    def available_states(self) -> list[str]:
        if self._df is None:
            return []
        return sorted(self._df["state"].unique().tolist())

    @property
    def available_categories(self) -> list[str]:
        if self._df is None:
            return ["General", "OBC", "EWS", "SC", "ST"]
        return sorted(self._df["category"].unique().tolist())

    @property
    def available_quotas(self) -> list[str]:
        if self._df is None:
            return []
        return sorted(self._df["quota"].unique().tolist())

    # ── Main prediction entry point ────────────────────────────────────────
    def predict(
        self,
        student_rank: int,
        category: str,
        state: str,
        quota=None,
        gender: str = "All",
        year: int = 2025,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with columns:
          college | state | quota | category | round | closing_rank_2025
          | chance_pct | best_round | best_round_num
        """
        if self._df is None:
            return pd.DataFrame()

        df = self._df.copy()
        if "gender" not in df.columns:
            df["gender"] = "All"

        # Filter by category / state / quota
        mask = df["category"] == category
        if state != "All States":
            mask &= df["state"] == state
        if quota and quota != "All":
            mask &= df["quota"] == quota

        # Gender filter: "All" rows always show; specific-gender rows only show
        # when the student's gender matches.  e.g. Male student sees All + Male rows.
        if gender != "All":
            mask &= df["gender"].isin(["All", gender])
        # else: show everything (All + Male + Female rows)

        subset = df[mask].copy()
        if subset.empty:
            return pd.DataFrame()

        if self._mode == "dnn":
            subset = self._dnn_predict(subset, student_rank, year)
        else:
            subset = self._lookup_predict(subset, student_rank)

        # Add gender column to output if missing
        if "gender" not in subset.columns:
            subset["gender"] = "All"

        # Best round: earliest round where chance > 50 %
        subset["round_num"] = subset["round"].map(ROUND_ORDER).fillna(4).astype(int)
        subset = subset.sort_values(["college", "round_num"])

        best = (
            subset[subset["chance_pct"] >= 50]
            .groupby("college")["round_num"]
            .min()
            .reset_index()
            .rename(columns={"round_num": "best_round_num"})
        )
        subset = subset.merge(best, on="college", how="left")
        subset["best_round_num"] = subset["best_round_num"].fillna(5).astype(int)
        subset["best_round"] = subset["best_round_num"].map(ROUND_LABEL).fillna("Unlikely")

        # Keep one row per (college, round, gender) — keep max chance if duplicates
        subset = (
            subset.sort_values("chance_pct", ascending=False)
            .drop_duplicates(subset=["college", "round", "gender"])
            .reset_index(drop=True)
        )

        return subset[[
            "college", "state", "quota", "category", "gender", "round",
            "closing_rank", "chance_pct", "best_round", "best_round_num",
        ]]

    def _lookup_predict(self, df: pd.DataFrame, student_rank: int) -> pd.DataFrame:
        df["chance_pct"] = df["closing_rank"].apply(
            lambda cr: round(_logistic_chance(student_rank, cr) * 100, 1)
        )
        return df

    def _dnn_predict(self, df: pd.DataFrame, student_rank: int, year: int) -> pd.DataFrame:
        log_r   = math.log1p(student_rank)
        yr_norm = (year - 2020) / 5.0

        gender_col = df["gender"] if "gender" in df.columns else ["All"] * len(df)
        c_ids  = self._encs["college"].transform(df["college"])
        s_ids  = self._encs["state"].transform(df["state"])
        ca_ids = self._encs["category"].transform(df["category"])
        q_ids  = self._encs["quota"].transform(df["quota"])
        r_ids  = [min(ROUND_ORDER.get(r, 4), 4) for r in df["round"]]
        g_ids  = self._encs.get("gender", None)
        if g_ids is not None:
            g_ids = g_ids.transform(gender_col)
        else:
            g_ids = [0] * len(df)

        n = len(df)
        with torch.no_grad():
            probs = self._model.predict_proba(
                torch.full((n,), log_r,   dtype=torch.float32).to(self._device),
                torch.full((n,), yr_norm, dtype=torch.float32).to(self._device),
                torch.tensor(c_ids,  dtype=torch.long).to(self._device),
                torch.tensor(s_ids,  dtype=torch.long).to(self._device),
                torch.tensor(ca_ids, dtype=torch.long).to(self._device),
                torch.tensor(q_ids,  dtype=torch.long).to(self._device),
                torch.tensor(r_ids,  dtype=torch.long).to(self._device),
                torch.tensor(g_ids,  dtype=torch.long).to(self._device),
            ).cpu().numpy()

        df = df.copy()
        df["chance_pct"] = np.round(probs * 100, 1)
        return df

    # ── Survival curve data for a single college ───────────────────────────
    def survival_curve(
        self,
        college: str,
        category: str,
        quota: str,
        state: str,
        year: int = 2025,
        n_points: int = 200,
    ) -> pd.DataFrame:
        """
        Returns DataFrame(rank, round, chance_pct) — one curve per round.
        """
        if self._df is None:
            return pd.DataFrame()

        rows = self._df[
            (self._df["college"]  == college)  &
            (self._df["category"] == category) &
            (self._df["quota"]    == quota)
        ]
        if rows.empty:
            return pd.DataFrame()

        results = []
        for _, rec in rows.iterrows():
            cr = int(rec["closing_rank"])
            rank_range = np.logspace(
                np.log10(max(1, cr // 5)),
                np.log10(min(800_000, cr * 3)),
                n_points,
            ).astype(int)

            if self._mode == "dnn":
                log_ranks = np.log1p(rank_range)
                yr_norm   = (year - 2020) / 5.0
                n         = len(rank_range)
                c_id  = self._encs["college"].transform([college])[0]
                s_id  = self._encs["state"].transform([state])[0]
                ca_id = self._encs["category"].transform([category])[0]
                q_id  = self._encs["quota"].transform([quota])[0]
                r_id  = min(ROUND_ORDER.get(rec["round"], 4), 4)
                rec_gender = rec.get("gender", "All") if hasattr(rec, "get") else "All"
                g_enc = self._encs.get("gender", None)
                g_id  = g_enc.transform([rec_gender])[0] if g_enc else 0
                with torch.no_grad():
                    probs = self._model.predict_proba(
                        torch.tensor(log_ranks, dtype=torch.float32).to(self._device),
                        torch.full((n,), yr_norm, dtype=torch.float32).to(self._device),
                        torch.full((n,), c_id,  dtype=torch.long).to(self._device),
                        torch.full((n,), s_id,  dtype=torch.long).to(self._device),
                        torch.full((n,), ca_id, dtype=torch.long).to(self._device),
                        torch.full((n,), q_id,  dtype=torch.long).to(self._device),
                        torch.full((n,), r_id,  dtype=torch.long).to(self._device),
                        torch.full((n,), g_id,  dtype=torch.long).to(self._device),
                    ).cpu().numpy()
            else:
                probs = np.array([_logistic_chance(int(r), cr) for r in rank_range])

            for rank_val, prob in zip(rank_range, probs):
                results.append({
                    "rank":       int(rank_val),
                    "round":      rec["round"],
                    "chance_pct": round(float(prob) * 100, 1),
                })

        return pd.DataFrame(results)
