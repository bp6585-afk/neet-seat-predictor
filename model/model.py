"""
Discrete-time survival DNN for NEET seat availability.

Frame: given (rank, college, category, state, quota, round),
predict P(seat still available) — i.e. P(student_rank ≤ closing_rank).

The model learns shared representations across colleges so that
similar-tier colleges with sparse data still get reasonable estimates.
"""

import torch
import torch.nn as nn


class SeatAvailabilityDNN(nn.Module):
    def __init__(
        self,
        n_colleges: int,
        n_states: int,
        n_categories: int,
        n_quotas: int,
        n_rounds: int = 5,
        n_genders: int = 3,      # Male / Female / All
        college_dim: int = 32,
        state_dim: int = 16,
        cat_dim: int = 8,
        quota_dim: int = 8,
        round_dim: int = 4,
        gender_dim: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        # +1 for <UNK> token in each embedding
        self.college_emb  = nn.Embedding(n_colleges   + 1, college_dim, padding_idx=0)
        self.state_emb    = nn.Embedding(n_states      + 1, state_dim,  padding_idx=0)
        self.category_emb = nn.Embedding(n_categories  + 1, cat_dim,   padding_idx=0)
        self.quota_emb    = nn.Embedding(n_quotas      + 1, quota_dim,  padding_idx=0)
        self.round_emb    = nn.Embedding(n_rounds      + 1, round_dim,  padding_idx=0)
        self.gender_emb   = nn.Embedding(n_genders     + 1, gender_dim, padding_idx=0)

        # +2 for log_rank scalar + year scalar
        in_dim = college_dim + state_dim + cat_dim + quota_dim + round_dim + gender_dim + 2

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),   nn.LayerNorm(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),    nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        log_rank:    torch.Tensor,   # (B,) float
        year_norm:   torch.Tensor,   # (B,) float, normalised year
        college_id:  torch.Tensor,   # (B,) long
        state_id:    torch.Tensor,   # (B,) long
        category_id: torch.Tensor,   # (B,) long
        quota_id:    torch.Tensor,   # (B,) long
        round_id:    torch.Tensor,   # (B,) long
        gender_id:   torch.Tensor,   # (B,) long
    ) -> torch.Tensor:               # (B,) float, logits

        x = torch.cat([
            log_rank.unsqueeze(1),
            year_norm.unsqueeze(1),
            self.college_emb(college_id),
            self.state_emb(state_id),
            self.category_emb(category_id),
            self.quota_emb(quota_id),
            self.round_emb(round_id),
            self.gender_emb(gender_id),
        ], dim=1)

        return self.trunk(x).squeeze(1)  # logits, apply sigmoid at inference

    def predict_proba(self, *args, **kwargs) -> torch.Tensor:
        """Convenience: returns probability in [0, 1]."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(*args, **kwargs))
