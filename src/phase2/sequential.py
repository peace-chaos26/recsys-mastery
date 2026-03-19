"""
sequential.py
-------------
SASRec-lite: Self-Attentive Sequential Recommendation.

Paper: Kang & McAuley (2018) "Self-Attentive Sequential Recommendation" ICDM.

Key ideas:
  1. Positional embeddings  -- model knows item order
  2. Causal self-attention  -- each position attends to past items only
  3. Last-position output   -- encodes full sequential context for prediction

Why sequential models?
  MF and NCF aggregate history into one vector -- order is lost.
  SASRec treats history as an ordered sequence and learns which
  past items to focus on when predicting the next one.
  Recent items naturally get higher attention weights.

Use case: "next item prediction" -- what will the user interact
  with next, given their recent activity? This is the core problem
  at Netflix (next show), Spotify (next song), TikTok (next video).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


# --------------------------------------------------------------------------
# 1. Dataset -- sequential sessions
# --------------------------------------------------------------------------

class SequentialDataset(Dataset):
    """
    Builds fixed-length interaction sequences per user.

    For each user with interactions [i1, i2, i3, i4, i5]:
      We create training samples by sliding a window:
        input=[i1,i2,i3], target=i4
        input=[i2,i3,i4], target=i5

    max_seq_len: maximum sequence length to consider.
      Longer sequences are truncated to the most recent max_seq_len items.
      Shorter sequences are left-padded with 0 (pad token).

    Why fixed length?
      Transformer self-attention is O(L^2) in sequence length.
      Capping at max_seq_len=50 keeps memory and compute bounded.
      In practice, most users have <50 interactions anyway.
    """

    def __init__(self, df: pd.DataFrame, max_seq_len: int = 50):
        self.max_seq_len = max_seq_len

        # Build index maps -- 0 is reserved for padding
        self.items = sorted(df["item_id"].unique())
        self.item2idx = {it: i + 1 for i, it in enumerate(self.items)}
        self.n_items = len(self.items) + 1  # +1 for pad token

        self.users = sorted(df["user_id"].unique())
        self.user2idx = {u: i for i, u in enumerate(self.users)}

        # Sort each user's interactions by row order (proxy for time)
        # In production you'd sort by actual timestamp
        user_sequences = defaultdict(list)
        for _, row in df.iterrows():
            user_sequences[row["user_id"]].append(
                self.item2idx[row["item_id"]]
            )

        # Build (input_seq, target) pairs
        self.sequences = []   # input sequences (padded to max_seq_len)
        self.targets   = []   # next item to predict
        self.user_idxs = []   # which user (for evaluation)

        for user_id, item_seq in user_sequences.items():
            u_idx = self.user2idx[user_id]
            # Need at least 2 items to form one training pair
            if len(item_seq) < 2:
                continue
            # Slide window over full sequence
            for end in range(2, len(item_seq) + 1):
                target = item_seq[end - 1]
                seq    = item_seq[max(0, end - 1 - max_seq_len): end - 1]
                # Left-pad with zeros to max_seq_len
                padded = [0] * (max_seq_len - len(seq)) + seq
                self.sequences.append(padded)
                self.targets.append(target)
                self.user_idxs.append(u_idx)

        self.sequences = torch.tensor(self.sequences, dtype=torch.long)
        self.targets   = torch.tensor(self.targets,   dtype=torch.long)
        self.user_idxs = torch.tensor(self.user_idxs, dtype=torch.long)

        print(f"[seq_dataset] {len(self.users)} users, "
              f"{len(self.items)} items (+1 pad), "
              f"{len(self.sequences)} training sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


# --------------------------------------------------------------------------
# 2. Multi-head self-attention block
# --------------------------------------------------------------------------

class SelfAttentionBlock(nn.Module):
    """
    Single transformer block: multi-head attention + FFN + layer norm.

    Causal masking: attention mask is a lower-triangular matrix.
    Position j can only attend to positions <= j.
    This prevents the model from seeing future items during training.

    Why multi-head?
      Different heads learn to attend to different aspects of history.
      One head might focus on recent items, another on category patterns,
      another on price-level patterns -- all in parallel.
    """

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,   # (B, L, D) convention
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, L, D)
        padding_mask: (B, L) — True where position is a pad token

        Returns: (B, L, D)
        """
        B, L, D = x.shape

        # Causal mask: (L, L) lower triangular
        # attn_mask[i,j] = -inf means position i cannot attend to j
        causal_mask = torch.triu(
            torch.ones(L, L, device=x.device) * float("-inf"),
            diagonal=1,
        )

        # Pre-norm architecture (more stable than post-norm)
        # Attention with residual connection
        normed = self.norm1(x)
        attn_out, _ = self.attention(
            normed, normed, normed,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
        )
        x = x + self.dropout(attn_out)

        # FFN with residual connection
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# --------------------------------------------------------------------------
# 3. SASRec model
# --------------------------------------------------------------------------

class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation (lite version).

    Architecture:
      item_embedding(n_items, hidden_dim)
      + positional_embedding(max_seq_len, hidden_dim)
      → dropout
      → N x SelfAttentionBlock
      → LayerNorm
      → last position vector
      → dot product with all item embeddings
      → scores over all items

    The last position vector encodes "what does this user want next?"
    given their entire sequence history. We dot it against all item
    embeddings to score every candidate.

    Training task: predict the next item in the sequence.
    Loss: cross-entropy over all items (including negatives).
    """

    def __init__(self, n_items: int, hidden_dim: int = 64,
                 n_layers: int = 2, n_heads: int = 2,
                 max_seq_len: int = 50, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim   = hidden_dim
        self.max_seq_len  = max_seq_len

        # Item embedding -- index 0 is padding (kept as zeros via padding_idx)
        self.item_emb = nn.Embedding(n_items, hidden_dim, padding_idx=0)

        # Positional embedding -- one vector per position in the sequence
        self.pos_emb = nn.Embedding(max_seq_len + 1, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight,  std=0.02)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq: (B, L) item index sequences, 0 = padding

        Returns: (B, hidden_dim) -- last position representation
        """
        B, L = seq.shape

        # Padding mask: True where seq == 0 (pad positions)
        # MultiheadAttention ignores these positions
        padding_mask = (seq == 0)  # (B, L)

        # Item embeddings
        x = self.item_emb(seq)  # (B, L, D)

        # Add positional embeddings
        # positions: 1, 2, ..., L (0 reserved for pad)
        positions = torch.arange(1, L + 1, device=seq.device).unsqueeze(0)
        x = x + self.pos_emb(positions)  # (B, L, D)
        x = self.dropout(x)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, padding_mask)

        x = self.final_norm(x)  # (B, L, D)

        # Return last non-padding position
        # Find last real item in each sequence
        seq_lengths = (seq != 0).sum(dim=1) - 1  # (B,) 0-indexed
        seq_lengths = seq_lengths.clamp(min=0)
        last = x[torch.arange(B), seq_lengths]   # (B, D)
        return last

    def score_items(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Score all items for a batch of sequences.
        Returns (B, n_items) logits.
        """
        last = self.forward(seq)          # (B, D)
        # Dot with all item embeddings: (B, D) x (D, n_items) -> (B, n_items)
        return last @ self.item_emb.weight.T


# --------------------------------------------------------------------------
# 4. Training
# --------------------------------------------------------------------------

def train(model, dataset, n_epochs=20, batch_size=128,
          lr=1e-3, weight_decay=1e-5):
    """
    Train SASRec with cross-entropy loss over all items.

    Loss: cross-entropy where the target is the next item index.
    This is equivalent to in-batch softmax but over ALL items,
    not just items in the batch -- much harder task, better model.
    """
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )

    print(f"[sasrec] Training: {n_epochs} epochs, "
          f"batch={batch_size}, lr={lr}")

    epoch_losses = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        batch_losses = []

        for seqs, targets in loader:
            optimizer.zero_grad()
            logits = model.score_items(seqs)   # (B, n_items)
            loss   = F.cross_entropy(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(loss.item())

        scheduler.step()
        epoch_loss = np.mean(batch_losses)
        epoch_losses.append(epoch_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{n_epochs}  "
                  f"loss={epoch_loss:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.5f}")

    return epoch_losses


# --------------------------------------------------------------------------
# 5. Evaluation
# --------------------------------------------------------------------------

def get_user_sequence(user_id, train_df, dataset):
    """Build the input sequence for a user from their training history."""
    user_items = train_df[train_df["user_id"] == user_id]["item_id"].tolist()
    item_indices = [dataset.item2idx.get(it, 0) for it in user_items]
    L = dataset.max_seq_len
    seq = item_indices[-L:]                    # take last L items
    padded = [0] * (L - len(seq)) + seq        # left-pad
    return torch.tensor([padded], dtype=torch.long)


def evaluate(model, dataset, train_df, test_df, k=10):
    """HR@K, NDCG@K, Precision@K for next-item prediction."""
    model.eval()

    user_test_items = defaultdict(set)
    for _, row in test_df.iterrows():
        it = dataset.item2idx.get(row["item_id"])
        if it is not None:
            user_test_items[row["user_id"]].add(it)

    user_train_items = defaultdict(set)
    for _, row in train_df.iterrows():
        it = dataset.item2idx.get(row["item_id"])
        if it is not None:
            user_train_items[row["user_id"]].add(it)

    hr, ndcg, prec = [], [], []
    n_covered = 0

    for user_id, relevant in user_test_items.items():
        seq = get_user_sequence(user_id, train_df, dataset)

        with torch.no_grad():
            scores = model.score_items(seq).squeeze(0).numpy()  # (n_items,)

        # Mask out pad token and training items
        scores[0] = -np.inf
        for it in user_train_items[user_id]:
            scores[it] = -np.inf

        ranked = np.argsort(scores)[::-1][:k]
        recommended = ranked.tolist()

        if recommended:
            n_covered += 1

        hr.append(float(any(it in relevant for it in recommended)))
        dcg  = sum(1.0 / np.log2(r + 2)
                   for r, it in enumerate(recommended) if it in relevant)
        idcg = sum(1.0 / np.log2(r + 2)
                   for r in range(min(len(relevant), k)))
        ndcg.append(dcg / idcg if idcg > 0 else 0.0)
        prec.append(sum(1 for it in recommended if it in relevant) / k)

    n_users = len(user_test_items)
    return {
        f"HR@{k}"        : float(np.mean(hr)),
        f"NDCG@{k}"      : float(np.mean(ndcg)),
        f"Precision@{k}" : float(np.mean(prec)),
        "coverage"       : n_covered / n_users if n_users else 0,
        "n_users_eval"   : n_users,
    }


# --------------------------------------------------------------------------
# 6. Sanity check
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_loader import (download_data, load_ratings,
                             filter_min_interactions, train_test_split)

    fp = download_data()
    df = load_ratings(fp)
    df = filter_min_interactions(df)
    train_df, test_df = train_test_split(df)

    dataset = SequentialDataset(train_df, max_seq_len=50)

    model = SASRec(
        n_items=dataset.n_items,
        hidden_dim=64,
        n_layers=2,
        n_heads=2,
        max_seq_len=50,
        dropout=0.2,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[sasrec] Parameters: {total_params:,}")

    losses = train(model, dataset, n_epochs=20,
                   batch_size=128, lr=1e-3)

    print()
    for k in [5, 10]:
        res = evaluate(model, dataset, train_df, test_df, k=k)
        print(f"SASRec K={k}: HR={res[f'HR@{k}']:.4f}  "
              f"NDCG={res[f'NDCG@{k}']:.4f}  "
              f"Coverage={res['coverage']:.4f}")

    print("\n--- Phase 2 model comparison ---")
    print(f"{'Model':<25} {'HR@10':<10} {'NDCG@10':<10} {'Coverage'}")
    print("-" * 55)
    print(f"{'Two-Tower':<25} {'0.121':<10} {'0.051':<10} 1.000")
    print(f"{'NeuMF':<25} {'0.441':<10} {'0.261':<10} 1.000")
    res10 = evaluate(model, dataset, train_df, test_df, k=10)
    print(f"{'SASRec':<25} {res10['HR@10']:<10.4f} "
          f"{res10['NDCG@10']:<10.4f} {res10['coverage']:.4f}")