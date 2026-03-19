"""
two_tower.py
------------
Two-Tower (Dual Encoder) model for candidate retrieval.

Architecture:
  - User tower: embedding lookup + MLP -> dense user vector
  - Item tower: embedding lookup + MLP -> dense item vector
  - Score: dot product of user and item vectors
  - Loss: in-batch softmax (sampled softmax)

Why two-tower for Phase 2?
  Phase 1 models computed similarity at query time over all items.
  Two-tower pre-computes item vectors offline and uses ANN search
  at serving time -- scales to 100M+ items with <20ms latency.

Real-world users: YouTube, Pinterest, TikTok, LinkedIn, Airbnb.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


# --------------------------------------------------------------------------
# 1. Dataset
# --------------------------------------------------------------------------

class InteractionDataset(Dataset):
    """
    Converts (user_id, item_id) pairs to integer indices.

    Two-tower uses implicit feedback -- we only need to know a user
    interacted with an item, not the rating value. Each __getitem__
    returns (user_idx, item_idx). Other items in the batch become
    in-batch negatives automatically.
    """

    def __init__(self, df: pd.DataFrame):
        self.users = sorted(df["user_id"].unique())
        self.items = sorted(df["item_id"].unique())
        self.user2idx = {u: i for i, u in enumerate(self.users)}
        self.item2idx = {it: i for i, it in enumerate(self.items)}

        self.user_indices = torch.tensor(
            [self.user2idx[u] for u in df["user_id"]], dtype=torch.long
        )
        self.item_indices = torch.tensor(
            [self.item2idx[it] for it in df["item_id"]], dtype=torch.long
        )
        print(f"[dataset] {len(self.users)} users, {len(self.items)} items, "
              f"{len(df)} interactions")

    def __len__(self):
        return len(self.user_indices)

    def __getitem__(self, idx):
        return self.user_indices[idx], self.item_indices[idx]

    @property
    def n_users(self):
        return len(self.users)

    @property
    def n_items(self):
        return len(self.items)


# --------------------------------------------------------------------------
# 2. Tower architecture
# --------------------------------------------------------------------------

class Tower(nn.Module):
    """
    Single tower: embedding -> MLP -> L2-normalized output.

    Both user and item towers share this architecture but have
    separate weights. L2 normalization at output makes dot product
    equal cosine similarity, bounded in [-1, 1], which stabilizes
    training and makes ANN indexing predictable.
    """

    def __init__(self, n_entities: int, embed_dim: int = 64,
                 hidden_dim: int = 128, output_dim: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(n_entities, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        x = self.embedding(indices)
        x = self.mlp(x)
        return F.normalize(x, p=2, dim=1)


# --------------------------------------------------------------------------
# 3. Two-tower model
# --------------------------------------------------------------------------

class TwoTowerModel(nn.Module):
    """
    Full two-tower model.

    Forward pass returns (B x B) score matrix for a batch.
    Diagonal = positive pairs. Off-diagonal = in-batch negatives.

    Temperature tau: score = (u . v) / tau
      Lower tau -> sharper distribution -> harder training signal.
      Typical values: 0.05 to 0.1.
    """

    def __init__(self, n_users: int, n_items: int,
                 embed_dim: int = 64, hidden_dim: int = 128,
                 output_dim: int = 64, dropout: float = 0.2,
                 temperature: float = 0.07):
        super().__init__()
        self.user_tower = Tower(n_users, embed_dim, hidden_dim,
                                output_dim, dropout)
        self.item_tower = Tower(n_items, embed_dim, hidden_dim,
                                output_dim, dropout)
        self.temperature = temperature

    def forward(self, user_indices, item_indices):
        user_vecs = self.user_tower(user_indices)  # (B, D)
        item_vecs = self.item_tower(item_indices)  # (B, D)
        # (B, D) x (D, B) -> (B, B) score matrix
        return torch.matmul(user_vecs, item_vecs.T) / self.temperature

    def get_user_vector(self, user_indices):
        with torch.no_grad():
            return self.user_tower(user_indices)

    def get_item_vectors(self, item_indices):
        with torch.no_grad():
            return self.item_tower(item_indices)


# --------------------------------------------------------------------------
# 4. In-batch softmax loss
# --------------------------------------------------------------------------

def in_batch_softmax_loss(scores: torch.Tensor) -> torch.Tensor:
    """
    scores: (B, B) where scores[i,j] = dot(user_i, item_j) / tau

    For each user i, item i is the positive class, all others are
    negatives. Standard cross-entropy where label[i] = i (diagonal).

    With batch_size=128: each user gets 127 free negatives.
    With batch_size=2048: each user gets 2047 negatives (YouTube scale).
    """
    batch_size = scores.shape[0]
    labels = torch.arange(batch_size, device=scores.device)
    return F.cross_entropy(scores, labels)


# --------------------------------------------------------------------------
# 5. Training loop
# --------------------------------------------------------------------------

def train(model, dataset, n_epochs=20, batch_size=128,
          lr=1e-3, weight_decay=1e-5):
    """Train with in-batch softmax loss + cosine LR annealing."""
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )
    device = next(model.parameters()).device
    epoch_losses = []

    print(f"[two_tower] Training: {n_epochs} epochs, "
          f"batch={batch_size}, lr={lr}")

    for epoch in range(1, n_epochs + 1):
        model.train()
        batch_losses = []
        for user_idx, item_idx in loader:
            user_idx = user_idx.to(device)
            item_idx = item_idx.to(device)
            optimizer.zero_grad()
            scores = model(user_idx, item_idx)
            loss = in_batch_softmax_loss(scores)
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
# 6. Evaluation
# --------------------------------------------------------------------------

def build_item_index(model, dataset):
    """
    Pre-compute all item vectors offline.
    In production: runs once per model update, stored in FAISS.
    Returns (n_items, output_dim) tensor.
    """
    all_item_idx = torch.arange(dataset.n_items)
    item_vecs = model.get_item_vectors(all_item_idx)
    print(f"[two_tower] Item index built: {item_vecs.shape}")
    return item_vecs


def recommend_for_user(user_idx, model, item_index,
                       train_item_indices, n=10):
    """Dot product retrieval — replaced by FAISS in ann_index.py."""
    user_vec = model.get_user_vector(torch.tensor([user_idx]))
    scores = (user_vec @ item_index.T).squeeze(0).numpy()
    ranked = np.argsort(scores)[::-1]
    recs = [(int(i), float(scores[i]))
            for i in ranked if i not in train_item_indices]
    return recs[:n]


def evaluate(model, dataset, item_index, train_df, test_df, k=10):
    """HR@K, NDCG@K, Precision@K — same protocol as Phase 1."""
    user_train_items = defaultdict(set)
    for _, row in train_df.iterrows():
        u = dataset.user2idx.get(row["user_id"])
        it = dataset.item2idx.get(row["item_id"])
        if u is not None and it is not None:
            user_train_items[u].add(it)

    user_test_items = defaultdict(set)
    for _, row in test_df.iterrows():
        u = dataset.user2idx.get(row["user_id"])
        it = dataset.item2idx.get(row["item_id"])
        if u is not None and it is not None:
            user_test_items[u].add(it)

    hr, ndcg, prec = [], [], []
    n_covered = 0

    for u_idx, relevant in user_test_items.items():
        recs = recommend_for_user(u_idx, model, item_index,
                                  user_train_items[u_idx], n=k)
        recommended = [idx for idx, _ in recs]
        if recommended:
            n_covered += 1
        hr.append(float(any(it in relevant for it in recommended[:k])))
        dcg  = sum(1.0 / np.log2(r + 2)
                   for r, it in enumerate(recommended[:k]) if it in relevant)
        idcg = sum(1.0 / np.log2(r + 2)
                   for r in range(min(len(relevant), k)))
        ndcg.append(dcg / idcg if idcg > 0 else 0.0)
        prec.append(sum(1 for it in recommended[:k] if it in relevant) / k)

    n_users = len(user_test_items)
    return {
        f"HR@{k}"        : float(np.mean(hr)),
        f"NDCG@{k}"      : float(np.mean(ndcg)),
        f"Precision@{k}" : float(np.mean(prec)),
        "coverage"       : n_covered / n_users if n_users else 0,
        "n_users_eval"   : n_users,
    }


# --------------------------------------------------------------------------
# 7. Sanity check
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

    dataset = InteractionDataset(train_df)

    model = TwoTowerModel(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        embed_dim=64, hidden_dim=128, output_dim=64,
        temperature=0.07,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[two_tower] Parameters: {total_params:,}")

    losses = train(model, dataset, n_epochs=20, batch_size=64, lr=1e-3)
    item_index = build_item_index(model, dataset)

    print()
    for k in [5, 10]:
        res = evaluate(model, dataset, item_index, train_df, test_df, k=k)
        print(f"Two-Tower K={k}: HR={res[f'HR@{k}']:.4f}  "
              f"NDCG={res[f'NDCG@{k}']:.4f}  "
              f"Coverage={res['coverage']:.4f}")

    print("\n--- Phase 1 vs Phase 2 ---")
    print(f"{'Model':<25} {'HR@10':<10} {'NDCG@10':<10} {'Coverage'}")
    print("-" * 55)
    print(f"{'Hybrid (alpha=0.9)':<25} {'0.485':<10} {'0.279':<10} 1.000")
    res10 = evaluate(model, dataset, item_index, train_df, test_df, k=10)
    print(f"{'Two-Tower':<25} {res10['HR@10']:<10.4f} "
          f"{res10['NDCG@10']:<10.4f} {res10['coverage']:.4f}")