"""
ncf.py
------
Neural Collaborative Filtering (NCF) for candidate ranking.

Paper: He et al. (2017) "Neural Collaborative Filtering" WWW.

NCF combines two ideas:
  1. GMF (Generalised Matrix Factorisation) -- element-wise product
     of user and item embeddings. Learns linear interactions.
     Equivalent to MF but trained with gradient descent.

  2. MLP -- concatenated user+item embeddings through deep layers.
     Learns non-linear, complex interactions that dot product cannot.

NeuMF (Neural Matrix Factorisation) fuses both:
  final_score = sigmoid( W * [GMF_output || MLP_output] )

Role in the pipeline:
  Two-tower retrieves top-1000 candidates from millions of items.
  NCF re-ranks those 1000 to produce the final top-10 the user sees.
  NCF is too slow to run on all items -- only on the small candidate set.

Training: binary cross-entropy on positive/negative pairs.
  Positives: observed interactions.
  Negatives: sampled randomly (not in-batch -- NCF is a pointwise model).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


# --------------------------------------------------------------------------
# 1. Dataset -- pointwise with explicit negatives
# --------------------------------------------------------------------------

class NCFDataset(Dataset):
    """
    Pointwise dataset: each sample is (user, item, label).
      label = 1 for observed interactions (positives)
      label = 0 for randomly sampled non-interactions (negatives)

    Why pointwise negatives (not in-batch like two-tower)?
      NCF predicts a relevance score for each (user, item) pair
      independently. It doesn't need to compare items within a batch.
      Pointwise training with BCE loss is simpler and works well for
      re-ranking where you score each candidate individually.

    Negative sampling ratio:
      For each positive, we sample `neg_ratio` negatives.
      neg_ratio=4 is standard from the NCF paper.
      More negatives = harder training, better calibration, slower.
    """

    def __init__(self, df: pd.DataFrame, neg_ratio: int = 4,
                 seed: int = 42):
        self.users = sorted(df["user_id"].unique())
        self.items = sorted(df["item_id"].unique())
        self.user2idx = {u: i for i, u in enumerate(self.users)}
        self.item2idx = {it: i for i, it in enumerate(self.items)}

        # Set of observed (user_idx, item_idx) pairs -- used to
        # avoid sampling false negatives
        self.observed = set(
            zip(
                [self.user2idx[u] for u in df["user_id"]],
                [self.item2idx[it] for it in df["item_id"]],
            )
        )

        rng = np.random.default_rng(seed)
        user_idxs, item_idxs, labels = [], [], []

        for u_str, it_str in zip(df["user_id"], df["item_id"]):
            u = self.user2idx[u_str]
            it = self.item2idx[it_str]

            # Positive sample
            user_idxs.append(u)
            item_idxs.append(it)
            labels.append(1.0)

            # Negative samples
            for _ in range(neg_ratio):
                neg_it = rng.integers(0, len(self.items))
                # Resample if it happens to be a positive
                while (u, neg_it) in self.observed:
                    neg_it = rng.integers(0, len(self.items))
                user_idxs.append(u)
                item_idxs.append(neg_it)
                labels.append(0.0)

        self.user_indices = torch.tensor(user_idxs, dtype=torch.long)
        self.item_indices = torch.tensor(item_idxs, dtype=torch.long)
        self.labels       = torch.tensor(labels, dtype=torch.float32)

        n_pos = len(df)
        n_neg = n_pos * neg_ratio
        print(f"[ncf_dataset] {n_pos} positives, {n_neg} negatives, "
              f"ratio 1:{neg_ratio}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.user_indices[idx],
                self.item_indices[idx],
                self.labels[idx])

    @property
    def n_users(self):
        return len(self.users)

    @property
    def n_items(self):
        return len(self.items)


# --------------------------------------------------------------------------
# 2. GMF component
# --------------------------------------------------------------------------

class GMF(nn.Module):
    """
    Generalised Matrix Factorisation.

    output = user_emb * item_emb  (element-wise product)

    This is equivalent to classic MF but trained end-to-end.
    The element-wise product captures bilinear interactions --
    dimension k is activated if BOTH user and item have high
    value in dimension k.
    """

    def __init__(self, n_users: int, n_items: int, embed_dim: int = 32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user_idx, item_idx):
        u = self.user_emb(user_idx)  # (B, embed_dim)
        v = self.item_emb(item_idx)  # (B, embed_dim)
        return u * v                 # (B, embed_dim) element-wise


# --------------------------------------------------------------------------
# 3. MLP component
# --------------------------------------------------------------------------

class MLP(nn.Module):
    """
    MLP on concatenated user+item embeddings.

    Captures non-linear, higher-order interactions that GMF cannot.
    Architecture follows the NCF paper: halve hidden dim each layer.

    Example with embed_dim=32:
      input : [user_emb || item_emb] = 64-dim
      layer1: 64 -> 64, ReLU, Dropout
      layer2: 64 -> 32, ReLU, Dropout
    """

    def __init__(self, n_users: int, n_items: int,
                 embed_dim: int = 32, layers: list = None,
                 dropout: float = 0.2):
        super().__init__()
        if layers is None:
            layers = [64, 32]

        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

        # Build MLP: input size = embed_dim * 2 (concat)
        mlp_layers = []
        in_size = embed_dim * 2
        for out_size in layers:
            mlp_layers += [
                nn.Linear(in_size, out_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_size = out_size
        self.mlp = nn.Sequential(*mlp_layers)
        self.output_dim = in_size

    def forward(self, user_idx, item_idx):
        u = self.user_emb(user_idx)     # (B, embed_dim)
        v = self.item_emb(item_idx)     # (B, embed_dim)
        x = torch.cat([u, v], dim=1)   # (B, embed_dim*2)
        return self.mlp(x)             # (B, output_dim)


# --------------------------------------------------------------------------
# 4. NeuMF -- fuses GMF and MLP
# --------------------------------------------------------------------------

class NeuMF(nn.Module):
    """
    Neural Matrix Factorisation -- the full NCF model.

    Fuses GMF (linear) and MLP (non-linear) paths:
      output = sigmoid( W * concat([GMF_out, MLP_out]) )

    Why fuse both?
      GMF captures linear interactions efficiently.
      MLP captures non-linear interactions expressively.
      Together they're more powerful than either alone.
      The fusion layer learns how to weight each contribution.

    Output is a scalar in [0, 1] -- predicted interaction probability.
    Trained with Binary Cross-Entropy loss.
    """

    def __init__(self, n_users: int, n_items: int,
                 gmf_dim: int = 32, mlp_embed_dim: int = 32,
                 mlp_layers: list = None, dropout: float = 0.2):
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [64, 32]

        self.gmf = GMF(n_users, n_items, embed_dim=gmf_dim)
        self.mlp = MLP(n_users, n_items, embed_dim=mlp_embed_dim,
                       layers=mlp_layers, dropout=dropout)

        # Fusion layer: GMF output dim + MLP output dim -> 1
        fusion_input_dim = gmf_dim + self.mlp.output_dim
        self.fusion = nn.Linear(fusion_input_dim, 1)
        nn.init.kaiming_uniform_(self.fusion.weight)

    def forward(self, user_idx, item_idx):
        gmf_out = self.gmf(user_idx, item_idx)  # (B, gmf_dim)
        mlp_out = self.mlp(user_idx, item_idx)  # (B, mlp_out_dim)
        fused   = torch.cat([gmf_out, mlp_out], dim=1)
        score   = self.fusion(fused).squeeze(1)
        return torch.sigmoid(score)             # (B,) in [0, 1]


# --------------------------------------------------------------------------
# 5. Training
# --------------------------------------------------------------------------

def train(model, dataset, n_epochs=20, batch_size=256,
          lr=1e-3, weight_decay=1e-5):
    """
    Train NeuMF with Binary Cross-Entropy loss.

    BCE loss: for each (user, item, label) triple:
      loss = -[label * log(score) + (1-label) * log(1-score)]

    Why BCE (not softmax like two-tower)?
      NCF is a pointwise model -- it scores each (user, item) pair
      independently. BCE treats this as binary classification:
      "will this user interact with this item: yes or no?"
      Two-tower's softmax needs to compare items within a batch,
      which requires all items to be seen together.
    """
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )

    print(f"[ncf] Training NeuMF: {n_epochs} epochs, "
          f"batch={batch_size}, lr={lr}")

    epoch_losses = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        batch_losses = []

        for user_idx, item_idx, labels in loader:
            optimizer.zero_grad()
            scores = model(user_idx, item_idx)
            loss   = F.binary_cross_entropy(scores, labels)
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

def recommend_for_user(user_idx, model, dataset,
                       train_item_indices, n=10):
    """
    Score all items for a user and return top-N.

    In the real pipeline, NCF only scores the ~1000 candidates
    returned by two-tower retrieval -- not all items.
    We score all items here for evaluation purposes.
    """
    model.eval()
    with torch.no_grad():
        all_items = torch.arange(dataset.n_items)
        user_tensor = torch.full((dataset.n_items,), user_idx,
                                 dtype=torch.long)
        scores = model(user_tensor, all_items).numpy()

    ranked = np.argsort(scores)[::-1]
    recs = [(int(i), float(scores[i]))
            for i in ranked if i not in train_item_indices]
    return recs[:n]


def evaluate(model, dataset, train_df, test_df, k=10):
    """HR@K, NDCG@K, Precision@K -- same protocol as Phase 1."""
    user_train_items = defaultdict(set)
    for _, row in train_df.iterrows():
        u  = dataset.user2idx.get(row["user_id"])
        it = dataset.item2idx.get(row["item_id"])
        if u is not None and it is not None:
            user_train_items[u].add(it)

    user_test_items = defaultdict(set)
    for _, row in test_df.iterrows():
        u  = dataset.user2idx.get(row["user_id"])
        it = dataset.item2idx.get(row["item_id"])
        if u is not None and it is not None:
            user_test_items[u].add(it)

    hr, ndcg, prec = [], [], []
    n_covered = 0

    for u_idx, relevant in user_test_items.items():
        recs = recommend_for_user(u_idx, model, dataset,
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

    dataset = NCFDataset(train_df, neg_ratio=4)

    model = NeuMF(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        gmf_dim=32,
        mlp_embed_dim=32,
        mlp_layers=[64, 32],
        dropout=0.2,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[ncf] NeuMF parameters: {total_params:,}")

    losses = train(model, dataset, n_epochs=20,
                   batch_size=256, lr=1e-3)

    print()
    for k in [5, 10]:
        res = evaluate(model, dataset, train_df, test_df, k=k)
        print(f"NeuMF K={k}: HR={res[f'HR@{k}']:.4f}  "
              f"NDCG={res[f'NDCG@{k}']:.4f}  "
              f"Coverage={res['coverage']:.4f}")

    print("\n--- Phase 2 comparison ---")
    print(f"{'Model':<25} {'HR@10':<10} {'NDCG@10':<10} {'Coverage'}")
    print("-" * 55)
    print(f"{'Two-Tower':<25} {'0.121':<10} {'0.051':<10} 1.000")
    res10 = evaluate(model, dataset, train_df, test_df, k=10)
    print(f"{'NeuMF':<25} {res10['HR@10']:<10.4f} "
          f"{res10['NDCG@10']:<10.4f} {res10['coverage']:.4f}")