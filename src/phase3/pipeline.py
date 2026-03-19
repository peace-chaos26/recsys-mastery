"""
pipeline.py
-----------
Multi-stage recommendation pipeline.

Stage 1 — Retrieval   : Two-Tower + FAISS → top-500 candidates
Stage 2 — Scoring     : NeuMF pointwise scoring → ranked shortlist
Stage 3 — Re-ranking  : MMR diversity + business rules → final top-N

This is the standard production architecture used at:
  YouTube  : candidate generation → ranking → post-processing
  LinkedIn : ANN retrieval → GBT scoring → diversity filter
  Pinterest: two-tower → PointWise NN → re-rank

Why three stages?
  Retrieval must be fast (ms) so it's approximate — high recall, low precision.
  Scoring is more expensive so it only runs on the small candidate set.
  Re-ranking applies constraints (diversity, freshness) scoring can't express.
  Each stage narrows the funnel: millions → 500 → 50 → 10.

Key design principle: each stage optimises a different objective.
  Retrieval  : maximise recall (don't miss good items)
  Scoring    : maximise precision (rank good items higher)
  Re-ranking : maximise utility (balance relevance + diversity + constraints)
"""

import numpy as np
import pandas as pd
import torch
import time
from collections import defaultdict
from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# 1. Pipeline config
# --------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    Controls the funnel sizes and behaviour at each stage.

    retrieval_k   : candidates from two-tower retrieval
    scoring_k     : top items passed from scoring to re-ranking
    final_k       : final recommendations shown to user
    diversity_lambda: MMR trade-off (0 = pure relevance, 1 = pure diversity)
    use_debiasing : apply IPS position bias correction in scoring
    """
    retrieval_k      : int   = 100
    scoring_k        : int   = 20
    final_k          : int   = 10
    diversity_lambda : float = 0.3
    use_debiasing    : bool  = True


# --------------------------------------------------------------------------
# 2. Retrieval stage
# --------------------------------------------------------------------------

def retrieval_stage(
    user_idx: int,
    user_vector: np.ndarray,
    faiss_index,
    train_item_indices: set,
    k: int = 100,
) -> list[int]:
    """
    Stage 1: Fast approximate retrieval using FAISS.

    Returns top-k item indices, excluding already-seen items.
    This is the only stage that touches the full item catalogue.

    user_vector: (1, dim) L2-normalized user embedding from two-tower
    """
    # Search k + buffer to account for filtering already-seen items
    buffer = len(train_item_indices) + k
    distances, indices = faiss_index.search(
        user_vector.astype(np.float32), buffer
    )
    # Filter already-seen items, keep top-k
    candidates = [
        int(idx) for idx in indices[0]
        if idx >= 0 and idx not in train_item_indices
    ]
    return candidates[:k]


# --------------------------------------------------------------------------
# 3. Scoring stage
# --------------------------------------------------------------------------

def scoring_stage(
    user_idx: int,
    candidate_item_indices: list[int],
    ncf_model,
    ncf_dataset,
    position_propensities: np.ndarray = None,
) -> list[tuple[int, float]]:
    """
    Stage 2: NeuMF pointwise scoring of candidates.

    Scores each candidate item individually using the trained NeuMF model.
    Optionally applies IPS debiasing if position_propensities is provided.

    position_propensities: array of P(seen | position k) for k=1..K
      Used to correct for the fact that items shown at higher positions
      in historical data received more clicks regardless of quality.

    Returns list of (item_idx, score) sorted descending.
    """
    if not candidate_item_indices:
        return []

    ncf_model.eval()
    with torch.no_grad():
        user_tensor = torch.full(
            (len(candidate_item_indices),), user_idx, dtype=torch.long
        )
        item_tensor = torch.tensor(candidate_item_indices, dtype=torch.long)

        # NeuMF scores in [0, 1]
        scores = ncf_model(user_tensor, item_tensor).numpy()

    # IPS debiasing: upweight items that appear at lower positions
    # in historical data (they were seen less often → less biased signal)
    if position_propensities is not None:
        n = len(scores)
        # Assign each candidate a position propensity based on its rank
        propensities = np.array([
            position_propensities[min(i, len(position_propensities) - 1)]
            for i in range(n)
        ])
        # IPS weight: 1 / P(seen at position i)
        # Clip to avoid extreme weights
        ips_weights = np.clip(1.0 / propensities, 1.0, 10.0)
        scores = scores * ips_weights

    scored = list(zip(candidate_item_indices, scores.tolist()))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# --------------------------------------------------------------------------
# 4. Re-ranking stage (MMR diversity)
# --------------------------------------------------------------------------

def reranking_stage(
    scored_candidates: list[tuple[int, float]],
    item_vectors: np.ndarray,
    k: int = 10,
    lambda_: float = 0.3,
) -> list[tuple[int, float]]:
    """
    Stage 3: MMR (Maximal Marginal Relevance) re-ranking for diversity.

    MMR iteratively selects items that balance:
      - Relevance to the user (high NeuMF score)
      - Diversity from already-selected items (low similarity to basket)

    Formula:
      MMR(item) = lambda * relevance(item)
                - (1 - lambda) * max_similarity(item, selected_items)

    lambda=0   → pure diversity (select maximally different items)
    lambda=1   → pure relevance (ignore diversity, same as no re-ranking)
    lambda=0.3 → 30% relevance, 70% diversity (good default)

    Why MMR matters:
      Without diversity, the top-10 might all be variations of the same item.
      At Amazon, recommending 10 variants of the same phone case is useless.
      MMR ensures the final list covers different regions of item space.

    item_vectors: (n_items, dim) item embeddings from two-tower item tower.
                  Used to compute inter-item similarity.
    """
    if not scored_candidates:
        return []

    # Normalise relevance scores to [0, 1]
    scores = np.array([s for _, s in scored_candidates])
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        scores = np.ones_like(scores)

    candidate_ids = [idx for idx, _ in scored_candidates]

    selected = []
    selected_vecs = []
    remaining = list(range(len(candidate_ids)))

    for _ in range(min(k, len(candidate_ids))):
        if not remaining:
            break

        if not selected:
            # First item: pick highest relevance
            best_pos = int(np.argmax([scores[i] for i in remaining]))
            best_idx = remaining[best_pos]
        else:
            # MMR: balance relevance and diversity
            best_score = -np.inf
            best_pos = 0

            for pos, i in enumerate(remaining):
                item_vec = item_vectors[candidate_ids[i]]
                # Max cosine similarity to already-selected items
                sims = [
                    float(np.dot(item_vec, sv) /
                          (np.linalg.norm(item_vec) * np.linalg.norm(sv) + 1e-8))
                    for sv in selected_vecs
                ]
                max_sim = max(sims) if sims else 0.0

                mmr_score = (lambda_ * scores[i]
                             - (1 - lambda_) * max_sim)
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_pos = pos
                    best_idx = i

        selected.append((candidate_ids[best_idx],
                         float(scored_candidates[best_idx][1])))
        selected_vecs.append(item_vectors[candidate_ids[best_idx]])
        remaining.pop(best_pos)

    return selected


# --------------------------------------------------------------------------
# 5. Full pipeline
# --------------------------------------------------------------------------

class RecommendationPipeline:
    """
    End-to-end multi-stage recommendation pipeline.

    Combines two-tower retrieval, NeuMF scoring, and MMR re-ranking
    into a single inference call per user.

    Usage:
        pipeline = RecommendationPipeline(config, tt_model, tt_dataset,
                                          ncf_model, ncf_dataset,
                                          faiss_index, item_vectors)
        recs = pipeline.recommend(user_id, train_df)
    """

    def __init__(self, config: PipelineConfig,
                 tt_model, tt_dataset,
                 ncf_model, ncf_dataset,
                 faiss_index, item_vectors: np.ndarray):
        self.config      = config
        self.tt_model    = tt_model
        self.tt_dataset  = tt_dataset
        self.ncf_model   = ncf_model
        self.ncf_dataset = ncf_dataset
        self.faiss_index = faiss_index
        self.item_vectors = item_vectors

        # Simple position propensity model: P(seen | position k) = 1/k
        # In production this is estimated from randomised experiments
        max_pos = config.retrieval_k
        self.propensities = np.array([1.0 / (k + 1)
                                       for k in range(max_pos)])

    def recommend(self, user_id: str,
                  train_df: pd.DataFrame) -> list[tuple[str, float]]:
        """
        Run the full pipeline for a single user.
        Returns list of (item_id, score) tuples.
        """
        # Map user_id to indices in each model
        tt_user_idx  = self.tt_dataset.user2idx.get(user_id)
        ncf_user_idx = self.ncf_dataset.user2idx.get(user_id)

        if tt_user_idx is None:
            return []

        # Training items to exclude
        user_train = train_df[train_df["user_id"] == user_id]["item_id"]
        tt_train_indices = {
            self.tt_dataset.item2idx[it]
            for it in user_train
            if it in self.tt_dataset.item2idx
        }

        # Stage 1: Retrieval
        user_vec = self.tt_model.get_user_vector(
            torch.tensor([tt_user_idx])
        ).numpy()
        candidates_tt = retrieval_stage(
            tt_user_idx, user_vec, self.faiss_index,
            tt_train_indices, k=self.config.retrieval_k
        )

        # Map two-tower item indices → item_ids → NCF item indices
        # (the two models have separate index spaces)
        candidates_ncf = []
        tt_to_ncf = {}
        for tt_idx in candidates_tt:
            item_id = self.tt_dataset.items[tt_idx]
            ncf_idx = self.ncf_dataset.item2idx.get(item_id)
            if ncf_idx is not None:
                candidates_ncf.append(ncf_idx)
                tt_to_ncf[ncf_idx] = tt_idx

        if not candidates_ncf or ncf_user_idx is None:
            # Fallback: return retrieval results directly
            return [
                (self.tt_dataset.items[idx], 0.0)
                for idx in candidates_tt[:self.config.final_k]
            ]

        # Stage 2: Scoring
        propensities = (self.propensities
                        if self.config.use_debiasing else None)
        scored = scoring_stage(
            ncf_user_idx, candidates_ncf,
            self.ncf_model, self.ncf_dataset,
            propensities
        )
        top_scored = scored[:self.config.scoring_k]

        # Stage 3: Re-ranking with MMR
        # Map NCF indices back to two-tower item vectors for similarity
        reranked = reranking_stage(
            top_scored,
            self.item_vectors,
            k=self.config.final_k,
            lambda_=self.config.diversity_lambda,
        )

        # Map back to item_ids
        results = []
        for ncf_idx, score in reranked:
            item_id = self.ncf_dataset.items[ncf_idx]
            results.append((item_id, score))

        return results

    def evaluate(self, test_df: pd.DataFrame,
                 train_df: pd.DataFrame, k: int = 10) -> dict:
        """
        Evaluate pipeline — HR@K, NDCG@K, Precision@K, latency.
        """
        user_test_items = defaultdict(set)
        for _, row in test_df.iterrows():
            user_test_items[row["user_id"]].add(row["item_id"])

        hr, ndcg, prec = [], [], []
        n_covered = 0
        latencies = []

        for user_id, relevant in user_test_items.items():
            t0 = time.perf_counter()
            recs = self.recommend(user_id, train_df)
            latencies.append((time.perf_counter() - t0) * 1000)

            recommended = [item_id for item_id, _ in recs]
            if recommended:
                n_covered += 1

            hr.append(float(any(it in relevant for it in recommended[:k])))
            dcg  = sum(1.0 / np.log2(r + 2)
                       for r, it in enumerate(recommended[:k])
                       if it in relevant)
            idcg = sum(1.0 / np.log2(r + 2)
                       for r in range(min(len(relevant), k)))
            ndcg.append(dcg / idcg if idcg > 0 else 0.0)
            prec.append(
                sum(1 for it in recommended[:k] if it in relevant) / k
            )

        n_users = len(user_test_items)
        return {
            f"HR@{k}"          : float(np.mean(hr)),
            f"NDCG@{k}"        : float(np.mean(ndcg)),
            f"Precision@{k}"   : float(np.mean(prec)),
            "coverage"         : n_covered / n_users if n_users else 0,
            "n_users_eval"     : n_users,
            "mean_latency_ms"  : float(np.mean(latencies)),
            "p95_latency_ms"   : float(np.percentile(latencies, 95)),
        }


# --------------------------------------------------------------------------
# 6. Sanity check
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, ".")
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"

    import faiss
    from src.data_loader import (download_data, load_ratings,
                             filter_min_interactions, train_test_split)
    from src.phase2.two_tower import (InteractionDataset, TwoTowerModel,
                                   train as tt_train, build_item_index)
    from src.phase2.ncf import NCFDataset, NeuMF, train as ncf_train
    from src.phase2.ann_index import build_index, IndexConfig

    print("Loading data...")
    fp = download_data()
    df = load_ratings(fp)
    df = filter_min_interactions(df)
    train_df, test_df = train_test_split(df)

    # Train two-tower
    print("\nTraining two-tower...")
    tt_dataset = InteractionDataset(train_df)
    tt_model = TwoTowerModel(
        n_users=tt_dataset.n_users, n_items=tt_dataset.n_items,
        embed_dim=64, hidden_dim=128, output_dim=64, temperature=0.07,
    )
    tt_train(tt_model, tt_dataset, n_epochs=20, batch_size=64, lr=1e-3)

    # Build FAISS index
    item_vecs_tensor = build_item_index(tt_model, tt_dataset)
    item_vecs_np = item_vecs_tensor.numpy().astype(np.float32)
    cfg = IndexConfig(index_type="hnsw", dim=64, hnsw_m=32)
    faiss_index = build_index(item_vecs_np, cfg)

    # Train NeuMF
    print("\nTraining NeuMF...")
    ncf_dataset = NCFDataset(train_df, neg_ratio=4)
    ncf_model = NeuMF(
        n_users=ncf_dataset.n_users, n_items=ncf_dataset.n_items,
        gmf_dim=32, mlp_embed_dim=32, mlp_layers=[64, 32],
    )
    ncf_train(ncf_model, ncf_dataset, n_epochs=20, batch_size=256, lr=1e-3)

    # Build pipeline — compare configs
    print("\n" + "=" * 65)
    print(f"{'Config':<30} {'HR@10':<10} {'NDCG@10':<10} "
          f"{'Coverage':<10} {'Latency(ms)'}")
    print("-" * 65)

    configs = [
        ("No diversity (λ=1.0)",
         PipelineConfig(diversity_lambda=1.0, use_debiasing=False)),
        ("Diversity only (λ=0.3)",
         PipelineConfig(diversity_lambda=0.3, use_debiasing=False)),
        ("Debiasing only",
         PipelineConfig(diversity_lambda=1.0, use_debiasing=True)),
        ("Full pipeline (λ=0.3 + IPS)",
         PipelineConfig(diversity_lambda=0.3, use_debiasing=True)),
    ]

    for name, config in configs:
        pipeline = RecommendationPipeline(
            config, tt_model, tt_dataset,
            ncf_model, ncf_dataset,
            faiss_index, item_vecs_np,
        )
        res = pipeline.evaluate(test_df, train_df, k=10)
        print(f"{name:<30} {res['HR@10']:<10.4f} "
              f"{res['NDCG@10']:<10.4f} "
              f"{res['coverage']:<10.4f} "
              f"{res['mean_latency_ms']:.2f}")

    print("=" * 65)