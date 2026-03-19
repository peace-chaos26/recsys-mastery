"""
diversity.py
------------
Diversity algorithms for recommendation re-ranking.

The filter bubble problem:
  Pure relevance maximisation produces lists of near-identical items.
  A user who clicked one Taylor Swift song gets 10 Taylor Swift songs.
  This hurts long-term engagement — users stop coming back.

Two algorithms implemented:
  1. MMR (Maximal Marginal Relevance) -- greedy, fast, interpretable
  2. DPP (Determinantal Point Process) -- principled, slower, better diversity

Both take a scored candidate list and return a re-ranked diverse subset.

When to use which:
  MMR  : production systems, latency-sensitive, easy to tune lambda
  DPP  : offline experiments, research, when you need guaranteed diversity
         with theoretical properties

References:
  Carbonell & Goldstein (1998) "The Use of MMR in Text Summarization"
  Chen et al. (2018) "Fast Greedy MAP Inference for DPP for Large-scale Diversification"
"""

import numpy as np
from typing import Optional


# --------------------------------------------------------------------------
# 1. MMR (Maximal Marginal Relevance)
# --------------------------------------------------------------------------

def mmr(
    candidates: list[tuple[int, float]],
    item_vectors: np.ndarray,
    k: int = 10,
    lambda_: float = 0.5,
) -> list[tuple[int, float]]:
    """
    MMR re-ranking for diversity.

    Greedily selects items that maximise:
      MMR(i) = lambda * relevance(i) - (1-lambda) * max_sim(i, selected)

    lambda=0 → pure diversity
    lambda=1 → pure relevance (no re-ranking)
    lambda=0.5 → equal weight (good default)

    Time complexity: O(k * n * d) where n=candidates, d=embedding dim.
    Fast enough for n=500, k=10 in <1ms.

    candidates  : list of (item_idx, relevance_score) sorted by score
    item_vectors: (n_items, dim) L2-normalized item embeddings
    """
    if not candidates:
        return []

    # Normalise relevance scores to [0, 1]
    scores = np.array([s for _, s in candidates])
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        scores = np.ones_like(scores)

    item_ids = [idx for idx, _ in candidates]
    selected, selected_vecs, remaining = [], [], list(range(len(item_ids)))

    for _ in range(min(k, len(item_ids))):
        if not remaining:
            break

        if not selected:
            best_i = int(np.argmax([scores[i] for i in remaining]))
        else:
            best_score, best_i = -np.inf, 0
            for i in remaining:
                vec = item_vectors[item_ids[i]]
                # Cosine similarity to each selected item
                sims = [
                    float(np.dot(vec, sv) /
                          (np.linalg.norm(vec) * np.linalg.norm(sv) + 1e-8))
                    for sv in selected_vecs
                ]
                mmr_score = lambda_ * scores[i] - (1 - lambda_) * max(sims)
                if mmr_score > best_score:
                    best_score, best_i = mmr_score, i

        selected.append((item_ids[best_i], float(candidates[best_i][1])))
        selected_vecs.append(item_vectors[item_ids[best_i]])
        remaining.remove(best_i)

    return selected


# --------------------------------------------------------------------------
# 2. DPP (Determinantal Point Process)
# --------------------------------------------------------------------------

def dpp_greedy(
    candidates: list[tuple[int, float]],
    item_vectors: np.ndarray,
    k: int = 10,
    epsilon: float = 1e-10,
) -> list[tuple[int, float]]:
    """
    Greedy MAP inference for a quality-diversity DPP.

    A DPP defines a probability distribution over subsets where the
    probability of selecting a set S is proportional to det(L_S),
    the determinant of the kernel submatrix for S.

    The kernel matrix L combines quality and similarity:
      L[i,j] = quality(i) * similarity(i,j) * quality(j)

    det(L_S) is large when:
      - Items in S have high quality (large diagonal)
      - Items in S are diverse (small off-diagonal similarity)

    Greedy MAP finds the approximate maximum probability subset
    by iteratively adding the item that most increases det(L_S).

    The determinant can be computed incrementally using Cholesky
    decomposition — adding one item at a time in O(k^2 * n) total.

    Time complexity: O(k^2 * n) — faster than exact MAP (NP-hard).

    Why DPP over MMR?
      MMR uses a simple max-similarity penalty.
      DPP considers the full geometry of selected items — it avoids
      not just pairwise similarity but also collinear item clusters.
      Theoretical guarantee: selected set covers the item space uniformly.
    """
    if not candidates:
        return []

    n = len(candidates)
    k = min(k, n)

    # Quality scores (relevance)
    qualities = np.array([max(s, 1e-6) for _, s in candidates])
    # Normalise to [0.1, 1]
    qualities = 0.1 + 0.9 * (qualities - qualities.min()) / \
                (qualities.max() - qualities.min() + 1e-8)

    item_ids = [idx for idx, _ in candidates]

    # Item vectors for candidates
    vecs = np.array([item_vectors[idx] for idx in item_ids],
                    dtype=np.float32)
    # L2 normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / (norms + 1e-8)

    # Kernel matrix: L[i,j] = q_i * cos_sim(i,j) * q_j
    cos_sim = vecs @ vecs.T
    L = np.outer(qualities, qualities) * cos_sim

    # Greedy MAP via incremental Cholesky
    # Based on Chen et al. (2018) fast greedy DPP
    selected_indices = []
    remaining = list(range(n))
    c = np.zeros((k, n))   # Cholesky vectors
    d = np.diag(L).copy()  # Diagonal values (marginal gains)

    for j in range(k):
        if not remaining:
            break

        # Select item with highest marginal gain
        best_i = remaining[int(np.argmax([d[i] for i in remaining]))]
        selected_indices.append(best_i)
        remaining.remove(best_i)

        if j < k - 1 and remaining:
            # Update Cholesky factors
            e_i = L[best_i, :] if j == 0 else \
                  L[best_i, :] - c[:j, best_i] @ c[:j, :]
            c[j, :] = e_i / np.sqrt(max(d[best_i], epsilon))
            # Update residual diagonal
            d -= c[j, :] ** 2
            d = np.maximum(d, epsilon)

    return [(item_ids[i], float(candidates[i][1]))
            for i in selected_indices]


# --------------------------------------------------------------------------
# 3. Diversity metrics
# --------------------------------------------------------------------------

def intra_list_diversity(
    recommended_indices: list[int],
    item_vectors: np.ndarray,
) -> float:
    """
    ILD: average pairwise distance between recommended items.

    ILD = (2 / (k*(k-1))) * sum_{i<j} distance(item_i, item_j)

    Higher ILD = more diverse recommendation list.
    distance = 1 - cosine_similarity (so ILD in [0, 1])

    This is the standard diversity metric in RecSys evaluation.
    Use alongside NDCG to measure the relevance-diversity tradeoff.
    """
    if len(recommended_indices) < 2:
        return 0.0

    vecs = np.array([item_vectors[i] for i in recommended_indices],
                    dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / (norms + 1e-8)

    # Pairwise cosine similarity
    sim_matrix = vecs @ vecs.T
    k = len(recommended_indices)

    # Average pairwise distance (upper triangle only)
    total_dist = 0.0
    n_pairs = 0
    for i in range(k):
        for j in range(i + 1, k):
            total_dist += 1.0 - sim_matrix[i, j]
            n_pairs += 1

    return total_dist / n_pairs if n_pairs > 0 else 0.0


def coverage(
    all_recommendations: list[list[int]],
    n_items: int,
) -> float:
    """
    Catalogue coverage: fraction of items that appear in any recommendation.

    coverage = |union of all recommended items| / n_items

    Low coverage = popularity bias (same items recommended to everyone).
    High coverage = system explores the catalogue broadly.

    Target: >10% for a healthy RecSys. Netflix aims for >50%.
    """
    all_items = set()
    for recs in all_recommendations:
        all_items.update(recs)
    return len(all_items) / n_items if n_items > 0 else 0.0


# --------------------------------------------------------------------------
# 4. Sanity check — compare MMR vs DPP
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_loader import (download_data, load_ratings,
                             filter_min_interactions, train_test_split)
    from src.phase2.two_tower import (InteractionDataset, TwoTowerModel,
                                   train as tt_train, build_item_index)

    fp = download_data()
    df = load_ratings(fp)
    df = filter_min_interactions(df)
    train_df, _ = train_test_split(df)

    # Train two-tower to get item vectors
    tt_dataset = InteractionDataset(train_df)
    tt_model = TwoTowerModel(
        n_users=tt_dataset.n_users, n_items=tt_dataset.n_items,
        embed_dim=64, hidden_dim=128, output_dim=64, temperature=0.07,
    )
    tt_train(tt_model, tt_dataset, n_epochs=20, batch_size=64, lr=1e-3)
    item_vecs = build_item_index(tt_model, tt_dataset).numpy()

    # Create fake scored candidates (top 50 items with random scores)
    rng = np.random.default_rng(42)
    n_candidates = 50
    fake_scores = rng.random(n_candidates)
    candidates = [(i, float(fake_scores[i])) for i in range(n_candidates)]
    candidates.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'Method':<25} {'ILD (diversity)':<20} {'Top-5 items'}")
    print("-" * 60)

    # No re-ranking
    top10_no_rerank = candidates[:10]
    ild_no = intra_list_diversity(
        [i for i, _ in top10_no_rerank], item_vecs
    )
    print(f"{'No re-ranking':<25} {ild_no:<20.4f} "
          f"{[i for i, _ in top10_no_rerank[:5]]}")

    # MMR at different lambdas
    for lam in [0.7, 0.5, 0.3]:
        mmr_result = mmr(candidates, item_vecs, k=10, lambda_=lam)
        ild_mmr = intra_list_diversity(
            [i for i, _ in mmr_result], item_vecs
        )
        print(f"{'MMR (λ='+str(lam)+')':<25} {ild_mmr:<20.4f} "
              f"{[i for i, _ in mmr_result[:5]]}")

    # DPP
    dpp_result = dpp_greedy(candidates, item_vecs, k=10)
    ild_dpp = intra_list_diversity(
        [i for i, _ in dpp_result], item_vecs
    )
    print(f"{'DPP (greedy)':<25} {ild_dpp:<20.4f} "
          f"{[i for i, _ in dpp_result[:5]]}")

    # Catalogue coverage
    all_recs = [
        [i for i, _ in top10_no_rerank],
        [i for i, _ in mmr(candidates, item_vecs, k=10, lambda_=0.5)],
        [i for i, _ in dpp_result],
    ]
    cov = coverage(all_recs, n_items=len(item_vecs))
    print(f"\nCatalogue coverage across all methods: {cov:.2%}")
    print("(higher = system explores more of the item catalogue)")