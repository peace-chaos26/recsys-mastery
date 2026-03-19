"""
hybrid.py
---------
Hybrid Recommender — combines Content-Based and Matrix Factorization.

Two strategies:
  1. Weighted hybrid  : final_score = alpha * CB + (1-alpha) * MF
                        Blends both signals for every user.
                        Best when both models have good coverage.

  2. Switching hybrid : use CB when user has >= min_ratings history,
                        fall back to MF (ALS) otherwise.
                        Best for handling cold-start users gracefully.

Why hybrid at all?
  CB alone → filter bubble (over-specialisation, no serendipity)
  MF alone → weaker on datasets where item similarity dominates
  Hybrid   → CB provides precision, MF provides diversity/discovery

In production (Netflix, Spotify, LinkedIn) hybrid systems are the norm.
The weights are typically learned via A/B testing or bandit algorithms
(which we cover in Phase 5).
"""

import numpy as np
import pandas as pd
from collections import defaultdict


# --------------------------------------------------------------------------
# 1. Score normalisation helper
# --------------------------------------------------------------------------

def normalise_scores(recs: list[tuple[str, float]]) -> dict[str, float]:
    """
    Min-max normalise recommendation scores to [0, 1].

    Why: CB scores (cosine similarity) and MF scores (dot products /
    ALS confidence) live on different scales. We can't add them directly.
    Min-max normalisation puts both on [0, 1] so alpha is meaningful.

    Returns dict of {item_id: normalised_score}.
    """
    if not recs:
        return {}
    scores = [s for _, s in recs]
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return {item: 1.0 for item, _ in recs}
    return {item: (s - min_s) / (max_s - min_s) for item, s in recs}


# --------------------------------------------------------------------------
# 2. Weighted hybrid
# --------------------------------------------------------------------------

def weighted_recommend(
    user_id: str,
    cb_model: dict,
    mf_model,
    train_matrix: pd.DataFrame,
    alpha: float = 0.7,
    n: int = 10,
    candidate_pool: int = 50,
) -> list[tuple[str, float]]:
    """
    Weighted combination of CB and MF scores.

    final_score(item) = alpha * CB_score + (1 - alpha) * MF_score

    Args:
      alpha         : weight for CB (0 = pure MF, 1 = pure CB, 0.7 = default)
      candidate_pool: number of candidates to fetch from each model before
                      merging. Larger pool = better coverage, slower.

    Why alpha=0.7 default?
      CB dominated our evaluation (NDCG=0.275 vs ALS=0.088), so we weight
      it higher. In practice this is tuned via offline eval or A/B testing.
    """
    from content_based import recommend as cb_recommend
    from content_based import build_user_taste_profile

    # Get candidates from both models
    cb_recs = cb_recommend(
        user_id, cb_model["matrix"], cb_model["item_sim_df"], n=candidate_pool
    )
    mf_recs = mf_model.recommend(user_id, train_matrix, n=candidate_pool)

    if not cb_recs and not mf_recs:
        return []

    # Normalise scores to [0, 1] independently
    cb_norm  = normalise_scores(cb_recs)
    mf_norm  = normalise_scores(mf_recs)

    # Union of all candidate items
    all_items = set(cb_norm.keys()) | set(mf_norm.keys())

    # Already-rated items to exclude
    rated = set(train_matrix.loc[user_id].dropna().index) \
            if user_id in train_matrix.index else set()

    # Combine scores
    combined = []
    for item_id in all_items:
        if item_id in rated:
            continue
        cb_score = cb_norm.get(item_id, 0.0)
        mf_score = mf_norm.get(item_id, 0.0)
        final    = alpha * cb_score + (1 - alpha) * mf_score
        combined.append((item_id, final))

    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:n]


# --------------------------------------------------------------------------
# 3. Switching hybrid
# --------------------------------------------------------------------------

def switching_recommend(
    user_id: str,
    cb_model: dict,
    mf_model,
    train_matrix: pd.DataFrame,
    min_ratings_for_cb: int = 3,
    n: int = 10,
) -> tuple[list[tuple[str, float]], str]:
    """
    Switch between CB and MF based on user interaction history.

    Logic:
      - User has >= min_ratings_for_cb rated items → use CB
        (enough history to build a reliable taste profile)
      - User has < min_ratings_for_cb rated items → use MF
        (not enough history for CB; MF latent vectors handle sparse users)

    Returns (recommendations, model_used) so we can track which path fired.

    Why min=3?
      With 1-2 ratings, the taste profile is too narrow — CB would
      over-index on a single item's neighbours. MF's global latent
      structure is more robust for sparse users.
    """
    from content_based import recommend as cb_recommend

    if user_id in train_matrix.index:
        n_rated = train_matrix.loc[user_id].notna().sum()
    else:
        n_rated = 0

    if n_rated >= min_ratings_for_cb:
        recs = cb_recommend(
            user_id, cb_model["matrix"], cb_model["item_sim_df"], n=n
        )
        return recs, "content-based"
    else:
        recs = mf_model.recommend(user_id, train_matrix, n=n)
        return recs, "matrix-factorization"


# --------------------------------------------------------------------------
# 4. Evaluate both hybrid strategies
# --------------------------------------------------------------------------

def evaluate_hybrid(
    cb_model: dict,
    mf_model,
    test_df: pd.DataFrame,
    train_matrix: pd.DataFrame,
    mode: str = "weighted",
    alpha: float = 0.7,
    k: int = 10,
) -> dict:
    """
    Evaluate a hybrid strategy.

    mode: "weighted" or "switching"
    """
    user_test_items = defaultdict(set)
    for _, row in test_df.iterrows():
        if row["user_id"] in train_matrix.index:
            user_test_items[row["user_id"]].add(row["item_id"])

    hr, ndcg, prec = [], [], []
    n_covered = 0
    model_usage = defaultdict(int)  # track switching hybrid path usage

    for user_id, relevant in user_test_items.items():
        if mode == "weighted":
            recs = weighted_recommend(
                user_id, cb_model, mf_model, train_matrix, alpha=alpha, n=k
            )
            recommended = [item for item, _ in recs]
        else:
            recs, used_model = switching_recommend(
                user_id, cb_model, mf_model, train_matrix, n=k
            )
            recommended = [item for item, _ in recs]
            model_usage[used_model] += 1

        if recommended:
            n_covered += 1

        hr.append(float(any(item in relevant for item in recommended[:k])))

        dcg  = sum(1.0 / np.log2(r + 2)
                   for r, item in enumerate(recommended[:k]) if item in relevant)
        idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(relevant), k)))
        ndcg.append(dcg / idcg if idcg > 0 else 0.0)

        prec.append(sum(1 for item in recommended[:k] if item in relevant) / k)

    n_users = len(user_test_items)
    result = {
        f"HR@{k}"        : np.mean(hr),
        f"NDCG@{k}"      : np.mean(ndcg),
        f"Precision@{k}" : np.mean(prec),
        "coverage"       : n_covered / n_users if n_users else 0,
        "n_users_eval"   : n_users,
        "mode"           : mode,
    }
    if mode == "switching":
        result["cb_users"]  = model_usage["content-based"]
        result["mf_users"]  = model_usage["matrix-factorization"]

    return result


# --------------------------------------------------------------------------
# 5. Sanity check — full comparison across all alphas
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_loader import (download_data, load_ratings,
                             filter_min_interactions, train_test_split,
                             build_user_item_matrix)
    from content_based import fit as cb_fit
    from matrix_factor import ALSRecommender

    fp          = download_data()
    df          = load_ratings(fp)
    df          = filter_min_interactions(df)
    train, test = train_test_split(df)
    matrix      = build_user_item_matrix(train)

    # Fit both models
    print("Fitting content-based model...")
    cb_model = cb_fit(matrix)

    print("\nFitting ALS model...")
    import os
    als = ALSRecommender(n_factors=50, iterations=20).fit(matrix)

    print("\n" + "="*65)
    print(f"{'Strategy':<28} {'HR@10':<10} {'NDCG@10':<10} {'Coverage':<10}")
    print("-"*65)

    # Weighted hybrid at different alphas
    for alpha in [0.3, 0.5, 0.7, 0.9]:
        res = evaluate_hybrid(cb_model, als, test, matrix,
                              mode="weighted", alpha=alpha, k=10)
        label = f"Weighted (α={alpha})"
        print(f"{label:<28} {res['HR@10']:<10.4f} "
              f"{res['NDCG@10']:<10.4f} {res['coverage']:<10.4f}")

    # Switching hybrid
    res = evaluate_hybrid(cb_model, als, test, matrix,
                          mode="switching", k=10)
    label = "Switching (min=3)"
    print(f"{label:<28} {res['HR@10']:<10.4f} "
          f"{res['NDCG@10']:<10.4f} {res['coverage']:<10.4f}")
    print(f"  → CB used for {res['cb_users']} users, "
          f"MF for {res['mf_users']} users")

    print("="*65)