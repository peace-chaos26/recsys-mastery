"""
evaluate.py
-----------
Offline evaluation metrics for recommender systems.

Implements:
  - HR@K     (Hit Rate)       -- did we recommend anything relevant?
  - NDCG@K   (Normalized DCG) -- are relevant items ranked highly?
  - Precision@K               -- fraction of top-K that are relevant

These are the three metrics you'll see in every RecSys paper and interview.
NDCG@10 is the single most commonly reported number in the field.

Usage:
  from evaluate import evaluate_model
  results = evaluate_model(model, test_df, train_matrix, K=10)
"""

import numpy as np
import pandas as pd
from collections import defaultdict


# --------------------------------------------------------------------------
# 1. Core metric functions (single-user)
# --------------------------------------------------------------------------

def hit_rate_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    HR@K = 1 if any of the top-K recommended items is relevant, else 0.

    Binary metric — doesn't care about rank within top-K.
    Good for measuring whether the system is useful at all.
    """
    return float(any(item in relevant for item in recommended[:k]))


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    NDCG@K = DCG@K / IDCG@K

    DCG@K  = sum of (1/log2(rank+1)) for each relevant item in top-K
    IDCG@K = DCG of a perfect ranking (all relevant items at top)

    The log2 discount means rank 1 contributes 1.0, rank 2 contributes
    0.63, rank 5 contributes 0.39 — being higher in the list matters.

    Normalized by IDCG so the score is always in [0, 1].
    """
    dcg = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            dcg += 1.0 / np.log2(rank + 1)

    # Ideal DCG: if we had placed all relevant items at the top
    n_relevant_in_k = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, n_relevant_in_k + 1))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Precision@K = (# relevant items in top-K) / K

    Simple fraction. Doesn't account for rank order within top-K.
    Useful alongside NDCG to understand coverage vs ranking quality.
    """
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


# --------------------------------------------------------------------------
# 2. Full evaluation loop
# --------------------------------------------------------------------------

def evaluate_model(
    model: dict,
    test_df: pd.DataFrame,
    k: int = 10,
) -> dict:
    """
    Evaluate a fitted CF model against held-out test ratings.

    Protocol:
      For each user in the test set:
        1. Generate top-K recommendations (excluding training items)
        2. Treat items the user rated in test as "relevant"
        3. Compute HR@K, NDCG@K, Precision@K

    Only evaluates users who appear in both test set and training matrix.
    Users with no coverage (no neighbours rated their test items) are
    counted as 0.0 for all metrics — not skipped. This is the conservative,
    honest evaluation approach.

    Args:
      model   : dict returned by user_cf.fit()
      test_df : DataFrame with columns [user_id, item_id, rating]
      k       : cutoff for all metrics

    Returns:
      dict with mean HR@K, NDCG@K, Precision@K and coverage stats
    """
    from user_cf import recommend

    matrix     = model["matrix"]
    centered   = model["centered"]
    sim_df     = model["sim_df"]
    user_means = model["user_means"]

    # Group test items by user
    # relevant items = anything the user rated in test
    # (we treat all test interactions as relevant — binary relevance)
    user_test_items = defaultdict(set)
    for _, row in test_df.iterrows():
        if row["user_id"] in matrix.index:
            user_test_items[row["user_id"]].add(row["item_id"])

    hr_scores, ndcg_scores, prec_scores = [], [], []
    n_covered = 0  # users where we could predict at least 1 item

    eval_users = list(user_test_items.keys())
    print(f"[evaluate] Evaluating {len(eval_users)} users at K={k} ...")

    for user_id in eval_users:
        relevant = user_test_items[user_id]

        # Get top-K recommendations (items not in training)
        recs = recommend(
            user_id, matrix, centered, sim_df, user_means, n=k, k=30
        )
        recommended_items = [item for item, _ in recs]

        if recommended_items:
            n_covered += 1

        hr_scores.append(hit_rate_at_k(recommended_items, relevant, k))
        ndcg_scores.append(ndcg_at_k(recommended_items, relevant, k))
        prec_scores.append(precision_at_k(recommended_items, relevant, k))

    results = {
        f"HR@{k}"        : np.mean(hr_scores),
        f"NDCG@{k}"      : np.mean(ndcg_scores),
        f"Precision@{k}" : np.mean(prec_scores),
        "coverage"       : n_covered / len(eval_users) if eval_users else 0,
        "n_users_eval"   : len(eval_users),
    }
    return results


# --------------------------------------------------------------------------
# 3. Pretty printer
# --------------------------------------------------------------------------

def print_results(results: dict, model_name: str = "Model") -> None:
    """Print evaluation results in a clean table format."""
    print(f"\n{'='*45}")
    print(f"  {model_name}")
    print(f"{'='*45}")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"  {metric:<18} {value:.4f}")
        else:
            print(f"  {metric:<18} {value}")
    print(f"{'='*45}\n")


# --------------------------------------------------------------------------
# 4. Sanity check
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from data_loader import (download_data, load_ratings,
                             filter_min_interactions, train_test_split,
                             build_user_item_matrix)
    from user_cf import fit

    # Load and prepare data
    fp          = download_data()
    df          = load_ratings(fp)
    df          = filter_min_interactions(df)
    train, test = train_test_split(df)
    matrix      = build_user_item_matrix(train)

    # Fit model
    model = fit(matrix)

    # Evaluate at K=5 and K=10
    for k in [5, 10]:
        results = evaluate_model(model, test, k=k)
        print_results(results, model_name=f"User-based CF (cosine, K={k})")