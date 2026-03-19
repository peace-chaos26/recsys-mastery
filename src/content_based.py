"""
content_based.py
----------------
Content-Based Filtering for recommendation systems.

Without rich item metadata (descriptions, categories, tags), we use the
item's co-rater profile as its "content" — treating the set of users who
rated an item as that item's document. This is item-based CF, which is
conceptually equivalent to content-based filtering.

TF-IDF weighting:
  - TF  (term frequency)    : how often user u rated item i (binary here)
  - IDF (inverse doc freq)  : users who rate fewer items are more informative
                              signals — a user who rated 2 items and both
                              overlap with item X is a stronger signal than
                              a user who rated 200 items

In Phase 4, we replace user-co-rater profiles with real LLM embeddings
of product descriptions — same cosine similarity logic, richer features.

Key advantage over user-CF:
  - No cross-user overlap needed — we compare items directly
  - Naturally handles the "what else is like this item?" question
  - Works well even with a single user (pure cold-start on user side)

Key limitation:
  - Filter bubble: recommends more of the same, less serendipity
  - Still needs item interaction history (cold-start on item side)
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


# --------------------------------------------------------------------------
# 1. Build item profiles
# --------------------------------------------------------------------------

def build_item_profiles(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Build a TF-IDF weighted item profile matrix.

    Each item is represented as a vector over users:
        rows    = items
        columns = users
        values  = TF-IDF weight (higher = more informative co-rater signal)

    TF:  binary (1 if user rated this item, 0 otherwise)
    IDF: log(n_items / n_items_rated_by_user)
         Users who rate many items carry less signal per item.
         Users who rate very few items are highly informative when they
         do rate something — they're selective.

    Returns:
      item_profiles: DataFrame (items x users) with TF-IDF weights
    """
    # Transpose: now rows=items, cols=users, values=rating or NaN
    item_user = matrix.T

    # Binary TF: 1 if rated, 0 if not
    tf = item_user.notna().astype(float)

    # IDF: log(n_items / number of items each user has rated)
    # user_item_count[u] = how many items user u has rated
    user_item_count = tf.sum(axis=0)  # sum across items for each user
    n_items = len(item_user)
    idf = np.log(n_items / (user_item_count + 1))  # +1 to avoid division by zero

    # TF-IDF = TF * IDF (broadcast IDF across rows)
    tfidf = tf.multiply(idf, axis=1)

    print(f"[cb] Item profiles: {tfidf.shape[0]} items × {tfidf.shape[1]} users")
    return tfidf


# --------------------------------------------------------------------------
# 2. Build item-item similarity matrix
# --------------------------------------------------------------------------

def compute_item_similarity(item_profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cosine similarity between all pairs of items.

    Input : item_profiles (items x users) TF-IDF matrix
    Output: symmetric similarity matrix (items x items), values in [0, 1]

    We L2-normalize each item vector before computing cosine — this is
    mathematically equivalent to cosine similarity and slightly faster
    for large matrices via dot product.

    Scale note: 154 items → 154×154 matrix, trivially small.
    At 1M items (Amazon scale) you'd use ANN (FAISS) to find
    approximate top-K similar items — we'll cover this in Phase 2.
    """
    profiles = item_profiles.fillna(0).values.astype(np.float32)

    # L2-normalize rows so dot product = cosine similarity
    profiles_norm = normalize(profiles, norm="l2", axis=1)
    sim_matrix = profiles_norm @ profiles_norm.T  # (n_items, n_items)

    sim_df = pd.DataFrame(
        sim_matrix,
        index=item_profiles.index,
        columns=item_profiles.index,
    )

    # Zero out self-similarity
    np.fill_diagonal(sim_df.values, 0)

    print(f"[cb] Similarity matrix: {sim_df.shape[0]}×{sim_df.shape[1]}")
    return sim_df


# --------------------------------------------------------------------------
# 3. Build user taste profile
# --------------------------------------------------------------------------

def build_user_taste_profile(
    user_id: str,
    matrix: pd.DataFrame,
    item_sim_df: pd.DataFrame,
) -> pd.Series | None:
    """
    Aggregate a user's rated items into a single taste profile.

    Method: weighted average of item similarity vectors, weighted by
    the user's rating for each item (higher rating → stronger signal).

    This gives us a vector over all items representing how similar
    each candidate item is to this user's overall taste.

    Returns None if user has no rated items in the similarity matrix.
    """
    if user_id not in matrix.index:
        return None

    user_ratings = matrix.loc[user_id].dropna()
    rated_items  = [it for it in user_ratings.index if it in item_sim_df.index]

    if not rated_items:
        return None

    # Weighted sum of similarity vectors
    taste = pd.Series(0.0, index=item_sim_df.index)
    total_weight = 0.0

    for item_id in rated_items:
        rating = user_ratings[item_id]
        taste += item_sim_df.loc[item_id] * rating
        total_weight += rating

    if total_weight == 0:
        return None

    return taste / total_weight


# --------------------------------------------------------------------------
# 4. Generate top-N recommendations
# --------------------------------------------------------------------------

def recommend(
    user_id: str,
    matrix: pd.DataFrame,
    item_sim_df: pd.DataFrame,
    n: int = 10,
) -> list[tuple[str, float]]:
    """
    Return top-N content-based recommendations for a user.

    Steps:
      1. Build user's taste profile (weighted average of item sim vectors)
      2. Score all candidate items by their similarity to the taste profile
      3. Exclude already-rated items
      4. Return top-N by score

    Returns list of (item_id, score) tuples sorted descending.
    """
    taste = build_user_taste_profile(user_id, matrix, item_sim_df)
    if taste is None:
        return []

    # Items the user has already rated — exclude from recommendations
    rated = set(matrix.loc[user_id].dropna().index)

    recs = [
        (item_id, float(score))
        for item_id, score in taste.items()
        if item_id not in rated and score > 0
    ]
    recs.sort(key=lambda x: x[1], reverse=True)
    return recs[:n]


# --------------------------------------------------------------------------
# 5. Fit function — clean interface matching user_cf and matrix_factor
# --------------------------------------------------------------------------

def fit(matrix: pd.DataFrame) -> dict:
    """
    Fit the content-based model on the training user-item matrix.

    Returns a model dict for use with recommend() and evaluate().
    """
    print("[cb] Building item profiles (TF-IDF)...")
    item_profiles = build_item_profiles(matrix)

    print("[cb] Computing item-item similarity matrix...")
    item_sim_df = compute_item_similarity(item_profiles)

    print(f"[cb] Fit complete.")
    return {
        "matrix"      : matrix,
        "item_profiles": item_profiles,
        "item_sim_df"  : item_sim_df,
    }


# --------------------------------------------------------------------------
# 6. Evaluation wrapper (same protocol as evaluate.py)
# --------------------------------------------------------------------------

def evaluate_cb(model: dict, test_df: pd.DataFrame, k: int = 10) -> dict:
    """Evaluate content-based model — HR@K, NDCG@K, Precision@K."""
    from collections import defaultdict

    matrix      = model["matrix"]
    item_sim_df = model["item_sim_df"]

    user_test_items = defaultdict(set)
    for _, row in test_df.iterrows():
        if row["user_id"] in matrix.index:
            user_test_items[row["user_id"]].add(row["item_id"])

    hr, ndcg, prec = [], [], []
    n_covered = 0

    for user_id, relevant in user_test_items.items():
        recs = recommend(user_id, matrix, item_sim_df, n=k)
        recommended = [item for item, _ in recs]

        if recommended:
            n_covered += 1

        hr.append(float(any(item in relevant for item in recommended[:k])))

        dcg  = sum(1.0 / np.log2(r + 2)
                   for r, item in enumerate(recommended[:k]) if item in relevant)
        idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(relevant), k)))
        ndcg.append(dcg / idcg if idcg > 0 else 0.0)

        prec.append(sum(1 for item in recommended[:k] if item in relevant) / k)

    n_users = len(user_test_items)
    return {
        f"HR@{k}"        : np.mean(hr),
        f"NDCG@{k}"      : np.mean(ndcg),
        f"Precision@{k}" : np.mean(prec),
        "coverage"       : n_covered / n_users if n_users else 0,
        "n_users_eval"   : n_users,
    }


# --------------------------------------------------------------------------
# 7. Sanity check
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_loader import (download_data, load_ratings,
                             filter_min_interactions, train_test_split,
                             build_user_item_matrix)

    fp          = download_data()
    df          = load_ratings(fp)
    df          = filter_min_interactions(df)
    train, test = train_test_split(df)
    matrix      = build_user_item_matrix(train)

    # Fit
    model = fit(matrix)

    # Sample recommendations
    sample_user = matrix.index[10]
    recs = recommend(sample_user, model["matrix"], model["item_sim_df"], n=5)
    print(f"\nTop-5 CB recommendations for user '{sample_user[:16]}...':")
    for item_id, score in recs:
        print(f"  {item_id}  →  score: {score:.4f}")

    # Evaluate
    print()
    for k in [5, 10]:
        res = evaluate_cb(model, test, k=k)
        print(f"CB  K={k}: HR={res[f'HR@{k}']:.4f}  "
              f"NDCG={res[f'NDCG@{k}']:.4f}  "
              f"Prec={res[f'Precision@{k}']:.4f}  "
              f"Coverage={res['coverage']:.4f}")