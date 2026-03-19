"""
user_cf.py
----------
User-based Collaborative Filtering (memory-based CF).

Algorithm in three steps:
  1. Mean-center each user's ratings (removes rating scale bias)
  2. Compute pairwise cosine similarity between all users
  3. For a target user, find K nearest neighbours → predict ratings
     for unseen items → return top-N recommendations

Key concepts introduced here:
  - Mean-centered cosine similarity (= Pearson correlation)
  - Neighbourhood-based prediction
  - The sparsity challenge: most users share very few co-rated items
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------------------------------
# 1. Mean-center the user-item matrix
# --------------------------------------------------------------------------

def mean_center(matrix: pd.DataFrame) -> tuple:
    """
    Subtract each user's mean rating from all their ratings.

    Why:
      Some users rate generously (everything 4-5), others harshly (1-2).
      Raw cosine would see these users as dissimilar even if their
      relative preferences are identical. Mean-centering normalizes this
      away — we compare taste patterns, not rating scales.

      Mean-centered cosine similarity is mathematically equivalent to
      Pearson correlation coefficient — you'll see both terms in papers.

    Returns:
      centered  : DataFrame with NaN preserved, ratings shifted by user mean
      user_means: Series of each user's mean rating (needed to un-center
                  predictions back to the original 1-5 scale)
    """
    # axis=1 → compute mean across items (columns) for each user (row)
    # skipna=True (default) → ignore NaN, only average rated items
    user_means = matrix.mean(axis=1)

    # Subtract mean row-wise. We use .sub() with axis=0 to align on index.
    # NaN cells stay NaN — subtracting from NaN is still NaN.
    centered = matrix.sub(user_means, axis=0)

    return centered, user_means


# --------------------------------------------------------------------------
# 2. Build the user-user similarity matrix
# --------------------------------------------------------------------------

def compute_user_similarity(centered: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cosine similarity between every pair of users.

    Input  : mean-centered user-item matrix (users x items), NaN for unrated
    Output : symmetric similarity matrix (users x users), values in [-1, 1]

    Implementation note:
      sklearn's cosine_similarity treats NaN as 0, which is exactly what
      we want — an unrated item contributes nothing to similarity.
      This is the standard approach for sparse CF matrices.

    Scale note:
      For 1,430 users this produces a 1430x1430 matrix (~16MB float64).
      Totally fine in memory. For 10M users (Netflix scale) you'd compute
      similarity lazily, only for the query user at inference time, using
      approximate methods (LSH, FAISS). We'll see that in Phase 2.
    """
    # Fill NaN with 0 for cosine computation (unrated = no signal)
    filled = centered.fillna(0).values  # numpy array for sklearn

    sim_matrix = cosine_similarity(filled)  # shape: (n_users, n_users)

    # Wrap back in DataFrame so we can look up by user_id
    sim_df = pd.DataFrame(
        sim_matrix,
        index=centered.index,
        columns=centered.index,
    )

    # A user's similarity with themselves is always 1.0 — set to 0
    # so they don't appear in their own neighbourhood.
    np.fill_diagonal(sim_df.values, 0)

    return sim_df


# --------------------------------------------------------------------------
# 3. Predict a rating for one (user, item) pair
# --------------------------------------------------------------------------

def predict_rating(
    user_id: str,
    item_id: str,
    matrix: pd.DataFrame,
    centered: pd.DataFrame,
    sim_df: pd.DataFrame,
    user_means: pd.Series,
    k: int = 30,
) -> float | None:
    """
    Predict what `user_id` would rate `item_id` using their K nearest
    neighbours who have actually rated that item.

    Formula (weighted average of neighbour deviations):

        pred(u, i) = mean(u) + sum( sim(u,v) * centered(v,i) )
                                    ---------------------------------
                                        sum( |sim(u,v)| )

        where the sum is over the K neighbours v who rated item i.

    Why add mean(u) back?
      We predicted the *deviation* from the user's mean. Adding mean(u)
      back gives a rating on the original 1-5 scale.

    Returns None if no neighbour has rated the item (can't predict).
    """
    # Item must exist in training matrix
    if item_id not in matrix.columns:
        return None

    # Get all neighbours sorted by similarity (descending)
    user_similarities = sim_df.loc[user_id].sort_values(ascending=False)

    # Keep only neighbours who have actually rated this item
    # (non-NaN in the centered matrix for this item)
    item_ratings = centered[item_id].dropna()  # users who rated this item

    # Intersection: neighbours who rated the item
    common = user_similarities.index.intersection(item_ratings.index)
    if len(common) == 0:
        return None

    # Take top-K from the common set
    top_k = user_similarities.loc[common].head(k)

    # Weighted average of their mean-centered ratings
    neighbour_centered_ratings = centered.loc[top_k.index, item_id]
    weights = top_k.values
    weight_sum = np.abs(weights).sum()

    if weight_sum == 0:
        return None

    deviation = np.dot(weights, neighbour_centered_ratings) / weight_sum

    # Un-center: add back the target user's mean
    prediction = user_means[user_id] + deviation

    # Clip to valid rating range [1, 5]
    return float(np.clip(prediction, 1.0, 5.0))


# --------------------------------------------------------------------------
# 4. Generate top-N recommendations for a user
# --------------------------------------------------------------------------

def recommend(
    user_id: str,
    matrix: pd.DataFrame,
    centered: pd.DataFrame,
    sim_df: pd.DataFrame,
    user_means: pd.Series,
    n: int = 10,
    k: int = 30,
) -> list[tuple[str, float]]:
    """
    Return the top-N item recommendations for a user.

    Only predicts ratings for items the user has NOT yet rated.
    Items with no neighbour coverage are skipped (no prediction possible).

    Returns:
      List of (item_id, predicted_rating) tuples, sorted descending.
    """
    # Items this user has already rated — we exclude these
    rated_items = set(matrix.loc[user_id].dropna().index)

    # Candidate items: everything in the matrix the user hasn't rated
    candidate_items = [
        item for item in matrix.columns if item not in rated_items
    ]

    predictions = []
    for item_id in candidate_items:
        pred = predict_rating(
            user_id, item_id, matrix, centered, sim_df, user_means, k=k
        )
        if pred is not None:
            predictions.append((item_id, pred))

    # Sort by predicted rating descending, return top-N
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n]


# --------------------------------------------------------------------------
# 5. Batch predict for evaluation (predict held-out test ratings)
# --------------------------------------------------------------------------

def batch_predict(
    test_df: pd.DataFrame,
    matrix: pd.DataFrame,
    centered: pd.DataFrame,
    sim_df: pd.DataFrame,
    user_means: pd.Series,
    k: int = 30,
) -> pd.DataFrame:
    """
    Predict ratings for all (user, item) pairs in test_df.

    Used in evaluate.py to compute NDCG, HR, MAP on held-out ratings.

    Returns test_df with an added 'predicted_rating' column.
    NaN where we couldn't predict (no neighbour coverage).
    """
    results = []
    for _, row in test_df.iterrows():
        user_id = row["user_id"]
        item_id = row["item_id"]

        # Skip users/items not seen during training
        if user_id not in sim_df.index:
            results.append(None)
            continue

        pred = predict_rating(
            user_id, item_id, matrix, centered, sim_df, user_means, k=k
        )
        results.append(pred)

    test_df = test_df.copy()
    test_df["predicted_rating"] = results
    return test_df


# --------------------------------------------------------------------------
# 6. Fit function — wraps steps 1 & 2 into a clean interface
# --------------------------------------------------------------------------

def fit(matrix: pd.DataFrame) -> dict:
    """
    Fit the User CF model on the training user-item matrix.

    Returns a model dict containing everything needed for inference.
    We use a plain dict (not a class) for now — clean, inspectable,
    easy to serialise. We'll move to a proper class in Phase 3.
    """
    print("[user_cf] Mean-centering ratings...")
    centered, user_means = mean_center(matrix)

    print("[user_cf] Computing user-user similarity matrix...")
    sim_df = compute_user_similarity(centered)

    print(f"[user_cf] Fit complete. {len(matrix)} users, {len(matrix.columns)} items.")
    return {
        "matrix": matrix,
        "centered": centered,
        "user_means": user_means,
        "sim_df": sim_df,
    }


# --------------------------------------------------------------------------
# 7. Sanity check
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from data_loader import download_data, load_ratings, filter_min_interactions
    from data_loader import train_test_split, build_user_item_matrix

    # Load data
    fp          = download_data()
    df          = load_ratings(fp)
    df          = filter_min_interactions(df)
    train, test = train_test_split(df)
    matrix      = build_user_item_matrix(train)

    # Fit model
    model = fit(matrix)

    # Pick the first user and generate recommendations
    sample_user = matrix.index[0]
    recs = recommend(
        sample_user,
        model["matrix"],
        model["centered"],
        model["sim_df"],
        model["user_means"],
        n=5,
    )

    print(f"\nTop-5 recommendations for user '{sample_user[:12]}...':")
    for item_id, score in recs:
        print(f"  {item_id}  →  predicted rating: {score:.2f}")

    # Show similarity stats
    sim = model["sim_df"]
    print(f"\nSimilarity matrix shape : {sim.shape}")
    print(f"Mean similarity         : {sim.values[sim.values > 0].mean():.4f}")
    print(f"Max similarity (non-self): {sim.values.max():.4f}")