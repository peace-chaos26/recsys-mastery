"""
matrix_factor.py
----------------
Matrix Factorization for recommendation systems.

Implements two approaches:
  1. SVD  — classic truncated SVD on the explicit rating matrix
  2. ALS  — Alternating Least Squares on implicit feedback

Key concept:
  Both decompose R (users x items) into U (users x k) and V (items x k).
  predict(u, i) = dot(U[u], V[i]) — always computable, no overlap needed.
  Coverage jumps from 12% (user CF) to ~95%+.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


# --------------------------------------------------------------------------
# 1. SVD-based Matrix Factorization (explicit ratings)
# --------------------------------------------------------------------------

class SVDRecommender:
    """
    Truncated SVD on the mean-centered user-item matrix.

    R_centered ≈ U × Σ × Vᵀ

    We keep only the top-k singular values/vectors — same idea as PCA,
    finding the k directions of maximum variance in the rating matrix.
    """

    def __init__(self, n_factors: int = 20):
        self.n_factors    = n_factors
        self.user_factors = None
        self.item_factors = None
        self.user_means   = None
        self.user_index   = None
        self.item_index   = None

    def fit(self, matrix: pd.DataFrame) -> "SVDRecommender":
        print(f"[svd] Fitting SVD with {self.n_factors} factors ...")

        self.user_index = matrix.index.tolist()
        self.item_index = matrix.columns.tolist()
        self.user_means = matrix.mean(axis=1)

        centered = matrix.sub(self.user_means, axis=0).fillna(0)
        R = centered.values.astype(np.float32)

        k = min(self.n_factors, min(R.shape) - 1)
        U, sigma, Vt = svds(R, k=k)

        # Sort descending
        idx = np.argsort(sigma)[::-1]
        U, sigma, Vt = U[:, idx], sigma[idx], Vt[idx, :]

        self.user_factors = U * sigma   # (n_users, k)
        self.item_factors = Vt.T        # (n_items, k)

        print(f"[svd] Fit complete. "
              f"{len(self.user_index)} users, {len(self.item_index)} items, k={k}")
        return self

    def predict(self, user_id: str, item_id: str) -> float | None:
        if user_id not in self.user_index or item_id not in self.item_index:
            return None
        u = self.user_index.index(user_id)
        i = self.item_index.index(item_id)
        pred = np.dot(self.user_factors[u], self.item_factors[i])
        pred += self.user_means[user_id]
        return float(np.clip(pred, 1.0, 5.0))

    def recommend(self, user_id: str, train_matrix: pd.DataFrame,
                  n: int = 10) -> list:
        if user_id not in self.user_index:
            return []

        u = self.user_index.index(user_id)
        scores = self.user_factors[u] @ self.item_factors.T
        scores += self.user_means[user_id]
        scores = np.clip(scores, 1.0, 5.0)

        rated = set(train_matrix.loc[user_id].dropna().index)
        recs = [
            (item_id, float(scores[i]))
            for i, item_id in enumerate(self.item_index)
            if item_id not in rated
        ]
        recs.sort(key=lambda x: x[1], reverse=True)
        return recs[:n]


# --------------------------------------------------------------------------
# 2. ALS-based Matrix Factorization (implicit feedback)
# --------------------------------------------------------------------------

class ALSRecommender:
    """
    Alternating Least Squares on implicit feedback.

    Treats ratings as confidence signals:
      confidence(u, i) = 1 + alpha * rating(u, i)

    This is the Hu, Koren & Volinsky (2008) formulation — still widely used.

    Key API note for implicit 0.7.x:
      - fit()       expects user_items as (users x items) CSR
      - recommend() expects the same (users x items) CSR, one row per user
    """

    def __init__(self, n_factors: int = 50, iterations: int = 20,
                 regularization: float = 0.01, alpha: float = 40.0):
        self.n_factors      = n_factors
        self.iterations     = iterations
        self.regularization = regularization
        self.alpha          = alpha
        self.model          = None
        self.user_index     = None
        self.item_index     = None
        self.user_map       = None
        self.item_map       = None
        self.user_items_csr = None  # stored for recommend()

    def fit(self, matrix: pd.DataFrame) -> "ALSRecommender":
        try:
            import implicit
        except ImportError:
            raise ImportError("Run: pip install implicit==0.7.2")

        print(f"[als] Fitting ALS: factors={self.n_factors}, "
              f"iterations={self.iterations}, alpha={self.alpha} ...")

        self.user_index = matrix.index.tolist()
        self.item_index = matrix.columns.tolist()
        self.user_map   = {u: i for i, u in enumerate(self.user_index)}
        self.item_map   = {it: i for i, it in enumerate(self.item_index)}

        # Confidence matrix C = 1 + alpha * R, NaN treated as 0
        R = matrix.fillna(0).values.astype(np.float32)
        C = (1 + self.alpha * R).astype(np.float32)

        # Store as CSR (users x items) — reused in recommend()
        self.user_items_csr = csr_matrix(C)

        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.n_factors,
            iterations=self.iterations,
            regularization=self.regularization,
            use_gpu=False,
        )
        # implicit 0.7.x: fit() takes (users x items) directly
        self.model.fit(self.user_items_csr)

        print(f"[als] Fit complete. "
              f"{len(self.user_index)} users, {len(self.item_index)} items")
        return self

    def recommend(self, user_id: str, train_matrix: pd.DataFrame,
                  n: int = 10) -> list:
        """Return top-N recommendations, excluding already-rated items."""
        if user_id not in self.user_map:
            return []

        u_idx = self.user_map[user_id]

        # Pass the single user's row from the stored CSR matrix
        user_row = self.user_items_csr[u_idx]

        item_ids, scores = self.model.recommend(
            u_idx,
            user_row,
            N=n,
            filter_already_liked_items=True,
        )

        results = []
        for item_idx, score in zip(item_ids, scores):
            if 0 <= int(item_idx) < len(self.item_index):
                results.append((self.item_index[int(item_idx)], float(score)))

        return results


# --------------------------------------------------------------------------
# 3. Shared evaluation wrapper
# --------------------------------------------------------------------------

def evaluate_mf(model, test_df: pd.DataFrame,
                train_matrix: pd.DataFrame, k: int = 10) -> dict:
    """
    Evaluate an MF model — same protocol as evaluate.py.
    HR@K, NDCG@K, Precision@K over held-out test ratings.
    """
    from collections import defaultdict

    user_test_items = defaultdict(set)
    for _, row in test_df.iterrows():
        user_test_items[row["user_id"]].add(row["item_id"])

    hr, ndcg, prec = [], [], []
    n_covered = 0

    for user_id, relevant in user_test_items.items():
        recs = model.recommend(user_id, train_matrix, n=k)
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
# 4. Sanity check — compare SVD vs ALS
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

    results_table = []

    # SVD
    svd = SVDRecommender(n_factors=20).fit(matrix)
    for k in [5, 10]:
        res = evaluate_mf(svd, test, matrix, k=k)
        res["model"] = "SVD (k=20)"
        res["K"] = k
        results_table.append(res)

    # ALS
    als = ALSRecommender(n_factors=50, iterations=20).fit(matrix)
    for k in [5, 10]:
        res = evaluate_mf(als, test, matrix, k=k)
        res["model"] = "ALS (k=50)"
        res["K"] = k
        results_table.append(res)

    # Print comparison table
    print(f"\n{'Model':<14} {'K':<5} {'HR@K':<10} {'NDCG@K':<10} "
          f"{'Prec@K':<10} {'Coverage':<10}")
    print("-" * 60)
    for r in results_table:
        k = r["K"]
        print(f"{r['model']:<14} {k:<5} "
              f"{r[f'HR@{k}']:<10.4f} {r[f'NDCG@{k}']:<10.4f} "
              f"{r[f'Precision@{k}']:<10.4f} {r['coverage']:<10.4f}")