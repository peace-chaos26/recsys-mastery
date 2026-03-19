"""
data_loader.py
--------------
Loads the Amazon Reviews (small) dataset and builds a user-item rating matrix.

Key concepts:
  - Explicit feedback: users gave ratings 1-5 (we know what they liked)
  - Sparsity: most users rate very few items — the matrix is >99% empty
  - Cold-start filtering: users/items with too few ratings hurt similarity quality
"""

import os
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm


# --------------------------------------------------------------------------
# 1. Download
# --------------------------------------------------------------------------

DATA_URL = (
    "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
    "ratings_Digital_Music.csv"
)
# Using Digital Music — ~65k ratings, fast to download, manageable size.
# No header: columns are user_id, item_id, rating, timestamp.


def download_data(data_dir: str = "data") -> str:
    """Download the raw ratings CSV if not already present. Returns file path."""
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, "ratings_Digital_Music.csv")

    if os.path.exists(filepath):
        print(f"[data_loader] Already downloaded: {filepath}")
        return filepath

    print(f"[data_loader] Downloading from {DATA_URL} ...")
    response = requests.get(DATA_URL, stream=True, timeout=30)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(filepath, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"[data_loader] Saved to {filepath}")
    return filepath


# --------------------------------------------------------------------------
# 2. Load & clean
# --------------------------------------------------------------------------

def load_ratings(filepath: str) -> pd.DataFrame:
    """
    Load raw CSV into a DataFrame with clean column names.

    The raw file has no header — columns are positional:
        col 0: user_id   (string hash)
        col 1: item_id   (Amazon ASIN)
        col 2: rating    (float 1.0-5.0)
        col 3: timestamp (unix epoch, dropped)
    """
    df = pd.read_csv(
        filepath,
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    df = df.drop(columns=["timestamp"])
    df["rating"] = df["rating"].astype(float)
    print(f"[data_loader] Loaded {len(df):,} ratings, "
          f"{df['user_id'].nunique():,} users, "
          f"{df['item_id'].nunique():,} items")
    return df


# --------------------------------------------------------------------------
# 3. Filter cold-start users and items
# --------------------------------------------------------------------------

def filter_min_interactions(
    df: pd.DataFrame,
    min_user_ratings: int = 5,
    min_item_ratings: int = 5,
) -> pd.DataFrame:
    """
    Remove users and items with fewer than min_* ratings.

    Why this matters:
      - A user with 1 rating has an almost-empty vector — cosine similarity
        with everyone will be near-zero or meaningless.
      - An item rated by 1 person will only ever be recommended to
        near-identical users — noise at this stage.
      - This is the cold-start problem — we handle it properly in Phase 2/4
        with content signals and embeddings.

    We run 2 passes because removing sparse users may make some items
    drop below threshold, and vice versa.
    """
    before = len(df)
    for _ in range(2):
        user_counts = df["user_id"].value_counts()
        item_counts = df["item_id"].value_counts()
        df = df[
            df["user_id"].isin(user_counts[user_counts >= min_user_ratings].index)
            & df["item_id"].isin(item_counts[item_counts >= min_item_ratings].index)
        ]
    after = len(df)
    print(f"[data_loader] After filtering: {after:,} ratings "
          f"({before - after:,} removed), "
          f"{df['user_id'].nunique():,} users, "
          f"{df['item_id'].nunique():,} items")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# 4. Build user-item matrix
# --------------------------------------------------------------------------

def build_user_item_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the ratings DataFrame into a user-item matrix.

        rows    = users
        columns = items
        values  = ratings (NaN where no rating exists)

    This is the fundamental data structure for memory-based CF.
    In production this would be scipy.sparse — the matrix is >95% NaN
    even after filtering. Dense DataFrame here for clarity; we switch
    to sparse in Phase 2.
    """
    matrix = df.pivot_table(
        index="user_id",
        columns="item_id",
        values="rating",
        aggfunc="mean",  # handles duplicate (user, item) pairs by averaging
    )
    sparsity = 1 - matrix.notna().sum().sum() / matrix.size
    print(f"[data_loader] Matrix shape: {matrix.shape[0]} users x {matrix.shape[1]} items")
    print(f"[data_loader] Sparsity: {sparsity:.1%} empty")
    return matrix


# --------------------------------------------------------------------------
# 5. Train/test split
# --------------------------------------------------------------------------

def train_test_split(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Per-user split: for each user, hold out test_ratio of their ratings.

    Why per-user (not random row split)?
      - A random row split might put ALL of a user's ratings in test, leaving
        nothing to compute similarity from. Per-user guarantees every user
        has training data.
      - In RecSys this is called leave-last-N-out evaluation. We'll use
        a stricter time-ordered version in Phase 3.
    """
    rng = np.random.default_rng(random_state)
    train_rows, test_rows = [], []

    for _, group in df.groupby("user_id"):
        idx = group.index.tolist()
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * test_ratio))
        test_rows.extend(idx[:n_test])
        train_rows.extend(idx[n_test:])

    train = df.loc[train_rows].reset_index(drop=True)
    test  = df.loc[test_rows].reset_index(drop=True)
    print(f"[data_loader] Train: {len(train):,} | Test: {len(test):,} ratings")
    return train, test


# --------------------------------------------------------------------------
# 6. Sanity check — run this file directly to verify the pipeline
# --------------------------------------------------------------------------

if __name__ == "__main__":
    fp          = download_data()
    df          = load_ratings(fp)
    df          = filter_min_interactions(df)
    train, test = train_test_split(df)
    matrix      = build_user_item_matrix(train)
    print("\nSample of matrix (5x5):")
    print(matrix.iloc[:5, :5].to_string())