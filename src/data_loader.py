"""
data_loader.py
--------------
Loads the Amazon Reviews 2023 dataset (Digital Music) and builds a
user-item rating matrix.

Source: McAuley Lab, UCSD — Amazon Reviews 2023
  https://amazon-reviews-2023.github.io/

Key concepts:
  - Explicit feedback: users gave ratings 1-5 (we know what they liked)
  - Sparsity: most users rate very few items — the matrix is >99% empty
  - Cold-start filtering: users/items with too few ratings hurt similarity quality
"""

import gzip
import os
import shutil

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


# --------------------------------------------------------------------------
# 1. Download
# --------------------------------------------------------------------------

DATA_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
    "benchmark/0core/rating_only/Gift_Cards.csv.gz"
)
# Amazon Reviews 2023 — Digital Music category (~128K ratings).
# Gzipped CSV with a header row.
# Columns: user_id, parent_asin, rating, timestamp


def download_data(data_dir: str = "data") -> str:
    """
    Download and decompress the ratings CSV if not already present.
    Returns path to the decompressed CSV file.
    """
    os.makedirs(data_dir, exist_ok=True)
    gz_path  = os.path.join(data_dir, "Gift_Cards.csv.gz")
    csv_path = os.path.join(data_dir, "Gift_Cards.csv")

    if os.path.exists(csv_path):
        size_kb = os.path.getsize(csv_path) / 1024
        if size_kb > 100:
            print(f"[data_loader] Already downloaded: {csv_path} ({size_kb:.0f} KB)")
            return csv_path
        else:
            print(f"[data_loader] File looks corrupt ({size_kb:.1f} KB) — re-downloading...")
            os.remove(csv_path)

    print("[data_loader] Downloading Amazon Reviews 2023 — Digital Music ...")
    response = requests.get(DATA_URL, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(gz_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    print("[data_loader] Decompressing ...")
    with gzip.open(gz_path, "rb") as f_in, open(csv_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(gz_path)

    print(f"[data_loader] Saved to {csv_path}")
    return csv_path


# --------------------------------------------------------------------------
# 2. Load & clean
# --------------------------------------------------------------------------

def load_ratings(filepath: str) -> pd.DataFrame:
    """
    Load the CSV into a DataFrame with standardised column names.

    2023 format has a header row:
        user_id      -- string hash
        parent_asin  -- Amazon product ID (renamed to item_id)
        rating       -- float 1.0-5.0
        timestamp    -- unix epoch ms (dropped)
    """
    df = pd.read_csv(filepath)
    df = df.rename(columns={"parent_asin": "item_id"})
    df = df[["user_id", "item_id", "rating"]]
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
    """
    matrix = df.pivot_table(
        index="user_id",
        columns="item_id",
        values="rating",
        aggfunc="mean",
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
# 6. Sanity check
# --------------------------------------------------------------------------

if __name__ == "__main__":
    fp          = download_data()
    df          = load_ratings(fp)
    df          = filter_min_interactions(df)
    train, test = train_test_split(df)
    matrix      = build_user_item_matrix(train)
    print("\nSample of matrix (5x5):")
    print(matrix.iloc[:5, :5].to_string())