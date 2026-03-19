"""
debiasing.py
------------
Position bias estimation and Inverse Propensity Scoring (IPS).

The problem:
  Click logs are biased — items shown at position 1 get clicked more
  than items at position 5, regardless of actual quality. Training or
  evaluating on raw clicks teaches models to replicate exposure bias,
  not to find genuinely good items.

IPS solution:
  Weight each observation by 1 / P(seen | position k).
  A click at position 1 (always seen) counts less.
  A click at position 5 (rarely seen) counts more.
  This gives an unbiased estimate of true item quality.

Three propensity models implemented:
  1. Position-based  : P(seen|k) = 1/k  (simple, no data needed)
  2. Examination     : P(seen|k) = (1/k)^eta  (tunable decay)
  3. Empirical       : estimated from click-through rates per position
                       (requires randomisation experiment data)

Reference:
  Joachims et al. (2017) "Unbiased Learning-to-Rank with Biased Feedback"
  Wang et al. (2018) "Position-Biased PageRank for Personalized Suggestions"
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


# --------------------------------------------------------------------------
# 1. Propensity models
# --------------------------------------------------------------------------

def position_propensity(max_position: int = 20) -> np.ndarray:
    """
    Simplest propensity model: P(seen | position k) = 1/k.

    Assumes examination probability decreases inversely with position.
    Works well as a baseline when no click data is available.

    Returns array of shape (max_position,) with propensities for
    positions 1 through max_position.
    """
    return np.array([1.0 / (k + 1) for k in range(max_position)])


def examination_propensity(max_position: int = 20,
                           eta: float = 0.5) -> np.ndarray:
    """
    Power-law examination model: P(seen | position k) = (1/k)^eta.

    eta controls how steeply attention drops with position:
      eta=0.0 → uniform (no position bias)
      eta=0.5 → moderate decay (typical for search results)
      eta=1.0 → same as position_propensity (strong decay)
      eta>1.0 → very strong decay (users rarely scroll)

    This is the standard model in the unbiased LTR literature.
    eta is typically estimated from randomised result interleaving.
    """
    return np.array([(1.0 / (k + 1)) ** eta for k in range(max_position)])


def empirical_propensity(click_df: pd.DataFrame,
                         max_position: int = 20) -> np.ndarray:
    """
    Estimate propensities from observed click-through rates per position.

    Requires a DataFrame with columns: [position, clicked]
    where position is 1-indexed and clicked is 0/1.

    CTR(k) ≈ quality * P(seen|k)
    If we assume average quality is constant across positions
    (true under randomised exposure), then:
      P(seen|k) ∝ CTR(k)

    We normalise so P(seen|1) = 1.0 (position 1 is always seen).

    In production this data comes from interleaving experiments
    or propensity-scored A/B tests where some results are shown
    at random positions.
    """
    propensities = np.ones(max_position) * 0.1  # default fallback

    if "position" not in click_df.columns or "clicked" not in click_df.columns:
        return propensities

    ctr_by_pos = (click_df.groupby("position")["clicked"]
                  .mean()
                  .reindex(range(1, max_position + 1))
                  .fillna(0.01))

    ctr_values = ctr_by_pos.values
    # Normalise: P(seen|1) = 1.0
    if ctr_values[0] > 0:
        propensities = ctr_values / ctr_values[0]
    else:
        propensities = ctr_values

    return np.clip(propensities, 0.01, 1.0)


# --------------------------------------------------------------------------
# 2. IPS weighting
# --------------------------------------------------------------------------

def ips_weights(positions: np.ndarray,
                propensities: np.ndarray,
                clip_max: float = 10.0) -> np.ndarray:
    """
    Compute IPS weights for a set of observations.

    weight(i) = 1 / P(seen | position[i])

    positions   : array of 1-indexed positions for each observation
    propensities: array of P(seen|k) for k=0..max_position-1
    clip_max    : cap extreme weights to prevent high-variance estimates

    Clipping is the standard variance-reduction technique for IPS.
    Without clipping, a single low-propensity item can dominate the loss.
    """
    # Convert 1-indexed positions to 0-indexed
    idxs = np.clip(positions - 1, 0, len(propensities) - 1).astype(int)
    weights = 1.0 / np.maximum(propensities[idxs], 1e-6)
    return np.clip(weights, 1.0, clip_max)


# --------------------------------------------------------------------------
# 3. IPS-corrected NDCG
# --------------------------------------------------------------------------

def ips_ndcg(
    recommended: list,
    relevant: set,
    propensities: np.ndarray,
    k: int = 10,
) -> float:
    """
    IPS-corrected NDCG@K.

    Standard NDCG assumes all items in the test set were equally
    likely to be interacted with — but position-biased click data
    violates this. Items at lower positions had less chance to be
    clicked and are underrepresented in the test set.

    IPS-NDCG reweights each hit by 1/P(seen|position) to correct
    for this bias, giving a fairer estimate of ranking quality.

    Formula:
      IPS-DCG@K = sum_k [ rel(k) * (1/propensity(k)) / log2(k+2) ]
      IPS-NDCG  = IPS-DCG / IPS-IDCG

    In practice IPS-NDCG is less stable than standard NDCG due to
    high variance in IPS weights — use with sufficient test data.
    """
    if not recommended or not relevant:
        return 0.0

    dcg = 0.0
    for rank, item in enumerate(recommended[:k]):
        if item in relevant:
            prop = propensities[min(rank, len(propensities) - 1)]
            ips_w = 1.0 / max(prop, 1e-6)
            dcg += ips_w / np.log2(rank + 2)

    # IDCG: perfect ranking with IPS weights
    n_rel = min(len(relevant), k)
    idcg = sum(
        (1.0 / max(propensities[min(r, len(propensities) - 1)], 1e-6))
        / np.log2(r + 2)
        for r in range(n_rel)
    )

    return dcg / idcg if idcg > 0 else 0.0


# --------------------------------------------------------------------------
# 4. Simulate position bias in click data
# --------------------------------------------------------------------------

def simulate_biased_clicks(
    df: pd.DataFrame,
    propensities: np.ndarray,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate position-biased click data from interaction logs.

    In real systems, you'd use actual logged positions. Here we simulate
    by assigning random positions to interactions and applying
    position-dependent click probability.

    This lets us demonstrate IPS correction on our Gift Cards dataset
    even without real position logs.

    Returns DataFrame with columns: [user_id, item_id, position, clicked]
    """
    rng = np.random.default_rng(seed)
    n = len(df)

    # Assign random positions 1..max_position
    max_pos = len(propensities)
    positions = rng.integers(1, max_pos + 1, size=n)

    # Click probability = base_quality * P(seen | position)
    # Base quality: 1.0 for observed interactions (they were relevant)
    base_quality = 0.8
    prop_at_pos = propensities[positions - 1]
    click_prob = base_quality * prop_at_pos
    clicked = (rng.random(n) < click_prob).astype(int)

    result = df[["user_id", "item_id"]].copy()
    result["position"] = positions
    result["clicked"]  = clicked
    return result


# --------------------------------------------------------------------------
# 5. Sanity check
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

    # Compare propensity models
    print("Propensity models at positions 1-10:")
    print(f"{'Position':<10} {'1/k':<12} {'(1/k)^0.5':<12} {'(1/k)^0.3'}")
    print("-" * 46)
    p1 = position_propensity(10)
    p2 = examination_propensity(10, eta=0.5)
    p3 = examination_propensity(10, eta=0.3)
    for k in range(10):
        print(f"{k+1:<10} {p1[k]:<12.4f} {p2[k]:<12.4f} {p3[k]:.4f}")

    # Simulate biased clicks and estimate empirical propensities
    print("\nSimulating biased click data...")
    prop_true = examination_propensity(20, eta=0.5)
    click_df = simulate_biased_clicks(train_df, prop_true)
    prop_emp = empirical_propensity(click_df, max_position=20)

    print(f"\n{'Position':<10} {'True P(seen)':<16} {'Estimated P(seen)'}")
    print("-" * 44)
    for k in range(min(10, len(prop_true))):
        print(f"{k+1:<10} {prop_true[k]:<16.4f} {prop_emp[k]:.4f}")

    # IPS weight distribution
    positions = np.arange(1, 11)
    weights = ips_weights(positions, prop_true, clip_max=10.0)
    print(f"\nIPS weights (clip_max=10):")
    print(f"{'Position':<10} {'Propensity':<14} {'IPS weight'}")
    print("-" * 36)
    for k, (p, w) in enumerate(zip(prop_true[:10], weights)):
        print(f"{k+1:<10} {p:<14.4f} {w:.4f}")

    print("\nKey insight: position 1 click weight=1.0, "
          "position 10 click weight=3.16")
    print("IPS amplifies rare low-position signals to correct for bias.")