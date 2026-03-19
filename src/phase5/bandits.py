"""
bandits.py
----------
Contextual bandits for recommendation — exploration vs exploitation.

The problem static models ignore:
  Every recommendation is also a data collection decision.
  Showing item A tells you about A but not B, C, D.
  A bandit balances: show what you think is best (exploit)
  vs show something uncertain to learn its true quality (explore).

Three algorithms, increasing sophistication:

  1. UCB (Upper Confidence Bound)
     "Be optimistic about uncertain items."
     Score = estimated_reward + confidence_bonus
     confidence_bonus shrinks as you show the item more.
     Pure count-based — no features.

  2. Thompson Sampling
     "Sample from your belief, don't just take the mean."
     Maintain a Beta distribution over each item's click probability.
     Sample once per decision — uncertain items sometimes sample high.
     Naturally balances exploration without a tuning parameter.

  3. LinUCB (Linear UCB)
     "Use features to estimate reward, UCB on the residual."
     Fits a linear model: reward = theta * context_features
     Confidence bound from the covariance of feature estimates.
     Production-grade: handles user+item context, scales well.

Real-world use:
  - Google News: LinUCB for article recommendation (Li et al. 2010)
  - Yahoo! Front Page: UCB for content personalisation
  - Netflix: Thompson sampling for thumbnail selection
  - Spotify: bandits for playlist diversity exploration

Reference:
  Li et al. (2010) "A Contextual-Bandit Approach to Personalized News"
  Chapelle & Li (2011) "An Empirical Evaluation of Thompson Sampling"
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from collections import defaultdict


# --------------------------------------------------------------------------
# 1. Bandit environment (simulates user clicks)
# --------------------------------------------------------------------------

class RecommendationEnvironment:
    """
    Simulates a recommendation environment for bandit evaluation.

    In production, the environment is the real world — you show an item,
    the user either clicks or doesn't, you observe the reward.

    Here we simulate using the Gift Cards interaction data:
      - Items a user interacted with in the test set = reward 1.0
      - Items they didn't interact with = reward 0.0

    This lets us evaluate bandits offline against known ground truth.
    """

    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        self.items = sorted(
            set(train_df["item_id"]) | set(test_df["item_id"])
        )
        self.item2idx = {it: i for i, it in enumerate(self.items)}
        self.n_items  = len(self.items)

        # Ground truth: items each user clicked in test set
        self.user_positives = defaultdict(set)
        for _, row in test_df.iterrows():
            self.user_positives[row["user_id"]].add(row["item_id"])

        # Training history: items each user already saw
        self.user_history = defaultdict(set)
        for _, row in train_df.iterrows():
            self.user_history[row["user_id"]].add(row["item_id"])

        self.users = list(self.user_positives.keys())

    def get_reward(self, user_id: str, item_id: str) -> float:
        """
        Observe reward for recommending item to user.
        1.0 if user interacted with item in test set, else 0.0.
        """
        return 1.0 if item_id in self.user_positives.get(user_id, set()) else 0.0

    def get_candidates(self, user_id: str,
                        n: int = 20) -> list[str]:
        """
        Get candidate items for a user — items not in their history.
        Includes both positive (test) items and random negatives.
        """
        seen = self.user_history.get(user_id, set())
        positives = list(self.user_positives.get(user_id, set()) - seen)

        # Add random negatives to make it a realistic candidate set
        rng = np.random.default_rng(hash(user_id) % (2**31))
        negatives = [it for it in self.items if it not in seen
                     and it not in set(positives)]
        rng.shuffle(negatives)
        candidates = positives + negatives[:max(0, n - len(positives))]
        rng.shuffle(candidates)
        return candidates[:n]

    def get_context(self, user_id: str,
                    item_id: str,
                    dim: int = 16) -> np.ndarray:
        """
        Build a context feature vector for (user, item) pair.

        In production: rich features (user embeddings, item features,
        session context, time of day, device, etc.)

        Here: simple hash-based features as a proxy.
        """
        rng = np.random.default_rng(
            (hash(user_id) ^ hash(item_id)) % (2**31)
        )
        return rng.standard_normal(dim).astype(np.float32)


# --------------------------------------------------------------------------
# 2. UCB (Upper Confidence Bound)
# --------------------------------------------------------------------------

class UCBBandit:
    """
    UCB1 bandit for item recommendation.

    For each item, maintain:
      n_shown[i]  : how many times item i was recommended
      sum_reward[i]: total reward received from item i

    UCB score = mean_reward + sqrt(2 * log(t) / n_shown)
                  ^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^
                  exploitation        exploration bonus

    The exploration bonus is large when n_shown is small (uncertain)
    and shrinks as we show the item more (confident).

    At t=1: exploration bonus dominates, shows everything once.
    At t=1000: items with low mean reward rarely get shown.
    """

    def __init__(self, n_items: int, alpha: float = 1.0):
        """
        alpha: exploration weight (higher = more exploration).
        alpha=1.0 is the standard UCB1 setting.
        alpha=0.5 is more conservative (less exploration).
        """
        self.n_items    = n_items
        self.alpha      = alpha
        self.n_shown    = np.zeros(n_items)
        self.sum_reward = np.zeros(n_items)
        self.t          = 0  # total steps

    def select(self, candidate_indices: list[int]) -> int:
        """Select item index with highest UCB score."""
        self.t += 1

        ucb_scores = np.full(self.n_items, np.inf)  # unshown items = inf priority
        shown_mask = self.n_shown > 0
        if shown_mask.any():
            mean_rewards = np.where(
                shown_mask,
                self.sum_reward / np.maximum(self.n_shown, 1),
                0.0
            )
            exploration = np.where(
                shown_mask,
                self.alpha * np.sqrt(
                    2 * np.log(self.t) / np.maximum(self.n_shown, 1)
                ),
                np.inf
            )
            ucb_scores = mean_rewards + exploration

        # Select highest UCB among candidates
        candidate_scores = [ucb_scores[i] for i in candidate_indices]
        best_pos = int(np.argmax(candidate_scores))
        return candidate_indices[best_pos]

    def update(self, item_idx: int, reward: float) -> None:
        """Update reward estimate for selected item."""
        self.n_shown[item_idx]    += 1
        self.sum_reward[item_idx] += reward


# --------------------------------------------------------------------------
# 3. Thompson Sampling
# --------------------------------------------------------------------------

class ThompsonSamplingBandit:
    """
    Thompson Sampling for Bernoulli bandits (click/no-click).

    Model each item's click probability as Beta(alpha, beta):
      alpha = 1 + number of clicks
      beta  = 1 + number of non-clicks

    At each step:
      1. Sample theta_i ~ Beta(alpha_i, beta_i) for each item
      2. Select item with highest sampled theta

    Why this works:
      Items with few observations have wide Beta distributions —
      they sometimes sample high (exploration).
      Items with many observations have narrow distributions —
      they reliably sample near their true mean (exploitation).

    No tuning parameter needed — exploration is automatic.
    """

    def __init__(self, n_items: int):
        # Beta distribution parameters: alpha=1, beta=1 (uniform prior)
        self.alpha = np.ones(n_items)   # 1 + clicks
        self.beta  = np.ones(n_items)   # 1 + non-clicks

    def select(self, candidate_indices: list[int],
               rng: np.random.Generator = None) -> int:
        """Sample from Beta distributions and select highest."""
        if rng is None:
            rng = np.random.default_rng()

        # Sample click probability for each candidate
        samples = np.array([
            rng.beta(self.alpha[i], self.beta[i])
            for i in candidate_indices
        ])
        best_pos = int(np.argmax(samples))
        return candidate_indices[best_pos]

    def update(self, item_idx: int, reward: float) -> None:
        """Update Beta distribution based on observed reward."""
        if reward > 0:
            self.alpha[item_idx] += 1.0
        else:
            self.beta[item_idx]  += 1.0

    def get_confidence(self, item_idx: int) -> float:
        """
        Return confidence in our estimate of item i's click rate.
        Higher = more certain. Based on effective sample size.
        """
        n = self.alpha[item_idx] + self.beta[item_idx] - 2
        return min(1.0, n / 20.0)


# --------------------------------------------------------------------------
# 4. LinUCB (Contextual bandit)
# --------------------------------------------------------------------------

class LinUCBBandit:
    """
    LinUCB: Linear contextual bandit.

    Models reward as linear in context features:
      E[reward | context x] = theta^T * x

    Maintains per-item ridge regression estimates of theta.
    UCB = theta^T * x + alpha * sqrt(x^T * A^{-1} * x)
                                    ^^^^^^^^^^^^^^^^^^^
                                    uncertainty in this context direction

    Why contextual bandits?
      UCB and Thompson treat each item independently — they don't
      know that "user likes gaming" makes Steam card more likely.
      LinUCB uses features to generalise across similar (user, item) pairs.

    Reference: Li et al. (2010) "Contextual-Bandit Approach to News Rec."
    """

    def __init__(self, n_items: int, context_dim: int = 16,
                 alpha: float = 1.0, reg: float = 1.0):
        """
        context_dim: dimension of (user, item) context vector
        alpha      : exploration parameter (UCB confidence width)
        reg        : L2 regularisation for ridge regression
        """
        self.n_items     = n_items
        self.context_dim = context_dim
        self.alpha       = alpha

        # Per-item ridge regression state
        # A[i] = X^T X + reg * I  (context covariance)
        # b[i] = X^T y             (context-reward correlation)
        self.A = np.array([reg * np.eye(context_dim)] * n_items)
        self.b = np.zeros((n_items, context_dim))

    def _theta(self, item_idx: int) -> np.ndarray:
        """Compute ridge regression estimate for item."""
        return np.linalg.solve(self.A[item_idx], self.b[item_idx])

    def select(self, candidate_indices: list[int],
               contexts: dict[int, np.ndarray]) -> int:
        """
        Select item with highest LinUCB score.

        contexts: dict of {item_idx: context_vector}
        """
        best_score, best_idx = -np.inf, candidate_indices[0]

        for item_idx in candidate_indices:
            x     = contexts.get(item_idx, np.zeros(self.context_dim))
            theta = self._theta(item_idx)

            # Exploitation: linear prediction
            exploit = float(theta @ x)

            # Exploration: uncertainty in this context direction
            A_inv    = np.linalg.inv(self.A[item_idx])
            explore  = self.alpha * np.sqrt(float(x @ A_inv @ x))

            score = exploit + explore
            if score > best_score:
                best_score, best_idx = score, item_idx

        return best_idx

    def update(self, item_idx: int, context: np.ndarray,
               reward: float) -> None:
        """Update ridge regression with observed (context, reward)."""
        x = context.reshape(-1, 1)
        self.A[item_idx] += x @ x.T
        self.b[item_idx] += reward * context


# --------------------------------------------------------------------------
# 5. Evaluation loop
# --------------------------------------------------------------------------

def evaluate_bandit(
    bandit,
    env: RecommendationEnvironment,
    n_steps: int = 500,
    n_candidates: int = 20,
    use_context: bool = False,
    seed: int = 42,
) -> dict:
    """
    Run bandit for n_steps and measure cumulative reward and hit rate.

    Each step:
      1. Sample a random user
      2. Get candidate items
      3. Bandit selects one item
      4. Observe reward
      5. Update bandit

    Metrics:
      cumulative_reward : total clicks observed
      hit_rate          : fraction of steps where reward=1
      cumulative_regret : gap vs oracle (always picks a positive item)
    """
    rng = np.random.default_rng(seed)
    users = env.users
    total_reward = 0.0
    rewards = []

    for step in range(n_steps):
        user_id = users[step % len(users)]
        candidates = env.get_candidates(user_id, n=n_candidates)

        if not candidates:
            rewards.append(0.0)
            continue

        candidate_indices = [env.item2idx[it] for it in candidates]

        if use_context:
            # LinUCB: build context for each candidate
            contexts = {
                idx: env.get_context(user_id, candidates[j])
                for j, idx in enumerate(candidate_indices)
            }
            selected_idx = bandit.select(candidate_indices, contexts)
        else:
            if hasattr(bandit, 'beta'):
                selected_idx = bandit.select(candidate_indices, rng=rng)
            else:
                selected_idx = bandit.select(candidate_indices)

        selected_item = env.items[selected_idx]
        reward = env.get_reward(user_id, selected_item)

        if use_context:
            ctx = contexts[selected_idx]
            bandit.update(selected_idx, ctx, reward)
        else:
            bandit.update(selected_idx, reward)

        total_reward += reward
        rewards.append(reward)

    rewards_arr = np.array(rewards)
    return {
        "cumulative_reward" : float(total_reward),
        "hit_rate"          : float(total_reward / n_steps),
        "hit_rate_first_100": float(rewards_arr[:100].mean()),
        "hit_rate_last_100" : float(rewards_arr[-100:].mean()),
        "improvement"       : float(rewards_arr[-100:].mean() -
                                    rewards_arr[:100].mean()),
    }


# --------------------------------------------------------------------------
# 6. Sanity check
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

    env = RecommendationEnvironment(train_df, test_df)
    n_items = env.n_items

    print(f"Environment: {len(env.users)} users, "
          f"{n_items} items, "
          f"{sum(len(v) for v in env.user_positives.values())} positive interactions\n")

    N_STEPS = 1000

    print(f"{'Algorithm':<25} {'Hit Rate':<12} {'First 100':<12} "
          f"{'Last 100':<12} {'Improvement'}")
    print("-" * 65)

    # Random baseline
    rng_base = np.random.default_rng(42)
    random_rewards = []
    for step in range(N_STEPS):
        user_id = env.users[step % len(env.users)]
        candidates = env.get_candidates(user_id, n=20)
        if candidates:
            item = rng_base.choice(candidates)
            random_rewards.append(env.get_reward(user_id, item))
        else:
            random_rewards.append(0.0)
    random_arr = np.array(random_rewards)
    print(f"{'Random':<25} {random_arr.mean():<12.4f} "
          f"{random_arr[:100].mean():<12.4f} "
          f"{random_arr[-100:].mean():<12.4f} "
          f"{random_arr[-100:].mean()-random_arr[:100].mean():.4f}")

    # UCB
    ucb = UCBBandit(n_items, alpha=1.0)
    res = evaluate_bandit(ucb, env, N_STEPS)
    print(f"{'UCB (alpha=1.0)':<25} {res['hit_rate']:<12.4f} "
          f"{res['hit_rate_first_100']:<12.4f} "
          f"{res['hit_rate_last_100']:<12.4f} "
          f"{res['improvement']:.4f}")

    # Thompson Sampling
    ts = ThompsonSamplingBandit(n_items)
    res = evaluate_bandit(ts, env, N_STEPS)
    print(f"{'Thompson Sampling':<25} {res['hit_rate']:<12.4f} "
          f"{res['hit_rate_first_100']:<12.4f} "
          f"{res['hit_rate_last_100']:<12.4f} "
          f"{res['improvement']:.4f}")

    # LinUCB
    linucb = LinUCBBandit(n_items, context_dim=16, alpha=0.5)
    res = evaluate_bandit(linucb, env, N_STEPS, use_context=True)
    print(f"{'LinUCB (alpha=0.5)':<25} {res['hit_rate']:<12.4f} "
          f"{res['hit_rate_first_100']:<12.4f} "
          f"{res['hit_rate_last_100']:<12.4f} "
          f"{res['improvement']:.4f}")

    print("\nKey metric: Improvement = last_100_hit_rate - first_100_hit_rate")
    print("Positive improvement = bandit is learning and getting better over time")
    print("Random has ~0 improvement — it never learns")