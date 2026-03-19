"""
offline_rl.py
-------------
Offline Reinforcement Learning for recommendation systems.

The problem bandits don't solve:
  Bandits optimise for the next click.
  But engagement is a long-term game — a user who gets great
  recommendations today comes back tomorrow.
  RL models this as a Markov Decision Process (MDP):
    State    : user's current interaction history
    Action   : which item to recommend
    Reward   : click, purchase, session length, return visit
    Transition: how user state changes after each interaction

Why OFFLINE RL (not online)?
  Online RL would explore by recommending random items to real users.
  That's expensive and harmful — you'd annoy millions of users to
  learn a marginally better policy.
  Offline RL learns from historical logged data without any live
  exploration. You learn the best policy you can from what you have.

The core challenge — distributional shift:
  Your logged data was collected by policy pi_behaviour (your old system).
  You want to learn policy pi_new that's better.
  But if pi_new takes actions not in the logs, you have no reward signal
  for those actions. Naive Q-learning exploits these gaps and produces
  wildly overconfident Q-values for out-of-distribution actions.

CQL (Conservative Q-Learning) fix:
  Add a penalty that pushes down Q-values for actions not in the logs.
  This makes the policy conservative — it sticks to well-supported actions.

  CQL loss = standard Bellman loss
           + alpha * (E[Q(s,a)] over random actions    <- push down OOD
                    - E[Q(s,a)] over logged actions)    <- push up in-dist

Reference:
  Kumar et al. (2020) "Conservative Q-Learning for Offline Reinforcement
  Learning" NeurIPS.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from dataclasses import dataclass


# --------------------------------------------------------------------------
# 1. MDP formulation for RecSys
# --------------------------------------------------------------------------

@dataclass
class Transition:
    """
    A single (state, action, reward, next_state, done) tuple.
    The fundamental unit of RL experience.

    state      : embedding of user's recent interaction history
    action     : item index recommended
    reward     : observed feedback (click=1, no-click=0)
    next_state : user state after the interaction
    done       : whether the session ended
    """
    state      : np.ndarray
    action     : int
    reward     : float
    next_state : np.ndarray
    done       : bool


def build_replay_buffer(
    train_df: pd.DataFrame,
    item2idx: dict,
    state_dim: int = 32,
    max_transitions: int = 5000,
) -> list:
    """
    Convert historical interaction logs into RL transitions.

    Each user's interaction sequence becomes a series of transitions:
      state[t]      = embedding of items seen so far (mean pooling)
      action[t]     = item shown at step t
      reward[t]     = 1.0 (all logged interactions are positives)
      next_state[t] = state after adding item t to history
      done[t]       = True if last interaction for this user

    State representation: mean of item one-hot encodings
    (simple but effective — in production use two-tower embeddings)
    """
    n_items = len(item2idx)
    transitions = []
    rng = np.random.default_rng(42)

    for user_id, group in train_df.groupby("user_id"):
        items = group["item_id"].tolist()
        if len(items) < 2:
            continue

        # Build state as running mean of seen item embeddings
        # Simple hash-based embeddings as proxy for real item vectors
        def item_embedding(item_id: str) -> np.ndarray:
            seed = hash(item_id) % (2**31)
            return np.random.default_rng(seed).standard_normal(state_dim)

        history_emb = np.zeros(state_dim)
        for t, item_id in enumerate(items[:-1]):
            action = item2idx.get(item_id, 0)
            item_emb = item_embedding(item_id)

            state = history_emb.copy()
            if t > 0:
                state = state / t  # mean of history

            next_item = items[t + 1]
            next_emb  = item_embedding(next_item)
            next_state = (history_emb + next_emb) / (t + 1)

            # Reward: 1.0 for all logged interactions (they were positive)
            # In production: use actual CTR, dwell time, purchase signal
            reward = 1.0

            # Add some negative transitions for training balance
            transitions.append(Transition(
                state=state.astype(np.float32),
                action=action,
                reward=reward,
                next_state=next_state.astype(np.float32),
                done=(t == len(items) - 2),
            ))

            # Negative transition: random unclicked item
            neg_item = items[rng.integers(0, len(items))]
            neg_action = item2idx.get(neg_item, 0)
            transitions.append(Transition(
                state=state.astype(np.float32),
                action=neg_action,
                reward=0.0,
                next_state=state.astype(np.float32),
                done=False,
            ))

            history_emb += item_emb

        if len(transitions) >= max_transitions:
            break

    print(f"[offline_rl] Replay buffer: {len(transitions)} transitions")
    return transitions


# --------------------------------------------------------------------------
# 2. Q-Network
# --------------------------------------------------------------------------

class QNetwork(nn.Module):
    """
    Q-function approximator: Q(state, action) -> expected return.

    Architecture: state embedding -> MLP -> Q-value per action
    Output shape: (batch, n_items) — Q-value for every action at once.

    This is the DQN (Deep Q-Network) architecture from Mnih et al. (2015),
    adapted for recommendation.
    """

    def __init__(self, state_dim: int, n_items: int,
                 hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_items),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)  # (batch, n_items)


# --------------------------------------------------------------------------
# 3. CQL training
# --------------------------------------------------------------------------

def train_cql(
    q_net: QNetwork,
    replay_buffer: list,
    n_epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    gamma: float = 0.99,
    cql_alpha: float = 1.0,
) -> list:
    """
    Train Q-network with Conservative Q-Learning.

    Standard DQN loss (Bellman):
      L_bellman = E[(Q(s,a) - (r + gamma * max_a' Q(s',a')))^2]

    CQL penalty:
      L_cql = E[log sum_a exp(Q(s,a))]   <- pushes down all Q-values
            - E[Q(s, a_logged)]           <- pushes up logged action Q-values

    Total loss = L_bellman + cql_alpha * L_cql

    The CQL penalty prevents the policy from assigning high Q-values
    to actions not in the logged data — the conservative constraint.

    gamma: discount factor. 0.99 means future rewards are nearly as
    valuable as immediate rewards. 0.9 means we care much less about
    rewards >10 steps away.
    """
    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
    # Target network for stable Bellman targets (standard DQN trick)
    target_net = QNetwork(
        q_net.net[0].in_features,
        q_net.net[-1].out_features,
    )
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    epoch_losses = []
    n_transitions = len(replay_buffer)

    print(f"[offline_rl] Training CQL: {n_epochs} epochs, "
          f"batch={batch_size}, alpha={cql_alpha}")

    for epoch in range(1, n_epochs + 1):
        # Shuffle replay buffer
        indices = np.random.permutation(n_transitions)
        batch_losses = []

        for start in range(0, n_transitions - batch_size, batch_size):
            batch_idx = indices[start:start + batch_size]
            batch = [replay_buffer[i] for i in batch_idx]

            states      = torch.tensor(
                np.array([t.state for t in batch]), dtype=torch.float32
            )
            actions     = torch.tensor(
                [t.action for t in batch], dtype=torch.long
            )
            rewards     = torch.tensor(
                [t.reward for t in batch], dtype=torch.float32
            )
            next_states = torch.tensor(
                np.array([t.next_state for t in batch]), dtype=torch.float32
            )
            dones       = torch.tensor(
                [t.done for t in batch], dtype=torch.float32
            )

            # Current Q-values for logged actions
            q_values = q_net(states)                        # (B, n_items)
            q_logged = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Bellman target using target network
            with torch.no_grad():
                q_next = target_net(next_states).max(1)[0]  # (B,)
                target = rewards + gamma * q_next * (1 - dones)

            # Bellman loss
            bellman_loss = F.mse_loss(q_logged, target)

            # CQL penalty: log-sum-exp over all actions minus logged Q-value
            # This pushes down Q-values for unlogged (OOD) actions
            cql_loss = (torch.logsumexp(q_values, dim=1).mean()
                        - q_logged.mean())

            loss = bellman_loss + cql_alpha * cql_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(loss.item())

        # Soft update target network
        for param, target_param in zip(
            q_net.parameters(), target_net.parameters()
        ):
            target_param.data.copy_(0.05 * param.data +
                                     0.95 * target_param.data)

        epoch_loss = np.mean(batch_losses)
        epoch_losses.append(epoch_loss)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{n_epochs}  loss={epoch_loss:.4f}")

    return epoch_losses


# --------------------------------------------------------------------------
# 4. Policy evaluation
# --------------------------------------------------------------------------

def evaluate_policy(
    q_net: QNetwork,
    env,
    item2idx: dict,
    items: list,
    state_dim: int = 32,
    n_episodes: int = 200,
    episode_len: int = 5,
) -> dict:
    """
    Evaluate the learned Q-policy on the recommendation environment.

    Each episode:
      1. Sample a user
      2. Build their state from training history
      3. Greedily select the action with highest Q-value (exploit)
      4. Observe reward from environment
      5. Update state

    Metrics: hit rate per step, cumulative reward per episode.
    """
    def item_embedding(item_id: str, dim: int) -> np.ndarray:
        seed = hash(item_id) % (2**31)
        return np.random.default_rng(seed).standard_normal(dim)

    q_net.eval()
    episode_rewards = []

    for ep in range(n_episodes):
        user_id = env.users[ep % len(env.users)]
        user_history = list(env.user_history.get(user_id, set()))

        # Build initial state from user history
        if user_history:
            embs = np.array([item_embedding(it, state_dim)
                             for it in user_history[-5:]])
            state = embs.mean(axis=0).astype(np.float32)
        else:
            state = np.zeros(state_dim, dtype=np.float32)

        ep_reward = 0.0
        seen = set(user_history)

        for step in range(episode_len):
            # Get Q-values for all items
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_vals = q_net(state_t).squeeze(0).numpy()

            # Mask seen items
            for it in seen:
                idx = item2idx.get(it, -1)
                if 0 <= idx < len(q_vals):
                    q_vals[idx] = -np.inf

            # Greedy action
            action_idx = int(np.argmax(q_vals))
            if action_idx < len(items):
                selected_item = items[action_idx]
            else:
                break

            # Observe reward
            reward = env.get_reward(user_id, selected_item)
            ep_reward += reward
            seen.add(selected_item)

            # Update state
            item_emb = item_embedding(selected_item, state_dim)
            state = (state + item_emb) / 2.0

        episode_rewards.append(ep_reward / episode_len)

    return {
        "mean_hit_rate"   : float(np.mean(episode_rewards)),
        "std_hit_rate"    : float(np.std(episode_rewards)),
        "n_episodes"      : n_episodes,
    }


# --------------------------------------------------------------------------
# 5. Sanity check
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_loader import (download_data, load_ratings,
                             filter_min_interactions, train_test_split)
    from bandits import RecommendationEnvironment

    fp = download_data()
    df = load_ratings(fp)
    df = filter_min_interactions(df)
    train_df, test_df = train_test_split(df)

    # Build environment and item index
    env = RecommendationEnvironment(train_df, test_df)
    items   = env.items
    item2idx = env.item2idx
    n_items  = env.n_items
    STATE_DIM = 32

    # Build replay buffer from interaction logs
    buffer = build_replay_buffer(
        train_df, item2idx,
        state_dim=STATE_DIM,
        max_transitions=5000,
    )

    # Compare: vanilla DQN (no CQL) vs CQL
    print("\nTraining vanilla DQN (no conservative penalty)...")
    dqn = QNetwork(STATE_DIM, n_items, hidden_dim=128)
    train_cql(dqn, buffer, n_epochs=20, batch_size=128,
              cql_alpha=0.0)  # alpha=0 → standard DQN

    print("\nTraining CQL (conservative Q-learning)...")
    cql = QNetwork(STATE_DIM, n_items, hidden_dim=128)
    train_cql(cql, buffer, n_epochs=20, batch_size=128,
              cql_alpha=1.0)  # alpha=1 → conservative

    # Evaluate both policies
    print("\nEvaluating policies...")
    dqn_res = evaluate_policy(dqn, env, item2idx, items,
                               state_dim=STATE_DIM, n_episodes=200)
    cql_res = evaluate_policy(cql, env, item2idx, items,
                               state_dim=STATE_DIM, n_episodes=200)

    # Random baseline
    rng = np.random.default_rng(42)
    random_rewards = []
    for ep in range(200):
        user_id = env.users[ep % len(env.users)]
        cands = env.get_candidates(user_id, n=10)
        ep_r = 0.0
        for _ in range(5):
            if cands:
                item = rng.choice(cands)
                ep_r += env.get_reward(user_id, item)
        random_rewards.append(ep_r / 5.0)

    print(f"\n{'Policy':<25} {'Hit Rate':<14} {'Std'}")
    print("-" * 45)
    print(f"{'Random':<25} {np.mean(random_rewards):<14.4f} "
          f"{np.std(random_rewards):.4f}")
    print(f"{'DQN (no CQL)':<25} {dqn_res['mean_hit_rate']:<14.4f} "
          f"{dqn_res['std_hit_rate']:.4f}")
    print(f"{'CQL (conservative)':<25} {cql_res['mean_hit_rate']:<14.4f} "
          f"{cql_res['std_hit_rate']:.4f}")

    print("\nKey insight:")
    print("  CQL's hit rate is often lower than DQN's on this dataset,")
    print("  but DQN is overconfident on OOD actions — its Q-values are")
    print("  inflated for items rarely seen in logs.")
    print("  CQL trades peak performance for reliability and safe deployment.")
    print("  In production: CQL policies are safer to A/B test — they won't")
    print("  catastrophically recommend items with zero interaction history.")