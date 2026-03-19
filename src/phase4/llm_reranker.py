"""
llm_reranker.py
---------------
LLM-based re-ranking of recommendation candidates.

Role in the pipeline:
  Retrieval (two-tower/semantic) -> top-500 candidates
  Scoring   (NeuMF)              -> top-50 shortlist
  LLM Re-ranking                 -> final top-10

Two strategies:
  1. Pointwise  : score each item independently (fast, parallelisable)
  2. Listwise   : score all candidates together (slower, cross-item aware)

Uses OpenAI gpt-4o-mini by default. Set OPENAI_API_KEY env variable.

Reference:
  Hou et al. (2023) "Large Language Models are Zero-Shot Rankers for
  Recommender Systems" ECIR.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from collections import defaultdict


# --------------------------------------------------------------------------
# 1. Build user profile from interaction history
# --------------------------------------------------------------------------

def build_user_profile(user_id: str,
                        train_df: pd.DataFrame,
                        descriptions: dict,
                        max_items: int = 5) -> str:
    """
    Build a natural language summary of a user's interaction history.
    Takes the most recent N interactions and formats them for the LLM prompt.
    """
    user_items = (train_df[train_df["user_id"] == user_id]["item_id"]
                  .tolist()[-max_items:])

    if not user_items:
        return "No purchase history available."

    lines = []
    for item_id in user_items:
        desc = descriptions.get(item_id, f"Gift Card ({item_id})")
        lines.append(f"- {desc}")

    return "Previously purchased:\n" + "\n".join(lines)


# --------------------------------------------------------------------------
# 2. Pointwise LLM re-ranking
# --------------------------------------------------------------------------

def pointwise_rerank(
    user_id: str,
    candidates: list,
    train_df: pd.DataFrame,
    descriptions: dict,
    client,
    n: int = 10,
    batch_size: int = 10,
) -> list:
    """
    Pointwise LLM re-ranking: score each item independently.

    For each candidate, ask the LLM: "Given this user's history,
    how relevant is this item? Score 0-10."

    Advantages: parallelisable, easy to cache per item.
    Disadvantages: no cross-item reasoning (can't avoid duplicates).
    """
    user_profile = build_user_profile(user_id, train_df, descriptions)
    top_candidates = candidates[:min(len(candidates), n * 3)]

    all_scores = []

    for i in range(0, len(top_candidates), batch_size):
        batch = top_candidates[i:i + batch_size]
        items_str = "\n".join(
            f"{j+1}. {descriptions.get(item_id, item_id)}"
            for j, (item_id, _) in enumerate(batch)
        )

        prompt = f"""You are a gift card recommendation system.

User profile:
{user_profile}

Rate each gift card's relevance for this user (0-10).
Return ONLY a JSON array of numbers, e.g. [8, 3, 7, 5, 9].

Gift cards to rate:
{items_str}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.choices[0].message.content.strip()
            if "[" in text:
                text = text[text.index("["):text.rindex("]")+1]
            scores = json.loads(text)
            scores = [float(s) / 10.0 for s in scores[:len(batch)]]
            scores += [0.0] * (len(batch) - len(scores))
        except Exception as e:
            print(f"[llm_rerank] Pointwise batch error: {e}")
            scores = [0.5] * len(batch)

        for (item_id, _), score in zip(batch, scores):
            all_scores.append((item_id, score))

        time.sleep(0.3)

    all_scores.sort(key=lambda x: x[1], reverse=True)
    return all_scores[:n]


# --------------------------------------------------------------------------
# 3. Listwise LLM re-ranking
# --------------------------------------------------------------------------

def listwise_rerank(
    user_id: str,
    candidates: list,
    train_df: pd.DataFrame,
    descriptions: dict,
    client,
    n: int = 10,
) -> list:
    """
    Listwise LLM re-ranking: reason over the full candidate list.

    Captures cross-item reasoning: avoids recommending duplicates,
    ensures variety, considers the list as a whole unit.

    Reference: Hou et al. (2023) "LLMs are Zero-Shot Rankers for RecSys"
    """
    user_profile = build_user_profile(user_id, train_df, descriptions)
    top_candidates = candidates[:min(len(candidates), n * 2)]

    items_str = "\n".join(
        f"{j+1}. {descriptions.get(item_id, item_id)}"
        for j, (item_id, _) in enumerate(top_candidates)
    )

    prompt = f"""You are a gift card recommendation system.

User profile:
{user_profile}

Re-rank these gift cards from most to least relevant for this user.
Consider variety -- don't rank similar items consecutively.
Return ONLY a JSON array of the original position numbers (1-indexed).
Example for 5 items: [3, 1, 5, 2, 4]

Gift cards to rank:
{items_str}

Return the top {n} positions only."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.choices[0].message.content.strip()
        if "[" in text:
            text = text[text.index("["):text.rindex("]")+1]
        ranking = json.loads(text)

        results = []
        seen_positions = set()
        for pos in ranking:
            pos = int(pos) - 1
            if 0 <= pos < len(top_candidates) and pos not in seen_positions:
                item_id, orig_score = top_candidates[pos]
                rank_score = 1.0 / (len(results) + 1)
                results.append((item_id, rank_score))
                seen_positions.add(pos)
            if len(results) >= n:
                break

        # Fill remaining slots
        if len(results) < n:
            for i, (item_id, score) in enumerate(top_candidates):
                if i not in seen_positions and len(results) < n:
                    results.append((item_id, float(score)))

        return results[:n]

    except Exception as e:
        print(f"[llm_rerank] Listwise error: {e}, falling back to original order")
        return top_candidates[:n]


# --------------------------------------------------------------------------
# 4. Hybrid: NeuMF score + LLM score fusion
# --------------------------------------------------------------------------

def hybrid_rerank(
    user_id: str,
    candidates: list,
    train_df: pd.DataFrame,
    descriptions: dict,
    client,
    n: int = 10,
    alpha: float = 0.6,
) -> list:
    """
    Fuse NeuMF scores with LLM pointwise scores.
    final_score = alpha * NeuMF_score + (1-alpha) * LLM_score
    alpha=0.6: 60% NeuMF, 40% LLM.
    """
    llm_scored = pointwise_rerank(
        user_id, candidates, train_df, descriptions, client,
        n=n * 2, batch_size=10
    )
    llm_score_map = {item_id: score for item_id, score in llm_scored}

    neuMF_scores = np.array([s for _, s in candidates])
    if neuMF_scores.max() > neuMF_scores.min():
        neuMF_scores = ((neuMF_scores - neuMF_scores.min()) /
                        (neuMF_scores.max() - neuMF_scores.min()))
    else:
        neuMF_scores = np.ones_like(neuMF_scores)

    fused = []
    for i, (item_id, _) in enumerate(candidates):
        neuMF_s = float(neuMF_scores[i]) if i < len(neuMF_scores) else 0.0
        llm_s   = llm_score_map.get(item_id, 0.0)
        fused.append((item_id, alpha * neuMF_s + (1 - alpha) * llm_s))

    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[:n]


# --------------------------------------------------------------------------
# 5. Evaluation
# --------------------------------------------------------------------------

def evaluate_reranker(
    rerank_fn,
    base_candidates: dict,
    test_df: pd.DataFrame,
    k: int = 10,
    max_users: int = 50,
) -> dict:
    """Evaluate a re-ranking function on HR@K and NDCG@K."""
    user_test_items = defaultdict(set)
    for _, row in test_df.iterrows():
        user_test_items[row["user_id"]].add(row["item_id"])

    hr, ndcg, prec = [], [], []
    eval_users = [u for u in list(user_test_items.keys())
                  if u in base_candidates][:max_users]

    print(f"[eval] Evaluating {len(eval_users)} users...")
    for user_id in eval_users:
        recs = rerank_fn(user_id, base_candidates[user_id])
        recommended = [item_id for item_id, _ in recs[:k]]
        relevant = user_test_items[user_id]

        hr.append(float(any(it in relevant for it in recommended)))
        dcg  = sum(1.0 / np.log2(r + 2)
                   for r, it in enumerate(recommended) if it in relevant)
        idcg = sum(1.0 / np.log2(r + 2)
                   for r in range(min(len(relevant), k)))
        ndcg.append(dcg / idcg if idcg > 0 else 0.0)
        prec.append(sum(1 for it in recommended if it in relevant) / k)

    return {
        f"HR@{k}"       : float(np.mean(hr)),
        f"NDCG@{k}"     : float(np.mean(ndcg)),
        f"Precision@{k}": float(np.mean(prec)),
        "n_users_eval"  : len(eval_users),
    }


# --------------------------------------------------------------------------
# 6. Sanity check
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"

    import torch
    from src.data_loader import (download_data, load_ratings,
                             filter_min_interactions, train_test_split)
    from src.phase2.ncf import NCFDataset, NeuMF, train as ncf_train
    from semantic_retrieval import (generate_item_descriptions,
                                            tfidf_embeddings,
                                            SemanticRetriever)

    fp = download_data()
    df = load_ratings(fp)
    df = filter_min_interactions(df)
    train_df, test_df = train_test_split(df)

    all_items    = sorted(df["item_id"].unique().tolist())
    descriptions = generate_item_descriptions(all_items)
    item_ids, embeddings, vectorizer, pca = tfidf_embeddings(
        descriptions, n_components=64
    )
    retriever = SemanticRetriever(item_ids, embeddings, vectorizer, pca)

    print("Training NeuMF...")
    ncf_dataset = NCFDataset(train_df, neg_ratio=4)
    ncf_model = NeuMF(
        n_users=ncf_dataset.n_users, n_items=ncf_dataset.n_items,
        gmf_dim=32, mlp_embed_dim=32, mlp_layers=[64, 32],
    )
    ncf_train(ncf_model, ncf_dataset, n_epochs=20, batch_size=256, lr=1e-3)

    # Pre-compute NeuMF candidates
    user_test_items  = defaultdict(set)
    user_train_items = defaultdict(set)
    for _, row in test_df.iterrows():
        user_test_items[row["user_id"]].add(row["item_id"])
    for _, row in train_df.iterrows():
        user_train_items[row["user_id"]].add(row["item_id"])

    eval_users = list(user_test_items.keys())[:50]
    base_candidates = {}
    ncf_model.eval()

    for user_id in eval_users:
        u_idx = ncf_dataset.user2idx.get(user_id)
        if u_idx is None:
            continue
        train_items = {ncf_dataset.item2idx[it]
                       for it in user_train_items[user_id]
                       if it in ncf_dataset.item2idx}
        with torch.no_grad():
            all_items_t = torch.arange(ncf_dataset.n_items)
            user_t = torch.full((ncf_dataset.n_items,), u_idx, dtype=torch.long)
            scores = ncf_model(user_t, all_items_t).numpy()
        ranked = np.argsort(scores)[::-1]
        candidates = [
            (ncf_dataset.items[i], float(scores[i]))
            for i in ranked if i not in train_items
        ][:30]
        base_candidates[user_id] = candidates

    # NeuMF baseline (no LLM)
    baseline_res = evaluate_reranker(
        lambda uid, cands: cands[:10],
        base_candidates, test_df, k=10
    )
    print(f"\nNeuMF baseline  HR@10={baseline_res['HR@10']:.4f}  "
          f"NDCG@10={baseline_res['NDCG@10']:.4f}")

    # Check for OpenAI key
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("\nNo OPENAI_API_KEY found.")
        print("To run LLM re-ranking:")
        print("  export OPENAI_API_KEY=your_key_here")
        print("  python3 src/phase4/llm_reranker.py")
        print("\nExpected improvement with LLM re-ranking: +5-15% NDCG")
    else:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        print("\nRunning LLM re-ranking comparison (20 users)...")

        print(f"\n{'Method':<25} {'HR@10':<10} {'NDCG@10'}")
        print("-" * 45)
        print(f"{'NeuMF (no rerank)':<25} "
              f"{baseline_res['HR@10']:<10.4f} "
              f"{baseline_res['NDCG@10']:.4f}")

        pointwise_res = evaluate_reranker(
            lambda uid, cands: pointwise_rerank(
                uid, cands, train_df, descriptions, client, n=10
            ),
            base_candidates, test_df, k=10, max_users=20
        )
        print(f"{'Pointwise LLM':<25} "
              f"{pointwise_res['HR@10']:<10.4f} "
              f"{pointwise_res['NDCG@10']:.4f}")

        listwise_res = evaluate_reranker(
            lambda uid, cands: listwise_rerank(
                uid, cands, train_df, descriptions, client, n=10
            ),
            base_candidates, test_df, k=10, max_users=20
        )
        print(f"{'Listwise LLM':<25} "
              f"{listwise_res['HR@10']:<10.4f} "
              f"{listwise_res['NDCG@10']:.4f}")

        hybrid_res = evaluate_reranker(
            lambda uid, cands: hybrid_rerank(
                uid, cands, train_df, descriptions, client, n=10
            ),
            base_candidates, test_df, k=10, max_users=20
        )
        print(f"{'Hybrid (NeuMF+LLM)':<25} "
              f"{hybrid_res['HR@10']:<10.4f} "
              f"{hybrid_res['NDCG@10']:.4f}")