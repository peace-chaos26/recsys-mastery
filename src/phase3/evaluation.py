"""
evaluation.py
-------------
Comprehensive offline evaluation suite for Phase 3.

Extends Phase 1's basic HR/NDCG/Precision with:
  - IPS-corrected NDCG (debiased evaluation)
  - Intra-List Diversity (ILD)
  - Catalogue coverage
  - Novelty (how often items outside user history appear)
  - Latency tracking (p50, p95, p99)

Why IPS-corrected evaluation?
  Standard NDCG on biased click data underestimates the value of
  debiasing and diversity interventions. IPS-NDCG gives a fairer
  picture of what the model actually learned vs what it was exposed to.

Usage:
  from phase3.evaluation import evaluate_full
  results = evaluate_full(pipeline, test_df, train_df, item_vectors)
"""

import numpy as np
import pandas as pd
import time
from collections import defaultdict
from debiasing import examination_propensity, ips_ndcg
from diversity import intra_list_diversity, coverage


# --------------------------------------------------------------------------
# 1. Full evaluation suite
# --------------------------------------------------------------------------

def evaluate_full(
    recommend_fn,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    item_vectors: np.ndarray,
    k: int = 10,
    eta: float = 0.5,
    verbose: bool = True,
) -> dict:
    """
    Comprehensive evaluation across all metrics.

    recommend_fn: callable(user_id, train_df) -> list of (item_id, score)
    item_vectors: (n_items, dim) array for diversity computation
                  item_id strings must be mappable to integer indices

    Metrics computed:
      HR@K          : hit rate (did we recommend anything relevant?)
      NDCG@K        : ranking quality (are relevant items ranked high?)
      IPS-NDCG@K    : debiased NDCG (corrects for position bias in test set)
      Precision@K   : fraction of top-K that are relevant
      ILD           : intra-list diversity (avg pairwise distance)
      Coverage      : fraction of catalogue recommended to anyone
      Novelty       : avg fraction of recs outside user training history
      Latency p50   : median recommendation latency (ms)
      Latency p95   : 95th percentile latency (ms)
    """
    propensities = examination_propensity(max_position=k, eta=eta)

    # Build per-user ground truth and training history
    user_test_items = defaultdict(set)
    for _, row in test_df.iterrows():
        user_test_items[row["user_id"]].add(row["item_id"])

    user_train_items = defaultdict(set)
    for _, row in train_df.iterrows():
        user_train_items[row["user_id"]].add(row["item_id"])

    # Build item_id → integer index for diversity computation
    all_item_ids = sorted(set(train_df["item_id"]) | set(test_df["item_id"]))
    item_id_to_idx = {it: i for i, it in enumerate(all_item_ids)}
    n_items_total = len(all_item_ids)

    metrics = defaultdict(list)
    all_rec_indices = []
    latencies = []

    eval_users = list(user_test_items.keys())
    if verbose:
        print(f"[eval] Evaluating {len(eval_users)} users at K={k}...")

    for user_id in eval_users:
        relevant = user_test_items[user_id]
        train_items = user_train_items[user_id]

        # Get recommendations
        t0 = time.perf_counter()
        recs = recommend_fn(user_id, train_df)
        latencies.append((time.perf_counter() - t0) * 1000)

        recommended = [item_id for item_id, _ in recs[:k]]

        if not recommended:
            for m in ["hr", "ndcg", "ips_ndcg", "prec", "ild", "novelty"]:
                metrics[m].append(0.0)
            continue

        # HR@K
        metrics["hr"].append(
            float(any(it in relevant for it in recommended))
        )

        # NDCG@K
        dcg  = sum(1.0 / np.log2(r + 2)
                   for r, it in enumerate(recommended) if it in relevant)
        idcg = sum(1.0 / np.log2(r + 2)
                   for r in range(min(len(relevant), k)))
        metrics["ndcg"].append(dcg / idcg if idcg > 0 else 0.0)

        # IPS-NDCG@K
        metrics["ips_ndcg"].append(
            ips_ndcg(recommended, relevant, propensities, k=k)
        )

        # Precision@K
        metrics["prec"].append(
            sum(1 for it in recommended if it in relevant) / k
        )

        # ILD (diversity within this user's recommendation list)
        rec_indices = [item_id_to_idx[it]
                       for it in recommended if it in item_id_to_idx]
        if len(rec_indices) >= 2 and max(rec_indices) < len(item_vectors):
            metrics["ild"].append(
                intra_list_diversity(rec_indices, item_vectors)
            )

        # Novelty: fraction of recs NOT in user's training history
        novel = sum(1 for it in recommended if it not in train_items)
        metrics["novelty"].append(novel / len(recommended))

        # Track for catalogue coverage
        all_rec_indices.extend(rec_indices)

    # Catalogue coverage
    cat_coverage = coverage(
        [[item_id_to_idx[it] for it in
          [item_id for item_id, _ in recommend_fn(uid, train_df)[:k]]
          if it in item_id_to_idx]
         for uid in eval_users[:50]],  # sample 50 users for speed
        n_items_total
    )

    results = {
        f"HR@{k}"          : float(np.mean(metrics["hr"])),
        f"NDCG@{k}"        : float(np.mean(metrics["ndcg"])),
        f"IPS-NDCG@{k}"    : float(np.mean(metrics["ips_ndcg"])),
        f"Precision@{k}"   : float(np.mean(metrics["prec"])),
        "ILD"              : float(np.mean(metrics["ild"])) if metrics["ild"] else 0.0,
        "Coverage"         : cat_coverage,
        "Novelty"          : float(np.mean(metrics["novelty"])),
        "Latency_p50_ms"   : float(np.percentile(latencies, 50)),
        "Latency_p95_ms"   : float(np.percentile(latencies, 95)),
        "n_users_eval"     : len(eval_users),
    }
    return results


def print_eval_report(results: dict, model_name: str = "Model") -> None:
    """Print a formatted evaluation report."""
    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  {'Relevance metrics':}")
    print(f"    {'HR@K':<22} {results.get('HR@10', results.get('HR@5', 0)):.4f}")
    print(f"    {'NDCG@K':<22} {results.get('NDCG@10', results.get('NDCG@5', 0)):.4f}")
    print(f"    {'IPS-NDCG@K':<22} {results.get('IPS-NDCG@10', results.get('IPS-NDCG@5', 0)):.4f}")
    print(f"    {'Precision@K':<22} {results.get('Precision@10', results.get('Precision@5', 0)):.4f}")
    print(f"  {'Beyond-accuracy metrics':}")
    print(f"    {'ILD (diversity)':<22} {results.get('ILD', 0):.4f}")
    print(f"    {'Coverage':<22} {results.get('Coverage', 0):.2%}")
    print(f"    {'Novelty':<22} {results.get('Novelty', 0):.4f}")
    print(f"  {'Serving metrics':}")
    print(f"    {'Latency p50 (ms)':<22} {results.get('Latency_p50_ms', 0):.2f}")
    print(f"    {'Latency p95 (ms)':<22} {results.get('Latency_p95_ms', 0):.2f}")
    print(f"{'='*50}")


# --------------------------------------------------------------------------
# 2. Sanity check — full pipeline eval
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, ".")
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"

    import faiss
    from src.data_loader import (download_data, load_ratings,
                             filter_min_interactions, train_test_split)
    from src.phase2.two_tower import (InteractionDataset, TwoTowerModel,
                                   train as tt_train, build_item_index)
    from src.phase2.ncf import NCFDataset, NeuMF, train as ncf_train
    from src.phase2.ann_index import build_index, IndexConfig
    from src.phase3.pipeline import RecommendationPipeline, PipelineConfig

    fp = download_data()
    df = load_ratings(fp)
    df = filter_min_interactions(df)
    train_df, test_df = train_test_split(df)

    # Train models
    print("Training two-tower...")
    tt_dataset = InteractionDataset(train_df)
    tt_model = TwoTowerModel(
        n_users=tt_dataset.n_users, n_items=tt_dataset.n_items,
        embed_dim=64, hidden_dim=128, output_dim=64, temperature=0.07,
    )
    tt_train(tt_model, tt_dataset, n_epochs=20, batch_size=64, lr=1e-3)
    item_vecs_tensor = build_item_index(tt_model, tt_dataset)
    item_vecs_np = item_vecs_tensor.numpy().astype(np.float32)

    cfg = IndexConfig(index_type="hnsw", dim=64, hnsw_m=32)
    faiss_index = build_index(item_vecs_np, cfg)

    print("Training NeuMF...")
    ncf_dataset = NCFDataset(train_df, neg_ratio=4)
    ncf_model = NeuMF(
        n_users=ncf_dataset.n_users, n_items=ncf_dataset.n_items,
        gmf_dim=32, mlp_embed_dim=32, mlp_layers=[64, 32],
    )
    ncf_train(ncf_model, ncf_dataset, n_epochs=20, batch_size=256, lr=1e-3)

    # Build pipeline
    pipeline = RecommendationPipeline(
        PipelineConfig(diversity_lambda=0.3, use_debiasing=True),
        tt_model, tt_dataset, ncf_model, ncf_dataset,
        faiss_index, item_vecs_np,
    )

    # Full evaluation with all metrics
    # Use item_vecs_np indexed by position (proxy — real eval needs item_id→idx)
    def recommend_fn(user_id, train_df):
        return pipeline.recommend(user_id, train_df)

    results = evaluate_full(
        recommend_fn, test_df, train_df,
        item_vecs_np, k=10,
    )
    print_eval_report(results, "Full pipeline (two-tower + NeuMF + MMR + IPS)")