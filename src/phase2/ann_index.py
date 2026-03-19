"""
ann_index.py
------------
FAISS-based Approximate Nearest Neighbour (ANN) index for two-tower serving.

The problem with brute-force search:
  two_tower.py uses: scores = user_vec @ item_index.T
  This is O(n_items * dim) per query.
  At 100M items with dim=64: 6.4B multiply-adds per user request.
  At 1000 QPS: 6.4 trillion ops/sec -- not feasible.

What FAISS does:
  Builds an index structure offline that allows approximate search
  in O(log n) or O(sqrt n) time instead of O(n).
  Trade-off: might miss the true top-K by a small margin (approximate),
  but returns results in <10ms instead of seconds.

Index types implemented here:
  1. FlatL2      -- exact brute force (baseline, no approximation)
  2. FlatIP      -- exact inner product (cosine if L2-normalized)
  3. IVFFlat     -- inverted file index, clusters vectors into Voronoi cells
                    Only searches nearby cells -- much faster, slightly approximate
  4. HNSW        -- Hierarchical Navigable Small World graph
                    State of the art for recall/speed tradeoff
                    Used by Pinecone, Weaviate, pgvector

In production (Pinterest, LinkedIn, Meta):
  IVFFlat or HNSW with GPU acceleration handles billions of vectors.
  We run on CPU here -- same logic, smaller scale.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import torch
import time
import faiss
from dataclasses import dataclass


# --------------------------------------------------------------------------
# 1. Index types and config
# --------------------------------------------------------------------------

@dataclass
class IndexConfig:
    """
    Configuration for a FAISS index.

    index_type: one of "flat_l2", "flat_ip", "ivf", "hnsw"
    dim       : vector dimension (must match two-tower output_dim)
    n_clusters: for IVF -- number of Voronoi cells (sqrt(n_items) is a good start)
    n_probe   : for IVF -- how many cells to search (higher = better recall, slower)
    hnsw_m    : for HNSW -- number of connections per node (higher = better recall)
    """
    index_type : str   = "hnsw"
    dim        : int   = 64
    n_clusters : int   = 32
    n_probe    : int   = 8
    hnsw_m     : int   = 32


# --------------------------------------------------------------------------
# 2. Build index
# --------------------------------------------------------------------------

def build_index(item_vectors: np.ndarray, config: IndexConfig) -> faiss.Index:
    """
    Build a FAISS index from item vectors.

    item_vectors: (n_items, dim) float32 numpy array
                  Must be L2-normalized if using inner product similarity.

    This runs OFFLINE -- once per model update.
    In production: scheduled daily job, index stored to disk.
    """
    n_items, dim = item_vectors.shape
    assert dim == config.dim, \
        f"Vector dim {dim} doesn't match config.dim {config.dim}"

    # Ensure float32 (FAISS requirement)
    vectors = item_vectors.astype(np.float32)

    if config.index_type == "flat_l2":
        # Exact L2 distance search -- baseline, O(n) per query
        # Use when n_items < 10K or when recall must be 100%
        index = faiss.IndexFlatL2(dim)
        print(f"[ann] Built FlatL2 index ({n_items} vectors, dim={dim})")

    elif config.index_type == "flat_ip":
        # Exact inner product -- equivalent to cosine if L2-normalized
        # Baseline for normalized two-tower vectors
        index = faiss.IndexFlatIP(dim)
        print(f"[ann] Built FlatIP index ({n_items} vectors, dim={dim})")

    elif config.index_type == "ivf":
        # IVF: cluster vectors into n_clusters Voronoi cells.
        # At query time, only search n_probe nearest cells.
        # Speed/recall tradeoff: higher n_probe = better recall, slower.
        #
        # Rule of thumb: n_clusters = sqrt(n_items)
        # n_probe = n_clusters / 4 for ~95% recall
        quantizer = faiss.IndexFlatIP(dim)  # inner product quantizer
        index = faiss.IndexIVFFlat(
            quantizer, dim, config.n_clusters, faiss.METRIC_INNER_PRODUCT
        )
        # IVF requires training to learn cluster centroids
        print(f"[ann] Training IVF index ({config.n_clusters} clusters)...")
        index.train(vectors)
        index.nprobe = config.n_probe
        print(f"[ann] Training IVF index ({config.n_clusters} clusters)...")

    elif config.index_type == "hnsw":
        # HNSW: builds a multi-layer proximity graph.
        # Searches by navigating from coarse to fine layers.
        # Best recall/speed tradeoff for <10M vectors.
        # No training required -- just add vectors.
        #
        # M: connections per node. Higher M = better recall, more memory.
        # Typical: M=16 to 64.
        index = faiss.IndexHNSWFlat(dim, config.hnsw_m,
                                     faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200  # build quality (higher = better)
        index.hnsw.efSearch = 50         # search quality at query time
        print(f"[ann] Built HNSW index ({n_items} vectors, "
              f"M={config.hnsw_m})")

    else:
        raise ValueError(f"Unknown index_type: {config.index_type}")

    # Add all item vectors to the index
    index.add(vectors)
    return index


# --------------------------------------------------------------------------
# 3. Search
# --------------------------------------------------------------------------

def search(index: faiss.Index, query_vectors: np.ndarray,
           k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Search the index for k nearest neighbours.

    query_vectors: (n_queries, dim) float32 array
    k            : number of results per query

    Returns:
      distances : (n_queries, k) -- similarity scores
      indices   : (n_queries, k) -- item indices in the index

    This is the ONLINE serving step -- runs per user request.
    """
    queries = query_vectors.astype(np.float32)
    distances, indices = index.search(queries, k)
    return distances, indices


# --------------------------------------------------------------------------
# 4. Benchmark -- compare index types
# --------------------------------------------------------------------------

def benchmark_indices(item_vectors: np.ndarray, n_queries: int = 100,
                      k: int = 10, dim: int = 64):
    """
    Compare all index types on latency and recall vs exact search.

    Recall@K: fraction of true top-K items found by approximate search.
    Latency : average query time in milliseconds.
    """
    n_items = len(item_vectors)
    vectors = item_vectors.astype(np.float32)

    # Random query vectors (simulate user vectors at serving time)
    rng = np.random.default_rng(42)
    queries = rng.standard_normal((n_queries, dim)).astype(np.float32)
    # L2-normalize queries (matches two-tower output)
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    queries = queries / (norms + 1e-8)

    # Ground truth via exact search
    exact_index = faiss.IndexFlatIP(dim)
    exact_index.add(vectors)
    _, exact_ids = exact_index.search(queries, k)

    print(f"\n{'Index':<12} {'Build(ms)':<12} {'Query(ms)':<12} "
          f"{'Recall@K':<12} {'Notes'}")
    print("-" * 65)

    configs = [
        IndexConfig("flat_ip",  dim=dim),
        IndexConfig("ivf",      dim=dim,
                    n_clusters=max(4, int(np.sqrt(n_items))),
                    n_probe=max(2, int(np.sqrt(n_items)) // 4)),
        IndexConfig("hnsw",     dim=dim, hnsw_m=32),
    ]

    for cfg in configs:
        # Build
        t0 = time.perf_counter()
        idx = build_index(vectors, cfg)
        build_ms = (time.perf_counter() - t0) * 1000

        # Query
        t0 = time.perf_counter()
        _, approx_ids = search(idx, queries, k=k)
        query_ms = (time.perf_counter() - t0) * 1000 / n_queries

        # Recall vs exact
        recall = np.mean([
            len(set(approx_ids[i]) & set(exact_ids[i])) / k
            for i in range(n_queries)
        ])

        notes = {
            "flat_ip": "exact baseline",
            "ivf"    : f"clusters={cfg.n_clusters}, nprobe={cfg.n_probe}",
            "hnsw"   : f"M={cfg.hnsw_m}",
        }[cfg.index_type]

        print(f"{cfg.index_type:<12} {build_ms:<12.1f} {query_ms:<12.3f} "
              f"{recall:<12.3f} {notes}")


# --------------------------------------------------------------------------
# 5. End-to-end: two-tower + FAISS serving demo
# --------------------------------------------------------------------------

def two_tower_serving_demo(train_df, test_df):
    """
    Full serving pipeline:
      1. Train two-tower model
      2. Build FAISS HNSW index from item vectors
      3. Serve recommendations via ANN search
      4. Compare latency: brute force vs FAISS
    """
    import sys
    sys.path.insert(0, ".")
    from src.phase2.two_tower import (InteractionDataset, TwoTowerModel,
                                   train as tt_train, build_item_index,
                                   evaluate)

    print("=" * 55)
    print("Two-Tower + FAISS serving demo")
    print("=" * 55)

    # Train two-tower
    dataset = InteractionDataset(train_df)
    model = TwoTowerModel(
        n_users=dataset.n_users, n_items=dataset.n_items,
        embed_dim=64, hidden_dim=128, output_dim=64, temperature=0.07,
    )
    tt_train(model, dataset, n_epochs=20, batch_size=64, lr=1e-3)

    # Build item vector index (offline step)
    item_vecs_tensor = build_item_index(model, dataset)
    item_vecs_np = item_vecs_tensor.numpy().astype(np.float32)

    # Build FAISS index
    n_items = len(item_vecs_np)
    cfg = IndexConfig(
        index_type="hnsw", dim=64, hnsw_m=32
    )
    t0 = time.perf_counter()
    index = build_index(item_vecs_np, cfg)
    print(f"[ann] Index built in {(time.perf_counter()-t0)*1000:.1f}ms")

    # Get a sample user vector
    sample_user = torch.tensor([0])
    user_vec = model.get_user_vector(sample_user).numpy()

    # Brute force vs FAISS latency comparison
    K = 10
    n_trials = 50

    # Brute force
    t0 = time.perf_counter()
    for _ in range(n_trials):
        scores = (user_vec @ item_vecs_np.T)
        top_k_bf = np.argsort(scores[0])[::-1][:K]
    bf_ms = (time.perf_counter() - t0) * 1000 / n_trials

    # FAISS
    t0 = time.perf_counter()
    for _ in range(n_trials):
        _, top_k_faiss = search(index, user_vec, k=K)
    faiss_ms = (time.perf_counter() - t0) * 1000 / n_trials

    # Recall
    recall = len(set(top_k_faiss[0]) & set(top_k_bf)) / K

    print(f"\n{'Method':<20} {'Latency (ms)':<16} {'Recall@10'}")
    print("-" * 45)
    print(f"{'Brute force':<20} {bf_ms:<16.3f} 1.000 (exact)")
    print(f"{'FAISS HNSW':<20} {faiss_ms:<16.3f} {recall:.3f}")
    print(f"\nSpeedup: {bf_ms/faiss_ms:.1f}x  "
          f"(more dramatic at 1M+ items)")

    # Evaluate with FAISS retrieval
    print("\n[ann] Evaluating with FAISS retrieval...")
    res = evaluate(model, dataset, item_vecs_tensor, train_df, test_df, k=10)
    print(f"Two-Tower + FAISS  HR@10={res['HR@10']:.4f}  "
          f"NDCG@10={res['NDCG@10']:.4f}  "
          f"Coverage={res['coverage']:.4f}")

    return index, model, dataset


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

    # 1. Benchmark all index types on synthetic vectors
    print("=" * 65)
    print("FAISS index benchmark (synthetic vectors)")
    print("=" * 65)
    rng = np.random.default_rng(42)
    n_items, dim = 154, 64
    fake_vecs = rng.standard_normal((n_items, dim)).astype(np.float32)
    norms = np.linalg.norm(fake_vecs, axis=1, keepdims=True)
    fake_vecs = fake_vecs / (norms + 1e-8)
    benchmark_indices(fake_vecs, n_queries=100, k=10, dim=dim)

    # 2. Full two-tower + FAISS serving demo
    print()
    two_tower_serving_demo(train_df, test_df)