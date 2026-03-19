# RecSys Mastery — E2E Recommender Systems

> A progressive, end-to-end recommender systems project built from scratch —
> from classic collaborative filtering to GenAI-powered ranking and RL-based exploration.
> Built on Amazon Reviews 2023 data. Each phase has a companion blog post.

---

## Overview

This project covers the full spectrum of modern recommendation systems,
implemented and evaluated on real Amazon review data. Each phase builds on
the last — mirroring how the industry evolved from deterministic memory-based
methods to neural two-tower models to LLM re-ranking to reinforcement learning.

---

## Roadmap

| Phase | Topic | Status |
|-------|-------|--------|
| 1 | Collaborative filtering, matrix factorization, content-based, hybrid, offline eval | ✅ Complete |
| 2 | Neural RecSys — two-tower, NeuMF, SASRec, FAISS ANN | ✅ Complete |
| 3 | Production pipeline — multi-stage, debiasing, MMR diversity | 🔄 In progress |
| 4 | GenAI × RecSys — LLM re-ranking, semantic retrieval, conversational | ⬜ Planned |
| 5 | RL for RecSys — bandits, deep RL, offline RL | ⬜ Planned |

---

## Phase 1 — Foundations ✅

**Dataset**: Amazon Reviews 2023 — Gift Cards  
**Stats**: 553 users × 154 items · 97% sparse · 3,171 ratings

| Model | HR@10 | NDCG@10 | Coverage | Notes |
|-------|-------|---------|----------|-------|
| User-based CF (K=30) | 0.009 | 0.003 | 12.5% | Sparsity kills coverage |
| SVD (k=20) | 0.122 | 0.042 | 99.6% | Dense vectors fix coverage |
| ALS (k=50, α=40) | 0.112 | 0.088 | 99.6% | Confidence weighting improves ranking |
| Content-based (TF-IDF) | 0.477 | 0.275 | 100% | Item similarity dominates on Gift Cards |
| **Hybrid (α=0.9)** | **0.485** | **0.279** | **100%** | 10% MF diversity improves CB |

**Key findings**: Coverage jumps from 12.5% (User CF) → 99.6% (MF) → 100% (CB/Hybrid). ALS beats SVD on NDCG (0.088 vs 0.042) because confidence weighting surfaces relevant items at rank 1-2. CB dominates on this dataset due to strong item-item similarity patterns in gift card purchasing behaviour.

---

## Phase 2 — Neural RecSys ✅

**Same dataset as Phase 1.**

| Model | HR@10 | NDCG@10 | Coverage | Role in pipeline |
|-------|-------|---------|----------|-----------------|
| Two-Tower | 0.121 | 0.051 | 100% | Candidate retrieval |
| Two-Tower + FAISS HNSW | 0.112 | 0.044 | 100% | Production serving |
| NeuMF (GMF + MLP) | 0.441 | 0.261 | 100% | Candidate ranking |
| SASRec (2-layer) | 0.056 | 0.031 | 100% | Sequential / next-item |

**Key findings**:
- Two-tower enables production-scale retrieval — FAISS HNSW delivers 1.8x speedup at 154 items (100-500x at 1M+ items) with 100% recall
- NeuMF's joint user+item MLP captures non-linear interactions that dot product cannot — NDCG 5x better than two-tower on this dataset
- SASRec underperforms on sparse data (median 5 interactions/user) — sequential models need 100+ interactions/user to learn meaningful patterns; would dominate at Netflix/TikTok scale
- Two-tower vs NeuMF gap illustrates the retrieval/ranking split: two-tower retrieves broadly from millions, NeuMF re-ranks the shortlist precisely

---

## Project Structure

```
src/
├── data_loader.py        Amazon Reviews 2023, cold-start filtering, matrix construction
├── user_cf.py            Mean-centered cosine CF, neighbourhood prediction
├── matrix_factor.py      SVDRecommender + ALSRecommender (Hu-Koren-Volinsky 2008)
├── content_based.py      TF-IDF item profiles, item-item cosine similarity
├── hybrid.py             Weighted + switching hybrid strategies
├── evaluate.py           HR@K, NDCG@K, Precision@K offline evaluation
└── phase2/
    ├── two_tower.py      Dual encoder, in-batch softmax loss, item index
    ├── ncf.py            GMF + MLP fusion (NeuMF), BCE pointwise training
    ├── sequential.py     SASRec — causal self-attention, positional embeddings
    └── ann_index.py      FAISS FlatIP / IVF / HNSW index + benchmark
```

---

## Setup

```bash
git clone https://github.com/peace-chaos26/recsys-mastery.git
cd recsys-mastery
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Run Phase 1
```bash
python3 src/data_loader.py
python3 src/user_cf.py
OPENBLAS_NUM_THREADS=1 python3 src/matrix_factor.py
python3 src/content_based.py
OPENBLAS_NUM_THREADS=1 python3 src/hybrid.py
python3 src/evaluate.py
```

### Run Phase 2
```bash
python3 src/phase2/two_tower.py
python3 src/phase2/ncf.py
python3 src/phase2/sequential.py
python3 src/phase2/ann_index.py   # Mac: add KMP_DUPLICATE_LIB_OK=TRUE prefix
```

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Data processing | pandas, numpy |
| Classical ML | scikit-learn, scipy |
| Matrix factorization | implicit (ALS) |
| Deep learning | PyTorch |
| ANN search | FAISS (IVF, HNSW) |
| Evaluation | custom — HR, NDCG, Precision@K |
| Phase 3+ | LambdaMART, HuggingFace, LLM APIs |

---

## Blog Series

1. **[Phase 1]** Building a Recommender System from Scratch — CF, MF, Content-Based & Hybrid *(coming soon)*
2. **[Phase 2]** Neural RecSys — Two-Tower, NeuMF, SASRec and FAISS at Scale *(coming soon)*
3. **[Phase 3]** Production-Grade RecSys — Multi-Stage Pipelines, Debiasing, Diversity
4. **[Phase 4]** LLMs Meet RecSys — Semantic Retrieval and Conversational Recommendations
5. **[Phase 5]** RL for RecSys — Bandits, Exploration, and Long-Horizon Optimization

---

## References

- Hu, Y., Koren, Y., & Volinsky, C. (2008). *Collaborative Filtering for Implicit Feedback Datasets*. ICDM.
- He, X., et al. (2017). *Neural Collaborative Filtering*. WWW.
- Kang, W., & McAuley, J. (2018). *Self-Attentive Sequential Recommendation*. ICDM.
- Hou, Y., et al. (2024). *Bridging Language and Items for Retrieval and Recommendation*. arXiv:2403.03952.

---

## License

MIT