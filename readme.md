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
| 2 | Neural RecSys — two-tower, NCF, sequential models, ANN | 🔄 In progress |
| 3 | Production pipeline — multi-stage, debiasing, MMR diversity | ⬜ Planned |
| 4 | GenAI × RecSys — LLM re-ranking, semantic retrieval, conversational | ⬜ Planned |
| 5 | RL for RecSys — bandits, deep RL, offline RL | ⬜ Planned |

---

## Phase 1 — Foundations

**Dataset**: Amazon Reviews 2023 — Gift Cards category  
**Source**: [McAuley Lab, UCSD](https://amazon-reviews-2023.github.io/)  
**Stats after 5-core filtering**: 553 users × 154 items · 97% sparse · 3,171 ratings

### Models implemented

| Model | HR@5 | HR@10 | NDCG@5 | NDCG@10 | Coverage |
|-------|------|-------|--------|---------|----------|
| User-based CF (cosine, K=30) | 0.007 | 0.009 | 0.003 | 0.003 | 12.5% |
| SVD (k=20) | 0.034 | 0.122 | 0.017 | 0.042 | 99.6% |
| ALS (k=50, α=40) | 0.023 | 0.112 | 0.012 | **0.088** | 99.6% |
| Content-based (TF-IDF) | TBD | TBD | TBD | TBD | — |
| Hybrid | TBD | TBD | TBD | TBD | — |

### Key findings so far

- **User-based CF** works conceptually but suffers from severe coverage problems (12.5%) on sparse datasets — most user pairs share zero co-rated items, making similarity meaningless
- **Matrix factorization** solves coverage entirely (99.6%) by learning dense latent vectors for every user and item — predictions are always possible without any overlap
- **ALS beats SVD on NDCG** (0.088 vs 0.042) despite lower HR — confidence weighting (`1 + α × rating`) surfaces relevant items at rank 1-2 rather than rank 8-9
- **5-core filtering** is essential for CF — without it, the Gift Cards dataset collapses to near-zero usable interactions

### Files

```
src/
├── data_loader.py      Amazon Reviews 2023 download, cold-start filtering,
│                       user-item matrix construction, train/test split
├── user_cf.py          Mean-centered cosine similarity, neighbourhood-based
│                       prediction, top-N recommendation
├── matrix_factor.py    SVDRecommender (truncated SVD) + ALSRecommender
│                       (Hu-Koren-Volinsky 2008, implicit library)
└── evaluate.py         HR@K, NDCG@K, Precision@K offline evaluation
```

---

## Setup

```bash
git clone https://github.com/peace-chaos26/recsys-mastery.git
cd recsys-mastery

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Run Phase 1

```bash
# Data loader (downloads ~3.7MB, decompresses, filters)
python3 src/data_loader.py

# User-based CF
python3 src/user_cf.py

# Matrix factorization (SVD + ALS comparison)
OPENBLAS_NUM_THREADS=1 python3 src/matrix_factor.py

# Offline evaluation
python3 src/evaluate.py
```

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Data processing | pandas, numpy |
| Similarity / linear algebra | scikit-learn, scipy |
| Matrix factorization | scipy (SVD), implicit (ALS) |
| Evaluation | custom — HR, NDCG, Precision@K |
| Phase 2+ | PyTorch, FAISS, HuggingFace |

---

## Blog Series

Each phase has a companion blog post:

1. **[Phase 1]** Building a Recommender System from Scratch — CF, MF & Offline Eval *(coming soon)*
2. **[Phase 2]** Two-Tower Models and Neural Collaborative Filtering
3. **[Phase 3]** Production-Grade RecSys — Multi-Stage Pipelines, Debiasing, Diversity
4. **[Phase 4]** LLMs Meet RecSys — Semantic Retrieval and Conversational Recommendations
5. **[Phase 5]** RL for RecSys — Bandits, Exploration, and Long-Horizon Optimization

---

## References

- Hu, Y., Koren, Y., & Volinsky, C. (2008). *Collaborative Filtering for Implicit Feedback Datasets*. ICDM.
- Hou, Y., et al. (2024). *Bridging Language and Items for Retrieval and Recommendation*. arXiv:2403.03952.

---

## License

MIT