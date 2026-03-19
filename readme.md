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

## Phase 1 — Foundations ✅

**Dataset**: Amazon Reviews 2023 — Gift Cards category
**Source**: [McAuley Lab, UCSD](https://amazon-reviews-2023.github.io/)
**Stats after 5-core filtering**: 553 users × 154 items · 97% sparse · 3,171 ratings

### Results

| Model | HR@5 | HR@10 | NDCG@5 | NDCG@10 | Coverage |
|-------|------|-------|--------|---------|----------|
| User-based CF (cosine, K=30) | 0.007 | 0.009 | 0.003 | 0.003 | 12.5% |
| SVD (k=20) | 0.034 | 0.122 | 0.017 | 0.042 | 99.6% |
| ALS (k=50, α=40) | 0.023 | 0.112 | 0.012 | 0.088 | 99.6% |
| Content-based (TF-IDF) | 0.374 | 0.477 | 0.241 | 0.275 | 100.0% |
| **Hybrid weighted (α=0.9)** | — | **0.485** | — | **0.279** | **100.0%** |
| Hybrid switching (min=3) | — | 0.459 | — | 0.265 | 100.0% |

### Key findings

- **User-based CF** works conceptually but collapses on sparse data — only 12.5% coverage because most user pairs share zero co-rated items
- **Matrix factorization** (SVD, ALS) solves coverage entirely (99.6%) by learning dense latent vectors — predictions always possible without overlap. ALS beats SVD on NDCG (0.088 vs 0.042) because confidence weighting surfaces relevant items at rank 1-2 rather than rank 8-9
- **Content-based filtering** dominates on Gift Cards (NDCG=0.275, 100% coverage) — item similarity is the strongest signal in this dataset; people who buy one gift card denomination buy similar ones
- **Hybrid (α=0.9)** edges past pure CB (NDCG 0.279 vs 0.275) — adding 10% MF contribution introduces just enough diversity to improve ranking without losing CB's precision
- **Switching hybrid** correctly routes 512/553 users (92%) to CB, confirming most users have sufficient history (≥3 ratings) for reliable taste profiles

### Files

```
src/
├── data_loader.py      Amazon Reviews 2023 download, cold-start filtering,
│                       user-item matrix construction, per-user train/test split
├── user_cf.py          Mean-centered cosine similarity, neighbourhood-based
│                       prediction (Pearson correlation equivalent)
├── matrix_factor.py    SVDRecommender (truncated SVD on explicit ratings) +
│                       ALSRecommender (Hu-Koren-Volinsky 2008, implicit lib)
├── content_based.py    TF-IDF item profiles, item-item cosine similarity,
│                       user taste profile aggregation
├── hybrid.py           Weighted hybrid (score fusion) + switching hybrid
│                       (route by user history depth)
└── evaluate.py         HR@K, NDCG@K, Precision@K offline evaluation suite
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
# Downloads ~3.7MB, decompresses, filters to 5-core
python3 src/data_loader.py

# User-based CF
python3 src/user_cf.py

# Matrix factorization — SVD vs ALS comparison
OPENBLAS_NUM_THREADS=1 python3 src/matrix_factor.py

# Content-based filtering
python3 src/content_based.py

# Hybrid — weighted and switching strategies
OPENBLAS_NUM_THREADS=1 python3 src/hybrid.py

# Offline evaluation suite
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

1. **[Phase 1]** Building a Recommender System from Scratch — CF, MF, Content-Based & Hybrid *(coming soon)*
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