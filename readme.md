# RecSys Mastery — E2E Recommender Systems

> A progressive, end-to-end recommender systems project built from scratch —
> from classic collaborative filtering to GenAI-powered ranking and RL-based exploration.

---

## Overview

This project is a structured learning and portfolio series covering the full
spectrum of modern recommendation systems, implemented on Amazon Review data.

Each phase builds on the last, mirroring how the industry evolved — from
deterministic memory-based methods to neural two-tower models to LLM re-ranking
to reinforcement learning for online exploration.

---

## Roadmap

| Phase | Topic | Status |
|-------|-------|--------|
| 1 | Collaborative filtering, content-based, hybrid, offline eval | 🔄 In progress |
| 2 | Neural RecSys — two-tower, NCF, sequential models, ANN | ⬜ Planned |
| 3 | Production pipeline — multi-stage, debiasing, MMR diversity | ⬜ Planned |
| 4 | GenAI × RecSys — LLM re-ranking, semantic retrieval, conversational | ⬜ Planned |
| 5 | RL for RecSys — bandits, deep RL, offline RL | ⬜ Planned |

---

## Phase 1 — Foundations (this branch)

**Dataset**: Amazon Digital Music Reviews (~65k ratings, explicit 1-5 stars)

**What's implemented:**
- User-item matrix construction with cold-start filtering
- User-based collaborative filtering (cosine similarity + top-N)
- Matrix factorization (SVD, ALS on implicit feedback)
- Content-based filtering (TF-IDF item profiles)
- Hybrid system (weighted combination)
- Offline evaluation: NDCG@K, MAP@K, HR@K, Precision@K

**Key results:**

| Model | NDCG@10 | HR@10 | MAP@10 |
|-------|---------|-------|--------|
| User CF (cosine) | TBD | TBD | TBD |
| ALS (implicit) | TBD | TBD | TBD |
| Content-based | TBD | TBD | TBD |
| Hybrid | TBD | TBD | TBD |

---

## Project Structure

```
recsys-mastery/
├── data/               # Raw data — gitignored, auto-downloaded
├── src/
│   ├── data_loader.py  # Dataset download, filtering, matrix construction
│   ├── user_cf.py      # User-based collaborative filtering
│   ├── matrix_factor.py# SVD and ALS matrix factorization
│   ├── content_based.py# TF-IDF content-based filtering
│   ├── hybrid.py       # Hybrid recommender
│   └── evaluate.py     # Offline evaluation metrics
├── notebooks/          # EDA and experimentation
├── outputs/            # Saved results and plots — gitignored
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/recsys-mastery.git
cd recsys-mastery

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Run Phase 1 pipeline end-to-end
python src/data_loader.py
```

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Data processing | pandas, numpy |
| Similarity / algebra | scikit-learn, scipy |
| Matrix factorization | implicit (ALS), scipy SVD |
| Evaluation | custom (NDCG, MAP, HR) |
| Phase 2+ | PyTorch, FAISS, HuggingFace |

---

## Blog Series

Each phase has a companion blog post:

1. **[Phase 1]** Building a Recommender System from Scratch — CF, Content-Based & Offline Eval
2. **[Phase 2]** Two-Tower Models and Neural Collaborative Filtering
3. **[Phase 3]** Production-Grade RecSys — Multi-Stage Pipelines, Debiasing, Diversity
4. **[Phase 4]** LLMs Meet RecSys — Semantic Retrieval and Conversational Recommendations
5. **[Phase 5]** RL for RecSys — Bandits, Exploration, and Long-Horizon Optimization

---

## License

MIT