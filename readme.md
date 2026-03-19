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
| 1 | Collaborative filtering, matrix factorization, content-based, hybrid | ✅ Complete |
| 2 | Neural RecSys — two-tower, NeuMF, SASRec, FAISS ANN | ✅ Complete |
| 3 | Production pipeline — multi-stage, IPS debiasing, MMR+DPP diversity | ✅ Complete |
| 4 | GenAI × RecSys — semantic retrieval, LLM reranking, conversational | ✅ Complete |
| 5 | RL for RecSys — UCB, Thompson sampling, LinUCB, CQL offline RL | ✅ Complete |

---

## Phase 1 — Foundations ✅

**Dataset**: Amazon Reviews 2023 — Gift Cards · 553 users × 154 items · 97% sparse

| Model | HR@10 | NDCG@10 | Coverage | Notes |
|-------|-------|---------|----------|-------|
| User-based CF (K=30) | 0.009 | 0.003 | 12.5% | Sparsity kills coverage |
| SVD (k=20) | 0.122 | 0.042 | 99.6% | Dense vectors fix coverage |
| ALS (k=50, α=40) | 0.112 | 0.088 | 99.6% | Confidence weighting improves ranking |
| Content-based (TF-IDF) | 0.477 | 0.275 | 100% | Item similarity dominates on Gift Cards |
| **Hybrid (α=0.9)** | **0.485** | **0.279** | **100%** | Best Phase 1 model |

---

## Phase 2 — Neural RecSys ✅

| Model | HR@10 | NDCG@10 | Coverage | Role |
|-------|-------|---------|----------|------|
| Two-Tower | 0.121 | 0.051 | 100% | Candidate retrieval |
| Two-Tower + FAISS HNSW | 0.112 | 0.044 | 100% | Production serving (1.8x speedup) |
| NeuMF (GMF + MLP) | 0.441 | 0.261 | 100% | Candidate re-ranking |
| SASRec (2-layer) | 0.056 | 0.031 | 100% | Sequential (needs more data to shine) |

---

## Phase 3 — Production Pipeline ✅

| Config | HR@10 | NDCG@10 | IPS-NDCG@10 | ILD | Latency p50 |
|--------|-------|---------|-------------|-----|-------------|
| No diversity (λ=1.0) | 0.440 | 0.261 | — | — | 2.39ms |
| Diversity only (λ=0.3) | 0.386 | 0.228 | — | — | 2.39ms |
| Full pipeline (λ=0.3 + IPS) | 0.373 | 0.222 | 0.341 | 0.831 | 2.38ms |

IPS-NDCG (0.341) is 52% higher than standard NDCG (0.222) — position bias in the
test set was systematically undervaluing items at lower positions.

---

## Phase 4 — GenAI × RecSys ✅

| Model | HR@10 | NDCG@10 | Cold-start |
|-------|-------|---------|------------|
| CB Phase 1 (co-rater TF-IDF) | 0.477 | 0.275 | No |
| Semantic (text TF-IDF) | 0.088 | 0.036 | Yes |
| NeuMF baseline | 0.540 | 0.358 | No |
| Listwise LLM reranking | 0.550 | 0.335 | No |
| **Hybrid (NeuMF + LLM)** | **0.500** | **0.380** | No |

Hybrid NeuMF+LLM achieves best NDCG (+6% over NeuMF alone).
Semantic retrieval trades accuracy for cold-start capability.

---

## Phase 5 — RL for RecSys ✅

| Algorithm | Hit Rate | Improvement (first→last 100 steps) | Notes |
|-----------|----------|--------------------------------------|-------|
| Random | 0.058 | -0.050 | Never learns |
| UCB (α=1.0) | 0.061 | -0.010 | Count-based, no features |
| **Thompson Sampling** | **0.084** | **+0.070** | Best — no tuning needed |
| LinUCB (α=0.5) | 0.066 | -0.010 | Contextual, needs richer features |
| DQN (no CQL) | 0.003 | — | Catastrophic OOD exploitation |
| CQL (α=1.0) | 0.051 | — | Conservative, safe deployment |

Thompson Sampling is the only algorithm with positive improvement (+0.07) —
it explores efficiently and learns which items are genuinely good over time.
DQN without CQL collapses (0.003 hit rate) due to distributional shift.

---

## Project Structure

```
src/
├── data_loader.py         Amazon Reviews 2023, cold-start filtering, matrix build
├── user_cf.py             Mean-centered cosine CF, neighbourhood prediction
├── matrix_factor.py       SVD + ALS (Hu-Koren-Volinsky 2008)
├── content_based.py       TF-IDF item profiles, item-item similarity
├── hybrid.py              Weighted + switching hybrid strategies
├── evaluate.py            HR@K, NDCG@K, Precision@K
├── phase2/
│   ├── two_tower.py       Dual encoder, in-batch softmax, item index
│   ├── ncf.py             GMF + MLP fusion (NeuMF), BCE training
│   ├── sequential.py      SASRec — causal self-attention, positional embeddings
│   └── ann_index.py       FAISS FlatIP / IVF / HNSW benchmark
├── phase3/
│   ├── pipeline.py        Multi-stage: two-tower → NeuMF → MMR + IPS
│   ├── debiasing.py       IPS propensity models, IPS-corrected NDCG
│   ├── diversity.py       MMR + DPP diversity, ILD metric, coverage
│   └── evaluation.py      Full eval suite — relevance + diversity + latency
├── phase4/
│   ├── semantic_retrieval.py  TF-IDF + LLM embeddings, cold-start support
│   ├── llm_reranker.py        Pointwise + listwise + hybrid LLM reranking
│   └── conversational.py      Multi-turn dialogue agent, preference extraction
└── phase5/
    ├── bandits.py         UCB, Thompson Sampling, LinUCB
    └── offline_rl.py      DQN vs CQL offline RL, replay buffer, policy eval
```

---

## Setup

```bash
git clone https://github.com/peace-chaos26/recsys-mastery.git
cd recsys-mastery
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Run all phases

```bash
# Phase 1
python3 src/data_loader.py
python3 src/user_cf.py
OPENBLAS_NUM_THREADS=1 python3 src/matrix_factor.py
python3 src/content_based.py
OPENBLAS_NUM_THREADS=1 python3 src/hybrid.py

# Phase 2
python3 src/phase2/two_tower.py
python3 src/phase2/ncf.py
python3 src/phase2/sequential.py
python3 src/phase2/ann_index.py        # Mac: prepend KMP_DUPLICATE_LIB_OK=TRUE

# Phase 3
python3 src/phase3/pipeline.py
python3 src/phase3/debiasing.py
OPENBLAS_NUM_THREADS=1 python3 src/phase3/diversity.py
OPENBLAS_NUM_THREADS=1 python3 src/phase3/evaluation.py

# Phase 4
python3 src/phase4/semantic_retrieval.py
python3 src/phase4/llm_reranker.py     # set OPENAI_API_KEY for LLM calls
python3 src/phase4/conversational.py   # set OPENAI_API_KEY for LLM dialogue

# Phase 5
python3 src/phase5/bandits.py
python3 src/phase5/offline_rl.py
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
| LLM reranking | OpenAI gpt-4o-mini |
| Evaluation | custom — HR, NDCG, IPS-NDCG, ILD, coverage |

---

## Blog Series

1. **[Phase 1]** Building a Recommender System from Scratch — CF, MF, Content-Based & Hybrid
2. **[Phase 2]** Neural RecSys — Two-Tower, NeuMF, SASRec and FAISS at Scale
3. **[Phase 3]** Production RecSys — Multi-Stage Pipelines, IPS Debiasing, MMR Diversity
4. **[Phase 4]** LLMs Meet RecSys — Semantic Retrieval, Re-ranking, Conversational Agents
5. **[Phase 5]** RL for RecSys — Bandits, Exploration and Conservative Q-Learning

---

## References

- Hu, Y., Koren, Y., & Volinsky, C. (2008). *Collaborative Filtering for Implicit Feedback Datasets*. ICDM.
- He, X., et al. (2017). *Neural Collaborative Filtering*.
- Kang, W., & McAuley, J. (2018). *Self-Attentive Sequential Recommendation*. ICDM.
- Li, L., et al. (2010). *A Contextual-Bandit Approach to Personalized News Recommendation*.
- Kumar, A., et al. (2020). *Conservative Q-Learning for Offline Reinforcement Learning*. NeurIPS.
- Hou, Y., et al. (2024). *Bridging Language and Items for Retrieval and Recommendation*. arXiv:2403.03952.

---

## License

MIT