# 🏆 League of Legends — Draft Recommendation System

An ML-powered draft assistant that recommends the optimal next champion pick
given the current draft state (picks, bans, team compositions).

Instead of simply **predicting** match outcomes, this system **optimises
draft decisions** — treating champion selection as a structured prediction /
ranking problem over the champion pool.

---

## ✨ Key Features

| Tier | Model | Approach |
|------|-------|----------|
| Baseline | **XGBoost / Random Forest** | Multi-hot draft-state → champion ranking |
| Neural | **MLP with Champion Embeddings** | Learned embeddings + pick/ban context |
| Advanced | **Causal Transformer** | Draft sequence as ordered tokens |

- **Synergy & counter matrices** built from historical match data
- **FastAPI serving layer** with interactive Swagger docs
- **MLflow experiment tracking** out-of-the-box
- Evaluation with **Top-k accuracy, MRR, NDCG**

---

## 📁 Project Structure

```
league-of-legends-Draft-Recommendation-System/
├── data/
│   ├── raw/          # Raw Riot API match JSON files
│   ├── processed/    # Clean Parquet files + synergy/counter matrices
│   └── external/     # Static champion metadata (Data Dragon)
│
├── src/
│   ├── data/
│   │   ├── ingest.py       # Riot API data pipeline
│   │   └── preprocess.py   # JSON → draft-event DataFrame
│   ├── features/
│   │   ├── champion_encoder.py   # Champion ID ↔ index + multi-hot encoding
│   │   └── synergy_counter.py    # Pairwise synergy & counter matrices
│   ├── models/
│   │   ├── baseline.py     # XGBoost / Random Forest recommender
│   │   ├── neural.py       # MLP with champion embeddings (PyTorch)
│   │   ├── transformer.py  # Causal Transformer (PyTorch)
│   │   └── train.py        # End-to-end training CLI
│   └── utils/
│       ├── config.py       # Centralised config.yaml loader
│       └── logger.py       # Standardised logging
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
├── models/            # Saved model checkpoints (.pkl / .pt)
│
├── evaluation/
│   ├── metrics.py     # Top-k accuracy, MRR, NDCG
│   └── evaluate.py    # Model comparison report
│
├── api/
│   ├── app.py         # FastAPI application factory
│   ├── routes.py      # /health + /recommend endpoints
│   └── schemas.py     # Pydantic request/response models
│
├── config.yaml        # All hyperparameters & paths
├── requirements.txt
└── setup.py
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Ingest match data

```bash
python -m src.data.ingest \
  --api-key RGAPI-your-key-here \
  --region na1 \
  --pages 5 \
  --matches-per-player 20
```

> **Tip:** You can also place any existing Riot match v5 JSON files directly
> in `data/raw/` and skip the ingestion step.

### 3. Preprocess raw data

```bash
python -m src.data.preprocess \
  --input data/raw \
  --output data/processed/drafts.parquet
```

You can also pass a JSONL file (e.g. Kaggle's `match_data.jsonl`) directly:

```bash
python -m src.data.preprocess --input data/raw/match_data.jsonl
```

### 4. Train a model

```bash
# XGBoost baseline (fast)
python -m src.models.train --model xgb

# MLP with embeddings
python -m src.models.train --model mlp --epochs 50

# Transformer (best accuracy, slowest)
python -m src.models.train --model transformer --epochs 100
```

MLflow runs are logged to `mlruns/`.  Launch the UI with:
```bash
mlflow ui
```

### 5. Evaluate models

```bash
python -m evaluation.evaluate --data data/processed/drafts.parquet
```

### 6. Run the API server

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

### 7. Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

The app lets you enter the current draft (picks/bans) and returns top-5
recommendations with a win-probability signal.

#### Example API request

```bash
curl -X POST http://localhost:8000/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "blue_picks": [157, 64],
    "red_picks": [238],
    "blue_bans": [11, 99],
    "red_bans": [517, 235],
    "pick_order": 2,
    "team": "blue",
    "role": "MID",
    "top_k": 5
  }'
```

---

## 🧠 ML Framing

```
Agent:   Draft assistant
Inputs:  Current picks, bans, roles, team side
Actions: Recommend next champion
Goal:    Maximise win probability
```

The problem is framed as **multi-class ranking** rather than binary win
prediction:

- **Input**: Draft-state feature vector (multi-hot picks/bans + role/team
  one-hot encodings)
- **Output**: Probability distribution over the champion pool → rank and serve
  the top-k

The Transformer model treats the draft as a sequence and predicts the next
token (champion) at each step — analogous to language modelling.

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Top-1 / Top-3 / Top-5 accuracy | Was the correct pick in the model's top-k? |
| MRR | Mean Reciprocal Rank — how high does the correct pick appear? |
| NDCG@k | Normalised Discounted Cumulative Gain at cutoff k |

---

## 🗺️ Roadmap

- [ ] Role-aware pick ordering (respect the true Bo5 ban/pick sequence)
- [ ] Patch-aware embeddings (champion stats shift between patches)
- [ ] Reinforcement learning agent that plays out full drafts
- [ ] Web front-end with real-time pick/ban UI
