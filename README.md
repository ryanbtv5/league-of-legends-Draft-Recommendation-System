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
| Baseline | **Random Forest** | Multi-hot draft-state → champion ranking |
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
│   │   ├── baseline.py     # Random Forest recommender
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
raw data set used:

https://www.kaggle.com/datasets/californianbill/patch-25-14-lol-league-of-legends-ranked-games/data?select=match_data.jsonl

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
# Random Forest baseline (fast)
python -m src.models.train --model rf

# MLP with embeddings
python -m src.models.train --model mlp --epochs 30

# Resume MLP training from the best checkpoint
python -m src.models.train --model mlp --epochs 30 --resume-from models/mlp_recommender_best.pt

# Transformer (best accuracy, slowest)
python -m src.models.train --model transformer --epochs 60

# Resume Transformer training from the best checkpoint
python -m src.models.train --model transformer --epochs 60 --resume-from models/transformer_recommender_best.pt

```

Additional: train the win-probability predictor (**Note**: win outcome prediction from draft state alone is ~50% baseline—difficult without player skill, patch context, and macro play):

```bash
# Train only the win predictor (optional; difficult task)
python -m src.models.train --model win

# When you train the transformer (above) the win predictor is also trained
# automatically as part of the pipeline (configurable via `model.win_predictor`).
```

MLflow runs are logged to `mlruns/`.  Launch the UI with:
```bash
mlflow ui
```

### 5. Evaluate models

```bash
python -m evaluation.evaluate --data data/processed/drafts.parquet
```

This evaluation command works with the current draft-state schema and reports
metrics for the saved Random Forest and MLP models.

### 6. Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

The Streamlit app is the main interface. It lets you enter the current draft
(picks/bans) and returns top-5 recommendations with a win-probability signal.
It includes Random Forest, MLP, and Transformer model choices when the
corresponding checkpoints are present in `models/`.

**Startup Optimization:** Models are lazy-loaded only when you click "Recommend",
so the app UI appears instantly. Champion names are cached after first load.
On subsequent runs, everything loads from cache (< 1 second).

> The API server is optional and only needed if you want programmatic access
> or the Swagger docs.

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
Inputs:  Current picks, bans, team side
Actions: Recommend next champion
Goal:    Maximise win probability
```

The problem is framed as **multi-class ranking** rather than binary win
prediction:

  **Input**: Draft-state feature vector (multi-hot picks/bans + team one-hot encoding)
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
