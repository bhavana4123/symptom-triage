# Symptom Triage Assistant

An AI-powered symptom triage web application that takes free-text patient
symptoms and returns a structured urgency assessment, possible conditions, and
plain-language next steps — without requiring a network call to an external AI
service for every request.

> **Disclaimer:** This tool is for educational and research purposes only.
> It is **not** a substitute for professional medical advice, diagnosis, or treatment.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) |
| ML | scikit-learn (TF-IDF + Logistic Regression / Random Forest / Gradient Boosting) |
| NLP | [spaCy](https://spacy.io/) `en_core_web_sm` |
| Experiment tracking | [MLflow](https://mlflow.org/) |
| Rule-based reasoning | Custom `TriageReasoner` (no external API) |
| Frontend | React 18 + [Tailwind CSS](https://tailwindcss.com/) + [Vite](https://vitejs.dev/) |
| Database | SQLite (triage history) |
| Containerisation | Docker + Docker Compose + nginx |

---

## Project Structure

```
symptom-triage/
├── backend/                  FastAPI application
│   ├── main.py               App factory, lifespan, all endpoints
│   ├── api/routes.py         Secondary Claude-backed route (optional)
│   ├── models/schemas.py     Pydantic schemas
│   └── services/triage.py   Claude API integration
│
├── ml/                       Machine-learning layer
│   ├── data_pipeline.py      Load → deduplicate → encode → split → save parquet
│   ├── train.py              MLflow experiment: 3 classifiers, best model saved
│   ├── predict.py            Inference wrapper (loads best_model.pkl)
│   ├── triage_reasoner.py    Rule-based explanation engine (no external API)
│   └── features.py           spaCy NER + lemmatisation helpers
│
├── frontend/                 React + Tailwind UI
│   └── src/
│       ├── App.jsx
│       └── components/
│           ├── SymptomForm.jsx
│           └── TriageResult.jsx
│
├── docker/                   Container configuration
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── nginx.conf
│
├── data/
│   ├── raw/                  Source CSV (symptom_disease.csv)
│   └── processed/            train / val / test parquet splits
│
├── ml/models/                Saved model artefacts (git-ignored)
├── tests/                    pytest test suites
├── docker-compose.yml
├── pyproject.toml
├── Makefile
└── .env.example
```

---

## Running Locally

### Prerequisites

- Python 3.11+
- Node.js 20+

### 1. Install and configure

```bash
git clone <repo-url>
cd symptom-triage

cp .env.example .env
# Edit .env — at minimum set ANTHROPIC_API_KEY if you want the Claude fallback path
```

### 2. Install Python dependencies

```bash
make install
# Installs all packages + downloads the spaCy en_core_web_sm model
```

### 3. Prepare data and train the model

Place `symptom_disease.csv` (columns: `Disease`, `Fever`, `Cough`, `Fatigue`,
`Difficulty Breathing`, `Age`, `Gender`, `Blood Pressure`, `Cholesterol Level`,
`Outcome Variable`) in `data/raw/`, then:

```bash
make pipeline   # deduplicate + encode + split → data/processed/*.parquet
make train      # trains 3 classifiers, saves best to ml/models/best_model.pkl
```

MLflow tracks every training run. Open the UI with:

```bash
mlflow ui
# http://localhost:5000
```

### 4. Start the API

```bash
make serve
# API:  http://localhost:8000
# Docs: http://localhost:8000/docs
```

### 5. Start the frontend

```bash
cd frontend
npm install
npm run dev
# UI: http://localhost:5173
```

### 6. Run tests

```bash
make test
```

---

## Running with Docker

```bash
# Build and start both services
docker compose up --build

# Or in the background
docker compose up --build -d
```

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API docs | http://localhost:8000/docs |

Stop and remove containers:

```bash
docker compose down
```

### Environment variables

Copy `.env.example` to `.env` and fill in values before running:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Claude API key (optional — only needed for the `/api/v1/triage` Claude path) |
| `DATABASE_URL` | SQLite path, e.g. `sqlite:///./symptom_triage.db` |
| `MODEL_PATH` | Override path to `best_model.pkl` |

---

## API Reference

### `POST /api/triage`

Runs the local ML pipeline and returns a structured triage report.

**Request**
```json
{
  "symptoms": "fever, cough, difficulty breathing",
  "duration_days": 3,
  "patient_age": 42
}
```

**Response**
```json
{
  "urgency_label": "urgent",
  "confidence": 0.87,
  "explanation": "Your symptoms suggest you should see a doctor within 24 hours.",
  "next_steps": [
    "Book a same-day doctor appointment",
    "Monitor symptoms closely",
    "Go to ER if symptoms worsen"
  ],
  "possible_conditions": ["Acute infection", "Moderate injury", "Severe pain"],
  "disclaimer": "This is not a medical diagnosis. Always consult a qualified doctor."
}
```

**Urgency levels**

| Label | Meaning |
|---|---|
| `emergency` | Go to the ER or call 108 immediately |
| `urgent` | See a doctor within 24 hours |
| `routine` | Schedule a routine appointment |

### `GET /api/health`

```json
{ "status": "ok", "model_loaded": true, "uptime_seconds": 143.2 }
```

### `GET /api/history`

Returns the last 20 triage results from SQLite, newest first.

---


---

## Makefile Targets

```
make install       Install Python deps + spaCy model
make pipeline      Run data_pipeline.py → data/processed/*.parquet
make train         Run train.py → ml/models/best_model.pkl
make serve         Start FastAPI with uvicorn --reload
make test          Run pytest
make docker-build  docker compose build
```
