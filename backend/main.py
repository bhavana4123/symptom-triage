"""FastAPI application — ML-powered symptom triage with SQLite history."""

from __future__ import annotations

import logging
import sqlite3
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ml.predict import load_disease_index, load_pipeline, predict_urgency
from ml.triage_reasoner import TriageReasoner

# ── paths ─────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).parent.parent
_MODEL_PATH = _ROOT / "ml" / "models" / "best_model.pkl"
_VECTORIZER_PATH = _ROOT / "ml" / "models" / "vectorizer.pkl"
_DB_PATH = _ROOT / "symptom_triage.db"

# ── logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("triage")

# ── pydantic models ───────────────────────────────────────────────────────────


class TriageRequest(BaseModel):
    symptoms: str = Field(..., min_length=3, description="Free-text symptom description")
    duration_days: int = Field(..., ge=0, description="How many days symptoms have been present")
    patient_age: int = Field(..., ge=0, le=120, description="Patient age in years")


class TriageResponse(BaseModel):
    urgency_label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    possible_conditions: list[str]
    explanation: str
    next_steps: list[str]
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float


class HistoryItem(BaseModel):
    id: int
    timestamp: str
    symptoms: str
    urgency_label: str
    confidence: float


# ── sqlite helpers ────────────────────────────────────────────────────────────


@contextmanager
def _db():
    """Yield a connected, auto-committing sqlite3 connection."""
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _init_db() -> None:
    """Create the triage_results table if it does not exist."""
    with _db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS triage_results (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT    NOT NULL DEFAULT (datetime('now')),
                symptoms      TEXT    NOT NULL,
                urgency_label TEXT    NOT NULL,
                confidence    REAL    NOT NULL
            )
            """
        )


def _insert_result(symptoms: str, urgency_label: str, confidence: float) -> None:
    """Persist a single triage result to SQLite."""
    with _db() as conn:
        conn.execute(
            "INSERT INTO triage_results (symptoms, urgency_label, confidence) VALUES (?, ?, ?)",
            (symptoms, urgency_label, confidence),
        )


def _fetch_history(limit: int = 20) -> list[dict[str, Any]]:
    """Return the most recent *limit* triage results, newest first."""
    with _db() as conn:
        rows = conn.execute(
            """
            SELECT id, timestamp, symptoms, urgency_label, confidence
            FROM   triage_results
            ORDER  BY id DESC
            LIMIT  ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


# ── lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and initialise SQLite on startup; nothing to clean up on shutdown."""
    _init_db()

    model_loaded = False
    if _MODEL_PATH.exists() and _VECTORIZER_PATH.exists():
        app.state.pipeline = load_pipeline(_MODEL_PATH)
        load_disease_index()  # warm up the disease index cache
        model_loaded = True
        log.info("Models loaded from %s", _MODEL_PATH.parent)
    else:
        app.state.pipeline = None
        log.warning(
            "ML models not found at %s — serving /api/health only until `make train` is run",
            _MODEL_PATH.parent,
        )

    app.state.model_loaded = model_loaded
    app.state.start_time = time.time()
    app.state.reasoner = TriageReasoner()

    yield  # application runs here

    # shutdown — SQLite connections are short-lived, nothing to close


# ── app ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Symptom Triage API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── request logging middleware ────────────────────────────────────────────────


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log method, path, latency, and status for every request.

    Business data (symptoms, prediction) is attached to request.state by
    the endpoint and flushed here after the response is complete, so a single
    log line captures the full picture at the end of each triage call.
    """
    start = time.perf_counter()
    request.state.triage_log: dict[str, Any] = {}

    response = await call_next(request)

    latency_ms = (time.perf_counter() - start) * 1000
    tl = request.state.triage_log

    if tl:
        log.info(
            "POST /api/triage  symptoms=%r  urgency=%s  conf=%.2f  %.1fms  status=%d",
            tl.get("symptoms", "")[:80],
            tl.get("urgency_label"),
            tl.get("confidence", 0.0),
            latency_ms,
            response.status_code,
        )
    else:
        log.info(
            "%s %s  %.1fms  status=%d",
            request.method,
            request.url.path,
            latency_ms,
            response.status_code,
        )

    return response


# ── endpoints ─────────────────────────────────────────────────────────────────


@app.post("/api/triage", response_model=TriageResponse)
def triage(body: TriageRequest, request: Request) -> TriageResponse:
    """Run ML inference + rule-based reasoning on the submitted symptoms.

    1. Validates the request with Pydantic.
    2. Runs the ColumnTransformer + classifier pipeline to get urgency_label and confidence.
    3. Calls TriageReasoner.explain() for the human-readable report.
    4. Persists the result to SQLite.
    5. Attaches log data to request.state for the logging middleware.
    """
    if not request.app.state.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="ML model not available. Run `make train` and restart the server.",
        )

    ml_prediction = predict_urgency(
        symptoms=body.symptoms,
        age=body.patient_age,
        duration_days=body.duration_days,
        pipeline=request.app.state.pipeline,
    )
    urgency_label = ml_prediction["urgency_label"]
    confidence = ml_prediction["confidence"]

    reasoner: TriageReasoner = request.app.state.reasoner
    result = reasoner.explain(symptoms=body.symptoms, ml_prediction=ml_prediction)

    _insert_result(body.symptoms, urgency_label, confidence)

    # expose to the logging middleware
    request.state.triage_log = {
        "symptoms": body.symptoms,
        "urgency_label": urgency_label,
        "confidence": confidence,
    }

    return TriageResponse(**result)


@app.get("/api/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Return server liveness, model status, and uptime."""
    return HealthResponse(
        status="ok",
        model_loaded=request.app.state.model_loaded,
        uptime_seconds=round(time.time() - request.app.state.start_time, 2),
    )


@app.get("/api/history", response_model=list[HistoryItem])
def history() -> list[HistoryItem]:
    """Return the last 20 triage results from the SQLite database, newest first."""
    return _fetch_history(limit=20)
