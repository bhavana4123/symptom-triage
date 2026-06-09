"""Inference: feature extraction -> model prediction -> real disease lookup."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

_ROOT = Path(__file__).parent.parent
_DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "best_model.pkl"
_TRAIN_PATH = _ROOT / "data" / "processed" / "train.parquet"

# ── keyword map ───────────────────────────────────────────────────────────────
# Keys match the binary column names used during training (data_pipeline.py).
# Values are free-text patterns to detect from a patient's symptom description.

SYMPTOM_KEYWORDS: dict[str, list[str]] = {
    "Fever": [
        "fever", "temperature", "febrile", "high temp",
    ],
    "Cough": [
        "cough", "coughing",
    ],
    "Fatigue": [
        "fatigue", "tired", "exhausted", "weakness", "weak", "letharg",
    ],
    "Difficulty Breathing": [
        "difficulty breathing", "shortness of breath", "breathless",
        "can't breathe", "cannot breathe", "dyspnea", "breathing difficulty",
        "hard to breathe",
    ],
}

# Module-level caches — populated on first call, reused for the lifetime of the process.
_pipeline: Any = None
_disease_index: dict[str, list[str]] | None = None


# ── loaders ───────────────────────────────────────────────────────────────────


def load_pipeline(model_path: Path | None = None) -> Any:
    """Load and cache best_model.pkl.

    Args:
        model_path: Override path; defaults to MODEL_PATH env var or
            ``ml/models/best_model.pkl`` relative to the project root.

    Returns:
        Fitted sklearn Pipeline.

    Raises:
        FileNotFoundError: If the pickle file is absent.
    """
    global _pipeline
    if _pipeline is None:
        path = Path(model_path or os.getenv("MODEL_PATH", str(_DEFAULT_MODEL_PATH)))
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {path}. Run `make train` first."
            )
        with open(path, "rb") as f:
            _pipeline = pickle.load(f)
    return _pipeline


def load_disease_index(train_path: Path = _TRAIN_PATH) -> dict[str, list[str]]:
    """Build and cache an urgency_label -> [Disease, ...] index from train.parquet.

    Diseases are ordered by frequency so the most common condition for each
    urgency level appears first.

    Args:
        train_path: Path to the training parquet split.

    Returns:
        Dict mapping each urgency label to an ordered list of disease names.
    """
    global _disease_index
    if _disease_index is None:
        if not train_path.exists():
            _disease_index = {}
        else:
            df = pd.read_parquet(train_path)
            index: dict[str, list[str]] = {}
            for label in df["urgency_label"].unique():
                index[label] = (
                    df.loc[df["urgency_label"] == label, "Disease"]
                    .value_counts()
                    .head(10)
                    .index.tolist()
                )
            _disease_index = index
    return _disease_index


# ── feature extraction ────────────────────────────────────────────────────────


def extract_features(symptoms: str, age: int) -> pd.DataFrame:
    """Map a free-text symptom description to the feature vector the pipeline expects.

    Replicates the logic of ``encode_binary`` + ``engineer_features`` in
    data_pipeline.py so that inference uses exactly the same columns the
    ColumnTransformer was fitted on:
    ``symptom_text``, ``Age``, ``symptom_count``, ``Gender``.

    Args:
        symptoms: Raw patient-supplied symptom string.
        age: Patient age in years.

    Returns:
        Single-row DataFrame ready to pass to ``pipeline.predict_proba()``.
    """
    text = symptoms.lower()

    # Detect which of the 4 binary symptoms are present
    flags: dict[str, int] = {
        col: 1 if any(kw in text for kw in keywords) else 0
        for col, keywords in SYMPTOM_KEYWORDS.items()
    }

    # Build symptom_text the same way _symptom_text() does in data_pipeline:
    # join the lowercase column names of active symptoms.
    symptom_text = " ".join(col.lower() for col, val in flags.items() if val == 1)
    symptom_count = sum(flags.values())

    # If no keyword matched, fall back to the raw text so the TF-IDF branch
    # of the ColumnTransformer still has signal to work with.
    if not symptom_text:
        symptom_text = text

    return pd.DataFrame([{
        "symptom_text": symptom_text,
        "Age": age,
        "symptom_count": symptom_count,
        "Gender": 0,   # not collected at inference time; neutral default
    }])


# ── public predict API ────────────────────────────────────────────────────────


def top_diseases(urgency_label: str, n: int = 3) -> list[str]:
    """Return up to *n* real disease names for a given urgency label.

    Args:
        urgency_label: One of ``emergency``, ``urgent``, ``routine``.
        n: How many disease names to return.

    Returns:
        List of disease name strings drawn from training data, most frequent first.
    """
    return load_disease_index().get(urgency_label, [])[:n]


def predict_urgency(
    symptoms: str,
    age: int,
    duration_days: int,
    pipeline: Any | None = None,
) -> dict:
    """Run the full prediction pipeline for a single patient record.

    Args:
        symptoms: Free-text symptom description from the patient.
        age: Patient age in years.
        duration_days: How long symptoms have been present (unused by the
            current classifier but kept for API symmetry with the request model).
        pipeline: Pre-loaded sklearn Pipeline; if ``None``, loaded lazily
            via :func:`load_pipeline`.

    Returns:
        Dict with keys:
        - ``urgency_label`` — one of ``emergency``, ``urgent``, ``routine``
        - ``confidence``    — float in [0, 1]
        - ``top_conditions`` — list of up to 3 real disease names from training data
    """
    if pipeline is None:
        pipeline = load_pipeline()

    row = extract_features(symptoms, age)
    proba = pipeline.predict_proba(row)[0]
    label: str = pipeline.classes_[proba.argmax()]
    confidence = float(proba.max())

    return {
        "urgency_label": label,
        "confidence": confidence,
        "top_conditions": top_diseases(label, n=3),
    }
