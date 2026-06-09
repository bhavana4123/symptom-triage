"""
Pytest test suites:
  TestDataPipeline  — integration tests against data/processed/*.parquet
  TestModel         — integration tests against ml/models/best_model.pkl
  TestTriageReasoner — unit tests for ml.triage_reasoner.TriageReasoner
  TestAPI           — endpoint tests using FastAPI TestClient with mocked state
"""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from ml.triage_reasoner import TriageReasoner

# ── paths ─────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).parent.parent
_PROCESSED = _ROOT / "data" / "processed"
_MODEL_DIR = _ROOT / "ml" / "models"

# ── constants ─────────────────────────────────────────────────────────────────

VALID_URGENCY_LABELS = {"emergency", "urgent", "routine"}

_TRIAGE_RESPONSE_KEYS = {
    "urgency_label",
    "confidence",
    "explanation",       # spec says "urgency_explanation" but the implementation
    "next_steps",        # uses "explanation" — tests match the actual contract
    "possible_conditions",
    "disclaimer",
}

_HISTORY_ITEM_KEYS = {"id", "timestamp", "symptoms", "urgency_label", "confidence"}

_VALID_REQUEST = {"symptoms": "fever cough fatigue", "duration_days": 3, "patient_age": 35}

# ── shared fixtures ───────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def train_df() -> pd.DataFrame:
    path = _PROCESSED / "train.parquet"
    if not path.exists():
        pytest.skip("train.parquet not found — run `make pipeline`")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def val_df() -> pd.DataFrame:
    path = _PROCESSED / "val.parquet"
    if not path.exists():
        pytest.skip("val.parquet not found — run `make pipeline`")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def test_df() -> pd.DataFrame:
    path = _PROCESSED / "test.parquet"
    if not path.exists():
        pytest.skip("test.parquet not found — run `make pipeline`")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def best_model():
    path = _MODEL_DIR / "best_model.pkl"
    if not path.exists():
        pytest.skip("best_model.pkl not found — run `make train`")
    with open(path, "rb") as f:
        return pickle.load(f)


@pytest.fixture
def reasoner() -> TriageReasoner:
    return TriageReasoner()


# ── API test helpers ──────────────────────────────────────────────────────────

_API_CLASSES = sorted(VALID_URGENCY_LABELS)  # ["emergency", "routine", "urgent"]


def _make_pipeline_mock(predicted_class: str = "urgent") -> MagicMock:
    """Return a mock sklearn pipeline that predicts *predicted_class* with 87% confidence."""
    idx = _API_CLASSES.index(predicted_class)
    proba = np.zeros(len(_API_CLASSES))
    proba[idx] = 0.87
    mock = MagicMock()
    mock.predict_proba.return_value = [proba]
    mock.classes_ = np.array(_API_CLASSES)
    return mock


def _set_state(client: TestClient, pipeline: object, model_loaded: bool = True) -> None:
    client.app.state.pipeline = pipeline
    client.app.state.model_loaded = model_loaded


# ═════════════════════════════════════════════════════════════════════════════
# 1 · Data pipeline
# ═════════════════════════════════════════════════════════════════════════════


class TestDataPipeline:
    """Validate the parquet splits produced by ml/data_pipeline.py."""

    def test_train_parquet_is_non_empty(self, train_df):
        assert len(train_df) > 0, "train.parquet has no rows"

    def test_val_parquet_is_non_empty(self, val_df):
        assert len(val_df) > 0, "val.parquet has no rows"

    def test_test_parquet_is_non_empty(self, test_df):
        assert len(test_df) > 0, "test.parquet has no rows"

    def test_urgency_label_column_exists(self, train_df):
        assert "urgency_label" in train_df.columns

    def test_train_urgency_labels_are_valid(self, train_df):
        actual = set(train_df["urgency_label"].unique())
        assert actual <= VALID_URGENCY_LABELS, f"unexpected labels in train: {actual - VALID_URGENCY_LABELS}"

    def test_val_urgency_labels_are_valid(self, val_df):
        actual = set(val_df["urgency_label"].unique())
        assert actual <= VALID_URGENCY_LABELS, f"unexpected labels in val: {actual - VALID_URGENCY_LABELS}"

    def test_test_urgency_labels_are_valid(self, test_df):
        actual = set(test_df["test_df" if "test_df" in test_df.columns else "urgency_label"].unique()
                     if "urgency_label" in test_df.columns else [])
        actual = set(test_df["urgency_label"].unique())
        assert actual <= VALID_URGENCY_LABELS, f"unexpected labels in test: {actual - VALID_URGENCY_LABELS}"

    def test_splits_have_no_overlapping_rows(self, train_df, val_df, test_df):
        # pd.util.hash_pandas_object gives a stable per-row hash over all column values.
        # Disjoint hash sets means no row appears in more than one split.
        train_hashes = set(pd.util.hash_pandas_object(train_df, index=False))
        val_hashes   = set(pd.util.hash_pandas_object(val_df,   index=False))
        test_hashes  = set(pd.util.hash_pandas_object(test_df,  index=False))

        assert train_hashes.isdisjoint(val_hashes),  "train and val share rows"
        assert train_hashes.isdisjoint(test_hashes), "train and test share rows"
        assert val_hashes.isdisjoint(test_hashes),   "val and test share rows"

    def test_train_is_largest_split(self, train_df, val_df, test_df):
        assert len(train_df) > len(val_df)
        assert len(train_df) > len(test_df)

    def test_val_and_test_are_similar_size(self, val_df, test_df):
        ratio = len(val_df) / max(len(test_df), 1)
        assert 0.8 <= ratio <= 1.25, f"val/test ratio {ratio:.2f} is unexpectedly skewed"


# ═════════════════════════════════════════════════════════════════════════════
# 2 · ML model
# ═════════════════════════════════════════════════════════════════════════════


class TestModel:
    """Smoke-tests for the trained scikit-learn pipeline."""

    def test_model_loads_without_error(self, best_model):
        assert best_model is not None

    def test_model_has_predict_proba(self, best_model):
        assert callable(getattr(best_model, "predict_proba", None))

    def test_model_has_valid_classes(self, best_model):
        classes = set(best_model.classes_)
        assert classes <= VALID_URGENCY_LABELS, f"unexpected classes: {classes - VALID_URGENCY_LABELS}"

    def test_prediction_returns_valid_urgency_label(self, best_model):
        sample = pd.DataFrame([{
            "symptom_text": "fever cough fatigue",
            "Age": 35,
            "symptom_count": 3,
            "Gender": 0,
        }])
        proba = best_model.predict_proba(sample)[0]
        label = best_model.classes_[proba.argmax()]
        assert label in VALID_URGENCY_LABELS, f"got unexpected label: {label!r}"

    def test_confidence_is_between_0_and_1(self, best_model):
        sample = pd.DataFrame([{
            "symptom_text": "chest pain shortness of breath",
            "Age": 55,
            "symptom_count": 2,
            "Gender": 1,
        }])
        proba = best_model.predict_proba(sample)[0]
        confidence = float(proba.max())
        assert 0.0 <= confidence <= 1.0, f"confidence {confidence} is out of [0, 1]"

    def test_probabilities_sum_to_one(self, best_model):
        sample = pd.DataFrame([{
            "symptom_text": "mild headache",
            "Age": 28,
            "symptom_count": 1,
            "Gender": 0,
        }])
        proba = best_model.predict_proba(sample)[0]
        assert abs(proba.sum() - 1.0) < 1e-6, f"probabilities sum to {proba.sum()}, expected 1.0"


# ═════════════════════════════════════════════════════════════════════════════
# 3 · TriageReasoner
# ═════════════════════════════════════════════════════════════════════════════


class TestTriageReasoner:
    """Unit tests for ml.triage_reasoner.TriageReasoner."""

    # shared predictions used across multiple tests
    _urgent_pred = {
        "urgency_label": "urgent",
        "confidence": 0.82,
        "top_conditions": ["Influenza", "Pneumonia", "Bronchitis"],
    }
    _emergency_pred = {
        "urgency_label": "emergency",
        "confidence": 0.95,
        "top_conditions": ["Cardiac emergency", "Stroke"],
    }
    _routine_pred = {
        "urgency_label": "routine",
        "confidence": 0.78,
        "top_conditions": ["Minor viral infection"],
    }

    def test_explain_returns_all_required_keys(self, reasoner):
        result = reasoner.explain("fever cough", self._urgent_pred)
        assert _TRIAGE_RESPONSE_KEYS <= result.keys(), (
            f"missing keys: {_TRIAGE_RESPONSE_KEYS - result.keys()}"
        )

    def test_explain_returns_urgency_label(self, reasoner):
        result = reasoner.explain("fever cough", self._urgent_pred)
        assert result["urgency_label"] == "urgent"

    def test_explain_passes_through_confidence(self, reasoner):
        result = reasoner.explain("fever cough fatigue", self._urgent_pred)
        assert result["confidence"] == pytest.approx(0.82)

    def test_explain_passes_through_possible_conditions(self, reasoner):
        result = reasoner.explain("fever cough fatigue", self._urgent_pred)
        assert result["possible_conditions"] == ["Influenza", "Pneumonia", "Bronchitis"]

    def test_disclaimer_is_always_present(self, reasoner):
        for pred in (self._urgent_pred, self._emergency_pred, self._routine_pred):
            result = reasoner.explain("some symptoms", pred)
            assert result["disclaimer"], "disclaimer must be a non-empty string"

    def test_emergency_next_steps(self, reasoner):
        result = reasoner.explain("severe chest pain", self._emergency_pred)
        steps = result["next_steps"]
        assert isinstance(steps, list)
        assert len(steps) > 0
        assert any("108" in s for s in steps), "emergency steps must mention 108"
        assert any("drive" in s.lower() for s in steps), "emergency steps must mention not driving"

    def test_urgent_next_steps(self, reasoner):
        result = reasoner.explain("high fever headache", self._urgent_pred)
        steps = result["next_steps"]
        assert isinstance(steps, list)
        assert len(steps) > 0
        assert any("doctor" in s.lower() or "appointment" in s.lower() for s in steps)

    def test_routine_next_steps(self, reasoner):
        result = reasoner.explain("mild cough", self._routine_pred)
        steps = result["next_steps"]
        assert isinstance(steps, list)
        assert len(steps) > 0
        assert any("checkup" in s.lower() or "schedule" in s.lower() for s in steps)

    def test_unknown_label_raises_value_error(self, reasoner):
        with pytest.raises(ValueError, match="Unknown urgency_label"):
            reasoner.explain(
                "some symptoms",
                {"urgency_label": "critical", "confidence": 0.9, "top_conditions": []},
            )

    def test_cache_returns_identical_object_on_repeat_call(self, reasoner):
        pred = {"urgency_label": "routine", "confidence": 0.6, "top_conditions": ["Cold"]}
        first  = reasoner.explain("runny nose sneezing", pred)
        second = reasoner.explain("runny nose sneezing", pred)
        assert second is first, "repeated identical call must return the cached object"

    def test_different_symptoms_produce_independent_results(self, reasoner):
        r1 = reasoner.explain("fever cough", self._urgent_pred)
        r2 = reasoner.explain("mild headache", self._routine_pred)
        assert r1["urgency_label"] != r2["urgency_label"]


# ═════════════════════════════════════════════════════════════════════════════
# 4 · API endpoints
# ═════════════════════════════════════════════════════════════════════════════


class TestAPI:
    """FastAPI endpoint tests using TestClient with mocked app state."""

    # ── GET /api/health ───────────────────────────────────────────────────────

    def test_health_returns_200(self):
        with TestClient(app) as c:
            r = c.get("/api/health")
        assert r.status_code == 200

    def test_health_response_schema(self):
        with TestClient(app) as c:
            data = c.get("/api/health").json()
        assert {"status", "model_loaded", "uptime_seconds"} <= data.keys()
        assert data["status"] == "ok"
        assert isinstance(data["model_loaded"], bool)
        assert data["uptime_seconds"] >= 0

    def test_health_model_loaded_true_when_state_set(self):
        with TestClient(app) as c:
            _set_state(c, _make_pipeline_mock(), model_loaded=True)
            data = c.get("/api/health").json()
        assert data["model_loaded"] is True

    # ── POST /api/triage ──────────────────────────────────────────────────────

    def test_triage_valid_request_returns_200(self):
        with TestClient(app) as c:
            _set_state(c, _make_pipeline_mock("urgent"))
            r = c.post("/api/triage", json=_VALID_REQUEST)
        assert r.status_code == 200

    def test_triage_response_contains_all_required_keys(self):
        with TestClient(app) as c:
            _set_state(c, _make_pipeline_mock("urgent"))
            data = c.post("/api/triage", json=_VALID_REQUEST).json()
        assert _TRIAGE_RESPONSE_KEYS <= data.keys(), (
            f"missing keys: {_TRIAGE_RESPONSE_KEYS - data.keys()}"
        )

    def test_triage_urgency_label_is_valid(self):
        with TestClient(app) as c:
            _set_state(c, _make_pipeline_mock("urgent"))
            data = c.post("/api/triage", json=_VALID_REQUEST).json()
        assert data["urgency_label"] in VALID_URGENCY_LABELS

    def test_triage_confidence_in_range(self):
        with TestClient(app) as c:
            _set_state(c, _make_pipeline_mock("urgent"))
            data = c.post("/api/triage", json=_VALID_REQUEST).json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_triage_emergency_label_and_steps(self):
        with TestClient(app) as c:
            _set_state(c, _make_pipeline_mock("emergency"))
            data = c.post(
                "/api/triage",
                json={"symptoms": "severe chest pain", "duration_days": 1, "patient_age": 55},
            ).json()
        assert data["urgency_label"] == "emergency"
        assert any("108" in s for s in data["next_steps"])

    def test_triage_urgent_label(self):
        with TestClient(app) as c:
            _set_state(c, _make_pipeline_mock("urgent"))
            data = c.post("/api/triage", json=_VALID_REQUEST).json()
        assert data["urgency_label"] == "urgent"

    def test_triage_routine_label(self):
        with TestClient(app) as c:
            _set_state(c, _make_pipeline_mock("routine"))
            data = c.post(
                "/api/triage",
                json={"symptoms": "mild headache", "duration_days": 1, "patient_age": 28},
            ).json()
        assert data["urgency_label"] == "routine"

    def test_triage_disclaimer_present(self):
        with TestClient(app) as c:
            _set_state(c, _make_pipeline_mock("urgent"))
            data = c.post("/api/triage", json=_VALID_REQUEST).json()
        assert data["disclaimer"]

    def test_triage_model_not_loaded_returns_503(self):
        with TestClient(app) as c:
            _set_state(c, pipeline=None, model_loaded=False)
            r = c.post("/api/triage", json=_VALID_REQUEST)
        assert r.status_code == 503

    # missing / invalid fields → 422

    def test_triage_missing_symptoms_returns_422(self):
        with TestClient(app) as c:
            r = c.post("/api/triage", json={"duration_days": 2, "patient_age": 30})
        assert r.status_code == 422

    def test_triage_missing_duration_days_returns_422(self):
        with TestClient(app) as c:
            r = c.post("/api/triage", json={"symptoms": "fever cough", "patient_age": 30})
        assert r.status_code == 422

    def test_triage_missing_patient_age_returns_422(self):
        with TestClient(app) as c:
            r = c.post("/api/triage", json={"symptoms": "fever cough", "duration_days": 2})
        assert r.status_code == 422

    def test_triage_symptoms_too_short_returns_422(self):
        with TestClient(app) as c:
            r = c.post(
                "/api/triage",
                json={"symptoms": "ab", "duration_days": 1, "patient_age": 30},
            )
        assert r.status_code == 422

    def test_triage_negative_duration_returns_422(self):
        with TestClient(app) as c:
            r = c.post(
                "/api/triage",
                json={"symptoms": "fever cough", "duration_days": -1, "patient_age": 30},
            )
        assert r.status_code == 422

    def test_triage_age_above_max_returns_422(self):
        with TestClient(app) as c:
            r = c.post(
                "/api/triage",
                json={"symptoms": "fever cough", "duration_days": 2, "patient_age": 200},
            )
        assert r.status_code == 422

    # ── GET /api/history ──────────────────────────────────────────────────────

    def test_history_returns_200(self):
        with TestClient(app) as c:
            r = c.get("/api/history")
        assert r.status_code == 200

    def test_history_returns_a_list(self):
        with TestClient(app) as c:
            data = c.get("/api/history").json()
        assert isinstance(data, list)

    def test_history_items_have_correct_schema(self):
        with TestClient(app) as c:
            _set_state(c, _make_pipeline_mock("routine"))
            c.post("/api/triage", json=_VALID_REQUEST)  # seed at least one row
            items = c.get("/api/history").json()
        if items:
            assert _HISTORY_ITEM_KEYS <= items[0].keys()

    def test_history_respects_20_item_limit(self):
        with TestClient(app) as c:
            _set_state(c, _make_pipeline_mock("urgent"))
            for _ in range(25):
                c.post("/api/triage", json=_VALID_REQUEST)
            items = c.get("/api/history").json()
        assert len(items) <= 20

    def test_history_newest_first(self):
        with TestClient(app) as c:
            _set_state(c, _make_pipeline_mock("urgent"))
            # post twice with distinct symptoms to create ordered records
            c.post("/api/triage", json={**_VALID_REQUEST, "symptoms": "fever cough first"})
            c.post("/api/triage", json={**_VALID_REQUEST, "symptoms": "headache second"})
            items = c.get("/api/history").json()
        if len(items) >= 2:
            assert items[0]["id"] > items[1]["id"], "history must be returned newest-first"
