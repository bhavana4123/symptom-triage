"""MLflow experiment: train 3 classifiers on processed parquet splits and save the best."""

from __future__ import annotations

import pickle
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── paths & constants ─────────────────────────────────────────────────────────

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).parent / "models"

TEXT_COL = "symptom_text"
NUMERIC_COLS = ["Age", "symptom_count", "Gender"]
TARGET_COL = "urgency_label"

EXPERIMENT_NAME = "symptom-triage"

CLASSIFIERS: dict[str, object] = {
    "RandomForest": RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    ),
    "LogisticRegression": LogisticRegression(
        max_iter=500, C=1.0, class_weight="balanced", random_state=42
    ),
}


@dataclass
class RunResult:
    """Holds evaluation metrics and the fitted pipeline for one MLflow run."""

    name: str
    run_id: str
    val_accuracy: float
    val_f1_macro: float
    pipeline: Pipeline = field(repr=False)


# ── helpers ───────────────────────────────────────────────────────────────────


def load_splits(processed_dir: Path = PROCESSED_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and val parquet files produced by data_pipeline.py.

    Args:
        processed_dir: Directory containing ``train.parquet`` and ``val.parquet``.

    Returns:
        Tuple of ``(train_df, val_df)``.

    Raises:
        FileNotFoundError: If either parquet file is missing.
    """
    for name in ("train", "val"):
        p = processed_dir / f"{name}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"{p} not found — run `make pipeline` first.")
    train = pd.read_parquet(processed_dir / "train.parquet")
    val = pd.read_parquet(processed_dir / "val.parquet")
    return train, val


def build_preprocessor() -> ColumnTransformer:
    """Construct the ColumnTransformer that combines TF-IDF and numeric features.

    Uses a bare string for the text column so that ColumnTransformer selects
    a Series (1-D), which TfidfVectorizer requires.

    Returns:
        Unfitted ``ColumnTransformer`` with a ``tfidf`` and a ``num`` branch.
    """
    return ColumnTransformer(
        transformers=[
            (
                "tfidf",
                TfidfVectorizer(max_features=5_000, ngram_range=(1, 2), sublinear_tf=True),
                TEXT_COL,  # string → Series selection; TfidfVectorizer needs 1-D
            ),
            (
                "num",
                StandardScaler(),
                NUMERIC_COLS,
            ),
        ],
        remainder="drop",
    )


def _log_confusion_matrix(
    y_true: pd.Series, y_pred: np.ndarray, labels: list[str], model_name: str
) -> None:
    """Save a confusion-matrix PNG to a temp file and log it as an MLflow artifact.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Ordered label list for the axes.
        model_name: Used in the figure title and artifact filename.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{model_name} — validation confusion matrix")
    fig.tight_layout()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"confusion_matrix_{model_name}.png"
        fig.savefig(path, dpi=120)
        mlflow.log_artifact(str(path), artifact_path="plots")

    plt.close(fig)


def train_one(
    name: str,
    estimator: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> RunResult:
    """Train a single classifier inside an MLflow run and return its results.

    autolog captures all hyperparameters from ``fit()``. Val metrics and the
    confusion-matrix artifact are logged manually so they appear alongside the
    auto-logged params in the same run.

    Args:
        name: Human-readable model name used as the MLflow run name.
        estimator: Unfitted sklearn estimator.
        X_train: Training feature DataFrame.
        y_train: Training labels.
        X_val: Validation feature DataFrame.
        y_val: Validation labels.

    Returns:
        ``RunResult`` containing the run ID, metrics, and fitted pipeline.
    """
    pipeline = Pipeline(
        [("preprocessor", build_preprocessor()), ("clf", estimator)],
        verbose=False,
    )

    with mlflow.start_run(run_name=name) as run:
        pipeline.fit(X_train, y_train)  # autolog captures params here

        y_pred = pipeline.predict(X_val)
        labels = sorted(y_val.unique().tolist())

        val_acc = accuracy_score(y_val, y_pred)
        val_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)

        mlflow.log_metrics({"val_accuracy": val_acc, "val_f1_macro": val_f1})
        mlflow.log_param("text_col", TEXT_COL)
        mlflow.log_param("numeric_cols", str(NUMERIC_COLS))

        _log_confusion_matrix(y_val, y_pred, labels, name)

    return RunResult(
        name=name,
        run_id=run.info.run_id,
        val_accuracy=val_acc,
        val_f1_macro=val_f1,
        pipeline=pipeline,
    )


def print_summary(results: list[RunResult]) -> None:
    """Print a formatted comparison table of all runs, marking the best by F1.

    Args:
        results: List of ``RunResult`` objects, one per classifier.
    """
    best_id = max(range(len(results)), key=lambda i: results[i].val_f1_macro)

    col_w = 22
    print(f"\n{'─' * (col_w + 32)}")
    print(f"  {'Model':<{col_w}} {'Val Acc':>9}  {'Val F1 Macro':>13}  {'Run ID':>8}")
    print(f"  {'─' * col_w} {'─' * 9}  {'─' * 13}  {'─' * 8}")
    for i, r in enumerate(results):
        marker = "  ← best" if i == best_id else ""
        print(
            f"  {r.name:<{col_w}} {r.val_accuracy:>9.4f}  {r.val_f1_macro:>13.4f}"
            f"  {r.run_id[:8]}{marker}"
        )
    print(f"{'─' * (col_w + 32)}\n")


def save_best(best: RunResult, model_dir: Path = MODEL_DIR) -> None:
    """Persist the best pipeline and its TF-IDF vectorizer as pickle files.

    The full pipeline is saved as ``best_model.pkl`` for end-to-end inference.
    The vectorizer is extracted and saved separately as ``vectorizer.pkl`` for
    use-cases that only need to transform text (e.g. nearest-neighbour search).

    Args:
        best: The ``RunResult`` with the highest validation F1.
        model_dir: Directory to write the pickle files into.
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "best_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best.pipeline, f)

    vectorizer = best.pipeline.named_steps["preprocessor"].named_transformers_["tfidf"]
    vec_path = model_dir / "vectorizer.pkl"
    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)

    # also log the winning artifacts back into the same MLflow run
    with mlflow.start_run(run_id=best.run_id):
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(vec_path), artifact_path="model")
        mlflow.set_tag("best_model", "true")

    print(f"Best model  ({best.name})  → {model_path}")
    print(f"Vectorizer               → {vec_path}")


# ── entry point ───────────────────────────────────────────────────────────────


def run(processed_dir: Path = PROCESSED_DIR) -> RunResult:
    """Execute the full training experiment and return the best RunResult.

    Steps:
    1. Load train / val parquet splits.
    2. Enable autolog for automatic hyperparameter capture.
    3. Train each classifier in its own MLflow run; log val metrics + confusion matrix.
    4. Print a summary table across all runs.
    5. Save the best pipeline and vectorizer to ``ml/models/``.

    Args:
        processed_dir: Directory with processed parquet files.

    Returns:
        The ``RunResult`` for the best-performing model.
    """
    train_df, val_df = load_splits(processed_dir)
    print(f"Train: {len(train_df):,} rows  |  Val: {len(val_df):,} rows")

    X_train, y_train = train_df.drop(columns=[TARGET_COL]), train_df[TARGET_COL]
    X_val, y_val = val_df.drop(columns=[TARGET_COL]), val_df[TARGET_COL]

    mlflow.set_experiment(EXPERIMENT_NAME)
    # autolog captures params/tags from fit(); we skip model artifact logging
    # since we manage pkl files ourselves
    mlflow.sklearn.autolog(log_models=False, log_input_examples=False, silent=True)

    results: list[RunResult] = []
    for name, estimator in CLASSIFIERS.items():
        print(f"Training {name} …")
        result = train_one(name, estimator, X_train, y_train, X_val, y_val)
        print(f"  acc={result.val_accuracy:.4f}  f1={result.val_f1_macro:.4f}")
        results.append(result)

    mlflow.sklearn.autolog(disable=True)

    print_summary(results)

    best = max(results, key=lambda r: r.val_f1_macro)
    save_best(best)

    return best


if __name__ == "__main__":
    run()
