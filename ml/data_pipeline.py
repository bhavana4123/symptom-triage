"""Data pipeline: raw CSV → cleaned features → stratified parquet splits."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ── paths ─────────────────────────────────────────────────────────────────────

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

CSV_PATH = RAW_DIR / "symptom_disease.csv"

SYMPTOM_COLS = ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]

REQUIRED_COLS = {
    "Disease", "Fever", "Cough", "Fatigue", "Difficulty Breathing",
    "Age", "Gender", "Blood Pressure", "Cholesterol Level", "Outcome Variable",
}

# ── pipeline steps ────────────────────────────────────────────────────────────


def load_raw(path: Path = CSV_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found: {path}")
    df = pd.read_csv(path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    print(f"Loaded {len(df):,} rows from {path}")
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows and report how many were removed.

    Args:
        df: Raw DataFrame, typically the output of ``load_raw``.

    Returns:
        Copy of *df* with duplicate rows removed.
    """
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        print(f"Removed {removed:,} duplicate row{'s' if removed != 1 else ''} ({before:,} -> {len(df):,})")
    else:
        print("No duplicate rows found.")
    return df


def encode_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Map Yes/No symptom columns to 1/0."""
    df = df.copy()
    for col in SYMPTOM_COLS:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    return df


def _symptom_text(row: pd.Series) -> str:
    return " ".join(col.lower() for col in SYMPTOM_COLS if row[col] == 1)


def _urgency_label(row: pd.Series) -> str:
    diff_breathing = row["Difficulty Breathing"] == 1
    high_bp = row["Blood Pressure"] == "High"
    if diff_breathing and high_bp:
        return "emergency"
    if diff_breathing or high_bp:
        return "urgent"
    return "routine"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["symptom_text"] = df.apply(_symptom_text, axis=1)
    df["urgency_label"] = df.apply(_urgency_label, axis=1)
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    df["symptom_count"] = df[SYMPTOM_COLS].sum(axis=1)
    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, temp = train_test_split(
        df,
        test_size=0.30,
        stratify=df["urgency_label"],
        random_state=42,
    )
    val, test = train_test_split(
        temp,
        test_size=0.50,
        stratify=temp["urgency_label"],
        random_state=42,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def print_distribution(split_name: str, df: pd.DataFrame) -> None:
    counts = df["urgency_label"].value_counts()
    pct = df["urgency_label"].value_counts(normalize=True) * 100
    print(f"\n{split_name}  ({len(df):,} rows)")
    print(f"  {'label':<14} {'count':>6}  {'%':>6}")
    print(f"  {'-'*14} {'-'*6}  {'-'*6}")
    for label in counts.index:
        print(f"  {label:<14} {counts[label]:>6}  {pct[label]:>5.1f}%")


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: Path = PROCESSED_DIR,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        dest = out_dir / f"{name}.parquet"
        split.to_parquet(dest, index=False)
        print(f"  saved {name}.parquet  ({len(split):,} rows)  ->  {dest}")


# ── entry point ───────────────────────────────────────────────────────────────


def run(data_path: Path = CSV_PATH) -> None:
    df = load_raw(data_path)
    df = deduplicate(df)
    df = encode_binary(df)
    df = engineer_features(df)

    train, val, test = split_data(df)

    print("\n-- class distribution -------------------------------------------")
    print_distribution("train", train)
    print_distribution("val  ", val)
    print_distribution("test ", test)

    print("\n-- saving splits ------------------------------------------------")
    save_splits(train, val, test)
    print("\nPipeline complete.")


if __name__ == "__main__":
    run()
