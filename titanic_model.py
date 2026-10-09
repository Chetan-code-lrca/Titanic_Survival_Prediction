"""Train and evaluate a compact Titanic survival baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Survived"
NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Sex", "Embarked"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = [TARGET, *FEATURES]


def load_training_data(csv_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load a Titanic training CSV and validate its required columns and labels."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {path}")

    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Training CSV contains no rows")

    target = frame[TARGET]
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError("Survived must contain only non-missing 0 or 1 labels")
    if target.nunique() != 2:
        raise ValueError("Survived must contain both classes (0 and 1)")
    if len(frame) < 8 or target.value_counts().min() < 2:
        raise ValueError("Training data needs at least 8 rows and two examples of each class")

    return frame[FEATURES].copy(), target.astype(int)


def build_pipeline() -> Pipeline:
    """Build a preprocessing and logistic-regression pipeline."""
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def evaluate_csv(csv_path: str | Path, random_state: int = 42) -> float:
    """Return held-out accuracy without training on the test partition."""
    features, target = load_training_data(csv_path)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=random_state,
        stratify=target,
    )
    model = build_pipeline()
    model.fit(x_train, y_train)
    return float(accuracy_score(y_test, model.predict(x_test)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Path to a local Titanic train.csv")
    args = parser.parse_args()
    try:
        score = evaluate_csv(args.csv_path)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        parser.error(str(exc))
    print(f"Held-out accuracy: {score:.3f}")


if __name__ == "__main__":
    main()
