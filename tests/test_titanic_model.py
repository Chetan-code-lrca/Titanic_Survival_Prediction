import pandas as pd
import pytest

from titanic_model import evaluate_csv, load_training_data


def make_passengers() -> pd.DataFrame:
    rows = []
    for index in range(24):
        rows.append(
            {
                "Survived": index % 2,
                "Pclass": 1 + (index % 3),
                "Sex": "female" if index % 2 else "male",
                "Age": None if index % 5 == 0 else 18 + index,
                "SibSp": index % 3,
                "Parch": index % 2,
                "Fare": None if index % 7 == 0 else 7.5 + index,
                "Embarked": None if index % 6 == 0 else ("S" if index % 2 else "C"),
            }
        )
    return pd.DataFrame(rows)


def test_pipeline_evaluates_csv_with_missing_features(tmp_path):
    path = tmp_path / "train.csv"
    make_passengers().to_csv(path, index=False)

    score = evaluate_csv(path)

    assert 0.0 <= score <= 1.0


def test_missing_required_columns_are_reported(tmp_path):
    path = tmp_path / "incomplete.csv"
    pd.DataFrame({"Survived": [0, 1], "Age": [20, 30]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_training_data(path)


def test_invalid_survival_labels_are_rejected(tmp_path):
    frame = make_passengers()
    frame.loc[0, "Survived"] = 2
    path = tmp_path / "invalid-labels.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="only non-missing 0 or 1"):
        load_training_data(path)


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="Training CSV not found"):
        load_training_data(tmp_path / "does-not-exist.csv")
