# Titanic Survival Prediction

A small, reproducible machine-learning baseline for the classic Titanic passenger-survival dataset. The project uses a scikit-learn preprocessing pipeline so missing ages/fares and unseen categories are handled consistently.

## Requirements

- Python 3.10 or newer
- A local Kaggle Titanic training CSV named `train.csv` (not included)

## Run

```bash
python -m pip install -r requirements.txt
python titanic_model.py path/to/train.csv
python -m pytest -q
```

The script prints hold-out accuracy from a stratified train/test split. It does not upload data, require API keys, or save a model automatically.

## Input columns

The CSV must contain `Survived`, `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, and `Embarked`. Missing feature values are imputed; missing or invalid labels are rejected.

## Scope

This is a baseline exercise, not a production predictor. The accuracy reported is measured on a held-out sample from the provided file and should not be presented as a guarantee for unseen data.
