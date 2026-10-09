"""
Titanic Survival Inference Module

Provides functions to:
- Load the trained model and feature list
- Preprocess a single passenger record
- Predict survival probability
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib
import re


def load_model_and_features(
    model_path: str = "models/titanic_model.pkl",
    features_path: str = "models/feature_cols.pkl",
) -> Tuple[Any, list]:
    """
    Load the trained model and feature column list.

    Returns
    -------
    model : object
        Trained scikit-learn model (e.g., RandomForestClassifier).
    feature_cols : list
        List of feature column names used during training.
    """
    model = joblib.load(model_path)
    feature_cols = joblib.load(features_path)
    return model, feature_cols


def preprocess_one_passenger(
    passenger: Dict[str, Any],
    feature_cols: list,
    age_median: float = 28.0,
    fare_median: float = 14.45,
) -> pd.DataFrame:
    """
    Preprocess a single passenger record into the feature matrix used during training.

    Parameters
    ----------
    passenger : dict
        Dictionary with keys:
        Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Name, Cabin
    feature_cols : list
        List of feature column names used during model training.
    age_median : float
        Median age used for imputation (from training data).
    fare_median : float
        Median fare used for imputation (from training data).

    Returns
    -------
    pd.DataFrame
        Processed feature matrix with one row and columns aligned to training.
    """
    df = pd.DataFrame([passenger])

    # ---- Extract title from name ----
    def extract_title(name: str) -> str:
        title_search = re.search(r" ([A-Za-z]+)\.", name)
        return title_search.group(1) if title_search else ""

    df["Title"] = df["Name"].apply(extract_title)

    rare_titles = [
        "Lady", "Countess", "Capt", "Col", "Don", "Dr",
        "Major", "Rev", "Sir", "Jonkheer", "Dona"
    ]
    df["Title"] = df["Title"].replace(rare_titles, "Rare")
    df["Title"] = df["Title"].replace(["Mlle", "Ms"], "Miss")
    df["Title"] = df["Title"].replace("Mme", "Mrs")

    # ---- Family size and cabin presence ----
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["HasCabin"] = df["Cabin"].notna().astype(int)

    # ---- Impute missing numeric values ----
    df["Age"] = df["Age"].fillna(age_median)
    df["Fare"] = df["Fare"].fillna(fare_median)

    # ---- One-hot encode categorical variables ----
    categorical_cols = ["Sex", "Embarked", "Pclass", "Title"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    # ---- Align columns with training ----
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]

    return df


def predict_survival(
    passenger: Dict[str, Any],
    model=None,
    feature_cols=None,
) -> Dict[str, Any]:
    """
    Predict survival for a single passenger.

    Returns a dictionary with:
      - predicted_class: 0 or 1
      - survival_probability: probability of survival (class 1)
    """
    if model is None or feature_cols is None:
        model, feature_cols = load_model_and_features()

    X = preprocess_one_passenger(passenger, feature_cols)

    pred_class = int(model.predict(X)[0])
    prob_survival = float(model.predict_proba(X)[0, 1])

    return {
        "predicted_class": pred_class,
        "survival_probability": prob_survival,
    }


if __name__ == "__main__":
    # Simple test when running this file directly
    passenger = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S",
        "Name": "Smith, Mr. John",
        "Cabin": np.nan,
    }

    result = predict_survival(passenger)
    print("Predicted class:", result["predicted_class"])
    print("Survival probability:", result["survival_probability"])