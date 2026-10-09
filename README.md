# Titanic Survival Prediction

An end-to-end machine-learning project that predicts whether a Titanic passenger survived using passenger information such as passenger class, sex, age, fare, family details, cabin availability, and port of embarkation.

The project demonstrates data preprocessing, feature engineering, missing-value handling, model training and evaluation, model serialization, and an interactive Streamlit application. SHAP is used to help explain which features influence a prediction.

## Project Overview

This project turns a machine-learning workflow into an interactive application. Users can enter passenger details in the web interface and receive a predicted survival outcome and estimated survival probability.

The project workflow includes:

- Loading and inspecting the Titanic dataset
- Exploring passenger attributes and missing values
- Extracting passenger titles from names
- Creating family-size and cabin-presence features
- Handling missing values
- Encoding categorical features
- Training and evaluating a Random Forest classifier
- Explaining model predictions with SHAP
- Saving the trained model and feature list
- Building an interactive frontend with Streamlit

## Project Structure

```text
titanic_app/
│
├── app.py
├── titanic_inference.py
├── TITANIC_SURVIVAL_PREDICTION.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── train.csv
│
└── models/
    ├── titanic_model.pkl
    └── feature_cols.pkl
```

## Dataset

The project uses the Titanic survival dataset from Kaggle:

[ Titanic Survival Datasets on Kaggle](https://www.kaggle.com/datasets/bhanupratapbiswas/titanic-survival-datasets)

The target column is `Survived`:

- `0` = did not survive
- `1` = survived

The passenger attributes include class, name, sex, age, family relationships, ticket fare, cabin, and embarkation port. The standard Titanic dataset defines `SibSp` as the number of siblings or spouses aboard and `Parch` as the number of parents or children aboard. [94]

## Features Created

- **Title:** Extracted from the passenger’s name, such as Mr, Mrs, Miss, or Master.
- **FamilySize:** Calculated as `SibSp + Parch + 1`.
- **HasCabin:** Indicates whether cabin information is available.

## Missing Data and Encoding

The preprocessing workflow handles missing values and converts categorical values into features suitable for model training.

- Missing `Age` values are filled with the training-data median.
- Missing `Embarked` values are filled with the most frequent value.
- Missing `Fare` values are filled with the training-data median when required.
- The raw `Cabin` value is represented by the engineered `HasCabin` feature.
- Categorical features such as sex, passenger class, embarkation port, and title are one-hot encoded.

## Model

The application uses a **Random Forest classifier** to predict the passenger’s survival class. The saved model is located at:

```text
models/titanic_model.pkl
```

The ordered feature names used by the model are saved at:

```text
models/feature_cols.pkl
```

## Model Evaluation

The notebook contains the model evaluation results, including the held-out test metrics and classification report.

> Add your actual test accuracy, precision, recall, and F1-score from the notebook here. Do not use example or estimated values.

## Explainability

The project uses **SHAP (SHapley Additive exPlanations)** to help explain model predictions. The Streamlit app displays a feature-importance chart for the passenger entered by the user.

SHAP explanations describe how features influenced the model’s output; they do not establish that a feature caused the historical outcome.

## Streamlit Application

The frontend allows a user to adjust passenger parameters, including:

- Passenger class
- Sex
- Age
- Siblings or spouses aboard
- Parents or children aboard
- Ticket fare
- Port of embarkation
- Passenger name
- Cabin availability

The app returns a predicted outcome, an estimated survival probability, and a SHAP feature explanation.

## Installation

Clone the repository:

```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/titanic-survival-prediction.git](https://github.com/YOUR_GITHUB_USERNAME/titanic-survival-prediction.git)
```

Move into the project directory:

```bash
cd titanic-survival-prediction
```

Create and activate a virtual environment on Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

From the project root directory, run:

```bash
streamlit run app.py
```

Streamlit will print a local address in the terminal, usually:

```text
http://localhost:8501
```

Open that address in a browser to use the application.

## Run the Notebook

Open the notebook:

```text
TITANIC_SURVIVAL_PREDICTION.ipynb
```

You can run it in Jupyter Notebook, JupyterLab, or Google Colab. Make sure the dataset path in the notebook matches your local project structure.

## Example Inference

The Streamlit interface accepts passenger details and displays a prediction. The inference module can also be used from Python:

```python
import numpy as np
from titanic_inference import (
    load_model_and_features,
    preprocess_one_passenger,
)

model, feature_cols = load_model_and_features()

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

features = preprocess_one_passenger(passenger, feature_cols)

prediction = int(model.predict(features))
survival_probability = float(model.predict_proba(features))[0][1]

print("Predicted survival:", prediction)
print("Survival probability:", f"{survival_probability:.2%}")
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- SHAP
- Matplotlib
- Joblib
- Jupyter Notebook

## Limitations

This project is an educational machine-learning demonstration. Its predictions depend on the dataset, selected features, preprocessing, and model. They should not be treated as certainty about an individual passenger or as a causal historical explanation.

## Author

**HarshaVarthini K**

Add your preferred professional or academic details here.
