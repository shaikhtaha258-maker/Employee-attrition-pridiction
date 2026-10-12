"""
model_utils.py
--------------
Web app (app.py) ke liye model banane ka code.
Steps wahi hain jo attrition_model.py me hain:
Load -> Clean -> Feature Selection -> Preprocess -> Train Random Forest
"""

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "data/employee_data.csv"
FEATURES = ["Age", "Salary", "JobSatisfaction", "OverTime", "YearsAtCompany", "JobRole"]


def train_model():
    # Dataset nahi hai to bana lo
    if not os.path.exists(DATA_PATH):
        import generate_data  # noqa: F401

    # 1. Load
    df = pd.read_csv(DATA_PATH)

    # 2. Cleaning
    df = df.drop_duplicates()
    for col in ["Age", "Salary", "JobSatisfaction"]:
        df[col] = df[col].fillna(df[col].median())

    # 3. Feature selection
    X = df[FEATURES].copy()
    y = df["Attrition"].map({"Yes": 1, "No": 0})

    # 4. Preprocessing
    encoders = {}
    for col in ["OverTime", "JobRole"]:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    # 5. Train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    importance = dict(zip(FEATURES, model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))

    averages = {
        "Age": float(df["Age"].mean()),
        "Salary": float(df["Salary"].mean()),
        "YearsAtCompany": float(df["YearsAtCompany"].mean()),
        "JobSatisfaction": float(df["JobSatisfaction"].mean()),
    }
    return model, encoders, accuracy, importance, len(df), averages


def predict(model, encoders, age, salary, satisfaction, overtime, years, role):
    """Ek employee ka data lo, leave ki probability (0-1) wapas do."""
    row = pd.DataFrame([{
        "Age": age,
        "Salary": salary,
        "JobSatisfaction": satisfaction,
        "OverTime": encoders["OverTime"].transform([overtime])[0],
        "YearsAtCompany": years,
        "JobRole": encoders["JobRole"].transform([role])[0],
    }])
    return float(model.predict_proba(row)[0][1])
