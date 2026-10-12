"""
Employee Attrition Prediction Using Machine Learning
----------------------------------------------------
Model  : Random Forest Classifier
Goal   : Predict whether an employee will STAY or LEAVE

Steps (PPT ki Methodology ke hisab se):
    1. Dataset load karna
    2. Data Cleaning
    3. Feature Selection
    4. Preprocessing (text -> numbers)
    5. Train Model (Random Forest)
    6. Evaluation (accuracy, confusion matrix)
    7. Feature Importance (kaun sa factor sabse important)
    8. Naye employee ke liye Stay / Leave prediction
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # plot ko file me save karne ke liye
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "data/employee_data.csv"
os.makedirs("outputs", exist_ok=True)

# Agar dataset nahi hai to pehle bana lo
if not os.path.exists(DATA_PATH):
    print("Dataset nahi mila, bana raha hoon...")
    import generate_data  # noqa: F401  (import karte hi data ban jata hai)

# =====================================================
# STEP 1: Dataset Load
# =====================================================
print("\n" + "=" * 55)
print("STEP 1: Dataset Load")
print("=" * 55)
df = pd.read_csv(DATA_PATH)
print("Shape (rows, columns):", df.shape)
print(df.head())

# =====================================================
# STEP 2: Data Cleaning
# =====================================================
print("\n" + "=" * 55)
print("STEP 2: Data Cleaning")
print("=" * 55)
print("Missing values (cleaning se pehle):")
print(df.isnull().sum())

df = df.drop_duplicates()                                  # duplicate rows hatao
df["Age"] = df["Age"].fillna(df["Age"].median())            # missing numbers -> median
df["Salary"] = df["Salary"].fillna(df["Salary"].median())
df["JobSatisfaction"] = df["JobSatisfaction"].fillna(df["JobSatisfaction"].median())

print("\nCleaning ke baad rows:", len(df))
print("Total missing values:", df.isnull().sum().sum())

# =====================================================
# STEP 3: Feature Selection
# =====================================================
print("\n" + "=" * 55)
print("STEP 3: Feature Selection")
print("=" * 55)
features = ["Age", "Salary", "JobSatisfaction", "OverTime", "YearsAtCompany", "JobRole"]
target = "Attrition"
print("Selected features:", features)

X = df[features].copy()
y = df[target].map({"Yes": 1, "No": 0})    # Yes = Leave (1), No = Stay (0)

# =====================================================
# STEP 4: Preprocessing (text columns ko numbers me badalna)
# =====================================================
print("\n" + "=" * 55)
print("STEP 4: Preprocessing")
print("=" * 55)
encoders = {}
for col in ["OverTime", "JobRole"]:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le
    print(f"{col} encoding:", dict(zip(le.classes_, le.transform(le.classes_))))

# =====================================================
# STEP 5: Train Model (Random Forest)
# =====================================================
print("\n" + "=" * 55)
print("STEP 5: Train Model - Random Forest")
print("=" * 55)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training data:", X_train.shape, "| Testing data:", X_test.shape)

model = RandomForestClassifier(
    n_estimators=100,          # 100 decision trees
    random_state=42,
    class_weight="balanced",   # Leave wale employees kam hote hain, isliye balance
)
model.fit(X_train, y_train)
print("Model train ho gaya!")

# =====================================================
# STEP 6: Evaluation
# =====================================================
print("\n" + "=" * 55)
print("STEP 6: Model Evaluation")
print("=" * 55)
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Stay", "Leave"]))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xticks([0, 1], ["Stay", "Leave"])
plt.yticks([0, 1], ["Stay", "Leave"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
plt.colorbar()
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.close()
print("Saved: outputs/confusion_matrix.png")

# =====================================================
# STEP 7: Feature Importance
# =====================================================
print("\n" + "=" * 55)
print("STEP 7: Feature Importance")
print("=" * 55)
importance = pd.Series(model.feature_importances_, index=features).sort_values()
print(importance.sort_values(ascending=False).round(3))

plt.figure(figsize=(7, 4))
importance.plot(kind="barh", color="#2a7ab0")
plt.title("Feature Importance - Kaun sa factor Attrition ko zyada affect karta hai?")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.close()
print("Saved: outputs/feature_importance.png")


# =====================================================
# STEP 8: Naye employee ka prediction
# =====================================================
def predict_employee(age, salary, satisfaction, overtime, years, role):
    """Ek employee ka data lo aur Stay / Leave batao."""
    row = pd.DataFrame([{
        "Age": age,
        "Salary": salary,
        "JobSatisfaction": satisfaction,
        "OverTime": encoders["OverTime"].transform([overtime])[0],
        "YearsAtCompany": years,
        "JobRole": encoders["JobRole"].transform([role])[0],
    }])
    prob_leave = model.predict_proba(row)[0][1]
    result = "LEAVE (Chhod sakta hai)" if prob_leave >= 0.5 else "STAY (Rukega)"
    return result, prob_leave


print("\n" + "=" * 55)
print("STEP 8: Sample Predictions")
print("=" * 55)

# Example 1: Risky employee (overtime, low satisfaction, kam salary)
res, p = predict_employee(24, 28000, 1, "Yes", 1, "Sales Executive")
print(f"Employee 1 -> {res} | Leave probability: {p * 100:.1f}%")

# Example 2: Safe employee (high satisfaction, no overtime, achi salary)
res, p = predict_employee(45, 80000, 4, "No", 15, "Manager")
print(f"Employee 2 -> {res} | Leave probability: {p * 100:.1f}%")

# ---------- Apna khud ka employee daal ke check karo ----------
print("\n--- Apna employee check karo ---")
while True:
    choice = input("Naya employee predict karna hai? (y/n): ").strip().lower()
    if choice != "y":
        break
    try:
        age = int(input("Age: "))
        salary = int(input("Monthly Salary: "))
        sat = int(input("Job Satisfaction (1-4): "))
        ot = input("OverTime (Yes/No): ").strip().capitalize()
        yrs = int(input("Years at Company: "))
        print("Job Roles:", list(encoders["JobRole"].classes_))
        role = input("Job Role (upar me se ek likho): ").strip()
        res, p = predict_employee(age, salary, sat, ot, yrs, role)
        print(f"\n>>> Prediction: {res} | Leave probability: {p * 100:.1f}%\n")
    except Exception as e:
        print("Galat input, dobara try karo. Error:", e)

print("\nProject complete! 'outputs' folder me graphs dekho.")
