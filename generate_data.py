"""
generate_data.py
----------------
Yeh script ek sample Employee dataset banata hai (data/employee_data.csv).

PPT ke Methodology slide ke hisab se columns:
    Age, Salary, JobSatisfaction, OverTime, YearsAtCompany, JobRole
Target column:
    Attrition  ->  "Yes" (Leave)  /  "No" (Stay)

Note: Yeh dummy (synthetic) data hai, sirf learning/demo ke liye.
Real project ke liye aap Kaggle ka "IBM HR Analytics Employee Attrition"
dataset use kar sakte ho (README me bataya hai).
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)          # taaki har baar same data bane
N = 1500                    # kitne employees

job_roles = ["Sales Executive", "Research Scientist", "Laboratory Technician",
             "Manager", "Human Resources", "Software Engineer"]

# ---------- Features banate hain ----------
age = np.random.randint(20, 60, N)
job_role = np.random.choice(job_roles, N)
years_at_company = np.clip(age - 20 - np.random.randint(0, 15, N), 0, 35)

# Salary (monthly, rupees) - role aur experience pe depend karti hai
base_salary = {"Sales Executive": 30000, "Research Scientist": 35000,
               "Laboratory Technician": 28000, "Manager": 70000,
               "Human Resources": 40000, "Software Engineer": 50000}
salary = np.array([base_salary[r] for r in job_role]) \
         + years_at_company * 1200 + np.random.randint(-5000, 5000, N)

job_satisfaction = np.random.randint(1, 5, N)           # 1 (low) se 4 (high)
overtime = np.random.choice(["Yes", "No"], N, p=[0.3, 0.7])

# ---------- Attrition ka rule (probability) ----------
# Low satisfaction, overtime, kam salary, kam experience, kam age => leave ka chance zyada
score = (
    -1.2
    + 2.0 * (overtime == "Yes")
    + 1.4 * (job_satisfaction == 1)
    + 0.7 * (job_satisfaction == 2)
    - 0.7 * (job_satisfaction == 4)
    - 0.00005 * (salary - 40000)
    - 0.10 * years_at_company
    - 0.03 * (age - 35)
)
prob_leave = 1 / (1 + np.exp(-score))
attrition = np.where(np.random.rand(N) < prob_leave, "Yes", "No")

df = pd.DataFrame({
    "Age": age,
    "Salary": salary,
    "JobSatisfaction": job_satisfaction,
    "OverTime": overtime,
    "YearsAtCompany": years_at_company,
    "JobRole": job_role,
    "Attrition": attrition,
})

# ---------- Thodi "ganda" data bhi daalte hain taaki Data Cleaning dikhe ----------
for col in ["Age", "Salary", "JobSatisfaction"]:
    idx = np.random.choice(df.index, 25, replace=False)
    df.loc[idx, col] = np.nan                       # missing values

df = pd.concat([df, df.sample(15, random_state=1)])  # duplicate rows

os.makedirs("data", exist_ok=True)
df.to_csv("data/employee_data.csv", index=False)
print("Dataset ban gaya: data/employee_data.csv")
print("Total rows:", len(df))
