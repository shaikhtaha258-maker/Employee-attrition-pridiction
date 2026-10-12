# Employee Attrition Prediction Using Machine Learning

Yeh project predict karta hai ki ek employee company me **Stay** karega ya **Leave** karega.
Model: **Random Forest Classifier** (beginner level).

## Project Structure
```
employee_attrition_project/
├── generate_data.py     # sample dataset banata hai
├── attrition_model.py   # ML code (terminal version)
├── app.py               # Browser interface (Flask) - isko run karna hai
├── model_utils.py       # app ke liye model training code
├── templates/index.html # web page ka design
├── requirements.txt     # libraries
├── data/                # employee_data.csv yahan banta hai
└── outputs/             # graphs yahan save hote hain
```

## VS Code me Run Kaise Kare

1. Python install hona chahiye (3.8+). Check: `python --version`
2. VS Code me **File > Open Folder** se `employee_attrition_project` folder kholo.
3. Terminal kholo (**Terminal > New Terminal**) aur libraries install karo:
   ```
   pip install -r requirements.txt
   ```
4. **Browser interface chalao:**
   ```
   python app.py
   ```
   Browser apne aap `http://127.0.0.1:5000` par khul jayega. Agar na khule to ye link manually kholo.
   Band karne ke liye terminal me `Ctrl + C` dabao.

5. (Optional) Terminal wala version, jisme graphs `outputs/` me save hote hain:
   ```
   python attrition_model.py
   ```
   (Mac/Linux par agar `python` na chale to `python3` use karo.)

Pehli baar run karne par dataset apne aap ban jayega.
Run hone ke baad `outputs/` folder me `confusion_matrix.png` aur `feature_importance.png` milenge.

## Project ke Steps (PPT ke hisab se)
| Step | Kya hota hai |
|------|--------------|
| 1 | Employee dataset load |
| 2 | Data Cleaning (missing values, duplicates) |
| 3 | Feature Selection: Age, Salary, Job Satisfaction, Overtime, Years at Company, Job Role |
| 4 | Preprocessing (text ko numbers me badalna) |
| 5 | Random Forest model train |
| 6 | Accuracy, Confusion Matrix |
| 7 | Feature Importance |
| 8 | Naye employee ka Stay / Leave prediction |

## Real Dataset use karna ho to
Kaggle ka **IBM HR Analytics Employee Attrition** dataset download karo, aur
`data/employee_data.csv` ke naam se save kar do. Column names ye rakho:
`Age, Salary (MonthlyIncome ko rename), JobSatisfaction, OverTime, YearsAtCompany, JobRole, Attrition`.

## Note
Is project ka default data *dummy (synthetic)* hai, sirf learning/demo ke liye.
