"""
app.py  -  Browser Interface

"""

import threading
import webbrowser
from flask import Flask, render_template, request

from model_utils import train_model, predict

app = Flask(__name__)

# Server start hote hi model train ho jata hai (1-2 second)
print("Training model...")
model, encoders, accuracy, importance, total_rows, averages = train_model()
print(f"Model ready. Accuracy: {accuracy * 100:.1f}%")

JOB_ROLES = list(encoders["JobRole"].classes_)


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    form = {"age": 30, "salary": 40000, "satisfaction": 3,
            "overtime": "No", "years": 3, "role": JOB_ROLES[0]}
    error = None
    compare = None

    if request.method == "POST":
        try:
            form = {
                "age": int(request.form["age"]),
                "salary": int(request.form["salary"]),
                "satisfaction": int(request.form["satisfaction"]),
                "overtime": request.form["overtime"],
                "years": int(request.form["years"]),
                "role": request.form["role"],
            }
            prob = predict(model, encoders, form["age"], form["salary"],
                           form["satisfaction"], form["overtime"],
                           form["years"], form["role"])
            result = {
                "leave": prob >= 0.5,
                "percent": round(prob * 100, 1),
            }
            compare = [
                {"label": "Age", "yours": form["age"], "avg": round(averages["Age"], 1)},
                {"label": "Monthly Salary", "yours": form["salary"], "avg": round(averages["Salary"])},
                {"label": "Years at Company", "yours": form["years"], "avg": round(averages["YearsAtCompany"], 1)},
                {"label": "Job Satisfaction", "yours": form["satisfaction"], "avg": round(averages["JobSatisfaction"], 1)},
            ]
        except Exception as e:
            error = f"Invalid input: {e}"

    return render_template(
        "index.html",
        form=form, result=result, error=error, roles=JOB_ROLES, compare=compare,
        accuracy=round(accuracy * 100, 1), total_rows=total_rows,
        importance={k: round(v * 100, 1) for k, v in importance.items()},
    )


def open_browser():
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    threading.Timer(1.5, open_browser).start()   # browser apne aap khulega
    app.run(debug=False, port=5000)
