from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load(r"C:\Users\Admin\Documents\AIML_Project\debt_risk\debt_risk\models\debt_model.pkl")

labels = {0: "Low", 1: "Medium", 2: "High"}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            gdp = float(request.form["gdp"])
            inflation = float(request.form["inflation"])
            debt = float(request.form["debt"])
            population = float(request.form["population"])

            data = pd.DataFrame([{
                "GDP_growth": gdp,
                "Inflation": inflation,
                "Debt_GDP": debt,
                "Population": population
            }])

            pred = model.predict(data)[0]
            prediction = labels[pred]

        except:
            prediction = "Invalid input"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)