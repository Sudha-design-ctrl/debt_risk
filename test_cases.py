import joblib
import pandas as pd

model = joblib.load("models/debt_model.pkl")

labels = {0: "Low", 1: "Medium", 2: "High"}

test_cases = [
    {"GDP_growth": 6, "Inflation": 4, "Debt_GDP": 50, "Population": 1_000_000},
    {"GDP_growth": 2, "Inflation": 8, "Debt_GDP": 70, "Population": 5_000_000},
    {"GDP_growth": -1, "Inflation": 10, "Debt_GDP": 120, "Population": 10_000_000},
    {"GDP_growth": 4, "Inflation": 3, "Debt_GDP": 30, "Population": 500_000},
    {
    "GDP_growth": 0, "Inflation": 12, "Debt_GDP": 95, "Population": 2_000_000
    },
    {
    "GDP_growth": 3, "Inflation": 5, "Debt_GDP": 80, "Population": 3_000_000
    }

]

for i, case in enumerate(test_cases):
    df = pd.DataFrame([case])
    pred = model.predict(df)[0]
    print(f"Test Case {i+1}: {labels[pred]}")