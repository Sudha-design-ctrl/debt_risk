import joblib
import pandas as pd

model = joblib.load("models/debt_model.pkl")

# Example input
sample = pd.DataFrame([{
    "GDP_growth": 5,
    "Inflation": 6,
    "Debt_GDP": 70,
    "Population": 1_400_000_000
}])

prediction = model.predict(sample)

labels = {0: "Low", 1: "Medium", 2: "High"}

print("Prediction:", labels[prediction[0]])