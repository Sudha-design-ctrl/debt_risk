import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train():
    df = pd.read_csv("data/final_dataset.csv")

    X = df[['GDP_growth', 'Inflation', 'Debt_GDP', 'Population']]
    y = df['Debt_Risk'].map({'Low': 0, 'Medium': 1, 'High': 2})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    joblib.dump(model, "models/debt_model.pkl")

    print("✅ Model trained & saved")


if __name__ == "__main__":
    train()