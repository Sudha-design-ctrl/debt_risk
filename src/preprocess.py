import pandas as pd

def preprocess():
    df = pd.read_csv("data/raw_data.csv")

    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

    for col in ['GDP_growth', 'Inflation', 'Population', 'Debt_GDP']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna()

    def classify_risk(debt):
        if debt > 90:
            return "High"
        elif debt > 60:
            return "Medium"
        else:
            return "Low"

    df['Debt_Risk'] = df['Debt_GDP'].apply(classify_risk)

    df.to_csv("data/final_dataset.csv", index=False)
    print("✅ Processed dataset saved")


if __name__ == "__main__":
    preprocess()