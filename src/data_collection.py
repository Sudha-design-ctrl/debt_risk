import pandas as pd
import requests

def fetch_world_bank_data():
    indicators = {
        "NY.GDP.MKTP.KD.ZG": "GDP_growth",
        "FP.CPI.TOTL.ZG": "Inflation",
        "SP.POP.TOTL": "Population",
        "GC.DOD.TOTL.GD.ZS": "Debt_GDP"
    }

    all_data = []

    for code, name in indicators.items():
        print(f"Fetching {name}...")

        url = f"https://api.worldbank.org/v2/country/all/indicator/{code}?format=json&per_page=20000"
        response = requests.get(url)
        data = response.json()

        if len(data) < 2:
            continue

        df = pd.json_normalize(data[1])

        df = df[['country.value', 'country.id', 'date', 'value']]
        df.columns = ['Country', 'ISO', 'Year', name]

        all_data.append(df)

    # Merge all indicators
    df_final = all_data[0]
    for df in all_data[1:]:
        df_final = pd.merge(df_final, df, on=['Country', 'ISO', 'Year'], how='outer')

    return df_final


if __name__ == "__main__":
    df = fetch_world_bank_data()
    df.to_csv("data/raw_data.csv", index=False)
    print("✅ Raw data saved successfully")