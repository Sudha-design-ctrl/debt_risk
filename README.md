# EcoSight — Sovereign Debt Risk Prediction & Analytics Platform 📊

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-black.svg)](https://palletsprojects.com/p/flask/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange.svg)](https://scikit-learn.org/)
[![ReportLab](https://img.shields.io/badge/ReportLab-4.0%2B-red.svg)](https://www.reportlab.com/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/Sudha-design-ctrl/debt_risk)

**EcoSight** is an end-to-end macroeconomic intelligence and sovereign debt risk assessment web application. Combining machine learning with Explainable AI (XAI), interactive dashboards, and automated PDF reporting, EcoSight provides analysts and policymakers with actionable insights into sovereign fiscal health.

---

## 📌 Key Features

* **🤖 Machine Learning Risk Prediction**: Predicts sovereign debt distress categories (`Low`, `Medium`, `High`) based on core economic indicators (GDP growth, inflation, debt-to-GDP ratio, population).
* **🎯 0–100 Risk Score & Gauge**: Converts probabilistic model outputs into an intuitive single risk score (0 = Safest, 100 = Highest Distress).
* **🔍 Explainable AI (XAI)**: Breaks down the positive and negative contributors to the risk score, highlighting whether risk is driven by inflation spikes, fiscal slippage, or contracting GDP.
* **💡 Actionable Policy Recommendations**: Generates tailored policy suggestions (fiscal consolidation, structural reforms, debt restructuring strategies) based on identified risk drivers.
* **📈 Macroeconomic Time-Series Forecasting**: Projects future debt trajectories and macroeconomic trends across countries.
* **⚖️ Cross-Country Comparison**: Interactive benchmarking tool allowing side-by-side fiscal analysis across multiple nations.
* **📂 Batch CSV Assessment**: Upload multi-country datasets to process bulk predictions in a single run.
* **📄 Automated PDF Report Generation**: Dynamically creates professional sovereign risk analysis PDF reports via ReportLab.

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend & Web Framework** | Python 3.13, Flask, Jinja2 Templates |
| **Machine Learning & Data** | Scikit-learn (RandomForestClassifier), Pandas, NumPy, SciPy, Joblib |
| **Document Generation** | ReportLab (pure Python PDF generation) |
| **Frontend UI & Visualization**| HTML5, CSS3, Bootstrap 5, Chart.js |
| **Data Sources** | World Bank API & IMF Macroeconomic Datasets |

---

## 📁 Project Structure

```text
debt_risk/
│
├── app/
│   ├── main.py               # Flask application server, routes & API endpoints
│   ├── templates/            # Jinja2 HTML templates
│   │   ├── base.html         # Base layout & navigation
│   │   ├── index.html        # Prediction form & XAI risk gauge
│   │   ├── dashboard.html    # Macroeconomic analytics & charts
│   │   ├── compare.html      # Cross-country comparison
│   │   ├── forecast.html     # Time-series forecasting
│   │   └── upload.html       # Batch CSV upload page
│   └── static/
│       ├── css/style.css     # Custom modern stylesheet
│       └── uploads/          # Temporary storage for batch files
│
├── data/
│   ├── raw_data.csv          # Macro indicators collected via World Bank API
│   └── final_dataset.csv     # Cleaned, labeled training dataset
│
├── models/
│   └── debt_model.pkl        # Serialized pre-trained Random Forest model
│
├── src/
│   ├── data_collection.py    # Automated World Bank API collector
│   ├── preprocess.py         # Data cleaning & labeling pipeline
│   ├── train.py              # Model training script
│   ├── risk_scorer.py        # 0–100 scoring & XAI explanation engine
│   ├── recommender.py        # Policy recommendation engine
│   ├── forecaster.py         # Time-series projection models
│   └── pdf_generator.py      # Automated PDF report generator
│
├── .gitignore                # Excludes cache, large datasets & temporary files
├── requirements.txt          # Python dependencies
├── test_cases.py             # Inference validation tests
└── README.md                 # Project documentation
```

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Sudha-design-ctrl/debt_risk.git
cd debt_risk
```

### 2. Create and Activate a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app/main.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🧪 Testing the Model

You can verify model inference directly from the command line:
```bash
python test_cases.py
```

Expected output:
```text
Test Case 1: Low
Test Case 2: Medium
Test Case 3: High
Test Case 4: Low
Test Case 5: High
Test Case 6: Medium
```

---

## 🌐 API Overview

EcoSight exposes RESTful JSON endpoints for integration:

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/stats` | GET | Global dataset summary metrics |
| `/api/risk-distribution` | GET | Distribution of risk categories across countries |
| `/api/debt-trend` | GET | Historical debt-to-GDP trends over time |
| `/api/countries` | GET | List of all available countries |
| `/api/country/<name>` | GET | Indicator history for a specific country |
| `/api/compare` | GET | Side-by-side country metrics |
| `/api/forecast/<country>` | GET | Time-series projection for a specific nation |
| `/download-report` | POST | Generates and streams a downloadable PDF report |

---

## 🔄 Methodology & Pipeline

```text
World Bank API / IMF Data
          ↓
Data Preprocessing & Cleaning (preprocess.py)
          ↓
Model Training with Random Forest (train.py)
          ↓
Probabilistic Inference & 0–100 Risk Scoring (risk_scorer.py)
          ↓
Explainable AI (XAI) & Dynamic Recommendations (recommender.py)
          ↓
Interactive Flask Dashboard & PDF Export (app/main.py, pdf_generator.py)
```

---

## 👩‍💻 Author

**Sudha Gowda**  
MCA Student at **PES University**, Bangalore  
Interested in Software Engineering, Data Analysis, Machine Learning, and Web Technologies.  

🔗 **GitHub**: [@Sudha-design-ctrl](https://github.com/Sudha-design-ctrl)  
📫 **Repository**: [https://github.com/Sudha-design-ctrl/debt_risk](https://github.com/Sudha-design-ctrl/debt_risk)
