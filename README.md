# Vectron 📈

> Sequential time-series forecasting for hedge fund financial data.

## 🧠 Overview
Vectron is a machine learning pipeline built for the Kaggle Hedge Fund 
Forecasting Competition. It predicts continuous financial values across 
multiple (code, sub-code, sub-category, horizon) combinations using 
strictly sequential, leakage-free predictions.

## 🏆 Competition
- **Platform:** Kaggle
- **Task:** Predict future financial values from time-series data
- **Metric:** Weighted RMSE Score
- **Team Size:** 5

## 🛠️ Tech Stack
- Python
- LightGBM / XGBoost
- Pandas, NumPy
- Scikit-learn
- Matplotlib / Seaborn

## 📁 Project Structure
Vectron/
├── data/               # Raw and processed data
├── notebooks/          # EDA and modeling notebooks
├── src/                # Source code
│   ├── features.py     # Feature engineering
│   ├── model.py        # Model training
│   ├── predict.py      # Sequential prediction pipeline
│   └── evaluate.py     # Metric calculation
├── submissions/        # CSV submission files
├── requirements.txt    # Dependencies
└── README.md

## 📊 Approach
1. Exploratory Data Analysis
2. Lag & rolling feature engineering
3. LightGBM model per group combination
4. Strict sequential prediction (no data leakage)
5. Walk-forward cross-validation

## 👥 Team
- Dev Shah
- Priyanshi Jain
- Lam Vu
- Jyotir Patel
- Hazem Masarani

