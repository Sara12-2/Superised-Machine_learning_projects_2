# ⚡ Household Power Consumption & Telco Churn Prediction Projects

This repository contains two end-to-end Machine Learning applications:

1. **Household Power Consumption Prediction (Streamlit App)**
2. **Telco Customer Churn Prediction (Tkinter GUI App)**

Both projects demonstrate full ML pipelines including preprocessing, training, evaluation, and real-time prediction interfaces.

---

# 📊 1. Household Power Consumption — Streamlit ML App

## 📌 Overview

This Streamlit application trains a **Random Forest Regressor** on the *Household Power Consumption dataset* and provides:

* Model training & evaluation
* Interactive visualizations
* Real-time prediction interface

---

## 🚀 Features

* 📂 Upload dataset (`.txt` or `.csv`)
* 🧹 Data preprocessing:

  * Combines `Date` + `Time`
  * Handles missing values
* 🌲 Model: `RandomForestRegressor`
* 📈 Model evaluation:

  * Mean Squared Error (MSE)
  * Mean Absolute Error (MAE)
  * R² Score
* 📊 Visualizations:

  * Distribution of `Global_active_power`
  * Actual vs Predicted scatter plot
* 🔮 Real-time prediction form via Streamlit

---

## 🛠️ Installation

```bash
pip install streamlit pandas numpy matplotlib scikit-learn
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📂 Dataset Requirements

* File: `household_power_consumption.txt`
* Separator: `;`
* Missing values: `?`

### Required Columns:

* Date
* Time
* Global_active_power (Target)
* Global_reactive_power
* Voltage
* Global_intensity
* Sub_metering_1
* Sub_metering_2
* Sub_metering_3

---

## ⚙️ Model Details

* Train/Test Split: 80/20
* Model: RandomForestRegressor
* n_estimators: 30
* max_depth: 10
* Scaling: StandardScaler

---

## 📌 Suggestions for Improvement

* Use `SimpleImputer` instead of dropping NaNs
* Add feature engineering (hour, day, month)
* Save model using `joblib`
* Add caching with Streamlit
* Hyperparameter tuning (GridSearchCV)

---

# 📞 2. Telco Customer Churn Predictor (Tkinter GUI)

## 📌 Overview

This desktop application predicts whether a telecom customer will churn using a **LightGBM classifier** with a simple Tkinter GUI.

It also handles class imbalance using **SMOTE**.

---

## 🚀 Features

* 📂 Load CSV dataset and train model instantly
* ⚡ LightGBM classifier (high performance)
* ⚖️ SMOTE for imbalance handling
* 📊 Evaluation metrics:

  * Accuracy
  * ROC-AUC Score
  * Precision
  * Recall
  * F1-score
* 🔮 Real-time prediction using GUI inputs

---

## 🛠️ Installation

```bash
pip install pandas scikit-learn imbalanced-learn lightgbm
```

---

## ▶️ Run the App

```bash
python churn_gui.py
```

---

## 📂 Dataset Requirements

### Required Columns:

* customerID
* gender
* SeniorCitizen
* Partner
* Dependents
* tenure
* PhoneService
* MultipleLines
* InternetService
* Contract
* PaymentMethod
* MonthlyCharges
* TotalCharges
* Churn (Target)

---

## 🔮 Prediction Inputs

Only 4 inputs required:

* tenure
* MonthlyCharges
* TotalCharges
* SeniorCitizen (0 = No, 1 = Yes)

---

## 🧪 Example

### Input:

```
tenure = 2
MonthlyCharges = 85
TotalCharges = 170
SeniorCitizen = 1
```

### Output:

```
Prediction: Churn
Probability: 0.67
```

---

## 📌 Notes & Improvements

* Use encoding pipelines for categorical features
* Add model persistence (`joblib`)
* Improve GUI layout (frames, validation)
* Try additional models (XGBoost, CatBoost)
* Add feature importance visualization

---

# ⭐ Author

Built as part of supervised machine learning practice projects.
