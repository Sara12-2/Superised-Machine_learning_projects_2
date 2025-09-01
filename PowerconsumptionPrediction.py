"""
Household Power Consumption — Streamlit GUI (Final Full Features)

Features:
1) Train RandomForest model on uploaded dataset
2) Show metrics (MSE, R2, MAE)
3) Display visualizations (distribution + actual vs predicted)
4) Predict at runtime by entering ALL feature values via Streamlit form

Usage:
- Run: streamlit run power_app.py
- Upload 'household_power_consumption.txt' in the GUI
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

st.set_page_config(page_title="Power Consumption ML App", layout="wide")
st.title("⚡ Household Power Consumption — ML App")

# --------------------------
# Upload Dataset
# --------------------------
data_file = st.file_uploader("Upload dataset", type=["txt", "csv", "data"])

if data_file is not None:
    # Read dataset
    df = pd.read_csv(
        data_file,
        sep=';',
        na_values=['?'],
        low_memory=False
    )

    # Combine Date + Time into datetime
    df['datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        format="%d/%m/%Y %H:%M:%S",
        errors='coerce'
    )

    df.dropna(inplace=True)
    df.drop(['Date', 'Time'], axis=1, inplace=True)

    # Target and Features
    y = df['Global_active_power']
    feature_cols = [col for col in df.columns if col not in ['Global_active_power','datetime']]
    X = df[feature_cols]

    # Split & Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Model
    model = RandomForestRegressor(n_estimators=30, random_state=42, max_depth=10, n_jobs=-1)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # --------------------------
    # Metrics
    # --------------------------
    st.subheader("📊 Model Performance Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("MSE", f"{mean_squared_error(y_test, y_pred):.4f}")
    col2.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.4f}")
    col3.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")

    # --------------------------
    # Plots
    # --------------------------
    st.subheader("📈 Visualizations")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5,3))
        ax.hist(y, bins=50, color='skyblue', edgecolor='black')
        ax.set_title("Distribution of Global Active Power")
        ax.set_xlabel("Global Active Power")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(5,3))
        ax.scatter(y_test, y_pred, alpha=0.3, color="teal")
        ax.set_title("Actual vs Predicted")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        st.pyplot(fig)

    # --------------------------
    # Runtime Prediction (All Features)
    # --------------------------
    st.subheader("🔮 Runtime Prediction")
    with st.form("prediction_form"):
        input_vals = []
        for feat in feature_cols:
            val = st.number_input(f"{feat}", value=0.0, format="%.4f")
            input_vals.append(val)
        submitted = st.form_submit_button("Predict")

        if submitted:
            try:
                arr = np.array([input_vals])   # shape (1,n_features)
                arr = scaler.transform(arr)
                pred = model.predict(arr)[0]
                st.success(f"Predicted Global Active Power: **{pred:.6f}**")
            except Exception as e:
                st.error(f"Prediction failed ❌: {e}")

else:
    st.info("Please upload the dataset file (household_power_consumption.txt).")
