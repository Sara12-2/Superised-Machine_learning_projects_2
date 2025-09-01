import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

# Global model and feature list
model = None
features = []

def train_model(file_path):
    global model, features
    data = pd.read_csv(file_path)

    # Preprocessing
    data.drop("customerID", axis=1, inplace=True)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["TotalCharges"].fillna(data["TotalCharges"].median(), inplace=True)

    # Encode target
    y = data["Churn"].map({"Yes": 1, "No": 0})
    X = data.drop("Churn", axis=1)

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)
    features = X.columns.tolist()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Handle imbalance
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    # LightGBM Model
    lgb = LGBMClassifier(
        random_state=42,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05
    )
    lgb.fit(X_train_sm, y_train_sm)

    model = lgb

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred)

    result_box.delete("1.0", tk.END)
    result_box.insert(
        tk.END,
        f"✅ Accuracy: {acc*100:.2f}%\nROC-AUC: {roc_auc:.3f}\n\n{report}"
    )

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        train_model(file_path)

def predict_churn():
    if model is None:
        messagebox.showerror("Error", "Model not trained yet.")
        return

    try:
        # Minimal inputs
        tenure = float(entry_dict["tenure"].get())
        monthly = float(entry_dict["MonthlyCharges"].get())
        total = float(entry_dict["TotalCharges"].get())
        senior = int(entry_dict["SeniorCitizen"].get())

        # Default values for missing features
        input_data = {
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "SeniorCitizen": senior,
            "Partner_No": 1,
            "Dependents_No": 1,
            "PhoneService_Yes": 1,
            "InternetService_No": 1,
            "Contract_Month-to-month": 1,
            "PaymentMethod_Electronic check": 1
        }

        input_df = pd.DataFrame([input_data])

        # Ensure same features
        for col in features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[features]

        # Predict
        proba = model.predict_proba(input_df)[0][1]
        prediction = 1 if proba >= 0.4 else 0   # threshold lowered for better recall

        result = f"🔮 Prediction: {'Churn' if prediction == 1 else 'No Churn'}\nProbability: {proba:.2f}"
        messagebox.showinfo("Prediction Result", result)

    except Exception as e:
        messagebox.showerror("Error", str(e))

# ------------------ GUI ------------------
root = tk.Tk()
root.title("Telco Churn Predictor")
root.geometry("500x600")

tk.Label(root, text="📊 Telco Churn Prediction", font=("Helvetica", 16, "bold")).pack(pady=10)
tk.Button(root, text="📂 Load CSV & Train Model", command=browse_file).pack(pady=10)

result_box = tk.Text(root, wrap=tk.WORD, font=("Courier", 10), height=8)
result_box.pack(padx=10, pady=10)

tk.Label(root, text="🧾 Enter New Customer Data", font=("Helvetica", 14)).pack(pady=10)

entry_frame = tk.Frame(root)
entry_frame.pack(pady=10)

entry_dict = {}

# Minimal inputs only
features_minimal = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
for i, feature in enumerate(features_minimal):
    tk.Label(entry_frame, text=feature).grid(row=i, column=0, padx=5, pady=5, sticky="w")
    entry = tk.Entry(entry_frame)
    entry.grid(row=i, column=1, padx=5, pady=5)
    entry_dict[feature] = entry

# Predict button
predict_btn = tk.Button(root, text="🔍 Predict Churn", command=predict_churn,
                        font=("Helvetica", 12, "bold"), bg="lightblue")
predict_btn.pack(pady=30, ipadx=20, ipady=5)

root.mainloop()
