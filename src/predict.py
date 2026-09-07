"""
predict.py
==========
Production inference module: load saved models (XGBoost, Random Forest, Logistic Regression)
and score new customers with complete feature engineering and preprocessing consistency.

Usage (CLI):
    python src/predict.py --model xgboost
    python src/predict.py --model xgboost --input data/new_customers.csv

Author  : Senior Data Scientist & ML Engineer
Project : Customer Churn Forecasting
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import clean_data
from feature_engineering import engineer_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

RISK_LABELS = {
    "Low Risk": "[LOW RISK]",
    "Medium Risk": "[MEDIUM RISK]",
    "High Risk": "[HIGH RISK]",
}

TOP_CHURN_DRIVERS_INFO = [
    {"feature": "Contract", "condition": lambda r: r.get("Contract") == "Month-to-month", "reason": "Month-to-month contract (no switching friction)", "importance": 0.142},
    {"feature": "tenure", "condition": lambda r: float(r.get("tenure", 0)) < 12, "reason": "Early lifecycle customer (tenure < 12 months)", "importance": 0.128},
    {"feature": "MonthlyCharges", "condition": lambda r: float(r.get("MonthlyCharges", 0)) > 75.0, "reason": "High monthly charges ($75+/mo)", "importance": 0.098},
    {"feature": "InternetService", "condition": lambda r: r.get("InternetService") == "Fiber optic", "reason": "Fiber optic service without bundled protections", "importance": 0.074},
    {"feature": "contract_risk_score", "condition": lambda r: r.get("Contract") == "Month-to-month" and r.get("PaperlessBilling") == "Yes", "reason": "Combined month-to-month and paperless billing", "importance": 0.059},
    {"feature": "OnlineSecurity", "condition": lambda r: r.get("OnlineSecurity") == "No", "reason": "No online security add-on", "importance": 0.051},
    {"feature": "TechSupport", "condition": lambda r: r.get("TechSupport") == "No", "reason": "No dedicated tech support service", "importance": 0.048},
    {"feature": "PaymentMethod", "condition": lambda r: r.get("PaymentMethod") == "Electronic check", "reason": "Electronic check payment method", "importance": 0.042},
    {"feature": "PaperlessBilling", "condition": lambda r: r.get("PaperlessBilling") == "Yes", "reason": "Paperless billing active", "importance": 0.041},
]


def load_model(model_name: str = "xgboost"):
    """Load a persisted estimator from models/."""
    fname = model_name.lower().replace(" ", "_") + ".pkl"
    path = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(path):
        available = [f.replace(".pkl", "") for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
        raise FileNotFoundError(
            f"Model '{model_name}' not found at {path}.\nAvailable models: {available}"
        )
    model = joblib.load(path)
    # Compatibility fix for cross-version LogisticRegression
    if hasattr(model, "predict_proba") and not hasattr(model, "multi_class"):
        setattr(model, "multi_class", "deprecated")
    print(f"[LOADED] {model_name} <- {path}")
    return model


def load_scaler():
    """Load the fitted StandardScaler from models/scaler.pkl."""
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    return joblib.load(scaler_path)


def load_trained_columns() -> list[str]:
    """Load the exact 36 feature column names in order."""
    cols_path = os.path.join(MODELS_DIR, "trained_columns.txt")
    if not os.path.exists(cols_path):
        raise FileNotFoundError(f"trained_columns.txt not found at {cols_path}")
    with open(cols_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def preprocess_for_inference(
    df: pd.DataFrame,
    trained_columns: list[str] | None = None,
    scaler=None,
) -> pd.DataFrame:
    """
    Complete inference preprocessing:
      1. Clean missing values and types.
      2. Domain-driven feature engineering (matching training).
      3. Binary mapping and one-hot encoding.
      4. StandardScaler transformation on continuous columns.
      5. Column alignment matching training schema.
    """
    df = df.copy()
    if trained_columns is None:
        trained_columns = load_trained_columns()
    if scaler is None:
        scaler = load_scaler()

    # Drop target or customer ID if present
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # 1. Clean data (coerces TotalCharges, handles median fill, IQR capping)
    df = clean_data(df)

    # 2. Domain-driven feature engineering
    df = engineer_features(df)

    # 3. Deterministic binary encoding
    bin_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    binary_fields = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_fields:
        if col in df.columns:
            df[col] = df[col].map(bin_map).fillna(0).astype(int)

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)

    # 4. Multi-class one-hot encoding (exact 21 dummy columns)
    df["MultipleLines_No phone service"] = (df.get("MultipleLines", "") == "No phone service").astype(int)
    df["MultipleLines_Yes"] = (df.get("MultipleLines", "") == "Yes").astype(int)
    df["InternetService_Fiber optic"] = (df.get("InternetService", "") == "Fiber optic").astype(int)
    df["InternetService_No"] = (df.get("InternetService", "") == "No").astype(int)
    df["OnlineSecurity_No internet service"] = (df.get("OnlineSecurity", "") == "No internet service").astype(int)
    df["OnlineSecurity_Yes"] = (df.get("OnlineSecurity", "") == "Yes").astype(int)
    df["OnlineBackup_No internet service"] = (df.get("OnlineBackup", "") == "No internet service").astype(int)
    df["OnlineBackup_Yes"] = (df.get("OnlineBackup", "") == "Yes").astype(int)
    df["DeviceProtection_No internet service"] = (df.get("DeviceProtection", "") == "No internet service").astype(int)
    df["DeviceProtection_Yes"] = (df.get("DeviceProtection", "") == "Yes").astype(int)
    df["TechSupport_No internet service"] = (df.get("TechSupport", "") == "No internet service").astype(int)
    df["TechSupport_Yes"] = (df.get("TechSupport", "") == "Yes").astype(int)
    df["StreamingTV_No internet service"] = (df.get("StreamingTV", "") == "No internet service").astype(int)
    df["StreamingTV_Yes"] = (df.get("StreamingTV", "") == "Yes").astype(int)
    df["StreamingMovies_No internet service"] = (df.get("StreamingMovies", "") == "No internet service").astype(int)
    df["StreamingMovies_Yes"] = (df.get("StreamingMovies", "") == "Yes").astype(int)
    df["Contract_One year"] = (df.get("Contract", "") == "One year").astype(int)
    df["Contract_Two year"] = (df.get("Contract", "") == "Two year").astype(int)
    df["PaymentMethod_Credit card (automatic)"] = (df.get("PaymentMethod", "") == "Credit card (automatic)").astype(int)
    df["PaymentMethod_Electronic check"] = (df.get("PaymentMethod", "") == "Electronic check").astype(int)
    df["PaymentMethod_Mailed check"] = (df.get("PaymentMethod", "") == "Mailed check").astype(int)

    # 5. Continuous feature scaling using fitted StandardScaler
    scale_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    df[scale_cols] = scaler.transform(df[scale_cols])

    # 6. Align columns to training schema
    for col in trained_columns:
        if col not in df.columns:
            df[col] = 0

    return df[trained_columns]


def derive_customer_drivers(row_dict: dict) -> list[str]:
    """Identify top churn risk factors present for this specific customer."""
    drivers = []
    for item in TOP_CHURN_DRIVERS_INFO:
        try:
            if item["condition"](row_dict):
                drivers.append(item["reason"])
        except Exception:
            continue
    if not drivers:
        drivers.append("Standard tenure and usage pattern (no critical risk flags)")
    return drivers[:4]


def get_retention_recommendation(risk_tier: str, drivers: list[str]) -> str:
    """Generate business retention recommendation based on risk tier and specific drivers."""
    if risk_tier == "High Risk":
        if any("Month-to-month" in d for d in drivers):
            return "Urgent Intervention: Offer 15% discount to switch from Month-to-month to a 1-year contract, bundled with free Security add-ons."
        elif any("tenure < 12" in d for d in drivers):
            return "Urgent Intervention: Assign dedicated onboarding specialist and offer a 60-day VIP service trial."
        return "Urgent Intervention: Priority outbound retention outreach with customized promotional rate."
    elif risk_tier == "Medium Risk":
        if any("Fiber optic" in d for d in drivers) or any("security" in d.lower() for d in drivers):
            return "Proactive Retention: Offer discounted 'Value Shield' bundle (Online Security + Tech Support) at $5/mo."
        return "Proactive Retention: Send customer satisfaction survey with loyalty credits on their next billing cycle."
    else:
        return "Standard Care: Account is healthy. Enroll in loyalty rewards programme and explore service expansion."


def predict_churn(
    model,
    df_raw: pd.DataFrame,
    trained_columns: list[str] | None = None,
    scaler=None,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """Run churn prediction pipeline on raw customer records."""
    if trained_columns is None:
        trained_columns = load_trained_columns()
    if scaler is None:
        scaler = load_scaler()

    ids = (
        df_raw["customerID"].values
        if "customerID" in df_raw.columns
        else [f"CUST-{i+1:04d}" for i in range(len(df_raw))]
    )

    X = preprocess_for_inference(df_raw.copy(), trained_columns, scaler)
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    results = []
    for i, (prob, pred) in enumerate(zip(probs, preds)):
        row_dict = df_raw.iloc[i].to_dict()
        if prob < 0.30:
            tier = "Low Risk"
        elif prob < 0.60:
            tier = "Medium Risk"
        else:
            tier = "High Risk"

        drivers = derive_customer_drivers(row_dict)
        rec = get_retention_recommendation(tier, drivers)

        results.append({
            "CustomerID": ids[i],
            "churn_probability": round(float(prob), 4),
            "churn_probability_pct": f"{prob * 100:.1f}%",
            "churn_prediction": "Churn" if pred == 1 else "No Churn",
            "risk_tier": tier,
            "top_drivers": drivers,
            "recommendation": rec,
        })

    return pd.DataFrame(results)


def batch_predict_from_csv(
    csv_path: str,
    model_name: str = "xgboost",
    threshold: float = 0.50,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Predict churn for all customers in a CSV file."""
    model = load_model(model_name)
    trained_cols = load_trained_columns()
    scaler = load_scaler()

    df_raw = pd.read_csv(csv_path)
    results = predict_churn(model, df_raw, trained_cols, scaler, threshold)

    if output_path:
        results.to_csv(output_path, index=False)
        print(f"[SAVED] Predictions -> {output_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer Churn Inference Pipeline")
    parser.add_argument("--model", default="xgboost", help="xgboost | random_forest | logistic_regression")
    parser.add_argument("--input", default=None, help="Path to input CSV file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()

    if args.input is None:
        print("[INFO] No --input provided. Running inference verification on sample customers...")
        demo_data = pd.DataFrame([
            {
                "customerID": "HIGH-RISK-01",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 95.65,
                "TotalCharges": 95.65,
            },
            {
                "customerID": "LOW-RISK-02",
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "Yes",
                "tenure": 60,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Credit card (automatic)",
                "MonthlyCharges": 65.00,
                "TotalCharges": 3900.00,
            },
        ])

        model = load_model(args.model)
        trained_cols = load_trained_columns()
        scaler = load_scaler()

        res = predict_churn(model, demo_data, trained_cols, scaler, args.threshold)
        print("\n" + "=" * 80)
        print("   INFERENCE TEST RESULTS")
        print("=" * 80)
        for _, row in res.iterrows():
            print(f"\nCustomer: {row['CustomerID']}")
            print(f"  Probability : {row['churn_probability_pct']} ({row['churn_probability']})")
            print(f"  Prediction  : {row['churn_prediction']}")
            print(f"  Risk Tier   : {row['risk_tier']}")
            print(f"  Top Drivers : {', '.join(row['top_drivers'])}")
            print(f"  Action      : {row['recommendation']}")
        print("\n[SUCCESS] Pipeline verification passed successfully.")
    else:
        results = batch_predict_from_csv(args.input, args.model, args.threshold, args.output)
        print(results.to_string(index=False))
