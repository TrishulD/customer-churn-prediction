"""
predict.py
==========
Production inference module: load a saved model and score new customers.
Can be used as a library or run from the command line.

Usage (CLI):
    python src/predict.py --model xgboost --input data/new_customers.csv

Author  : Senior Data Scientist
Project : Customer Churn Forecasting
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing       import clean_data, encode_features

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")

# Colour-coded risk labels
RISK_LABELS = {
    "Low Risk"   : "🟢",
    "Medium Risk": "🟡",
    "High Risk"  : "🔴",
}


# ──────────────────────────────────────────────────────────────────────────────
# PREDICTION PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def load_model(model_name: str):
    """Load a persisted model from the models/ directory."""
    fname = model_name.lower().replace(" ", "_") + ".pkl"
    path  = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(path):
        available = [f.replace(".pkl", "") for f in os.listdir(MODELS_DIR)
                     if f.endswith(".pkl")]
        raise FileNotFoundError(
            f"Model '{model_name}' not found.\n"
            f"Available models: {available}\n"
            f"Looked in: {MODELS_DIR}"
        )
    model = joblib.load(path)
    print(f"[LOADED] {model_name} ← {path}")
    return model


def preprocess_for_inference(df: pd.DataFrame, trained_columns: list) -> pd.DataFrame:
    """
    Apply the same cleaning + encoding as training, then align columns.

    Parameters
    ----------
    df              : Raw new-customer DataFrame.
    trained_columns : Feature columns the model was trained on (in order).

    Returns
    -------
    pd.DataFrame aligned to training schema (missing cols filled with 0).
    """
    # Target may be absent in inference data
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    df = clean_data(df)   # also drops customerID

    # Feature engineering (light version for inference)
    if "tenure" in df.columns and "MonthlyCharges" in df.columns:
        df["avg_monthly_spend"] = (
            df.get("TotalCharges", df["MonthlyCharges"]) / (df["tenure"] + 1)
        ).round(2)

    df = encode_features(df)

    # Align columns to training schema
    missing_cols = set(trained_columns) - set(df.columns)
    extra_cols   = set(df.columns) - set(trained_columns)
    for col in missing_cols:
        df[col] = 0         # fill unseen dummy columns with 0
    df = df.drop(columns=list(extra_cols), errors="ignore")
    df = df[trained_columns]  # enforce column order

    return df


def predict_churn(
    model,
    df_raw       : pd.DataFrame,
    trained_cols : list,
    threshold    : float = 0.5,
) -> pd.DataFrame:
    """
    Generate churn predictions and risk scores for new customers.

    Parameters
    ----------
    model        : Fitted sklearn-compatible model.
    df_raw       : Raw customer data (may include customerID and Churn).
    trained_cols : Feature columns the model expects.
    threshold    : Decision boundary for binary classification.

    Returns
    -------
    pd.DataFrame with columns: customerID (if present), churn_probability,
                                churn_prediction, risk_tier, recommendation.
    """
    # Preserve IDs for output
    ids = df_raw["customerID"].values if "customerID" in df_raw.columns else (
        np.arange(len(df_raw))
    )

    X = preprocess_for_inference(df_raw.copy(), trained_cols)

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    # Risk tier mapping
    tiers = pd.cut(
        probs,
        bins   = [0, 0.30, 0.60, 1.0],
        labels = ["Low Risk", "Medium Risk", "High Risk"],
    )

    # Recommendation logic
    def get_recommendation(tier):
        recs = {
            "Low Risk"   : "No immediate action. Enrol in loyalty programme.",
            "Medium Risk": "Proactive outreach. Offer a service bundle discount.",
            "High Risk"  : "Urgent intervention. Assign retention specialist.",
        }
        return recs.get(tier, "Review manually.")

    results = pd.DataFrame({
        "CustomerID"        : ids,
        "churn_probability" : np.round(probs, 4),
        "churn_prediction"  : ["Churn" if p else "No Churn" for p in preds],
        "risk_tier"         : tiers,
        "recommendation"    : [get_recommendation(t) for t in tiers],
    })

    print(f"\n[PREDICT] Scored {len(results)} customers.")
    print(results["risk_tier"].value_counts().to_string())
    return results


def batch_predict_from_csv(
    csv_path   : str,
    model_name : str = "xgboost",
    threshold  : float = 0.50,
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    End-to-end batch prediction from a CSV file.

    Parameters
    ----------
    csv_path    : Path to raw new-customer CSV.
    model_name  : Name of the saved model to use.
    threshold   : Decision threshold.
    output_path : If provided, save scored results to this CSV path.

    Returns
    -------
    Scored DataFrame.
    """
    model = load_model(model_name)

    # Infer trained columns from a reference dataset or fallback list
    trained_cols_path = os.path.join(MODELS_DIR, "trained_columns.txt")
    if os.path.exists(trained_cols_path):
        with open(trained_cols_path) as f:
            trained_cols = [l.strip() for l in f.readlines()]
    else:
        # Fallback: re-derive columns from the raw data
        from preprocessing import load_data, clean_data, encode_features, split_data
        from feature_engineering import engineer_features
        raw = load_data()
        enc = encode_features(engineer_features(clean_data(raw)))
        trained_cols = [c for c in enc.columns if c != "Churn"]

    df_raw  = pd.read_csv(csv_path)
    results = predict_churn(model, df_raw, trained_cols, threshold)

    if output_path:
        results.to_csv(output_path, index=False)
        print(f"[SAVED] Predictions → {output_path}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer Churn Predictor")
    parser.add_argument("--model", default="xgboost",
                        help="Model name: logistic_regression | random_forest | xgboost")
    parser.add_argument("--input", default=None,
                        help="Path to input CSV file with new customers")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold (default: 0.5)")
    parser.add_argument("--output", default=None,
                        help="Path to save predictions CSV")
    args = parser.parse_args()

    if args.input is None:
        # Demo: score a mini synthetic sample
        print("[INFO] No --input provided. Running demo on 5 synthetic customers.")
        demo = pd.DataFrame({
            "customerID"     : ["DEMO-001", "DEMO-002", "DEMO-003", "DEMO-004", "DEMO-005"],
            "gender"         : ["Male", "Female", "Male", "Female", "Male"],
            "SeniorCitizen"  : [0, 1, 0, 0, 1],
            "Partner"        : ["Yes", "No", "No", "Yes", "No"],
            "Dependents"     : ["No", "No", "Yes", "No", "No"],
            "tenure"         : [1, 60, 24, 12, 3],
            "PhoneService"   : ["Yes", "Yes", "Yes", "No", "Yes"],
            "MultipleLines"  : ["No", "Yes", "No", "No phone service", "No"],
            "InternetService": ["Fiber optic", "DSL", "No", "DSL", "Fiber optic"],
            "OnlineSecurity" : ["No", "Yes", "No internet service", "No", "No"],
            "OnlineBackup"   : ["No", "Yes", "No internet service", "Yes", "No"],
            "DeviceProtection": ["No", "Yes", "No internet service", "No", "No"],
            "TechSupport"    : ["No", "Yes", "No internet service", "No", "No"],
            "StreamingTV"    : ["No", "Yes", "No internet service", "No", "Yes"],
            "StreamingMovies": ["No", "Yes", "No internet service", "No", "Yes"],
            "Contract"       : ["Month-to-month", "Two year", "One year",
                                "Month-to-month", "Month-to-month"],
            "PaperlessBilling": ["Yes", "No", "No", "Yes", "Yes"],
            "PaymentMethod"  : ["Electronic check", "Bank transfer (automatic)",
                                "Mailed check", "Electronic check",
                                "Electronic check"],
            "MonthlyCharges" : [95.65, 25.10, 0.00, 45.20, 89.30],
            "TotalCharges"   : [95.65, 1505.0, 0.00, 542.40, 267.90],
        })
        results = batch_predict_from_csv.__wrapped__(demo, args.model, args.threshold) \
            if hasattr(batch_predict_from_csv, "__wrapped__") else None

        # Inline demo path
        model = load_model(args.model)
        tp = os.path.join(MODELS_DIR, "trained_columns.txt")
        if os.path.exists(tp):
            with open(tp) as f:
                trained_cols = [l.strip() for l in f]
        else:
            from preprocessing import load_data, clean_data, encode_features
            from feature_engineering import engineer_features
            raw  = load_data()
            enc  = encode_features(engineer_features(clean_data(raw)))
            trained_cols = [c for c in enc.columns if c != "Churn"]

        results = predict_churn(model, demo, trained_cols, args.threshold)
        print("\n── Prediction Results ─────────────────────────────────")
        print(results[["CustomerID", "churn_probability",
                        "churn_prediction", "risk_tier", "recommendation"]]
              .to_string(index=False))
    else:
        results = batch_predict_from_csv(
            csv_path   = args.input,
            model_name = args.model,
            threshold  = args.threshold,
            output_path= args.output,
        )
        print(results.to_string(index=False))
