"""
api/py/index.py
===============
Vercel Python Serverless Function — Self-contained FastAPI ML Inference Service.

Served at /api/py (and /api/py/* via vercel.json rewrites).

This file deliberately inlines all inference logic from src/predict.py,
src/preprocessing.py, and the inference-relevant parts of src/feature_engineering.py.

Reason: src/feature_engineering.py imports `imbalanced-learn` (SMOTE) at module
level, which is a training-only dependency (>80 MB). Including it in the Vercel
function would bloat the bundle unnecessarily.

This file is the Vercel entrypoint only. Local development still uses src/api.py
via: uvicorn src.api:app --host 127.0.0.1 --port 8000

Author  : Customer Churn Forecasting Project
Version : 2.0.0 (Vercel-ready)
"""


import os
import sys
from contextlib import asynccontextmanager
from typing import List, Optional
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Path resolution — works both locally and inside Vercel's Lambda filesystem
# ---------------------------------------------------------------------------
_THIS_FILE = os.path.abspath(__file__)
# api/py/index.py  ->  api/py/  ->  api/  ->  project_root/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_FILE)))
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")

# Fallback: if running directly from Lambda /var/task root
if not os.path.isdir(MODELS_DIR):
    MODELS_DIR = os.path.join("/var/task", "models")


# ---------------------------------------------------------------------------
# Model loading helpers (mirrors src/predict.py, without src/ import)
# ---------------------------------------------------------------------------

def _load_model(model_name: str):
    key = model_name.lower().replace(" ", "_")
    # Prefer XGBoost native JSON format for stable cross-version serialization
    if key == "xgboost":
        for candidate in ["xgboost.json", "xgboost_model.json"]:
            json_path = os.path.join(MODELS_DIR, candidate)
            if os.path.exists(json_path):
                import xgboost as xgb
                model = xgb.XGBClassifier()
                model.load_model(json_path)
                print(f"[LOADED] {model_name} <- {json_path} (native JSON format)")
                return model

    fname = key + ".pkl"
    path = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(path):
        available = [
            f.replace(".pkl", "")
            for f in os.listdir(MODELS_DIR)
            if f.endswith(".pkl")
        ]
        raise FileNotFoundError(
            f"Model '{model_name}' not found at {path}. Available: {available}"
        )
    model = joblib.load(path)
    # Compatibility fix for scikit-learn LogisticRegression across versions
    if hasattr(model, "predict_proba") and not hasattr(model, "multi_class"):
        setattr(model, "multi_class", "deprecated")
    print(f"[LOADED] {model_name} <- {path}")
    return model


def _load_scaler():
    path = os.path.join(MODELS_DIR, "scaler.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler not found at {path}")
    return joblib.load(path)


def _load_trained_columns() -> list[str]:
    path = os.path.join(MODELS_DIR, "trained_columns.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"trained_columns.txt not found at {path}")
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Inference preprocessing (mirrors src/predict.py::preprocess_for_inference)
# Feature engineering inlined from src/feature_engineering.py::engineer_features
# Data cleaning inlined from src/preprocessing.py::clean_data
# — without imblearn / matplotlib / seaborn / requests
# ---------------------------------------------------------------------------

def _clean_data_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inference-only clean_data subset:
    - Drop customerID / Churn if present
    - Coerce TotalCharges to numeric
    - Impute NaN TotalCharges with median (or 0 for single rows)
    - IQR outlier capping for numeric columns
    Does NOT encode the Churn target (not present at inference time).
    """
    df = df.copy()

    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)
    if "Churn" in df.columns:
        df.drop(columns=["Churn"], inplace=True)

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

    # Coerce TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Impute missing TotalCharges
    median_tc = df["TotalCharges"].median()
    if pd.isna(median_tc):
        median_tc = 0.0
    df["TotalCharges"] = df["TotalCharges"].fillna(median_tc)

    # IQR capping on continuous columns (only when IQR > 0)
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col in df.columns:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                df[col] = df[col].clip(lower, upper)

    return df


def _engineer_features_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inference-only engineer_features subset.
    Mirrors src/feature_engineering.py::engineer_features but:
    - No SMOTE import
    - Uses fixed benchmark threshold (89.85) for is_high_value on small batches
      to prevent quantile distortion on single rows.
    """
    df = df.copy()

    # Tenure group buckets
    bins = [-1, 12, 36, np.inf]
    labels = ["New (0-12m)", "Mid (13-36m)", "Loyal (37m+)"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels)

    # Spend efficiency
    df["avg_monthly_spend"] = (df["TotalCharges"] / (df["tenure"] + 1)).round(2)

    # Service count (count "Yes" across add-on columns)
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies",
    ]
    existing_svc = [c for c in service_cols if c in df.columns]
    df["service_count"] = df[existing_svc].apply(
        lambda row: (row == "Yes").sum(), axis=1
    )

    # Charges per active service (avoid div-by-zero)
    df["charges_per_service"] = (
        df["MonthlyCharges"] / (df["service_count"] + 1)
    ).round(2)

    # Flags
    df["has_multiple_services"] = (df["service_count"] >= 3).astype(int)

    # is_high_value: use fixed benchmark for single-row / small-batch inference
    # to match the training-time 75th percentile (89.85 from IBM Telco training set)
    if len(df) > 10 and df["MonthlyCharges"].nunique() > 1:
        high_val_threshold = df["MonthlyCharges"].quantile(0.75)
    else:
        high_val_threshold = 89.85  # benchmark 75th percentile
    df["is_high_value"] = (df["MonthlyCharges"] >= high_val_threshold).astype(int)

    # Contract risk score (month-to-month × paperless billing)
    if "Contract" in df.columns and "PaperlessBilling" in df.columns:
        df["contract_risk_score"] = (
            (df["Contract"] == "Month-to-month").astype(int)
            * (df["PaperlessBilling"] == "Yes").astype(int)
        )

    return df


def _preprocess_for_inference(
    df: pd.DataFrame,
    trained_columns: list[str],
    scaler,
) -> pd.DataFrame:
    """
    Full inference preprocessing pipeline:
    1. Clean (coerce types, impute, IQR cap)
    2. Feature engineering (domain features)
    3. Binary encoding
    4. Deterministic one-hot encoding (21 dummy columns)
    5. StandardScaler on continuous columns
    6. Column alignment to training schema
    """
    df = df.copy()

    # 1. Clean
    df = _clean_data_for_inference(df)

    # 2. Feature engineering
    df = _engineer_features_for_inference(df)

    # 3. Deterministic binary encoding
    bin_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    binary_fields = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_fields:
        if col in df.columns:
            df[col] = df[col].map(bin_map).fillna(0).astype(int)

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = (
            pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
        )

    # 4. Multi-class one-hot encoding — deterministic (no pd.get_dummies variance)
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

    # 5. StandardScaler on continuous columns
    scale_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    df[scale_cols] = scaler.transform(df[scale_cols])

    # 6. Align to training schema (add any missing cols as 0)
    for col in trained_columns:
        if col not in df.columns:
            df[col] = 0

    return df[trained_columns]


# ---------------------------------------------------------------------------
# Churn driver logic (matches src/predict.py — no changes)
# ---------------------------------------------------------------------------
_TOP_CHURN_DRIVERS = [
    {
        "feature": "Contract",
        "condition": lambda r: r.get("Contract") == "Month-to-month",
        "reason": "Month-to-month contract (no switching friction)",
    },
    {
        "feature": "tenure",
        "condition": lambda r: float(r.get("tenure", 0)) < 12,
        "reason": "Early lifecycle customer (tenure < 12 months)",
    },
    {
        "feature": "MonthlyCharges",
        "condition": lambda r: float(r.get("MonthlyCharges", 0)) > 75.0,
        "reason": "High monthly charges ($75+/mo)",
    },
    {
        "feature": "InternetService",
        "condition": lambda r: r.get("InternetService") == "Fiber optic",
        "reason": "Fiber optic service without bundled protections",
    },
    {
        "feature": "contract_risk_score",
        "condition": lambda r: (
            r.get("Contract") == "Month-to-month"
            and r.get("PaperlessBilling") == "Yes"
        ),
        "reason": "Combined month-to-month and paperless billing",
    },
    {
        "feature": "OnlineSecurity",
        "condition": lambda r: r.get("OnlineSecurity") == "No",
        "reason": "No online security add-on",
    },
    {
        "feature": "TechSupport",
        "condition": lambda r: r.get("TechSupport") == "No",
        "reason": "No dedicated tech support service",
    },
    {
        "feature": "PaymentMethod",
        "condition": lambda r: r.get("PaymentMethod") == "Electronic check",
        "reason": "Electronic check payment method",
    },
    {
        "feature": "PaperlessBilling",
        "condition": lambda r: r.get("PaperlessBilling") == "Yes",
        "reason": "Paperless billing active",
    },
]


def _derive_drivers(row_dict: dict) -> list[str]:
    drivers = []
    for item in _TOP_CHURN_DRIVERS:
        try:
            if item["condition"](row_dict):
                drivers.append(item["reason"])
        except Exception:
            continue
    if not drivers:
        drivers.append("Standard tenure and usage pattern (no critical risk flags)")
    return drivers[:4]


def _get_recommendation(risk_tier: str, drivers: list[str]) -> str:
    if risk_tier == "High Risk":
        if any("Month-to-month" in d for d in drivers):
            return (
                "Urgent Intervention: Offer 15% discount to switch from "
                "Month-to-month to a 1-year contract, bundled with free Security add-ons."
            )
        elif any("tenure < 12" in d for d in drivers):
            return (
                "Urgent Intervention: Assign dedicated onboarding specialist "
                "and offer a 60-day VIP service trial."
            )
        return "Urgent Intervention: Priority outbound retention outreach with customized promotional rate."
    elif risk_tier == "Medium Risk":
        if any("Fiber optic" in d for d in drivers) or any("security" in d.lower() for d in drivers):
            return (
                "Proactive Retention: Offer discounted 'Value Shield' bundle "
                "(Online Security + Tech Support) at $5/mo."
            )
        return (
            "Proactive Retention: Send customer satisfaction survey "
            "with loyalty credits on their next billing cycle."
        )
    return "Standard Care: Account is healthy. Enroll in loyalty rewards programme and explore service expansion."


def _run_inference(model, df_raw: pd.DataFrame, trained_cols: list[str], scaler, threshold: float = 0.50):
    ids = (
        df_raw["customerID"].values
        if "customerID" in df_raw.columns
        else [f"CUST-{i+1:04d}" for i in range(len(df_raw))]
    )
    X = _preprocess_for_inference(df_raw.copy(), trained_cols, scaler)
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

        drivers = _derive_drivers(row_dict)
        rec = _get_recommendation(tier, drivers)

        results.append({
            "CustomerID": ids[i],
            "churn_probability": round(float(prob), 4),
            "churn_probability_pct": f"{prob * 100:.1f}%",
            "churn_prediction": "Churn" if pred == 1 else "No Churn",
            "risk_tier": tier,
            "top_drivers": drivers,
            "recommendation": rec,
        })
    return results


# ---------------------------------------------------------------------------
# Global model cache (populated on startup / cold start)
# ---------------------------------------------------------------------------
_MODELS: dict = {}
_SCALER = None
_TRAINED_COLS: list[str] | None = None


def _bootstrap_models():
    global _SCALER, _TRAINED_COLS
    try:
        _SCALER = _load_scaler()
        _TRAINED_COLS = _load_trained_columns()
        for name in ["xgboost", "random_forest", "logistic_regression"]:
            try:
                _MODELS[name] = _load_model(name)
            except Exception as e:
                print(f"[WARN] Could not load {name}: {e}")
        print(f"[INFO] Models loaded: {list(_MODELS.keys())} | Features: {len(_TRAINED_COLS)}")
    except Exception as e:
        print(f"[ERROR] Model bootstrap failed: {e}")


# ---------------------------------------------------------------------------
# FastAPI application (lifespan — modern, non-deprecated)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _bootstrap_models()
    yield  # application runs
    # (cleanup on shutdown — nothing needed for read-only models)


app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "High-performance ML inference service for Telco Customer Churn Prediction. "
        "Powered by XGBoost, Random Forest, and Logistic Regression trained on IBM Telco dataset."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class CustomerInput(BaseModel):
    customerID: Optional[str] = Field("CUST-NEW", description="Unique customer ID")
    gender: str = Field("Female", description="Male / Female")
    SeniorCitizen: int = Field(0, description="1 if senior citizen, 0 otherwise")
    Partner: str = Field("No", description="Yes / No")
    Dependents: str = Field("No", description="Yes / No")
    tenure: float = Field(1.0, ge=0, le=100, description="Tenure in months")
    PhoneService: str = Field("Yes", description="Yes / No")
    MultipleLines: str = Field("No", description="No / Yes / No phone service")
    InternetService: str = Field("Fiber optic", description="DSL / Fiber optic / No")
    OnlineSecurity: str = Field("No", description="Yes / No / No internet service")
    OnlineBackup: str = Field("No", description="Yes / No / No internet service")
    DeviceProtection: str = Field("No", description="Yes / No / No internet service")
    TechSupport: str = Field("No", description="Yes / No / No internet service")
    StreamingTV: str = Field("Yes", description="Yes / No / No internet service")
    StreamingMovies: str = Field("Yes", description="Yes / No / No internet service")
    Contract: str = Field("Month-to-month", description="Month-to-month / One year / Two year")
    PaperlessBilling: str = Field("Yes", description="Yes / No")
    PaymentMethod: str = Field(
        "Electronic check",
        description=(
            "Electronic check / Mailed check / "
            "Bank transfer (automatic) / Credit card (automatic)"
        ),
    )
    MonthlyCharges: float = Field(85.0, ge=0, description="Monthly bill ($)")
    TotalCharges: Optional[float] = Field(None, ge=0, description="Cumulative spend ($)")


class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_probability_pct: str
    churn_prediction: str
    risk_tier: str
    confidence: float
    top_drivers: List[str]
    recommendation: str
    model_used: str
    timestamp: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return {
        "service": "Customer Churn Prediction ML API",
        "status": "healthy",
        "docs": "/api/py/docs",
        "best_model": "xgboost (ROC-AUC: 0.868)",
        "models_loaded": list(_MODELS.keys()),
    }


@app.get("/health")
def health_check():
    return {
        "status": "online",
        "models_loaded": list(_MODELS.keys()),
        "scaler_loaded": _SCALER is not None,
        "features_count": len(_TRAINED_COLS) if _TRAINED_COLS else 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/models")
def list_models():
    return {
        "available_models": [
            {
                "id": "xgboost",
                "name": "XGBoost Classifier",
                "roc_auc": 0.868,
                "accuracy": 0.820,
                "f1_score": 0.634,
                "is_recommended": True,
            },
            {
                "id": "random_forest",
                "name": "Random Forest Classifier",
                "roc_auc": 0.856,
                "accuracy": 0.812,
                "f1_score": 0.617,
                "is_recommended": False,
            },
            {
                "id": "logistic_regression",
                "name": "Logistic Regression",
                "roc_auc": 0.845,
                "accuracy": 0.800,
                "f1_score": 0.600,
                "is_recommended": False,
            },
        ]
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_single(
    customer: CustomerInput,
    model_name: str = Query("xgboost"),
):
    model_key = model_name.lower().strip()

    # Lazy-load model if not already cached
    if model_key not in _MODELS:
        try:
            _MODELS[model_key] = _load_model(model_key)
        except Exception as e:
            available = list(_MODELS.keys()) or ["xgboost", "random_forest", "logistic_regression"]
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model '{model_name}' not available. "
                    f"Choose from {available}. Error: {str(e)}"
                ),
            )

    if _SCALER is None or _TRAINED_COLS is None:
        raise HTTPException(
            status_code=503,
            detail="ML inference service not fully initialised. Scaler or feature columns missing.",
        )

    model = _MODELS[model_key]
    data_dict = customer.model_dump() if hasattr(customer, "model_dump") else customer.dict()

    # Default TotalCharges if omitted
    if data_dict.get("TotalCharges") is None:
        data_dict["TotalCharges"] = round(
            data_dict["MonthlyCharges"] * max(data_dict["tenure"], 1), 2
        )

    df_raw = pd.DataFrame([data_dict])

    try:
        rows = _run_inference(model, df_raw, _TRAINED_COLS, _SCALER, threshold=0.50)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference execution failed: {str(e)}",
        )

    row = rows[0]
    prob = row["churn_probability"]
    confidence = round(max(prob, 1 - prob), 4)

    return PredictionResponse(
        customer_id=row["CustomerID"],
        churn_probability=prob,
        churn_probability_pct=row["churn_probability_pct"],
        churn_prediction=row["churn_prediction"],
        risk_tier=row["risk_tier"],
        confidence=confidence,
        top_drivers=row["top_drivers"],
        recommendation=row["recommendation"],
        model_used=model_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Local dev entry point (not used on Vercel)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=False)
