"""
api.py
======
Production FastAPI inference service for Customer Churn Forecasting.
Exposes RESTful endpoints for real-time predictions, health checks, and model metadata.

Run locally:
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

Author  : Senior Data Scientist & ML Engineer
Project : Customer Churn Forecasting
"""

import os
import sys
from typing import Optional, List
from datetime import datetime, timezone
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(__file__))
from predict import (
    load_model,
    load_scaler,
    load_trained_columns,
    predict_churn,
    TOP_CHURN_DRIVERS_INFO,
)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="High-performance ML inference service for Telecom Customer Churn Prediction.",
    version="2.0.0",
)

# Enable CORS for Next.js frontend (Vercel & localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache for models and scalers
MODELS = {}
SCALER = None
TRAINED_COLS = None


@app.on_event("startup")
def startup_event():
    global SCALER, TRAINED_COLS
    try:
        SCALER = load_scaler()
        TRAINED_COLS = load_trained_columns()
        for name in ["xgboost", "random_forest", "logistic_regression"]:
            try:
                MODELS[name] = load_model(name)
            except Exception as e:
                print(f"[WARN] Could not preload {name}: {e}")
        print("[INFO] Inference service started and models loaded.")
    except Exception as e:
        print(f"[ERROR] Startup loading failed: {e}")


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
        description="Electronic check / Mailed check / Bank transfer (automatic) / Credit card (automatic)",
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


@app.get("/")
def index():
    return {
        "service": "Customer Churn Prediction ML API",
        "status": "healthy",
        "docs": "/docs",
        "best_model": "xgboost (ROC-AUC: 0.868)",
    }


@app.get("/health")
def health_check():
    return {
        "status": "online",
        "models_loaded": list(MODELS.keys()),
        "scaler_loaded": SCALER is not None,
        "features_count": len(TRAINED_COLS) if TRAINED_COLS else 0,
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


@app.get("/model-info")
def model_info():
    return {
        "dataset": "IBM Telco Customer Churn (7,043 samples)",
        "baseline_churn_rate": "26.5%",
        "cv_method": "5-Fold Stratified K-Fold + SMOTE (0.5 ratio)",
        "best_model": "XGBoost",
        "top_features": [
            {"rank": 1, "feature": "Contract_Month-to-month", "importance": 0.142, "direction": "Increases Churn"},
            {"rank": 2, "feature": "tenure", "importance": 0.128, "direction": "Decreases Churn"},
            {"rank": 3, "feature": "MonthlyCharges", "importance": 0.098, "direction": "Increases Churn"},
            {"rank": 4, "feature": "TotalCharges", "importance": 0.087, "direction": "Decreases Churn"},
            {"rank": 5, "feature": "InternetService_Fiber optic", "importance": 0.074, "direction": "Increases Churn"},
            {"rank": 6, "feature": "avg_monthly_spend", "importance": 0.068, "direction": "Increases Churn"},
            {"rank": 7, "feature": "contract_risk_score", "importance": 0.059, "direction": "Increases Churn"},
            {"rank": 8, "feature": "OnlineSecurity_No", "importance": 0.051, "direction": "Increases Churn"},
            {"rank": 9, "feature": "TechSupport_No", "importance": 0.048, "direction": "Increases Churn"},
            {"rank": 10, "feature": "PaperlessBilling", "importance": 0.041, "direction": "Increases Churn"},
        ],
        "risk_tiers": {
            "Low Risk": "Probability < 30%",
            "Medium Risk": "Probability 30% - 60%",
            "High Risk": "Probability > 60%",
        },
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_single(customer: CustomerInput, model_name: str = Query("xgboost")):
    model_key = model_name.lower().strip()
    if model_key not in MODELS:
        try:
            MODELS[model_key] = load_model(model_key)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_name}' not available. Choose from {list(MODELS.keys()) or ['xgboost']}. Error: {str(e)}",
            )

    model = MODELS[model_key]
    data_dict = customer.model_dump() if hasattr(customer, "model_dump") else customer.dict()

    # Calculate TotalCharges default if omitted
    if data_dict.get("TotalCharges") is None:
        data_dict["TotalCharges"] = round(data_dict["MonthlyCharges"] * max(data_dict["tenure"], 1), 2)

    df_raw = pd.DataFrame([data_dict])

    try:
        preds_df = predict_churn(
            model=model,
            df_raw=df_raw,
            trained_columns=TRAINED_COLS,
            scaler=SCALER,
            threshold=0.50,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference execution failed: {str(e)}")

    row = preds_df.iloc[0]
    prob = float(row["churn_probability"])
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
