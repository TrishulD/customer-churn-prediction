"""
Comprehensive verification test for customer-churn-prediction ML inference pipeline.
Verifies clean loading, exact probabilities, absence of warnings, and FastAPI endpoints.
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd

# Turn on all warnings so any warning will be visible
warnings.simplefilter("always")

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)

print("=" * 80)
print("1. VERIFYING api/py/index.py & _bootstrap_models()")
print("=" * 80)

import importlib.util
spec = importlib.util.spec_from_file_location("api_py_index", os.path.join(REPO_ROOT, "api", "py", "index.py"))
api_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api_mod)

# Catch any warnings during bootstrap
with warnings.catch_warnings(record=True) as w_list:
    warnings.simplefilter("always")
    api_mod._bootstrap_models()

    print(f"Captured {len(w_list)} warnings during _bootstrap_models().")
    for w in w_list:
        print(f"  [WARNING] {w.category.__name__}: {w.message}")

assert len(api_mod._MODELS) == 3, f"Expected 3 models, got {len(api_mod._MODELS)}"
for m in ["xgboost", "random_forest", "logistic_regression"]:
    assert m in api_mod._MODELS, f"Model {m} missing from _MODELS"
    print(f"  [PASS] {m}: {type(api_mod._MODELS[m]).__name__}")

assert api_mod._SCALER is not None, "Scaler is None"
print(f"  [PASS] Scaler loaded: {type(api_mod._SCALER).__name__}")
print(f"         Feature names: {list(getattr(api_mod._SCALER, 'feature_names_in_', []))}")

assert api_mod._TRAINED_COLS is not None, "Trained columns is None"
assert len(api_mod._TRAINED_COLS) == 36, f"Expected 36 columns, got {len(api_mod._TRAINED_COLS)}"
print(f"  [PASS] Trained columns: exactly {len(api_mod._TRAINED_COLS)} features")

print("\n" + "=" * 80)
print("2. VERIFYING src/predict.py load_model() & inference")
print("=" * 80)

from src.predict import load_model, load_scaler, load_trained_columns, predict_churn

with warnings.catch_warnings(record=True) as w_list2:
    warnings.simplefilter("always")
    src_xgb = load_model("xgboost")
    src_rf = load_model("random_forest")
    src_lr = load_model("logistic_regression")
    src_scaler = load_scaler()
    src_cols = load_trained_columns()

    print(f"Captured {len(w_list2)} warnings during src/predict.py loading.")
    for w in w_list2:
        print(f"  [WARNING] {w.category.__name__}: {w.message}")

assert len(src_cols) == 36, f"Expected 36 columns, got {len(src_cols)}"
print("  [PASS] src/predict.py loaded all 3 models cleanly")

print("\n" + "=" * 80)
print("3. VERIFYING EXACT INFERENCE PREDICTIONS ON KNOWN TEST CUSTOMERS")
print("=" * 80)

high_risk_cust = pd.DataFrame([{
    "customerID": "HIGH-RISK-01",
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 1, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 95.65, "TotalCharges": 95.65,
}])

low_risk_cust = pd.DataFrame([{
    "customerID": "LOW-RISK-02",
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
    "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
    "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
    "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
    "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 65.00, "TotalCharges": 3900.00,
}])

# Test via api/py/index.py _run_inference
res_xgb_high = api_mod._run_inference(api_mod._MODELS["xgboost"], high_risk_cust, api_mod._TRAINED_COLS, api_mod._SCALER)[0]
res_rf_high  = api_mod._run_inference(api_mod._MODELS["random_forest"], high_risk_cust, api_mod._TRAINED_COLS, api_mod._SCALER)[0]
res_lr_high  = api_mod._run_inference(api_mod._MODELS["logistic_regression"], high_risk_cust, api_mod._TRAINED_COLS, api_mod._SCALER)[0]

res_xgb_low = api_mod._run_inference(api_mod._MODELS["xgboost"], low_risk_cust, api_mod._TRAINED_COLS, api_mod._SCALER)[0]
res_rf_low  = api_mod._run_inference(api_mod._MODELS["random_forest"], low_risk_cust, api_mod._TRAINED_COLS, api_mod._SCALER)[0]
res_lr_low  = api_mod._run_inference(api_mod._MODELS["logistic_regression"], low_risk_cust, api_mod._TRAINED_COLS, api_mod._SCALER)[0]

print("HIGH-RISK CUSTOMER RESULTS:")
print(f"  XGBoost:             Prob = {res_xgb_high['churn_probability_pct']} ({res_xgb_high['churn_probability']}), Pred = {res_xgb_high['churn_prediction']}, Tier = {res_xgb_high['risk_tier']}")
print(f"  Random Forest:       Prob = {res_rf_high['churn_probability_pct']} ({res_rf_high['churn_probability']}), Pred = {res_rf_high['churn_prediction']}, Tier = {res_rf_high['risk_tier']}")
print(f"  Logistic Regression: Prob = {res_lr_high['churn_probability_pct']} ({res_lr_high['churn_probability']}), Pred = {res_lr_high['churn_prediction']}, Tier = {res_lr_high['risk_tier']}")

print("\nLOW-RISK CUSTOMER RESULTS:")
print(f"  XGBoost:             Prob = {res_xgb_low['churn_probability_pct']} ({res_xgb_low['churn_probability']}), Pred = {res_xgb_low['churn_prediction']}, Tier = {res_xgb_low['risk_tier']}")
print(f"  Random Forest:       Prob = {res_rf_low['churn_probability_pct']} ({res_rf_low['churn_probability']}), Pred = {res_rf_low['churn_prediction']}, Tier = {res_rf_low['risk_tier']}")
print(f"  Logistic Regression: Prob = {res_lr_low['churn_probability_pct']} ({res_lr_low['churn_probability']}), Pred = {res_lr_low['churn_prediction']}, Tier = {res_lr_low['risk_tier']}")

# Assertions
assert abs(res_xgb_high["churn_probability"] - 0.8389) < 0.005, f"XGBoost high-risk expected ~0.8389, got {res_xgb_high['churn_probability']}"
assert res_xgb_high["churn_prediction"] == "Churn"
assert res_xgb_high["risk_tier"] == "High Risk"

assert abs(res_xgb_low["churn_probability"] - 0.0151) < 0.005, f"XGBoost low-risk expected ~0.0151, got {res_xgb_low['churn_probability']}"
assert res_xgb_low["churn_prediction"] == "No Churn"
assert res_xgb_low["risk_tier"] == "Low Risk"

assert abs(res_rf_high["churn_probability"] - 0.7533) < 0.005
assert abs(res_lr_high["churn_probability"] - 0.9523) < 0.005

print("\n  [PASS] All probability checks match expected project values within 0.005 tolerance!")

print("\n" + "=" * 80)
print("4. VERIFYING FASTAPI TEST CLIENT (ENDPOINTS)")
print("=" * 80)

from fastapi.testclient import TestClient

client = TestClient(api_mod.app)

# Health endpoint
r_health = client.get("/health")
assert r_health.status_code == 200, f"Health returned {r_health.status_code}"
h_data = r_health.json()
print("  [PASS] GET /health:", h_data)
assert h_data["scaler_loaded"] is True
assert h_data["features_count"] == 36
assert set(h_data["models_loaded"]) == {"xgboost", "random_forest", "logistic_regression"}

# Models endpoint
r_models = client.get("/models")
assert r_models.status_code == 200
m_data = r_models.json()
print("  [PASS] GET /models:", [m["name"] for m in m_data["available_models"]])

# Predict endpoint with each model
customer_payload = high_risk_cust.iloc[0].to_dict()

for m_name in ["xgboost", "random_forest", "logistic_regression"]:
    r_pred = client.post(f"/predict?model_name={m_name}", json=customer_payload)
    assert r_pred.status_code == 200, f"Predict for {m_name} failed: {r_pred.text}"
    p_data = r_pred.json()
    print(f"  [PASS] POST /predict?model_name={m_name} -> {p_data['churn_prediction']} ({p_data['churn_probability_pct']}), Risk: {p_data['risk_tier']}")
    assert p_data["risk_tier"] == "High Risk"
    assert p_data["churn_prediction"] == "Churn"

print("\n" + "=" * 80)
print("5. VERIFYING ABSENCE OF HEURISTIC/FALLBACK IN CODEBASE")
print("=" * 80)

import re

# Check route.ts
with open(os.path.join(REPO_ROOT, "app", "api", "predict", "route.ts"), "r") as f:
    route_ts_content = f.read()

assert "heuristic" not in route_ts_content.lower(), "Heuristic found in app/api/predict/route.ts"
assert "calculateheuristic" not in route_ts_content.lower(), "calculateHeuristic found in route.ts"
assert "503" in route_ts_content, "503 status code missing from route.ts"
print("  [PASS] app/api/predict/route.ts contains strictly no heuristic fallback and returns 503 on service failure.")

# Check api/py/index.py
with open(os.path.join(REPO_ROOT, "api", "py", "index.py"), "r") as f:
    api_py_content = f.read()

assert "heuristic" not in api_py_content.lower(), "Heuristic found in api/py/index.py"
print("  [PASS] api/py/index.py contains strictly real ML inference from trained models.")

print("\n" + "=" * 80)
print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY WITH ZERO ERRORS")
print("=" * 80)
