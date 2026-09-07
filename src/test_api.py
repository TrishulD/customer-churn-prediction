import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi.testclient import TestClient
from api import app, startup_event

# Run startup
startup_event()
client = TestClient(app)

print("Testing /health ...")
r_health = client.get("/health")
print("Status:", r_health.status_code, r_health.json())
assert r_health.status_code == 200

print("\nTesting /models ...")
r_models = client.get("/models")
print("Status:", r_models.status_code, len(r_models.json()["available_models"]), "models available")
assert r_models.status_code == 200

print("\nTesting /predict with High Risk customer...")
high_risk_payload = {
    "customerID": "TEST-HIGH",
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
    "TotalCharges": 95.65
}

r_pred = client.post("/predict?model_name=xgboost", json=high_risk_payload)
print("Status:", r_pred.status_code)
data = r_pred.json()
print("Prediction response:")
for k, v in data.items():
    print(f"  {k}: {v}")
assert r_pred.status_code == 200
assert data["risk_tier"] == "High Risk"
assert data["churn_prediction"] == "Churn"
print("\n[ALL TESTS PASSED]")
