"""
feature_engineering.py
=======================
Advanced feature engineering and SMOTE-based class-imbalance handling
for the Customer Churn Forecasting project.

Author  : Senior Data Scientist
Project : Customer Churn Forecasting
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42


# ──────────────────────────────────────────────────────────────────────────────
# 1. DOMAIN-DRIVEN FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-informed interaction and ratio features BEFORE encoding.

    New features
    ────────────
    tenure_group         : Customer lifecycle bucket (New / Mid / Loyal)
    avg_monthly_spend    : TotalCharges / (tenure + 1)  – smoothed
    charges_per_service  : MonthlyCharges normalised by number of add-on services
    has_multiple_services: Flag for customers with ≥ 3 value-added services
    is_high_value        : Flag for MonthlyCharges in top quartile
    contract_risk_score  : Interaction of month-to-month contract × paperless billing

    Parameters
    ----------
    df : Cleaned (but not yet encoded) DataFrame.

    Returns
    -------
    pd.DataFrame : DataFrame with additional engineered columns.
    """
    df = df.copy()

    # ── Tenure buckets ──────────────────────────────────────────────────────
    bins   = [-1, 12, 36, np.inf]
    labels = ["New (0-12m)", "Mid (13-36m)", "Loyal (37m+)"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels)

    # ── Spend efficiency ────────────────────────────────────────────────────
    df["avg_monthly_spend"] = (df["TotalCharges"] / (df["tenure"] + 1)).round(2)

    # ── Service count (count 'Yes' responses across add-on service columns) ─
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

    # ── Flags ───────────────────────────────────────────────────────────────
    df["has_multiple_services"] = (df["service_count"] >= 3).astype(int)
    if len(df) > 10 and df["MonthlyCharges"].nunique() > 1:
        high_val_threshold = df["MonthlyCharges"].quantile(0.75)
    else:
        high_val_threshold = 89.85  # Benchmark 75th percentile of IBM Telco training set
    df["is_high_value"] = (df["MonthlyCharges"] >= high_val_threshold).astype(int)

    # ── Contract risk score (Month-to-month × Paperless Billing) ───────────
    # Will be binary after this mapping; later encoding handles it
    if "Contract" in df.columns and "PaperlessBilling" in df.columns:
        df["contract_risk_score"] = (
            (df["Contract"] == "Month-to-month").astype(int)
            * (df["PaperlessBilling"] == "Yes").astype(int)
        )

    print("[FEATURE ENG] New columns added:")
    new_cols = [
        "tenure_group", "avg_monthly_spend", "service_count",
        "charges_per_service", "has_multiple_services",
        "is_high_value", "contract_risk_score",
    ]
    for col in new_cols:
        if col in df.columns:
            print(f"   • {col}")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2. CLASS IMBALANCE HANDLING WITH SMOTE
# ──────────────────────────────────────────────────────────────────────────────
def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: float = 0.5,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE (Synthetic Minority Oversampling TEchnique) to balance
    the training set without leaking information from the test set.

    Why SMOTE?
    ──────────
    The IBM Telco dataset has ~26 % churn rate.  Training directly on this
    imbalanced data biases models toward the majority class.  SMOTE
    synthesises new minority-class samples by interpolating between
    existing ones in feature space, avoiding naive duplication.

    Parameters
    ----------
    X_train           : Training feature matrix.
    y_train           : Training labels.
    sampling_strategy : Desired minority/majority ratio after resampling.
    random_state      : Reproducibility seed.

    Returns
    -------
    X_resampled, y_resampled
    """
    print(f"\n[SMOTE] Before → Class distribution: {dict(y_train.value_counts())}")
    print(f"[SMOTE] Churn rate before : {y_train.mean():.2%}")

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        k_neighbors=5,
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)

    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name=y_train.name)

    print(f"[SMOTE] After  → Class distribution: {dict(y_res.value_counts())}")
    print(f"[SMOTE] Churn rate after  : {y_res.mean():.2%}")
    print(f"[SMOTE] Resampled training set shape: {X_res.shape}")

    return X_res, y_res


# ──────────────────────────────────────────────────────────────────────────────
# STANDALONE EXECUTION
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from preprocessing import load_data, clean_data, encode_features, split_data

    raw        = load_data()
    cleaned    = clean_data(raw)
    engineered = engineer_features(cleaned)
    encoded    = encode_features(engineered)
    X_tr, X_te, y_tr, y_te, feats = split_data(encoded)
    X_tr_bal, y_tr_bal = apply_smote(X_tr, y_tr)
    print("\n[DONE] Feature engineering & SMOTE complete.")
