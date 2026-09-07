"""
preprocessing.py
================
Production-quality data loading, cleaning, and preprocessing pipeline
for the IBM Telco Customer Churn dataset.

Author  : Senior Data Scientist
Project : Customer Churn Forecasting
"""

import os
import io
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
TARGET_COL = "Churn"
RANDOM_STATE = 42


# ──────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
def load_data(url: str = DATA_URL, local_path: str | None = None) -> pd.DataFrame:
    """
    Load IBM Telco Customer Churn dataset from a URL or local CSV file.

    Parameters
    ----------
    url        : Direct URL to the raw CSV file.
    local_path : If provided, load from disk instead of the network.

    Returns
    -------
    pd.DataFrame : Raw dataset.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if local_path and os.path.exists(local_path):
        print(f"[INFO] Loading data from local file: {local_path}")
        df = pd.read_csv(local_path)
    else:
        print(f"[INFO] Downloading dataset from:\n       {url}")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        save_path = os.path.join(DATA_DIR, "telco_churn_raw.csv")
        df.to_csv(save_path, index=False)
        print(f"[INFO] Dataset saved to: {save_path}")

    print(f"[INFO] Dataset shape : {df.shape}")
    print(f"[INFO] Columns       : {list(df.columns)}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2. DATA CLEANING
# ──────────────────────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform essential cleaning steps:
      - Drop uninformative ID column
      - Convert TotalCharges to numeric (handles hidden whitespace)
      - Impute missing TotalCharges with the median
      - Encode binary target (Yes/No → 1/0)
      - Strip leading/trailing whitespace from object columns

    Parameters
    ----------
    df : Raw DataFrame.

    Returns
    -------
    pd.DataFrame : Cleaned DataFrame.
    """
    df = df.copy()

    # Drop customer ID – not predictive
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # Strip whitespace from all string columns
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

    # TotalCharges is stored as string with spaces for new customers → coerce
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # ── Missing value analysis ──────────────────────────────────────────────
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_report = pd.DataFrame({"Missing": missing, "Pct (%)": missing_pct})
    missing_report = missing_report[missing_report["Missing"] > 0]
    print("\n[MISSING VALUES]")
    print(missing_report if not missing_report.empty else "  None found.")

    # Impute TotalCharges NaN with median (new customers with 0 tenure)
    median_tc = df["TotalCharges"].median()
    if pd.isna(median_tc):
        median_tc = 0.0
    df["TotalCharges"] = df["TotalCharges"].fillna(median_tc)

    # ── Outlier handling (IQR capping for numeric cols) ────────────────────
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in num_cols:
        if col in df.columns:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            if IQR > 0:
                lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                n_out = ((df[col] < lower) | (df[col] > upper)).sum()
                if n_out:
                    df[col] = df[col].clip(lower, upper)
                    print(f"[INFO] Capped {n_out} outliers in '{col}'")

    # Encode target → binary integer (if present)
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
        print(f"\n[INFO] Cleaned shape  : {df.shape}")
        print(f"[INFO] Churn rate     : {df[TARGET_COL].mean():.2%}")
    else:
        print(f"\n[INFO] Cleaned shape  : {df.shape} (Inference mode, no target)")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3. EXPLORATORY VISUALISATIONS
# ──────────────────────────────────────────────────────────────────────────────
def plot_churn_distribution(df: pd.DataFrame) -> None:
    """Bar + pie chart of churn vs. non-churn counts."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Count plot
    counts = df[TARGET_COL].value_counts()
    axes[0].bar(["No Churn", "Churn"], counts.values,
                color=["#2ecc71", "#e74c3c"], edgecolor="black", linewidth=0.8)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 30, f"{v}\n({v/len(df)*100:.1f}%)",
                     ha="center", fontweight="bold", fontsize=11)
    axes[0].set_title("Customer Churn Distribution", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].set_ylim(0, counts.max() * 1.15)
    axes[0].spines[["top", "right"]].set_visible(False)

    # Pie chart
    axes[1].pie(counts.values, labels=["No Churn", "Churn"],
                autopct="%1.1f%%", colors=["#2ecc71", "#e74c3c"],
                startangle=90, explode=(0, 0.07),
                textprops={"fontsize": 12})
    axes[1].set_title("Churn Proportion", fontsize=14, fontweight="bold")

    plt.suptitle("IBM Telco — Churn Class Distribution", fontsize=15, y=1.02)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "01_churn_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved → {path}")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Correlation heatmap of numeric + encoded binary features."""
    os.makedirs(REPORT_DIR, exist_ok=True)

    # Encode yes/no binary cols for correlation
    df_enc = df.copy()
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0,
                  "No phone service": 0, "No internet service": 0}
    str_cols = df_enc.select_dtypes(include="object").columns
    for col in str_cols:
        df_enc[col] = df_enc[col].replace(binary_map)
        df_enc[col] = pd.to_numeric(df_enc[col], errors="coerce")

    num_df = df_enc.select_dtypes(include=[np.number]).dropna(axis=1)
    corr   = num_df.corr()

    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.5,
                linecolor="white", ax=ax, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap", fontsize=15, fontweight="bold", pad=15)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "02_correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. ENCODE & SPLIT
# ──────────────────────────────────────────────────────────────────────────────
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply encoding strategy:
      - Binary yes/no columns  → Label Encoding (0/1)
      - Multi-class categoricals → One-Hot Encoding

    Returns a fully numeric DataFrame ready for ML.
    """
    df = df.copy()

    # Columns with only two unique values (excluding target)
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    binary_cols = [c for c in cat_cols if df[c].nunique() == 2]
    multi_cols  = [c for c in cat_cols if df[c].nunique() > 2]

    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])

    # One-hot encode multi-class columns (drop_first to avoid dummy trap)
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    print(f"\n[ENCODING] Binary label-encoded : {binary_cols}")
    print(f"[ENCODING] One-hot encoded       : {multi_cols}")
    print(f"[ENCODING] Final feature count   : {df.shape[1] - 1}")
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> tuple:
    """
    Stratified train/test split + StandardScaler on numeric features.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale continuous numeric features
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    num_cols = [c for c in num_cols if c in X_train.columns]
    scaler   = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    print(f"\n[SPLIT] Train : {X_train.shape}  |  Test : {X_test.shape}")
    print(f"[SPLIT] Train churn rate : {y_train.mean():.2%}")
    print(f"[SPLIT] Test  churn rate : {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test, list(X.columns)


# ──────────────────────────────────────────────────────────────────────────────
# STANDALONE EXECUTION
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    raw_df     = load_data()
    clean_df   = clean_data(raw_df)
    plot_churn_distribution(clean_df)
    plot_correlation_heatmap(clean_df)
    encoded_df = encode_features(clean_df)
    X_train, X_test, y_train, y_test, features = split_data(encoded_df)
    print("\n[DONE] Preprocessing complete.")
