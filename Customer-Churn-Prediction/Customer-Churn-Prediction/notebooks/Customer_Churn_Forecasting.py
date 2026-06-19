"""
Customer_Churn_Forecasting.py
==============================
Google Colab–compatible end-to-end script.
Copy this into a Colab notebook (one cell per section) OR run directly with:
    python notebooks/Customer_Churn_Forecasting.py

Sections
────────
0.  Install dependencies
1.  Data loading & cleaning
2.  Exploratory data analysis
3.  Feature engineering
4.  Train / test split + SMOTE
5.  Model training (LR, RF, XGB)
6.  Evaluation & visualisations
7.  Business insights
8.  Prediction demo

Author  : Senior Data Scientist
Project : Customer Churn Forecasting
"""

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 0 — INSTALL & IMPORTS                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
import subprocess, sys

def pip_install(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])

pip_install(
    "pandas==2.1.4", "numpy==1.26.3", "matplotlib==3.8.2",
    "seaborn==0.13.1", "scikit-learn==1.4.0",
    "xgboost==2.0.3",  "imbalanced-learn==0.11.0",
    "joblib==1.3.2",   "requests==2.31.0",
)

import os, io, time, warnings, joblib
import requests
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.model_selection  import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier
from sklearn.metrics          import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
from scipy.stats              import randint, uniform
from imblearn.over_sampling   import SMOTE
from xgboost                  import XGBClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

print("✅  All packages imported successfully.")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 — DATA LOADING & CLEANING                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
RANDOM_STATE = 42
TARGET       = "Churn"

# ── Load ──────────────────────────────────────────────────────────────────
print(f"\n📥 Downloading IBM Telco Churn dataset …")
resp = requests.get(DATA_URL, timeout=30)
resp.raise_for_status()
df_raw = pd.read_csv(io.StringIO(resp.text))
print(f"   Shape: {df_raw.shape}")
print(df_raw.head(3))

# ── Save raw copy ─────────────────────────────────────────────────────────
os.makedirs("data",    exist_ok=True)
os.makedirs("models",  exist_ok=True)
os.makedirs("reports", exist_ok=True)
df_raw.to_csv("data/telco_churn_raw.csv", index=False)

# ── Clean ─────────────────────────────────────────────────────────────────
df = df_raw.copy()

# Drop ID
df.drop(columns=["customerID"], inplace=True, errors="ignore")

# Strip whitespace
for col in df.select_dtypes("object"):
    df[col] = df[col].str.strip()

# Fix TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print(f"\nMissing TotalCharges: {df['TotalCharges'].isna().sum()}")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Cap outliers (IQR)
for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    df[col] = df[col].clip(Q1 - 1.5*(Q3-Q1), Q3 + 1.5*(Q3-Q1))

# Encode target
df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})

print(f"\n✅  Cleaned shape: {df.shape}")
print(f"   Churn rate   : {df[TARGET].mean():.2%}")
print(df.dtypes.value_counts())


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2 — EXPLORATORY DATA ANALYSIS                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── 2a: Churn Distribution ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
counts = df[TARGET].value_counts()
axes[0].bar(["No Churn", "Churn"], counts.values,
            color=["#2ecc71", "#e74c3c"], edgecolor="black")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 30, f"{v} ({v/len(df)*100:.1f}%)", ha="center", fontweight="bold")
axes[0].set_title("Churn Distribution", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Count")
axes[0].spines[["top","right"]].set_visible(False)

axes[1].pie(counts.values, labels=["No Churn","Churn"],
            autopct="%1.1f%%", colors=["#2ecc71","#e74c3c"],
            startangle=90, explode=(0, 0.07))
axes[1].set_title("Churn Proportion", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("reports/01_churn_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅  Plot 1/6 saved.")

# ── 2b: Numeric distributions by churn ───────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, col in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"]):
    for label, color in [(0, "#2ecc71"), (1, "#e74c3c")]:
        ax.hist(df[df[TARGET]==label][col], bins=30, alpha=0.6,
                color=color, label="No Churn" if label==0 else "Churn",
                edgecolor="white")
    ax.set_title(f"{col} Distribution by Churn", fontweight="bold")
    ax.legend()
    ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("reports/02_numeric_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅  Plot 2/6 saved.")

# ── 2c: Correlation Heatmap ───────────────────────────────────────────────
df_enc_hm = df.copy()
bin_map = {"Yes":1,"No":0,"Male":1,"Female":0,
           "No phone service":0,"No internet service":0}
for col in df_enc_hm.select_dtypes("object"):
    df_enc_hm[col] = df_enc_hm[col].replace(bin_map)
    df_enc_hm[col] = pd.to_numeric(df_enc_hm[col], errors="coerce")
num_df = df_enc_hm.select_dtypes(include=np.number).dropna(axis=1)
corr   = num_df.corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, linewidths=0.5, ax=ax,
            annot_kws={"size":7})
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("reports/03_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅  Plot 3/6 saved.")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3 — FEATURE ENGINEERING                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
df_feat = df.copy()

# Tenure buckets
df_feat["tenure_group"] = pd.cut(
    df_feat["tenure"], bins=[-1,12,36,np.inf],
    labels=["New (0-12m)","Mid (13-36m)","Loyal (37m+)"]
)

# Spend features
df_feat["avg_monthly_spend"] = (
    df_feat["TotalCharges"] / (df_feat["tenure"] + 1)
).round(2)

# Service count
svc_cols = ["PhoneService","MultipleLines","OnlineSecurity","OnlineBackup",
            "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
existing = [c for c in svc_cols if c in df_feat.columns]
df_feat["service_count"] = df_feat[existing].apply(
    lambda r: (r == "Yes").sum(), axis=1
)
df_feat["charges_per_service"] = (
    df_feat["MonthlyCharges"] / (df_feat["service_count"] + 1)
).round(2)

# Flags
df_feat["has_multiple_services"] = (df_feat["service_count"] >= 3).astype(int)
df_feat["is_high_value"] = (
    df_feat["MonthlyCharges"] >= df_feat["MonthlyCharges"].quantile(0.75)
).astype(int)

# Contract risk
df_feat["contract_risk_score"] = (
    (df_feat["Contract"] == "Month-to-month").astype(int)
    * (df_feat["PaperlessBilling"] == "Yes").astype(int)
)

print(f"✅  Feature engineering complete. Shape: {df_feat.shape}")
print("   New columns:", ["tenure_group","avg_monthly_spend","service_count",
                           "charges_per_service","has_multiple_services",
                           "is_high_value","contract_risk_score"])


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4 — ENCODING, SPLIT & SMOTE                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── Encoding ──────────────────────────────────────────────────────────────
df_enc = df_feat.copy()
cat_cols    = df_enc.select_dtypes("object").columns.tolist()
binary_cols = [c for c in cat_cols if df_enc[c].nunique() == 2]
multi_cols  = [c for c in cat_cols if df_enc[c].nunique() > 2]

le = LabelEncoder()
for col in binary_cols:
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))

df_enc = pd.get_dummies(df_enc, columns=multi_cols, drop_first=True)
print(f"✅  Encoding complete. Final feature count: {df_enc.shape[1]-1}")

# ── Train / Test Split ────────────────────────────────────────────────────
X = df_enc.drop(columns=[TARGET])
y = df_enc[TARGET]
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# Save column order for inference
with open("models/trained_columns.txt", "w") as f:
    f.write("\n".join(feature_names))

# Scale numeric columns
scaler   = StandardScaler()
num_cols = [c for c in ["tenure","MonthlyCharges","TotalCharges"] if c in X_train.columns]
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols]  = scaler.transform(X_test[num_cols])
joblib.dump(scaler, "models/scaler.pkl")

print(f"\n✅  Split → Train: {X_train.shape}  |  Test: {X_test.shape}")

# ── SMOTE ─────────────────────────────────────────────────────────────────
print(f"\nBefore SMOTE → {dict(y_train.value_counts())}")
smote = SMOTE(sampling_strategy=0.5, random_state=RANDOM_STATE, k_neighbors=5)
X_res, y_res = smote.fit_resample(X_train, y_train)
X_res = pd.DataFrame(X_res, columns=X_train.columns)
y_res = pd.Series(y_res, name=y_train.name)
print(f"After  SMOTE → {dict(y_res.value_counts())}")
print(f"✅  Resampled training set: {X_res.shape}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5 — MODEL TRAINING & HYPERPARAMETER TUNING                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

model_configs = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                           class_weight="balanced"),
        {"C": uniform(0.01,10), "solver":["lbfgs","liblinear"]}
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                               class_weight="balanced"),
        {"n_estimators":randint(100,400), "max_depth":[None,5,10,15],
         "min_samples_split":randint(2,15), "max_features":["sqrt","log2"]}
    ),
    "XGBoost": (
        XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE,
                      use_label_encoder=False, n_jobs=-1),
        {"n_estimators":randint(100,400), "max_depth":randint(3,8),
         "learning_rate":uniform(0.01,0.29), "subsample":uniform(0.6,0.4),
         "colsample_bytree":uniform(0.6,0.4), "gamma":uniform(0,0.5)}
    ),
}

trained_models = {}
for name, (est, params) in model_configs.items():
    print(f"\n{'─'*55}\n  Training: {name}\n{'─'*55}")
    t0 = time.time()
    search = RandomizedSearchCV(
        est, params, n_iter=20, scoring="roc_auc", cv=cv,
        n_jobs=-1, random_state=RANDOM_STATE, refit=True, verbose=0
    )
    search.fit(X_res, y_res)
    elapsed = time.time() - t0
    print(f"  ✔  Best CV ROC-AUC : {search.best_score_:.4f}  ({elapsed:.0f}s)")
    print(f"  ✔  Best params     : {search.best_params_}")

    # Save model
    fname = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(search.best_estimator_, f"models/{fname}", compress=3)
    trained_models[name] = search.best_estimator_

print("\n✅  All models trained and saved to models/")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 6 — EVALUATION & VISUALISATIONS                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
PALETTE = {"Logistic Regression":"#3498db",
           "Random Forest":"#2ecc71", "XGBoost":"#e74c3c"}

metrics_rows = []
for name, model in trained_models.items():
    y_prob = model.predict_proba(X_test)[:,1]
    y_pred = (y_prob >= 0.5).astype(int)
    row = {
        "Model"    : name,
        "Accuracy" : accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall"   : recall_score(y_test, y_pred, zero_division=0),
        "F1 Score" : f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC"  : roc_auc_score(y_test, y_prob),
        "_y_prob"  : y_prob,
        "_y_pred"  : y_pred,
    }
    metrics_rows.append(row)

metrics_df = pd.DataFrame(metrics_rows).set_index("Model")
display_cols = ["Accuracy","Precision","Recall","F1 Score","ROC-AUC"]
print("\n" + "="*60)
print("   MODEL COMPARISON TABLE")
print("="*60)
print(metrics_df[display_cols].round(4).to_string())

# ── Plot: ROC Curves ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8,6))
for name in metrics_df.index:
    fpr, tpr, _ = roc_curve(y_test, metrics_df.loc[name,"_y_prob"])
    ax.plot(fpr, tpr, lw=2.5, color=PALETTE[name],
            label=f"{name}  (AUC={metrics_df.loc[name,'ROC-AUC']:.4f})")
ax.plot([0,1],[0,1],"k--",lw=1,label="Random Classifier")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold")
ax.legend(loc="lower right"); ax.grid(alpha=0.3)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("reports/04_roc_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅  ROC curves saved.")

# ── Plot: Confusion Matrices ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, name in zip(axes, metrics_df.index):
    cm = confusion_matrix(y_test, metrics_df.loc[name,"_y_pred"], normalize="true")
    ConfusionMatrixDisplay(cm, display_labels=["No Churn","Churn"]).plot(
        ax=ax, colorbar=False, cmap="Blues", values_format=".2%")
    ax.set_title(f"{name}\nROC-AUC: {metrics_df.loc[name,'ROC-AUC']:.4f}",
                 fontweight="bold")
plt.suptitle("Normalised Confusion Matrices", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("reports/05_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅  Confusion matrices saved.")

# ── Plot: Feature Importance ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
for ax, name in zip(axes, ["Logistic Regression","Random Forest","XGBoost"]):
    model = trained_models[name]
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        title_sfx = "Gini Importance"
    else:
        imp = np.abs(model.coef_[0])
        title_sfx = "|Coefficient|"
    top = (pd.DataFrame({"Feature":feature_names,"Imp":imp})
             .sort_values("Imp", ascending=False).head(15))
    ax.barh(top["Feature"][::-1], top["Imp"][::-1],
            color=PALETTE[name], edgecolor="white")
    ax.set_title(f"{name}\n{title_sfx}", fontsize=12, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
plt.suptitle("Top-15 Feature Importances", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("reports/06_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅  Feature importance saved.")

# ── Plot: Metrics Bar Chart ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(display_cols)); w = 0.25
for i, name in enumerate(metrics_df.index):
    vals = metrics_df.loc[name, display_cols].values.astype(float)
    bars = ax.bar(x + (i-1)*w, vals, w, label=name,
                  color=list(PALETTE.values())[i], edgecolor="white")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(display_cols, fontsize=12)
ax.set_ylim(0, 1.12); ax.set_ylabel("Score"); ax.legend()
ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("reports/07_metrics_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅  Metrics comparison saved.")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 7 — BUSINESS INSIGHTS                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
best_name  = metrics_df["ROC-AUC"].idxmax()
best_model = trained_models[best_name]
print(f"\n🏆  Best Model: {best_name}  (ROC-AUC = {metrics_df.loc[best_name,'ROC-AUC']:.4f})")

y_prob_best = best_model.predict_proba(X_test)[:,1]
scored = X_test.copy()
scored["churn_probability"] = y_prob_best
scored["actual_churn"]      = y_test.values
scored["risk_tier"] = pd.cut(
    scored["churn_probability"],
    bins=[-0.001, 0.30, 0.60, 1.001],
    labels=["Low Risk","Medium Risk","High Risk"]
)

# Revenue impact
ARPU         = 65.0   # average revenue per user (USD/month)
RET_COST     = 10.0   # retention cost per customer
tier_counts  = scored["risk_tier"].value_counts()
high_risk_n  = tier_counts.get("High Risk", 0)
rev_at_risk  = high_risk_n * ARPU
ret_invest   = high_risk_n * RET_COST
net_saved    = rev_at_risk - ret_invest

print(f"\n{'═'*55}")
print("   BUSINESS INTELLIGENCE REPORT")
print(f"{'═'*55}")
print(f"\n📊  Risk Tier Distribution:")
for tier, cnt in tier_counts.sort_index().items():
    print(f"     {tier:<15}: {cnt:>4} customers ({cnt/len(scored)*100:.1f}%)")

print(f"\n💰  Revenue Impact:")
print(f"     High-risk customers        : {high_risk_n}")
print(f"     Monthly revenue at risk    : ${rev_at_risk:,.0f}")
print(f"     Estimated retention cost   : ${ret_invest:,.0f}")
print(f"     Net revenue protected      : ${net_saved:,.0f}")

print(f"\n🔑  Top Churn Drivers:")
for drv, why in [
    ("Month-to-month contract", "No lock-in → easiest path to switch."),
    ("Short tenure (< 12m)",    "Early customers haven't built inertia."),
    ("High charges, few services","Price-value mismatch."),
    ("No TechSupport / Security","Fewer protective services = lower perceived value."),
    ("Fiber Optic users",       "Higher-paying, higher-expectation segment."),
    ("Paperless billing",       "Digitally savvy → comparison-shops online."),
]:
    print(f"   ▸ {drv:35s} → {why}")

print(f"\n🎯  Retention Strategies:")
for s in [
    "Offer 10-15% discount to M2M customers who switch to annual contracts.",
    "New-customer onboarding programme: assign account manager for months 1-6.",
    "'Value Shield' bundle: Security + Backup + TechSupport at 20% off.",
    "Proactive call to high-risk Fiber Optic customers 30 days before renewal.",
    "Loyalty milestone rewards at 12, 24 and 36 months.",
    "Personalised pricing engine: flag customers above 75th-percentile spend.",
]:
    print(f"   ✔  {s}")

# ── Churn probability histogram ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for label, color, name in [(0,"#2ecc71","No Churn"),(1,"#e74c3c","Churn")]:
    mask = scored["actual_churn"] == label
    ax.hist(scored.loc[mask,"churn_probability"], bins=40,
            alpha=0.6, color=color, label=name, edgecolor="white")
ax.axvline(0.30, color="orange", ls="--", lw=1.5, label="Low/Med threshold (0.30)")
ax.axvline(0.60, color="red",    ls="--", lw=1.5, label="Med/High threshold (0.60)")
ax.set_xlabel("Predicted Churn Probability"); ax.set_ylabel("Count")
ax.set_title(f"Churn Probability Distribution — {best_name}",
             fontsize=13, fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("reports/08_churn_probability_dist.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅  Churn probability distribution saved.")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 8 — PREDICTION DEMO                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
new_customers = pd.DataFrame({
    "gender"         : ["Male","Female","Male","Female"],
    "SeniorCitizen"  : [0,1,0,0],
    "Partner"        : ["Yes","No","No","Yes"],
    "Dependents"     : ["No","No","Yes","No"],
    "tenure"         : [1,60,24,12],
    "PhoneService"   : ["Yes","Yes","Yes","No"],
    "MultipleLines"  : ["No","Yes","No","No phone service"],
    "InternetService": ["Fiber optic","DSL","No","DSL"],
    "OnlineSecurity" : ["No","Yes","No internet service","No"],
    "OnlineBackup"   : ["No","Yes","No internet service","Yes"],
    "DeviceProtection":["No","Yes","No internet service","No"],
    "TechSupport"    : ["No","Yes","No internet service","No"],
    "StreamingTV"    : ["No","Yes","No internet service","No"],
    "StreamingMovies": ["No","Yes","No internet service","No"],
    "Contract"       : ["Month-to-month","Two year","One year","Month-to-month"],
    "PaperlessBilling":["Yes","No","No","Yes"],
    "PaymentMethod"  : ["Electronic check","Bank transfer (automatic)",
                        "Mailed check","Electronic check"],
    "MonthlyCharges" : [95.65, 25.10, 0.00, 45.20],
    "TotalCharges"   : [95.65, 1505.0, 0.00, 542.40],
})

# Preprocess new customers
def preprocess_new(df_new, feature_cols):
    d = df_new.copy()
    for col in d.select_dtypes("object"):
        d[col] = d[col].str.strip()
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce").fillna(0)
    # Engineer
    d["avg_monthly_spend"] = (d["TotalCharges"] / (d["tenure"]+1)).round(2)
    svc = ["PhoneService","MultipleLines","OnlineSecurity","OnlineBackup",
           "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
    d["service_count"] = d[[c for c in svc if c in d.columns]].apply(
        lambda r: (r=="Yes").sum(), axis=1)
    d["charges_per_service"] = (d["MonthlyCharges"]/(d["service_count"]+1)).round(2)
    d["has_multiple_services"] = (d["service_count"]>=3).astype(int)
    d["is_high_value"] = (d["MonthlyCharges"]>=d["MonthlyCharges"].quantile(0.75)).astype(int)
    d["contract_risk_score"] = (
        (d["Contract"]=="Month-to-month").astype(int)
        *(d["PaperlessBilling"]=="Yes").astype(int)
    )
    # Encode
    cat  = d.select_dtypes("object").columns
    bin_ = [c for c in cat if d[c].nunique()==2]
    mul_ = [c for c in cat if d[c].nunique()>2]
    le2  = LabelEncoder()
    for c in bin_:
        d[c] = le2.fit_transform(d[c].astype(str))
    d = pd.get_dummies(d, columns=mul_, drop_first=True)
    # Align
    for c in feature_cols:
        if c not in d.columns:
            d[c] = 0
    return d[feature_cols]

X_new = preprocess_new(new_customers, feature_names)
X_new[num_cols] = scaler.transform(X_new[num_cols])

probs_new = best_model.predict_proba(X_new)[:,1]
preds_new = (probs_new >= 0.5).astype(int)
tiers_new = pd.cut(probs_new, bins=[-0.001,0.30,0.60,1.001],
                   labels=["Low Risk","Medium Risk","High Risk"])

demo_results = pd.DataFrame({
    "Customer"         : [f"C{i+1:03d}" for i in range(len(new_customers))],
    "Tenure (months)"  : new_customers["tenure"].values,
    "Monthly Charges"  : new_customers["MonthlyCharges"].values,
    "Contract"         : new_customers["Contract"].values,
    "Churn Probability": np.round(probs_new, 4),
    "Prediction"       : ["Churn" if p else "No Churn" for p in preds_new],
    "Risk Tier"        : tiers_new,
})

print("\n" + "="*75)
print("   PREDICTION DEMO — NEW CUSTOMERS")
print("="*75)
print(demo_results.to_string(index=False))

print(f"\n\n{'='*55}")
print("   ALL VISUALISATIONS SAVED TO: reports/")
print("   ALL MODELS SAVED TO: models/")
print(f"{'='*55}")
print("\n✅  Customer Churn Forecasting pipeline complete!")
