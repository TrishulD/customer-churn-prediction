"""
evaluate.py
===========
Comprehensive model evaluation:
  - Accuracy, Precision, Recall, F1, ROC-AUC
  - Confusion matrix (absolute + normalised)
  - ROC curves (all models on one plot)
  - Feature importance (RF + XGB) & coefficient magnitude (LR)
  - Business insights: high-risk segments, revenue impact

Author  : Senior Data Scientist
Project : Customer Churn Forecasting
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
)

sys.path.insert(0, os.path.dirname(__file__))

REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
PALETTE    = {"Logistic Regression": "#3498db",
              "Random Forest"      : "#2ecc71",
              "XGBoost"            : "#e74c3c"}


# ──────────────────────────────────────────────────────────────────────────────
# 1. METRICS
# ──────────────────────────────────────────────────────────────────────────────
def compute_metrics(
    name     : str,
    model,
    X_test   : pd.DataFrame,
    y_test   : pd.Series,
    threshold: float = 0.5,
) -> dict:
    """
    Compute all classification metrics for a single model.

    Parameters
    ----------
    name      : Display name of the model.
    model     : Fitted sklearn-compatible estimator.
    X_test    : Test feature matrix.
    y_test    : True labels.
    threshold : Decision threshold for class assignment.

    Returns
    -------
    dict with all metric values + predicted probabilities.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "Model"    : name,
        "Accuracy" : round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall"   : round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1 Score" : round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC"  : round(roc_auc_score(y_test, y_prob), 4),
        "_y_prob"  : y_prob,
        "_y_pred"  : y_pred,
    }

    print(f"\n── {name} ─{'─'*(40-len(name))}")
    for k, v in metrics.items():
        if not k.startswith("_"):
            print(f"   {k:<12}: {v}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])}")

    return metrics


def evaluate_all_models(
    results : list[dict],
    X_test  : pd.DataFrame,
    y_test  : pd.Series,
) -> pd.DataFrame:
    """
    Evaluate all trained models and return a comparison DataFrame.
    """
    all_metrics = []
    for r in results:
        m = compute_metrics(r["name"], r["best_estimator"], X_test, y_test)
        all_metrics.append(m)

    metrics_df = pd.DataFrame(all_metrics).set_index("Model")
    # Drop private columns from display
    display_df = metrics_df[[c for c in metrics_df.columns if not c.startswith("_")]]

    print("\n" + "="*55)
    print("   MODEL COMPARISON TABLE")
    print("="*55)
    print(display_df.to_string())

    return metrics_df


# ──────────────────────────────────────────────────────────────────────────────
# 2. CONFUSION MATRICES
# ──────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrices(
    metrics_df: pd.DataFrame,
    y_test    : pd.Series,
) -> None:
    """Plot side-by-side normalised confusion matrices for all models."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    models = metrics_df.index.tolist()
    n      = len(models)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, models):
        y_pred = metrics_df.loc[name, "_y_pred"]
        cm     = confusion_matrix(y_test, y_pred, normalize="true")
        disp   = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format=".2%")
        ax.set_title(f"{name}\nROC-AUC: {metrics_df.loc[name,'ROC-AUC']:.4f}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.suptitle("Normalised Confusion Matrices — All Models",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "03_confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. ROC CURVES
# ──────────────────────────────────────────────────────────────────────────────
def plot_roc_curves(
    metrics_df: pd.DataFrame,
    y_test    : pd.Series,
) -> None:
    """Overlay ROC curves for all models on a single plot."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    for name in metrics_df.index:
        y_prob = metrics_df.loc[name, "_y_prob"]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc  = metrics_df.loc[name, "ROC-AUC"]
        color = PALETTE.get(name, None)
        ax.plot(fpr, tpr, lw=2.5, color=color,
                label=f"{name}  (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "04_roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. FEATURE IMPORTANCE
# ──────────────────────────────────────────────────────────────────────────────
def plot_feature_importance(
    results      : list[dict],
    feature_names: list[str],
    top_n        : int = 20,
) -> None:
    """
    Plot feature importances for tree-based models and absolute
    coefficients for Logistic Regression.
    """
    os.makedirs(REPORT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, len(results), figsize=(8 * len(results), 8))
    if len(results) == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        name  = r["name"]
        model = r["best_estimator"]

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            title_suffix = "Feature Importance (Gini)"
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
            title_suffix = "|Coefficient| Magnitude"
        else:
            continue

        feat_df = (
            pd.DataFrame({"Feature": feature_names, "Importance": importances})
            .sort_values("Importance", ascending=False)
            .head(top_n)
        )

        color = PALETTE.get(name, "#7f8c8d")
        bars  = ax.barh(feat_df["Feature"][::-1], feat_df["Importance"][::-1],
                        color=color, edgecolor="white", linewidth=0.5)
        ax.set_xlabel(title_suffix, fontsize=11)
        ax.set_title(f"{name}\n{title_suffix}", fontsize=12, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle(f"Top-{top_n} Feature Importances by Model",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "05_feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. METRICS BAR CHART
# ──────────────────────────────────────────────────────────────────────────────
def plot_metrics_comparison(metrics_df: pd.DataFrame) -> None:
    """Grouped bar chart comparing all metrics across models."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    display_cols = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    plot_df = metrics_df[display_cols].copy()

    fig, ax = plt.subplots(figsize=(12, 6))
    x      = np.arange(len(display_cols))
    width  = 0.25
    n      = len(plot_df)

    for i, (name, row) in enumerate(plot_df.iterrows()):
        offset = (i - n // 2) * width + (width / 2 if n % 2 == 0 else 0)
        bars   = ax.bar(x + offset, row[display_cols], width,
                        label=name, color=list(PALETTE.values())[i],
                        edgecolor="white", linewidth=0.8)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(display_cols, fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison — All Metrics",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "06_metrics_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. BUSINESS INSIGHTS
# ──────────────────────────────────────────────────────────────────────────────
def generate_business_insights(
    best_model,
    X_test         : pd.DataFrame,
    y_test         : pd.Series,
    original_df    : pd.DataFrame,
    avg_revenue    : float = 65.0,    # approximate IBM Telco ARPU (monthly)
    retention_cost : float = 10.0,    # cost to retain one customer
) -> pd.DataFrame:
    """
    Attach churn probability scores to the test set and surface actionable
    business intelligence.

    Parameters
    ----------
    best_model     : Best performing trained model.
    X_test         : Test feature matrix.
    y_test         : True labels for test set.
    original_df    : Clean (pre-encoded) DataFrame for readable segment labels.
    avg_revenue    : Average monthly revenue per customer (USD).
    retention_cost : Estimated cost to retain one customer (USD).

    Returns
    -------
    pd.DataFrame : Scored test customers with risk tier.
    """
    print("\n" + "="*60)
    print("   BUSINESS INSIGHTS")
    print("="*60)

    # Score test set
    churn_prob = best_model.predict_proba(X_test)[:, 1]
    scored     = X_test.copy()
    scored["churn_probability"] = churn_prob
    scored["actual_churn"]      = y_test.values

    # Risk tiers
    scored["risk_tier"] = pd.cut(
        scored["churn_probability"],
        bins   = [0, 0.30, 0.60, 1.0],
        labels = ["Low Risk", "Medium Risk", "High Risk"],
    )

    tier_counts = scored["risk_tier"].value_counts().sort_index()
    print("\n── Risk Tier Distribution ───────────────────────────────")
    for tier, cnt in tier_counts.items():
        pct = cnt / len(scored) * 100
        print(f"   {tier:<15}: {cnt:>5} customers ({pct:.1f}%)")

    # Revenue at risk (high-risk segment)
    high_risk_n = tier_counts.get("High Risk", 0)
    rev_at_risk = high_risk_n * avg_revenue
    retention_investment = high_risk_n * retention_cost
    revenue_saved = rev_at_risk - retention_investment

    print("\n── Revenue Impact ────────────────────────────────────────")
    print(f"   High-risk customers        : {high_risk_n}")
    print(f"   Monthly revenue at risk    : ${rev_at_risk:,.0f}")
    print(f"   Est. retention investment  : ${retention_investment:,.0f}")
    print(f"   Net revenue protected      : ${revenue_saved:,.0f}")

    print("\n── Top Churn Drivers (domain knowledge) ─────────────────")
    drivers = [
        ("Month-to-month contract",
         "No lock-in means effortless switch; highest predictor of churn."),
        ("Short tenure (< 12 months)",
         "Early-lifecycle customers haven't yet built switching inertia."),
        ("High MonthlyCharges without bundled services",
         "Price-to-value mismatch triggers cost-conscious churn."),
        ("No OnlineSecurity / TechSupport",
         "Customers with fewer protective services have lower perceived value."),
        ("Paperless billing",
         "Correlates with digitally-savvy users who comparison-shop online."),
        ("Fiber optic internet",
         "Higher-paying segment is also higher-expectation; more likely to switch."),
    ]
    for drv, reason in drivers:
        print(f"\n   ▸ {drv}")
        print(f"     → {reason}")

    print("\n── Retention Strategies ──────────────────────────────────")
    strategies = [
        "Offer M2M customers a 10–15 % discount to switch to annual contracts.",
        "Target 0–6 month cohort with onboarding excellence and free service bundles.",
        "Create a 'value bundle' (Security + Backup + Support) for high-spend customers.",
        "Proactive outreach: call high-risk Fiber Optic users 30 days before renewal.",
        "Loyalty rewards after 12 / 24 / 36 month milestones to build switching friction.",
        "Personalised pricing: discount MonthlyCharges for customers above 75th percentile.",
    ]
    for i, s in enumerate(strategies, 1):
        print(f"   {i}. {s}")

    return scored


# ──────────────────────────────────────────────────────────────────────────────
# STANDALONE EXECUTION
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from preprocessing       import load_data, clean_data, encode_features, split_data
    from feature_engineering import engineer_features, apply_smote
    from train               import run_training_pipeline, load_model

    raw        = load_data()
    cleaned    = clean_data(raw)
    engineered = engineer_features(cleaned)
    encoded    = encode_features(engineered)
    X_tr, X_te, y_tr, y_te, feats = split_data(encoded)
    X_tr_bal, y_tr_bal = apply_smote(X_tr, y_tr)

    results    = run_training_pipeline(X_tr_bal, y_tr_bal, n_iter=10)
    metrics_df = evaluate_all_models(results, X_te, y_te)

    plot_confusion_matrices(metrics_df, y_te)
    plot_roc_curves(metrics_df, y_te)
    plot_feature_importance(results, feats)
    plot_metrics_comparison(metrics_df)

    # Pick best model by ROC-AUC
    best_name  = metrics_df["ROC-AUC"].idxmax()
    best_result = next(r for r in results if r["name"] == best_name)
    scored_df  = generate_business_insights(
        best_result["best_estimator"], X_te, y_te, cleaned
    )
    print("\n[DONE] Evaluation complete.")
