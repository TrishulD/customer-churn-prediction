"""
train.py
========
End-to-end model training pipeline:
  - Logistic Regression
  - Random Forest
  - XGBoost
with hyperparameter tuning via RandomizedSearchCV and model persistence.

Author  : Senior Data Scientist
Project : Customer Churn Forecasting
"""

import os
import sys
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost                import XGBClassifier

warnings.filterwarnings("ignore")

# ── Add src/ to path when run directly ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing      import load_data, clean_data, encode_features, split_data
from feature_engineering import engineer_features, apply_smote

MODELS_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")
RANDOM_STATE = 42
CV_FOLDS     = 5
N_ITER       = 30          # RandomizedSearch iterations per model


# ──────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS + HYPERPARAMETER GRIDS
# ──────────────────────────────────────────────────────────────────────────────
def get_model_configs() -> dict:
    """
    Return a dictionary of {name: (estimator, param_distribution)} pairs.

    Hyperparameter rationale
    ────────────────────────
    Logistic Regression : regularisation strength C and solver choice
    Random Forest       : tree count, depth, split criteria to reduce variance
    XGBoost             : learning rate, depth, subsampling to prevent overfitting
    """
    configs = {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                class_weight="balanced",   # handles residual imbalance
            ),
            {
                "C"      : uniform(0.01, 10),
                "solver" : ["lbfgs", "liblinear"],
                "penalty": ["l2"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced",
            ),
            {
                "n_estimators"     : randint(100, 500),
                "max_depth"        : [None, 5, 10, 15, 20],
                "min_samples_split": randint(2, 20),
                "min_samples_leaf" : randint(1, 10),
                "max_features"     : ["sqrt", "log2"],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                use_label_encoder=False,
                n_jobs=-1,
            ),
            {
                "n_estimators"      : randint(100, 500),
                "max_depth"         : randint(3, 10),
                "learning_rate"     : uniform(0.01, 0.3),
                "subsample"         : uniform(0.6, 0.4),
                "colsample_bytree"  : uniform(0.6, 0.4),
                "min_child_weight"  : randint(1, 10),
                "gamma"             : uniform(0, 0.5),
                "reg_alpha"         : uniform(0, 1),
                "reg_lambda"        : uniform(1, 2),
            },
        ),
    }
    return configs


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING WITH HYPERPARAMETER TUNING
# ──────────────────────────────────────────────────────────────────────────────
def train_model(
    name       : str,
    estimator,
    param_dist : dict,
    X_train    : pd.DataFrame,
    y_train    : pd.Series,
    n_iter     : int = N_ITER,
    cv_folds   : int = CV_FOLDS,
) -> dict:
    """
    Tune a single model via RandomizedSearchCV and return results.

    Parameters
    ----------
    name       : Human-readable model name.
    estimator  : Sklearn-compatible estimator object.
    param_dist : Hyperparameter search distribution.
    X_train    : SMOTE-resampled training features.
    y_train    : SMOTE-resampled training labels.
    n_iter     : Number of random parameter draws.
    cv_folds   : StratifiedKFold splits for cross-validation.

    Returns
    -------
    dict : {name, best_estimator, best_params, best_cv_score, train_time}
    """
    print(f"\n{'='*60}")
    print(f"  Training  : {name}")
    print(f"  CV Folds  : {cv_folds}   |   Iterations : {n_iter}")
    print(f"{'='*60}")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator          = estimator,
        param_distributions= param_dist,
        n_iter             = n_iter,
        scoring            = "roc_auc",    # primary metric for imbalanced data
        cv                 = cv,
        n_jobs             = -1,
        verbose            = 1,
        random_state       = RANDOM_STATE,
        refit              = True,
    )

    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"\n  ✔ Best CV ROC-AUC : {search.best_score_:.4f}")
    print(f"  ✔ Time taken      : {elapsed:.1f}s")
    print(f"  ✔ Best params     :")
    for k, v in search.best_params_.items():
        print(f"      {k}: {v}")

    return {
        "name"          : name,
        "best_estimator": search.best_estimator_,
        "best_params"   : search.best_params_,
        "best_cv_score" : search.best_score_,
        "train_time"    : elapsed,
    }


# ──────────────────────────────────────────────────────────────────────────────
# SAVE / LOAD HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def save_model(result: dict) -> str:
    """Persist a trained estimator to disk using joblib."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    fname = result["name"].lower().replace(" ", "_") + ".pkl"
    path  = os.path.join(MODELS_DIR, fname)
    joblib.dump(result["best_estimator"], path, compress=3)
    print(f"[SAVED] {result['name']} → {path}")
    return path


def load_model(model_name: str):
    """Load a persisted model by name (e.g. 'random_forest')."""
    fname = model_name.lower().replace(" ", "_") + ".pkl"
    path  = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    print(f"[LOADED] {model_name} from {path}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# MASTER TRAINING PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def run_training_pipeline(
    X_train : pd.DataFrame,
    y_train : pd.Series,
    n_iter  : int = N_ITER,
) -> list[dict]:
    """
    Train all three models, save them, and return results list.

    Parameters
    ----------
    X_train : SMOTE-balanced feature matrix.
    y_train : SMOTE-balanced target vector.
    n_iter  : RandomizedSearchCV iterations.

    Returns
    -------
    List of result dicts (one per model).
    """
    configs = get_model_configs()
    results = []

    for name, (est, params) in configs.items():
        res = train_model(name, est, params, X_train, y_train, n_iter=n_iter)
        save_model(res)
        results.append(res)

    # Summary table
    summary = pd.DataFrame([
        {
            "Model"        : r["name"],
            "CV ROC-AUC"   : round(r["best_cv_score"], 4),
            "Train Time(s)": round(r["train_time"], 1),
        }
        for r in results
    ]).sort_values("CV ROC-AUC", ascending=False)

    print("\n" + "="*50)
    print("   TRAINING SUMMARY")
    print("="*50)
    print(summary.to_string(index=False))

    return results


# ──────────────────────────────────────────────────────────────────────────────
# STANDALONE EXECUTION
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    raw        = load_data()
    cleaned    = clean_data(raw)
    engineered = engineer_features(cleaned)
    encoded    = encode_features(engineered)
    X_tr, X_te, y_tr, y_te, feats = split_data(encoded)
    X_tr_bal, y_tr_bal = apply_smote(X_tr, y_tr)

    results = run_training_pipeline(X_tr_bal, y_tr_bal)
    print("\n[DONE] All models trained and saved.")
