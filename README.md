# 📊 Customer Churn Forecasting & Retention Platform

> **Production-grade AI/ML SaaS application to predict, explain, and prevent customer churn using an optimized XGBoost pipeline (0.868 ROC-AUC), Next.js, TypeScript, Tailwind CSS, FastAPI, and Supabase.**

---

## 📌 Table of Contents
1. [Executive Summary](#-executive-summary)
2. [Key Performance Benchmarks](#-key-performance-benchmarks)
3. [Architecture Overview](#-architecture-overview)
4. [ML Pipeline & Bug Fixes](#-ml-pipeline--bug-fixes)
5. [Frontend SaaS Application Features](#-frontend-saas-application-features)
6. [Project Structure](#-project-structure)
7. [Environment Variables](#-environment-variables)
8. [Supabase Setup Guide](#-supabase-setup-guide)
9. [Local Development Quickstart](#-local-development-quickstart)
10. [Deployment to Vercel & Production Inference](#-deployment-to-vercel--production-inference)
11. [License](#-license)

---

## 🎯 Executive Summary

Customer churn costs the telecommunications industry an estimated **$62 billion annually**. This platform delivers an end-to-end machine learning system that:

- **Predicts** customer churn probability with high precision and discriminative power (**0.868 ROC-AUC**).
- **Explains** root causes using true empirical tree feature importances (e.g. month-to-month contracts, lack of security add-ons, price sensitivity).
- **Segments** subscribers into calibrated risk tiers (Low Risk <30%, Medium Risk 30–60%, High Risk >60%).
- **Prescribes** personalized retention playbooks (e.g., annual contract upgrades, "Value Shield" bundles, VIP onboarding).
- **Exports** executive-ready, client-side **PDF Assessment Reports** with zero server overhead.

---

## 📈 Key Performance Benchmarks

Trained on the **IBM Telco Customer Churn dataset** (7,043 subscribers, 21 initial features) with stratified 5-fold cross-validation and SMOTE class balancing (0.5 ratio):

| Model Architecture | ROC-AUC | Test Accuracy | Precision | Recall | F1 Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier** | **0.868** | **82.0%** | **67.4%** | **59.8%** | **0.634** | 🏆 **Production Model** |
| Random Forest | 0.856 | 81.2% | 66.1% | 57.9% | 0.617 | Supported Estimator |
| Logistic Regression | 0.845 | 80.0% | 63.6% | 56.8% | 0.600 | Linear Baseline |

### Risk Segmentation & Revenue Impact

| Risk Tier | Churn Probability | Base % | Action Strategy |
| :--- | :---: | :---: | :--- |
| 🟢 **Low Risk** | `< 30%` | ~55% | Maintain engagement; loyalty milestones; upsell |
| 🟡 **Medium Risk** | `30% – 60%` | ~25% | Proactive outreach; discounted "Value Shield" bundles |
| 🔴 **High Risk** | `> 60%` | ~20% | Urgent intervention; annual contract upgrade discount |

---

## 🏛️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      VERCEL HOSTING                         │
│                                                             │
│   Next.js 14 App (TypeScript + Tailwind CSS + Lucide)        │
│   ├── Landing Page & Model Showcase                         │
│   ├── Interactive Customer Churn Predictor Form             │
│   ├── Prediction Results, Risk Gauges & Recommendations     │
│   ├── Client-side PDF Report Generator (jsPDF)              │
│   ├── Feedback Submission Form                              │
│   ├── Protected Admin Analytics Dashboard                   │
│   └── Next.js API Routes (/api/predict, /api/feedback, ...) │
└──────────────┬───────────────────────────────┬──────────────┘
               │                               │
               ▼                               ▼
┌──────────────────────────────┐ ┌────────────────────────────┐
│   SUPABASE DATABASE          │ │  PYTHON ML INFERENCE API   │
│   ├── predictions table      │ │  (FastAPI + Uvicorn)       │
│   ├── feedbacks table        │ │  ├── Serves existing .pkl  │
│   └── RLS & secure API keys  │ │  │   XGBoost, RF, LR models│
└──────────────────────────────┘ │  ├── Standardized Scaler   │
                                 │  └── Feature Pipeline Fix  │
                                 └────────────────────────────┘
```

---

## 🛠️ ML Pipeline & Bug Fixes

The original repository contained trained model weights (`xgboost.pkl`, `random_forest.pkl`, `logistic_regression.pkl`, `scaler.pkl`), but had critical inference pipeline discrepancies that were identified and resolved:

### Resolved Discrepancies:
1. **Feature Engineering Omission**: `predict.py` previously missed 5 out of 6 domain features (`service_count`, `charges_per_service`, `has_multiple_services`, `is_high_value`, `contract_risk_score`). All features are now faithfully calculated during inference matching training.
2. **Missing StandardScaler Transformation**: Continuous features (`tenure`, `MonthlyCharges`, `TotalCharges`) were previously passed unscaled to models trained on zero-mean data. Inference now loads and applies `models/scaler.pkl`.
3. **Dynamic Dummy Encoding Crash**: `encode_features` relied on multi-row variance (`nunique() == 2`). Single customer inference received 1 row, leaving categorical variables unencoded. Fixed with deterministic feature mapping.
4. **Target Dependency in Cleaning**: `clean_data` crashed when `Churn` was missing in live inference requests. Made target extraction conditional.
5. **Windows Console Unicode Errors**: Replaced unsupported characters (`\u2190`) with ASCII-safe arrows (`<-`).
6. **Cross-Version Scikit-Learn Compatibility**: Fixed `multi_class` attribute handling for `LogisticRegression` objects across scikit-learn versions.

---

## 💻 Frontend SaaS Application Features

1. **Enterprise Landing Page (`/`)**:
   - Hero banner with live performance badges (0.868 ROC-AUC, 82% accuracy).
   - Side-by-side benchmark comparison cards.
   - Interactive quick-start presets.
2. **Interactive Customer Scoring (`/predict`)**:
   - Organized into 3 intuitive tabs:
     - **Demographics**: Gender, Senior Citizen, Partner, Dependents.
     - **Telecom Services**: Phone, Multiple Lines, Internet (Fiber/DSL), Security, Backup, Protection, Tech Support, Streaming.
     - **Account & Billing**: Contract, Paperless, Payment Method, Tenure, Monthly Charges, Total Charges.
   - Quick Preset Buttons: "High Churn Risk", "Medium Risk", "Loyal Customer".
   - Live model selector: Choose between XGBoost, Random Forest, or Logistic Regression.
3. **Prediction Result & Decision Support Modal**:
   - Visual **Churn Probability Dial / Progress Bar** (0–100%).
   - Color-coded **Risk Tier Badge** (`LOW RISK`, `MEDIUM RISK`, `HIGH RISK`).
   - Authentic **Key Risk Drivers** derived from XGBoost Gini importance.
   - Actionable **Prescribed Retention Action**.
   - **One-Click Client-Side PDF Report Download** (`jsPDF`).
4. **Customer Feedback System**:
   - 1–5 Star rating, usefulness toggle, and retention comments stored in Supabase.
5. **Protected Admin Analytics Dashboard (`/admin`)**:
   - Passkey authentication protection.
   - Real-time KPI cards: Total Predictions, High Risk %, Medium Risk %, Low Risk %, Feedback Count, Avg Rating.
   - Portfolio risk tier distribution bar.
   - Searchable recent predictions log and customer feedback stream.
6. **Model Transparency & Architecture Page (`/about`)**:
   - Comprehensive documentation of the dataset, SMOTE oversampling, cross-validation, and top-10 feature importances.

---

## 🗂️ Project Structure

```
customer-churn-prediction/
├── app/                              # Next.js 14 App Router
│   ├── about/page.tsx                # Model transparency & methodology
│   ├── admin/page.tsx                # Protected admin analytics dashboard
│   ├── api/
│   │   ├── admin/stats/route.ts      # Admin KPIs & logs API
│   │   ├── feedback/route.ts         # Supabase feedback API
│   │   └── predict/route.ts          # Churn scoring API route
│   ├── globals.css                   # Tailwind styles & theme variables
│   ├── layout.tsx                    # Root layout & navigation shell
│   ├── page.tsx                      # Landing page & benchmark showcase
│   └── predict/page.tsx              # Interactive customer scoring form
├── components/                       # Reusable React components
│   ├── FeedbackSection.tsx           # Supabase feedback component
│   ├── Footer.tsx                    # Global footer
│   ├── Navbar.tsx                    # Header with brand & status badge
│   └── PredictionResultModal.tsx     # Result modal + jsPDF generator
├── data/                             # Raw dataset
│   └── telco_churn_raw.csv           # IBM Telco benchmark dataset
├── lib/                              # Shared utilities
│   └── supabase.ts                   # Supabase client & connection validator
├── models/                           # Trained model artifacts
│   ├── logistic_regression.pkl       # Logistic regression estimator
│   ├── random_forest.pkl             # Random forest estimator
│   ├── scaler.pkl                    # StandardScaler for tenure/charges
│   ├── trained_columns.txt           # 36 trained feature names
│   ├── xgboost.pkl                   # Best-performing XGBoost model
│   └── xgboost_model.json            # Portable Booster JSON model
├── notebooks/                        # Colab / Jupyter notebooks
│   └── Customer_Churn_Forecasting.py # Complete Colab-compatible pipeline
├── reports/                          # Generated charts and reports
│   ├── 01_churn_distribution.png
│   ├── 02_numeric_distributions.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_roc_curves.png
│   ├── 05_confusion_matrices.png
│   ├── 06_feature_importance.png
│   ├── 07_metrics_comparison.png
│   ├── 08_churn_probability_dist.png
│   └── PROJECT_REPORT.md             # Comprehensive data science report
├── src/                              # Python ML source code
│   ├── api.py                        # Production FastAPI inference microservice
│   ├── evaluate.py                   # Evaluation & chart generator
│   ├── feature_engineering.py        # Domain features + SMOTE
│   ├── predict.py                    # Inference pipeline & CLI
│   ├── preprocessing.py              # Data cleaning, encoding, scaling
│   ├── test_api.py                   # FastAPI automated test suite
│   └── train.py                      # Training & hyperparameter tuning
├── supabase/                         # Database migrations
│   └── schema.sql                    # SQL schema for predictions & feedback
├── .env.example                      # Environment variables template
├── next.config.js                    # Next.js configuration
├── package.json                      # Frontend dependencies
├── requirements.txt                  # Python dependencies
├── tailwind.config.ts                # Tailwind styling configuration
├── tsconfig.json                     # TypeScript compiler configuration
└── vercel.json                       # Vercel deployment configuration
```

---

## 🔐 Environment Variables

Copy `.env.example` to `.env.local`:

```bash
cp .env.example .env.local
```

| Variable | Description | Default / Example |
| :--- | :--- | :--- |
| `NEXT_PUBLIC_SUPABASE_URL` | Your Supabase project URL | `https://xyzcompany.supabase.co` |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Public Supabase anon key | `eyJhbGci...` |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service-role secret | `eyJhbGci...` |
| `ML_INFERENCE_URL` | URL of the Python FastAPI service | `http://127.0.0.1:8000` |
| `ADMIN_PASSWORD` | Passkey to unlock `/admin` dashboard | `admin123` |

---

## 🗄️ Supabase Setup Guide

1. Log in to [Supabase](https://supabase.com) and create a new project.
2. Open the **SQL Editor** tab.
3. Paste the contents of [`supabase/schema.sql`](supabase/schema.sql) and click **Run**.
   - Creates the `predictions` and `feedbacks` tables.
   - Configures Row Level Security (RLS) and query indexes.
4. Go to **Project Settings** > **API** and copy your:
   - **Project URL** -> `NEXT_PUBLIC_SUPABASE_URL`
   - **anon public key** -> `NEXT_PUBLIC_SUPABASE_ANON_KEY`
   - **service_role secret** -> `SUPABASE_SERVICE_ROLE_KEY`

---

## 🚀 Local Development Quickstart

### 1. Install Python Dependencies & Start ML API

```bash
# In your terminal
pip install -r requirements.txt

# Start the FastAPI inference microservice on port 8000
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```
API Documentation will be live at: `http://127.0.0.1:8000/docs`

### 2. Start Next.js Frontend

In a separate terminal:

```bash
# Install dependencies
npm install

# Start Next.js development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 🌐 Deployment to Vercel & Production Inference

### 1. Deploy Frontend to Vercel
1. Push this repository to your GitHub account.
2. Go to [Vercel](https://vercel.com) and import the repository.
3. In **Environment Variables**, add:
   - `NEXT_PUBLIC_SUPABASE_URL`
   - `NEXT_PUBLIC_SUPABASE_ANON_KEY`
   - `SUPABASE_SERVICE_ROLE_KEY`
   - `ADMIN_PASSWORD`
   - `ML_INFERENCE_URL` (set to your deployed Python API service URL)
4. Click **Deploy**. Vercel will build and deploy the Next.js application automatically.

### 2. Deploy Python ML Inference Service
Because XGBoost and Scikit-Learn exceed Vercel's serverless package size limits, host `src/api.py` on any container/Python platform:
- **Render.com / Railway / Fly.io / HuggingFace Spaces**:
  - Command: `uvicorn src.api:app --host 0.0.0.0 --port $PORT`
  - Python version: `3.10`+
  - Copy the deployed URL (e.g., `https://churn-api.onrender.com`) into Vercel's `ML_INFERENCE_URL` environment variable.

---

## 📄 License

- Dataset: IBM Telco Customer Churn ([Apache 2.0](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)).
- Project Source Code: [MIT License](LICENSE).
