"use client";

import React, { useState } from "react";
import {
  User,
  Wifi,
  CreditCard,
  Sparkles,
  AlertCircle,
  HelpCircle,
  RefreshCw,
  Zap,
} from "lucide-react";
import PredictionResultModal, {
  PredictionData,
} from "@/components/PredictionResultModal";

const PRESET_PROFILES = {
  highRisk: {
    label: "High Churn Risk Profile",
    desc: "1 mo tenure, Month-to-month, Fiber optic, Electronic check",
    data: {
      customerID: "PRESET-HIGH-01",
      gender: "Female",
      SeniorCitizen: 0,
      Partner: "No",
      Dependents: "No",
      tenure: 1,
      PhoneService: "Yes",
      MultipleLines: "No",
      InternetService: "Fiber optic",
      OnlineSecurity: "No",
      OnlineBackup: "No",
      DeviceProtection: "No",
      TechSupport: "No",
      StreamingTV: "Yes",
      StreamingMovies: "Yes",
      Contract: "Month-to-month",
      PaperlessBilling: "Yes",
      PaymentMethod: "Electronic check",
      MonthlyCharges: 95.65,
      TotalCharges: 95.65,
    },
  },
  loyal: {
    label: "Loyal Customer Profile",
    desc: "60 mo tenure, 2-Year Contract, Bundled Security & Support",
    data: {
      customerID: "PRESET-LOYAL-02",
      gender: "Male",
      SeniorCitizen: 0,
      Partner: "Yes",
      Dependents: "Yes",
      tenure: 60,
      PhoneService: "Yes",
      MultipleLines: "Yes",
      InternetService: "DSL",
      OnlineSecurity: "Yes",
      OnlineBackup: "Yes",
      DeviceProtection: "Yes",
      TechSupport: "Yes",
      StreamingTV: "No",
      StreamingMovies: "No",
      Contract: "Two year",
      PaperlessBilling: "No",
      PaymentMethod: "Credit card (automatic)",
      MonthlyCharges: 65.0,
      TotalCharges: 3900.0,
    },
  },
  midRisk: {
    label: "Mid-Risk Profile",
    desc: "14 mo tenure, 1-Year Contract, No Tech Support",
    data: {
      customerID: "PRESET-MID-03",
      gender: "Female",
      SeniorCitizen: 0,
      Partner: "Yes",
      Dependents: "No",
      tenure: 14,
      PhoneService: "Yes",
      MultipleLines: "Yes",
      InternetService: "Fiber optic",
      OnlineSecurity: "No",
      OnlineBackup: "Yes",
      DeviceProtection: "No",
      TechSupport: "No",
      StreamingTV: "Yes",
      StreamingMovies: "No",
      Contract: "Month-to-month",
      PaperlessBilling: "Yes",
      PaymentMethod: "Bank transfer (automatic)",
      MonthlyCharges: 82.5,
      TotalCharges: 1155.0,
    },
  },
};

export default function PredictPage() {
  const [activeTab, setActiveTab] = useState<"demographics" | "services" | "billing">("demographics");
  const [model, setModel] = useState<string>("xgboost");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionData | null>(null);

  const [formData, setFormData] = useState({
    customerID: "CUST-1049",
    gender: "Female",
    SeniorCitizen: 0,
    Partner: "No",
    Dependents: "No",
    tenure: 1,
    PhoneService: "Yes",
    MultipleLines: "No",
    InternetService: "Fiber optic",
    OnlineSecurity: "No",
    OnlineBackup: "No",
    DeviceProtection: "No",
    TechSupport: "No",
    StreamingTV: "Yes",
    StreamingMovies: "Yes",
    Contract: "Month-to-month",
    PaperlessBilling: "Yes",
    PaymentMethod: "Electronic check",
    MonthlyCharges: 89.9,
    TotalCharges: 89.9,
  });

  const handleChange = (field: string, val: any) => {
    setFormData((prev) => {
      const updated = { ...prev, [field]: val };
      // Auto-recalculate TotalCharges when tenure or MonthlyCharges update if not manually edited
      if (field === "tenure" || field === "MonthlyCharges") {
        const t = field === "tenure" ? Number(val) : Number(prev.tenure);
        const m = field === "MonthlyCharges" ? Number(val) : Number(prev.MonthlyCharges);
        updated.TotalCharges = Math.round(m * Math.max(t, 1) * 100) / 100;
      }
      return updated;
    });
  };

  const applyPreset = (key: keyof typeof PRESET_PROFILES) => {
    setFormData(PRESET_PROFILES[key].data as any);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`/api/predict?model=${model}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.message || "Failed to score customer");
      }

      setPrediction(data);
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred during prediction.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-5xl px-4 py-12 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
          Customer Churn Risk Scoring
        </h1>
        <p className="mt-2 text-sm text-slate-400">
          Enter customer attributes to compute real-time churn probability, risk
          tier, and specific retention directives.
        </p>
      </div>

      {/* Preset Quick Loader */}
      <div className="mt-8 rounded-2xl border border-slate-800 bg-slate-900/60 p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <span className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-slate-400">
            <Zap className="h-4 w-4 text-amber-400" />
            Quick Presets:
          </span>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => applyPreset("highRisk")}
              className="rounded-lg border border-rose-500/30 bg-rose-950/40 px-3 py-1 text-xs font-medium text-rose-300 transition-colors hover:bg-rose-900/50"
            >
              High Risk Preset
            </button>
            <button
              type="button"
              onClick={() => applyPreset("midRisk")}
              className="rounded-lg border border-amber-500/30 bg-amber-950/40 px-3 py-1 text-xs font-medium text-amber-300 transition-colors hover:bg-amber-900/50"
            >
              Medium Risk Preset
            </button>
            <button
              type="button"
              onClick={() => applyPreset("loyal")}
              className="rounded-lg border border-emerald-500/30 bg-emerald-950/40 px-3 py-1 text-xs font-medium text-emerald-300 transition-colors hover:bg-emerald-900/50"
            >
              Loyal Customer Preset
            </button>
          </div>
        </div>
      </div>

      {/* Model Selection Bar */}
      <div className="mt-6 flex flex-wrap items-center justify-between gap-4 rounded-xl border border-slate-800 bg-slate-900/40 px-5 py-3">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-blue-400" />
          <span className="text-xs font-medium text-slate-300">Inference Estimator:</span>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs font-semibold text-white focus:border-blue-500 focus:outline-none"
          >
            <option value="xgboost">XGBoost (Recommended &bull; 0.868 AUC)</option>
            <option value="random_forest">Random Forest (0.856 AUC)</option>
            <option value="logistic_regression">Logistic Regression (0.845 AUC)</option>
          </select>
        </div>
      </div>

      {error && (
        <div className="mt-6 flex items-center gap-3 rounded-xl border border-rose-500/30 bg-rose-950/40 p-4 text-sm text-rose-300">
          <AlertCircle className="h-5 w-5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Form Container */}
      <form onSubmit={handleSubmit} className="mt-8">
        {/* Navigation Tabs */}
        <div className="flex border-b border-slate-800">
          <button
            type="button"
            onClick={() => setActiveTab("demographics")}
            className={`flex items-center gap-2 border-b-2 px-6 py-3 text-sm font-semibold transition-colors ${
              activeTab === "demographics"
                ? "border-blue-500 text-blue-400"
                : "border-transparent text-slate-400 hover:text-slate-200"
            }`}
          >
            <User className="h-4 w-4" />
            1. Demographics
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("services")}
            className={`flex items-center gap-2 border-b-2 px-6 py-3 text-sm font-semibold transition-colors ${
              activeTab === "services"
                ? "border-blue-500 text-blue-400"
                : "border-transparent text-slate-400 hover:text-slate-200"
            }`}
          >
            <Wifi className="h-4 w-4" />
            2. Telecom Services
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("billing")}
            className={`flex items-center gap-2 border-b-2 px-6 py-3 text-sm font-semibold transition-colors ${
              activeTab === "billing"
                ? "border-blue-500 text-blue-400"
                : "border-transparent text-slate-400 hover:text-slate-200"
            }`}
          >
            <CreditCard className="h-4 w-4" />
            3. Account & Billing
          </button>
        </div>

        {/* Tab 1: Demographics */}
        {activeTab === "demographics" && (
          <div className="grid gap-6 rounded-b-2xl border border-t-0 border-slate-800 bg-slate-900/40 p-6 sm:grid-cols-2">
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400">
                Customer Identifier
              </label>
              <input
                type="text"
                value={formData.customerID}
                onChange={(e) => handleChange("customerID", e.target.value)}
                className="mt-2 w-full rounded-lg border border-slate-800 bg-slate-950 px-3.5 py-2.5 text-sm text-white focus:border-blue-500 focus:outline-none"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400">
                Gender
              </label>
              <select
                value={formData.gender}
                onChange={(e) => handleChange("gender", e.target.value)}
                className="mt-2 w-full rounded-lg border border-slate-800 bg-slate-950 px-3.5 py-2.5 text-sm text-white focus:border-blue-500 focus:outline-none"
              >
                <option value="Female">Female</option>
                <option value="Male">Male</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400">
                Senior Citizen Status
              </label>
              <select
                value={formData.SeniorCitizen}
                onChange={(e) => handleChange("SeniorCitizen", Number(e.target.value))}
                className="mt-2 w-full rounded-lg border border-slate-800 bg-slate-950 px-3.5 py-2.5 text-sm text-white focus:border-blue-500 focus:outline-none"
              >
                <option value={0}>No (Under 65)</option>
                <option value={1}>Yes (Senior Citizen)</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400">
                Has Partner
              </label>
              <select
                value={formData.Partner}
                onChange={(e) => handleChange("Partner", e.target.value)}
                className="mt-2 w-full rounded-lg border border-slate-800 bg-slate-950 px-3.5 py-2.5 text-sm text-white focus:border-blue-500 focus:outline-none"
              >
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400">
                Has Dependents
              </label>
              <select
                value={formData.Dependents}
                onChange={(e) => handleChange("Dependents", e.target.value)}
                className="mt-2 w-full rounded-lg border border-slate-800 bg-slate-950 px-3.5 py-2.5 text-sm text-white focus:border-blue-500 focus:outline-none"
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
            </div>

            <div className="flex items-end justify-end sm:col-span-2">
              <button
                type="button"
                onClick={() => setActiveTab("services")}
                className="rounded-lg bg-slate-800 px-5 py-2.5 text-sm font-semibold text-white hover:bg-slate-700"
              >
                Next: Telecom Services &rarr;
              </button>
            </div>
          </div>
        )}

        {/* Tab 2: Telecom Services */}
        {activeTab === "services" && (
          <div className="grid gap-6 rounded-b-2xl border border-t-0 border-slate-800 bg-slate-900/40 p-6 sm:grid-cols-3">
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400">
                Phone Service
              </label>
              <select
                value={formData.PhoneService}
                onChange={(e) => handleChange("PhoneService", e.target.value)}
                className="mt-2 w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-sm text-white focus:border-blue-500 focus:outline-none"
              >
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400">
                Multiple Lines
              </label>
              <select
                value={formData.MultipleLines}
                onChange={(e) => handleChange("MultipleLines", e.target.value)}
                className="mt-2 w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-sm text-white focus:border-blue-500 focus:outline-none"
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
                <option value="No phone service">No phone service</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400">
                Internet Service
              </label>
              <select
                value={formData.InternetService}
                onChange={(e) => handleChange("InternetService", e.target.value)}
                className="mt-2 w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-sm text-white focus:border-blue-500 focus:outline-none"
              >
                <option value="Fiber optic">Fiber optic</option>
                <option value="DSL">DSL</option>
                <option value="No">No</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400">
                Online Security
              </label>
              <select
                value={formData.OnlineSecurity}
                onChange={(e) => handleChange("OnlineSecurity", e.target.value)}
                className="mt-2 w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-sm text-white focus:border-blue-500 focus:outline-none"
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
                <option value="No internet service">No internet service</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400">
                Online Backup
              </label>
              <select
                value={formData.OnlineBackup}
                onChange={(e) => handleChange("OnlineBackup", e.target.value)}
                className="mt-2 w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-sm text-white focus:border-blue-500 focus:outline-none"
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
                <option value="No internet service">No internet service</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400">
                Device Protection
              </label>
              <select
                value={formData.DeviceProtection}
                onChange={(e) => handleChange("DeviceProtection", e.target.value)}
                className="mt-2 w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-sm text-white focus:border-blue-500 focus:outline-none"
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
                <option value="No internet service">No internet service</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400">
                Tech Support
              </label>
              <select
                value={formData.TechSupport}
                onChange={(e) => handleChange("TechSupport", e.target.value)}
                className="mt-2 w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-sm text-white focus:border-blue-500 focus:outline-none"
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
                <option value="No internet service">No internet service</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400">
                Streaming TV
              </label>
              <select
                value={formData.StreamingTV}
                onChange={(e) => handleChange("StreamingTV", e.target.value)}
                className="mt-2 w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-sm text-white focus:border-blue-500 focus:outline-none"
              >
                <option value="Yes">Yes</option>
                <option value="No">No</option>
                <option value="No internet service">No internet service</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400">
                Streaming Movies
              </label>
              <select
                value={formData.StreamingMovies}
                onChange={(e) => handleChange("StreamingMovies", e.target.value)}
                className="mt-2 w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-sm text-white focus:border-blue-500 focus:outline-none"
              >
                <option value="Yes">Yes</option>
                <option value="No">No</option>
                <option value="No internet service">No internet service</option>
              </select>
            </div>

            <div className="flex items-end justify-between sm:col-span-3">
              <button
                type="button"
                onClick={() => setActiveTab("demographics")}
                className="rounded-lg border border-slate-700 bg-slate-800 px-4 py-2 text-xs font-semibold text-slate-300 hover:bg-slate-700"
              >
                &larr; Back
              </button>
              <button
                type="button"
                onClick={() => setActiveTab("billing")}
                className="rounded-lg bg-slate-800 px-5 py-2.5 text-sm font-semibold text-white hover:bg-slate-700"
              >
                Next: Account & Billing &rarr;
              </button>
            </div>
          </div>
        )}

        {/* Tab 3: Account & Billing */}
        {activeTab === "billing" && (
          <div className="grid gap-6 rounded-b-2xl border border-t-0 border-slate-800 bg-slate-900/40 p-6 sm:grid-cols-2">
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400">
                Contract Agreement
              </label>
              <select
                value={formData.Contract}
                onChange={(e) => handleChange("Contract", e.target.value)}
                className="mt-2 w-full rounded-lg border border-slate-800 bg-slate-950 px-3.5 py-2.5 text-sm text-white focus:border-blue-500 focus:outline-none"
              >
                <option value="Month-to-month">Month-to-month (High Churn Risk)</option>
                <option value="One year">One year</option>
                <option value="Two year">Two year (High Loyalty)</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400">
                Payment Method
              </label>
              <select
                value={formData.PaymentMethod}
                onChange={(e) => handleChange("PaymentMethod", e.target.value)}
                className="mt-2 w-full rounded-lg border border-slate-800 bg-slate-950 px-3.5 py-2.5 text-sm text-white focus:border-blue-500 focus:outline-none"
              >
                <option value="Electronic check">Electronic check</option>
                <option value="Mailed check">Mailed check</option>
                <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
                <option value="Credit card (automatic)">Credit card (automatic)</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400">
                Paperless Billing
              </label>
              <select
                value={formData.PaperlessBilling}
                onChange={(e) => handleChange("PaperlessBilling", e.target.value)}
                className="mt-2 w-full rounded-lg border border-slate-800 bg-slate-950 px-3.5 py-2.5 text-sm text-white focus:border-blue-500 focus:outline-none"
              >
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400">
                Tenure (Months): <span className="text-white">{formData.tenure}</span>
              </label>
              <input
                type="number"
                min="0"
                max="72"
                value={formData.tenure}
                onChange={(e) => handleChange("tenure", Number(e.target.value))}
                className="mt-2 w-full rounded-lg border border-slate-800 bg-slate-950 px-3.5 py-2.5 text-sm text-white focus:border-blue-500 focus:outline-none"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400">
                Monthly Charges ($)
              </label>
              <input
                type="number"
                step="0.05"
                min="10"
                max="150"
                value={formData.MonthlyCharges}
                onChange={(e) => handleChange("MonthlyCharges", Number(e.target.value))}
                className="mt-2 w-full rounded-lg border border-slate-800 bg-slate-950 px-3.5 py-2.5 text-sm text-white focus:border-blue-500 focus:outline-none"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400">
                Total Charges ($)
              </label>
              <input
                type="number"
                step="0.05"
                min="0"
                value={formData.TotalCharges}
                onChange={(e) => handleChange("TotalCharges", Number(e.target.value))}
                className="mt-2 w-full rounded-lg border border-slate-800 bg-slate-950 px-3.5 py-2.5 text-sm text-white focus:border-blue-500 focus:outline-none"
              />
            </div>

            <div className="flex items-center justify-between sm:col-span-2">
              <button
                type="button"
                onClick={() => setActiveTab("services")}
                className="rounded-lg border border-slate-700 bg-slate-800 px-4 py-2 text-xs font-semibold text-slate-300 hover:bg-slate-700"
              >
                &larr; Back to Services
              </button>
            </div>
          </div>
        )}

        {/* Submit Bar */}
        <div className="mt-8 flex justify-center">
          <button
            type="submit"
            disabled={loading}
            className="flex items-center gap-2.5 rounded-xl bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 px-8 py-4 text-base font-bold text-white shadow-xl shadow-blue-600/30 transition-all hover:opacity-95 hover:shadow-blue-500/40 disabled:opacity-50"
          >
            {loading ? (
              <>
                <RefreshCw className="h-5 w-5 animate-spin" />
                <span>Running Inference Pipeline...</span>
              </>
            ) : (
              <>
                <Sparkles className="h-5 w-5" />
                <span>Predict Churn & Generate Retention Plan</span>
              </>
            )}
          </button>
        </div>
      </form>

      {/* Result Modal */}
      {prediction && (
        <PredictionResultModal
          prediction={prediction}
          formData={formData}
          onClose={() => setPrediction(null)}
        />
      )}
    </div>
  );
}
