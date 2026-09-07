import React from "react";
import Link from "next/link";
import {
  Award,
  Layers,
  CheckCircle2,
  TrendingUp,
  Cpu,
  ArrowRight,
  ShieldCheck,
  BookOpen,
} from "lucide-react";

export default function AboutPage() {
  const topFeatures = [
    { rank: 1, feature: "Contract (Month-to-month)", importance: "14.2%", direction: "Increases Churn", reason: "Zero switching cost; effortless departure." },
    { rank: 2, feature: "tenure", importance: "12.8%", direction: "Decreases Churn", reason: "Longer tenure builds switching inertia." },
    { rank: 3, feature: "MonthlyCharges", importance: "9.8%", direction: "Increases Churn", reason: "Price-sensitive subscribers comparison-shop competitors." },
    { rank: 4, feature: "TotalCharges", importance: "8.7%", direction: "Decreases Churn", reason: "Cumulative customer investment in platform." },
    { rank: 5, feature: "InternetService (Fiber optic)", importance: "7.4%", direction: "Increases Churn", reason: "Higher-cost, higher-expectation customer segment." },
    { rank: 6, feature: "avg_monthly_spend", importance: "6.8%", direction: "Increases Churn", reason: "Engineered spend velocity feature." },
    { rank: 7, feature: "contract_risk_score", importance: "5.9%", direction: "Increases Churn", reason: "Interaction between month-to-month and paperless billing." },
    { rank: 8, feature: "OnlineSecurity (No)", importance: "5.1%", direction: "Increases Churn", reason: "Absence of protective add-on lowers perceived platform value." },
    { rank: 9, feature: "TechSupport (No)", importance: "4.8%", direction: "Increases Churn", reason: "Unresolved issues compound dissatisfaction over time." },
    { rank: 10, feature: "PaperlessBilling (Yes)", importance: "4.1%", direction: "Increases Churn", reason: "Digitally-savvy customers actively review bills online." },
  ];

  return (
    <div className="mx-auto max-w-5xl px-4 py-12 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="text-center">
        <div className="inline-flex items-center gap-2 rounded-full border border-blue-500/30 bg-blue-950/60 px-3 py-1 text-xs font-medium text-blue-300">
          <BookOpen className="h-3.5 w-3.5" />
          <span>Project Methodology & Benchmarks</span>
        </div>
        <h1 className="mt-4 text-3xl font-extrabold text-white sm:text-4xl">
          Machine Learning Architecture & Evaluation
        </h1>
        <p className="mx-auto mt-2 max-w-2xl text-sm text-slate-400">
          Complete transparency into data preprocessing, SMOTE class balancing,
          hyperparameter tuning, and cross-validated model evaluation.
        </p>
      </div>

      {/* Dataset Overview */}
      <section className="mt-12 rounded-2xl border border-slate-800 bg-slate-900/40 p-6 sm:p-8">
        <h2 className="text-xl font-bold text-white">1. Dataset & Preprocessing Pipeline</h2>
        <p className="mt-2 text-sm leading-relaxed text-slate-300">
          Trained on the benchmark IBM Telco Customer Churn dataset (7,043 subscriber records).
          Raw data was processed with strict leak-free practices:
        </p>

        <div className="mt-6 grid gap-4 sm:grid-cols-3">
          <div className="rounded-xl border border-slate-800 bg-slate-950 p-4">
            <h3 className="text-xs font-bold uppercase tracking-wider text-blue-400">
              Data Cleaning
            </h3>
            <p className="mt-2 text-xs leading-relaxed text-slate-400">
              CustomerID dropped. 11 whitespace nulls in TotalCharges imputed via median.
              Numeric features bounded by IQR outlier capping.
            </p>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-950 p-4">
            <h3 className="text-xs font-bold uppercase tracking-wider text-indigo-400">
              Feature Engineering
            </h3>
            <p className="mt-2 text-xs leading-relaxed text-slate-400">
              Domain interaction features created: <code className="text-slate-200">avg_monthly_spend</code>,{" "}
              <code className="text-slate-200">service_count</code>,{" "}
              <code className="text-slate-200">charges_per_service</code>, and{" "}
              <code className="text-slate-200">contract_risk_score</code>.
            </p>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-950 p-4">
            <h3 className="text-xs font-bold uppercase tracking-wider text-purple-400">
              SMOTE Oversampling
            </h3>
            <p className="mt-2 text-xs leading-relaxed text-slate-400">
              26.5% minority class balanced to 0.5 ratio on training sets only via
              Synthetic Minority Oversampling Technique (k=5 neighbors).
            </p>
          </div>
        </div>
      </section>

      {/* Model Benchmark Table */}
      <section className="mt-10 rounded-2xl border border-slate-800 bg-slate-900/40 p-6 sm:p-8">
        <h2 className="text-xl font-bold text-white">2. Test Set Evaluation Benchmark</h2>
        <p className="mt-2 text-sm text-slate-400">
          Stratified 80/20 train/test split. Cross-validation optimized for ROC-AUC.
        </p>

        <div className="mt-6 overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-slate-800 text-xs uppercase text-slate-400">
              <tr>
                <th className="py-3 pr-4">Model Architecture</th>
                <th className="py-3 px-4">ROC-AUC</th>
                <th className="py-3 px-4">Accuracy</th>
                <th className="py-3 px-4">Precision</th>
                <th className="py-3 px-4">Recall</th>
                <th className="py-3 pl-4">F1 Score</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-300">
              <tr className="bg-blue-950/20 font-semibold text-white">
                <td className="py-3.5 pr-4 text-blue-400">
                  XGBoost Classifier (Best)
                </td>
                <td className="py-3.5 px-4 font-bold text-blue-400">0.868</td>
                <td className="py-3.5 px-4">82.0%</td>
                <td className="py-3.5 px-4">67.4%</td>
                <td className="py-3.5 px-4">59.8%</td>
                <td className="py-3.5 pl-4">0.634</td>
              </tr>
              <tr>
                <td className="py-3.5 pr-4">Random Forest</td>
                <td className="py-3.5 px-4 font-semibold text-emerald-400">0.856</td>
                <td className="py-3.5 px-4">81.2%</td>
                <td className="py-3.5 px-4">66.1%</td>
                <td className="py-3.5 px-4">57.9%</td>
                <td className="py-3.5 pl-4">0.617</td>
              </tr>
              <tr>
                <td className="py-3.5 pr-4">Logistic Regression</td>
                <td className="py-3.5 px-4 font-semibold text-indigo-400">0.845</td>
                <td className="py-3.5 px-4">80.0%</td>
                <td className="py-3.5 px-4">63.6%</td>
                <td className="py-3.5 px-4">56.8%</td>
                <td className="py-3.5 pl-4">0.600</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* Feature Importance Rankings */}
      <section className="mt-10 rounded-2xl border border-slate-800 bg-slate-900/40 p-6 sm:p-8">
        <h2 className="text-xl font-bold text-white">3. Top Churn Drivers (XGBoost Gini Importance)</h2>
        <p className="mt-2 text-sm text-slate-400">
          Ranked by empirical feature importance derived from the trained XGBoost tree ensembles.
        </p>

        <div className="mt-6 space-y-3">
          {topFeatures.map((item) => (
            <div
              key={item.rank}
              className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-800/80 bg-slate-950/60 px-4 py-3 text-xs"
            >
              <div className="flex items-center gap-3">
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-slate-800 font-bold text-slate-300">
                  {item.rank}
                </span>
                <div>
                  <p className="font-semibold text-white">{item.feature}</p>
                  <p className="text-slate-400">{item.reason}</p>
                </div>
              </div>
              <div className="flex items-center gap-4 text-right">
                <span className="text-[11px] font-medium text-slate-400">{item.direction}</span>
                <span className="rounded-lg bg-blue-950/80 px-2.5 py-1 font-mono font-bold text-blue-400 border border-blue-500/20">
                  {item.importance}
                </span>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Business Revenue Impact */}
      <section className="mt-10 rounded-2xl border border-emerald-500/30 bg-emerald-950/10 p-6 sm:p-8">
        <div className="flex items-center gap-3 text-emerald-400">
          <TrendingUp className="h-6 w-6" />
          <h2 className="text-xl font-bold text-white">4. Commercial Impact & ROI Economics</h2>
        </div>
        <p className="mt-3 text-sm leading-relaxed text-slate-300">
          Based on an Average Revenue Per User (ARPU) of $65/month and a targeted retention
          intervention cost of $10 per customer:
        </p>

        <div className="mt-6 grid gap-4 sm:grid-cols-3">
          <div className="rounded-xl border border-slate-800 bg-slate-950 p-4">
            <span className="text-xs text-slate-400">Monthly Revenue at Risk</span>
            <p className="mt-1 text-2xl font-bold text-rose-400">~$9,100 / mo</p>
            <span className="text-[11px] text-slate-500">In the ~20% high-risk tier</span>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-950 p-4">
            <span className="text-xs text-slate-400">Retention Campaign Cost</span>
            <p className="mt-1 text-2xl font-bold text-slate-300">~$1,400 / mo</p>
            <span className="text-[11px] text-slate-500">At $10 per intervention</span>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-950 p-4">
            <span className="text-xs text-slate-400">Net Annual Protected Revenue</span>
            <p className="mt-1 text-2xl font-bold text-emerald-400">$92,400 / yr</p>
            <span className="text-[11px] text-emerald-400/60">Estimated bottom-line lift</span>
          </div>
        </div>

        <div className="mt-8 flex justify-center">
          <Link
            href="/predict"
            className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-6 py-3 text-sm font-semibold text-white shadow-lg shadow-blue-600/25 hover:bg-blue-500"
          >
            <span>Run Prediction on a Customer</span>
            <ArrowRight className="h-4 w-4" />
          </Link>
        </div>
      </section>
    </div>
  );
}
