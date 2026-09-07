import Link from "next/link";
import {
  ShieldAlert,
  TrendingUp,
  Award,
  Zap,
  CheckCircle2,
  ArrowRight,
  Database,
  BarChart,
  FileSpreadsheet,
  DownloadCloud,
} from "lucide-react";

export default function HomePage() {
  return (
    <div className="relative overflow-hidden">
      {/* Background Glow */}
      <div className="pointer-events-none absolute -top-40 left-1/2 -z-10 h-[500px] w-[800px] -translate-x-1/2 rounded-full bg-gradient-to-tr from-blue-600/20 to-indigo-500/20 blur-3xl" />

      {/* Hero Section */}
      <section className="mx-auto max-w-7xl px-4 pt-16 pb-20 sm:px-6 lg:px-8">
        <div className="text-center">
          <div className="inline-flex items-center gap-2 rounded-full border border-blue-500/30 bg-blue-950/60 px-3.5 py-1 text-xs font-medium text-blue-300 backdrop-blur-sm">
            <Award className="h-3.5 w-3.5 text-blue-400" />
            <span>State-of-the-Art Production ML Pipeline</span>
          </div>

          <h1 className="mt-6 text-4xl font-extrabold tracking-tight text-white sm:text-6xl lg:text-7xl">
            Forecast & Prevent{" "}
            <span className="bg-gradient-to-r from-blue-400 via-indigo-300 to-purple-400 bg-clip-text text-transparent">
              Customer Churn
            </span>
          </h1>

          <p className="mx-auto mt-6 max-w-3xl text-lg leading-relaxed text-slate-300 sm:text-xl">
            Proactively identify at-risk subscribers, pinpoint the exact behavioral
            drivers behind customer departures, and execute automated retention
            playbooks powered by our tuned <strong>XGBoost</strong> model.
          </p>

          <div className="mt-10 flex flex-wrap items-center justify-center gap-4">
            <Link
              href="/predict"
              className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 px-6 py-3.5 text-base font-semibold text-white shadow-lg shadow-blue-600/25 transition-all hover:from-blue-500 hover:to-indigo-500 hover:shadow-blue-500/35"
            >
              <Zap className="h-5 w-5" />
              <span>Launch Churn Predictor</span>
              <ArrowRight className="h-4 w-4" />
            </Link>

            <Link
              href="/about"
              className="inline-flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-900/80 px-6 py-3.5 text-base font-semibold text-slate-200 backdrop-blur-sm transition-all hover:bg-slate-800 hover:text-white"
            >
              <BarChart className="h-5 w-5 text-slate-400" />
              <span>Explore Model Reports</span>
            </Link>
          </div>
        </div>

        {/* Live Metrics Grid */}
        <div className="mt-16 grid grid-cols-2 gap-4 sm:grid-cols-4 lg:gap-6">
          <div className="rounded-2xl border border-slate-800/80 bg-slate-900/50 p-6 backdrop-blur-sm">
            <div className="flex items-center gap-2 text-blue-400">
              <Award className="h-5 w-5" />
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Best Model
              </span>
            </div>
            <p className="mt-2 text-3xl font-bold text-white">0.868</p>
            <p className="mt-1 text-xs text-slate-400">ROC-AUC (XGBoost)</p>
          </div>

          <div className="rounded-2xl border border-slate-800/80 bg-slate-900/50 p-6 backdrop-blur-sm">
            <div className="flex items-center gap-2 text-emerald-400">
              <CheckCircle2 className="h-5 w-5" />
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Test Accuracy
              </span>
            </div>
            <p className="mt-2 text-3xl font-bold text-white">82.0%</p>
            <p className="mt-1 text-xs text-slate-400">Held-out validation</p>
          </div>

          <div className="rounded-2xl border border-slate-800/80 bg-slate-900/50 p-6 backdrop-blur-sm">
            <div className="flex items-center gap-2 text-purple-400">
              <Database className="h-5 w-5" />
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Dataset Scale
              </span>
            </div>
            <p className="mt-2 text-3xl font-bold text-white">7,043</p>
            <p className="mt-1 text-xs text-slate-400">IBM Telco records</p>
          </div>

          <div className="rounded-2xl border border-slate-800/80 bg-slate-900/50 p-6 backdrop-blur-sm">
            <div className="flex items-center gap-2 text-amber-400">
              <TrendingUp className="h-5 w-5" />
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                F1 Score
              </span>
            </div>
            <p className="mt-2 text-3xl font-bold text-white">0.634</p>
            <p className="mt-1 text-xs text-slate-400">Precision-recall balance</p>
          </div>
        </div>
      </section>

      {/* Model Benchmark Comparison */}
      <section className="border-t border-slate-800/80 bg-slate-950/60 py-20">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h2 className="text-3xl font-bold text-white sm:text-4xl">
              Rigorous Multi-Model Evaluation
            </h2>
            <p className="mx-auto mt-3 max-w-2xl text-slate-400">
              Evaluated on stratified 5-fold cross-validation and tested on a 20%
              held-out sample using SMOTE class balancing.
            </p>
          </div>

          <div className="mt-12 grid gap-6 md:grid-cols-3">
            {/* XGBoost Card */}
            <div className="relative rounded-2xl border-2 border-blue-500/80 bg-gradient-to-b from-blue-950/40 to-slate-900/90 p-6 shadow-xl shadow-blue-500/10">
              <div className="absolute -top-3 right-6 rounded-full bg-blue-600 px-3 py-0.5 text-xs font-bold uppercase tracking-wider text-white">
                Best Performer
              </div>
              <h3 className="text-xl font-bold text-white">XGBoost Classifier</h3>
              <p className="mt-1 text-xs text-blue-300">
                Gradient boosted trees with L1/L2 regularization
              </p>

              <div className="mt-6 space-y-3">
                <div className="flex justify-between border-b border-slate-800 pb-2 text-sm">
                  <span className="text-slate-400">ROC-AUC Score</span>
                  <span className="font-semibold text-blue-400">0.868</span>
                </div>
                <div className="flex justify-between border-b border-slate-800 pb-2 text-sm">
                  <span className="text-slate-400">Accuracy</span>
                  <span className="font-semibold text-white">82.0%</span>
                </div>
                <div className="flex justify-between border-b border-slate-800 pb-2 text-sm">
                  <span className="text-slate-400">Precision</span>
                  <span className="font-semibold text-white">67.4%</span>
                </div>
                <div className="flex justify-between border-b border-slate-800 pb-2 text-sm">
                  <span className="text-slate-400">Recall</span>
                  <span className="font-semibold text-white">59.8%</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-slate-400">F1 Score</span>
                  <span className="font-semibold text-white">0.634</span>
                </div>
              </div>

              <div className="mt-6">
                <Link
                  href="/predict"
                  className="block w-full rounded-lg bg-blue-600 py-2.5 text-center text-sm font-semibold text-white transition-colors hover:bg-blue-500"
                >
                  Score with XGBoost
                </Link>
              </div>
            </div>

            {/* Random Forest Card */}
            <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 backdrop-blur-sm">
              <h3 className="text-xl font-bold text-white">Random Forest</h3>
              <p className="mt-1 text-xs text-slate-400">
                Ensemble of decorrelated decision trees
              </p>

              <div className="mt-6 space-y-3">
                <div className="flex justify-between border-b border-slate-800 pb-2 text-sm">
                  <span className="text-slate-400">ROC-AUC Score</span>
                  <span className="font-semibold text-emerald-400">0.856</span>
                </div>
                <div className="flex justify-between border-b border-slate-800 pb-2 text-sm">
                  <span className="text-slate-400">Accuracy</span>
                  <span className="font-semibold text-white">81.2%</span>
                </div>
                <div className="flex justify-between border-b border-slate-800 pb-2 text-sm">
                  <span className="text-slate-400">Precision</span>
                  <span className="font-semibold text-white">66.1%</span>
                </div>
                <div className="flex justify-between border-b border-slate-800 pb-2 text-sm">
                  <span className="text-slate-400">Recall</span>
                  <span className="font-semibold text-white">57.9%</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-slate-400">F1 Score</span>
                  <span className="font-semibold text-white">0.617</span>
                </div>
              </div>

              <div className="mt-6">
                <Link
                  href="/predict"
                  className="block w-full rounded-lg border border-slate-700 bg-slate-800/80 py-2.5 text-center text-sm font-semibold text-slate-300 transition-colors hover:bg-slate-700 hover:text-white"
                >
                  Score with Random Forest
                </Link>
              </div>
            </div>

            {/* Logistic Regression Card */}
            <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 backdrop-blur-sm">
              <h3 className="text-xl font-bold text-white">Logistic Regression</h3>
              <p className="mt-1 text-xs text-slate-400">
                Regularized linear model baseline
              </p>

              <div className="mt-6 space-y-3">
                <div className="flex justify-between border-b border-slate-800 pb-2 text-sm">
                  <span className="text-slate-400">ROC-AUC Score</span>
                  <span className="font-semibold text-indigo-400">0.845</span>
                </div>
                <div className="flex justify-between border-b border-slate-800 pb-2 text-sm">
                  <span className="text-slate-400">Accuracy</span>
                  <span className="font-semibold text-white">80.0%</span>
                </div>
                <div className="flex justify-between border-b border-slate-800 pb-2 text-sm">
                  <span className="text-slate-400">Precision</span>
                  <span className="font-semibold text-white">63.6%</span>
                </div>
                <div className="flex justify-between border-b border-slate-800 pb-2 text-sm">
                  <span className="text-slate-400">Recall</span>
                  <span className="font-semibold text-white">56.8%</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-slate-400">F1 Score</span>
                  <span className="font-semibold text-white">0.600</span>
                </div>
              </div>

              <div className="mt-6">
                <Link
                  href="/predict"
                  className="block w-full rounded-lg border border-slate-700 bg-slate-800/80 py-2.5 text-center text-sm font-semibold text-slate-300 transition-colors hover:bg-slate-700 hover:text-white"
                >
                  Score with Logistic Reg
                </Link>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Feature Highlights */}
      <section className="py-20">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="grid gap-8 lg:grid-cols-3">
            <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-8">
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-blue-500/10 text-blue-400">
                <ShieldAlert className="h-6 w-6" />
              </div>
              <h3 className="mt-5 text-xl font-bold text-white">
                Honest Explainability
              </h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-400">
                Predictions reveal authentic Gini feature importances. Identify whether
                churn is driven by short tenure, contract type, lack of security add-ons,
                or price sensitivity.
              </p>
            </div>

            <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-8">
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-indigo-500/10 text-indigo-400">
                <FileSpreadsheet className="h-6 w-6" />
              </div>
              <h3 className="mt-5 text-xl font-bold text-white">
                Actionable Retention Playbooks
              </h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-400">
                Never guess the next step. Every prediction includes segment-tailored
                interventions such as contract upgrade discounts, VIP onboarding calls,
                and discounted Value Shield bundles.
              </p>
            </div>

            <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-8">
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-purple-500/10 text-purple-400">
                <DownloadCloud className="h-6 w-6" />
              </div>
              <h3 className="mt-5 text-xl font-bold text-white">
                Executive PDF Reports
              </h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-400">
                Export client-side executive churn assessments formatted with customer
                demographics, probability dials, key drivers, and timestamped
                retention plans with zero server overhead.
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
