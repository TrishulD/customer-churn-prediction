"use client";

import React, { useState } from "react";
import {
  ShieldAlert,
  ShieldCheck,
  AlertTriangle,
  FileDown,
  X,
  MessageSquare,
  Sparkles,
  CheckCircle,
  Clock,
  User,
} from "lucide-react";
import jsPDF from "jspdf";
import FeedbackSection from "./FeedbackSection";

export interface PredictionData {
  customer_id: string;
  churn_probability: number;
  churn_probability_pct: string;
  churn_prediction: string;
  risk_tier: "Low Risk" | "Medium Risk" | "High Risk" | string;
  confidence: number;
  top_drivers: string[];
  recommendation: string;
  model_used: string;
  timestamp: string;
}

interface CustomerFormData {
  customerID?: string;
  gender: string;
  SeniorCitizen: number;
  Partner: string;
  Dependents: string;
  tenure: number;
  PhoneService: string;
  MultipleLines: string;
  InternetService: string;
  OnlineSecurity: string;
  OnlineBackup: string;
  DeviceProtection: string;
  TechSupport: string;
  StreamingTV: string;
  StreamingMovies: string;
  Contract: string;
  PaperlessBilling: string;
  PaymentMethod: string;
  MonthlyCharges: number;
  TotalCharges?: number;
}

interface ModalProps {
  prediction: PredictionData;
  formData: CustomerFormData;
  onClose: () => void;
}

export default function PredictionResultModal({
  prediction,
  formData,
  onClose,
}: ModalProps) {
  const [showFeedback, setShowFeedback] = useState(false);
  const [isGeneratingPdf, setIsGeneratingPdf] = useState(false);

  const probPct = Math.round(prediction.churn_probability * 100);

  // Risk styling
  const getRiskDetails = () => {
    switch (prediction.risk_tier) {
      case "High Risk":
        return {
          badgeBg: "bg-rose-500/10 text-rose-400 border-rose-500/30",
          barBg: "bg-rose-500",
          icon: ShieldAlert,
          titleColor: "text-rose-400",
        };
      case "Medium Risk":
        return {
          badgeBg: "bg-amber-500/10 text-amber-400 border-amber-500/30",
          barBg: "bg-amber-500",
          icon: AlertTriangle,
          titleColor: "text-amber-400",
        };
      default:
        return {
          badgeBg: "bg-emerald-500/10 text-emerald-400 border-emerald-500/30",
          barBg: "bg-emerald-500",
          icon: ShieldCheck,
          titleColor: "text-emerald-400",
        };
    }
  };

  const risk = getRiskDetails();
  const RiskIcon = risk.icon;

  const handleDownloadPdf = () => {
    setIsGeneratingPdf(true);
    try {
      const doc = new jsPDF();

      // Brand Header
      doc.setFillColor(15, 23, 42); // slate-950
      doc.rect(0, 0, 210, 35, "F");

      doc.setTextColor(255, 255, 255);
      doc.setFontSize(18);
      doc.setFont("helvetica", "bold");
      doc.text("Customer Churn Risk Assessment Report", 14, 18);

      doc.setFontSize(9);
      doc.setFont("helvetica", "normal");
      doc.setTextColor(148, 163, 184);
      doc.text(
        `Generated on: ${new Date(prediction.timestamp).toLocaleString()} | Model: ${prediction.model_used.toUpperCase()}`,
        14,
        26
      );

      // Section 1: Customer Profile Summary
      doc.setTextColor(15, 23, 42);
      doc.setFontSize(13);
      doc.setFont("helvetica", "bold");
      doc.text("1. Customer Profile", 14, 45);

      doc.setDrawColor(226, 232, 240);
      doc.line(14, 48, 196, 48);

      doc.setFontSize(10);
      doc.setFont("helvetica", "normal");
      doc.setTextColor(51, 65, 85);

      const col1 = 14;
      const col2 = 110;
      let y = 56;

      doc.text(`Customer ID: ${prediction.customer_id}`, col1, y);
      doc.text(`Contract Type: ${formData.Contract}`, col2, y);
      y += 7;
      doc.text(`Tenure: ${formData.tenure} months`, col1, y);
      doc.text(`Internet Service: ${formData.InternetService}`, col2, y);
      y += 7;
      doc.text(`Monthly Charges: $${Number(formData.MonthlyCharges).toFixed(2)}`, col1, y);
      doc.text(`Payment Method: ${formData.PaymentMethod}`, col2, y);
      y += 7;
      doc.text(`Total Charges: $${Number(formData.TotalCharges || 0).toFixed(2)}`, col1, y);
      doc.text(`Paperless Billing: ${formData.PaperlessBilling}`, col2, y);

      // Section 2: Churn Prediction & Risk
      y += 15;
      doc.setTextColor(15, 23, 42);
      doc.setFontSize(13);
      doc.setFont("helvetica", "bold");
      doc.text("2. Churn Prediction & Risk Score", 14, y);

      y += 3;
      doc.line(14, y, 196, y);

      y += 10;
      doc.setFillColor(248, 250, 252);
      doc.roundedRect(14, y, 182, 30, 2, 2, "F");

      doc.setFontSize(11);
      doc.setFont("helvetica", "bold");
      doc.setTextColor(15, 23, 42);
      doc.text("Risk Tier:", 20, y + 12);
      doc.text(prediction.risk_tier.toUpperCase(), 50, y + 12);

      doc.text("Churn Probability:", 100, y + 12);
      doc.text(`${prediction.churn_probability_pct}`, 145, y + 12);

      doc.setFontSize(10);
      doc.setFont("helvetica", "normal");
      doc.setTextColor(71, 85, 105);
      doc.text(`Prediction Verdict: Customer is ${prediction.churn_prediction.toUpperCase()}`, 20, y + 22);

      // Section 3: Key Risk Drivers
      y += 40;
      doc.setTextColor(15, 23, 42);
      doc.setFontSize(13);
      doc.setFont("helvetica", "bold");
      doc.text("3. Key Identified Churn Drivers", 14, y);

      y += 3;
      doc.line(14, y, 196, y);

      y += 8;
      doc.setFontSize(10);
      doc.setFont("helvetica", "normal");
      doc.setTextColor(51, 65, 85);

      prediction.top_drivers.forEach((driver) => {
        doc.text(`- ${driver}`, 18, y);
        y += 6;
      });

      // Section 4: Recommended Retention Playbook
      y += 6;
      doc.setTextColor(15, 23, 42);
      doc.setFontSize(13);
      doc.setFont("helvetica", "bold");
      doc.text("4. Prescribed Retention Action", 14, y);

      y += 3;
      doc.line(14, y, 196, y);

      y += 8;
      doc.setFontSize(10);
      doc.setFont("helvetica", "italic");
      doc.setTextColor(15, 23, 42);
      const splitRec = doc.splitTextToSize(prediction.recommendation, 180);
      doc.text(splitRec, 18, y);

      // Footer
      doc.setFontSize(8);
      doc.setFont("helvetica", "normal");
      doc.setTextColor(148, 163, 184);
      doc.text(
        "Confidential &bull; Customer Churn Prediction Engine &bull; Developed with XGBoost & Next.js",
        14,
        285
      );

      doc.save(`churn_report_${prediction.customer_id}.pdf`);
    } catch (err) {
      console.error("PDF generation failed:", err);
      alert("Failed to generate PDF. Please try again.");
    } finally {
      setIsGeneratingPdf(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 p-4 backdrop-blur-sm">
      <div className="relative max-h-[90vh] w-full max-w-2xl overflow-y-auto rounded-2xl border border-slate-800 bg-slate-900 p-6 shadow-2xl">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-5 right-5 rounded-lg p-1.5 text-slate-400 hover:bg-slate-800 hover:text-white"
        >
          <X className="h-5 w-5" />
        </button>

        {/* Header */}
        <div className="flex items-center gap-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-slate-800">
            <RiskIcon className={`h-7 w-7 ${risk.titleColor}`} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-xl font-bold text-white">Prediction Analysis</h2>
              <span
                className={`rounded-full border px-2.5 py-0.5 text-xs font-semibold uppercase ${risk.badgeBg}`}
              >
                {prediction.risk_tier}
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Customer ID: <span className="text-slate-200">{prediction.customer_id}</span> &bull; Scored by{" "}
              <span className="font-semibold text-blue-400">{prediction.model_used}</span>
            </p>
          </div>
        </div>

        {/* Probability Gauge & Progress */}
        <div className="mt-6 rounded-xl border border-slate-800 bg-slate-950/60 p-5">
          <div className="flex items-end justify-between">
            <div>
              <span className="text-xs font-medium uppercase tracking-wider text-slate-400">
                Churn Probability
              </span>
              <div className="flex items-baseline gap-2">
                <span className={`text-4xl font-extrabold ${risk.titleColor}`}>
                  {prediction.churn_probability_pct}
                </span>
                <span className="text-xs text-slate-400">
                  (Verdict: <strong>{prediction.churn_prediction}</strong>)
                </span>
              </div>
            </div>
            <div className="text-right">
              <span className="text-xs font-medium uppercase tracking-wider text-slate-400">
                Model Confidence
              </span>
              <p className="text-lg font-bold text-white">
                {Math.round(prediction.confidence * 100)}%
              </p>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="mt-4 h-3 w-full overflow-hidden rounded-full bg-slate-800">
            <div
              className={`h-full transition-all duration-500 ${risk.barBg}`}
              style={{ width: `${Math.min(probPct, 100)}%` }}
            />
          </div>

          <div className="mt-2 flex justify-between text-[10px] text-slate-500">
            <span>0% (Low Risk)</span>
            <span>30% Threshold</span>
            <span>60% Threshold</span>
            <span>100% (High Risk)</span>
          </div>
        </div>

        {/* Key Drivers */}
        <div className="mt-6">
          <h3 className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wider text-slate-400">
            <Sparkles className="h-4 w-4 text-blue-400" />
            Key Factors Influencing This Prediction
          </h3>
          <ul className="mt-3 space-y-2">
            {prediction.top_drivers.map((driver, idx) => (
              <li
                key={idx}
                className="flex items-start gap-2.5 rounded-lg border border-slate-800/80 bg-slate-950/40 px-3.5 py-2.5 text-sm text-slate-300"
              >
                <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-blue-400" />
                <span>{driver}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Recommended Retention Playbook */}
        <div className="mt-6 rounded-xl border border-blue-500/20 bg-blue-950/20 p-4">
          <div className="flex items-center gap-2 text-blue-400">
            <CheckCircle className="h-4 w-4" />
            <h4 className="text-sm font-semibold uppercase tracking-wide">
              Prescribed Retention Action
            </h4>
          </div>
          <p className="mt-2 text-sm leading-relaxed text-slate-200">
            {prediction.recommendation}
          </p>
        </div>

        {/* Feedback Section Toggle */}
        {showFeedback ? (
          <div className="mt-6 border-t border-slate-800 pt-5">
            <FeedbackSection
              predictionId={prediction.customer_id}
              customerId={prediction.customer_id}
              churnPrediction={prediction.churn_prediction}
              riskTier={prediction.risk_tier}
              onSubmitted={() => {}}
            />
          </div>
        ) : null}

        {/* Actions Footer */}
        <div className="mt-6 flex flex-wrap items-center justify-between gap-3 border-t border-slate-800 pt-5">
          <div className="flex items-center gap-2">
            <button
              onClick={handleDownloadPdf}
              disabled={isGeneratingPdf}
              className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white shadow-md shadow-blue-600/20 transition-all hover:bg-blue-500 disabled:opacity-50"
            >
              <FileDown className="h-4 w-4" />
              <span>{isGeneratingPdf ? "Generating..." : "Download PDF Report"}</span>
            </button>

            <button
              onClick={() => setShowFeedback(!showFeedback)}
              className="flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-800 px-3.5 py-2.5 text-sm font-semibold text-slate-300 transition-colors hover:bg-slate-700 hover:text-white"
            >
              <MessageSquare className="h-4 w-4" />
              <span>{showFeedback ? "Hide Feedback" : "Leave Feedback"}</span>
            </button>
          </div>

          <button
            onClick={onClose}
            className="rounded-lg px-4 py-2.5 text-sm font-medium text-slate-400 hover:bg-slate-800 hover:text-white"
          >
            Close Window
          </button>
        </div>
      </div>
    </div>
  );
}
