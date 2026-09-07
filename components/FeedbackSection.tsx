"use client";

import React, { useState } from "react";
import { Star, CheckCircle, AlertCircle, Send } from "lucide-react";

interface FeedbackProps {
  predictionId: string;
  customerId: string;
  churnPrediction: string;
  riskTier: string;
  onSubmitted?: () => void;
}

export default function FeedbackSection({
  predictionId,
  customerId,
  churnPrediction,
  riskTier,
  onSubmitted,
}: FeedbackProps) {
  const [rating, setRating] = useState<number>(5);
  const [isUseful, setIsUseful] = useState<boolean>(true);
  const [comments, setComments] = useState<string>("");
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [submitted, setSubmitted] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);

    try {
      const res = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prediction_id: predictionId,
          customer_id: customerId,
          rating,
          is_useful: isUseful,
          churn_prediction: churnPrediction,
          risk_tier: riskTier,
          comments,
        }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.message || "Failed to submit feedback");
      }

      setSubmitted(true);
      if (onSubmitted) onSubmitted();
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred.");
    } finally {
      setSubmitting(false);
    }
  };

  if (submitted) {
    return (
      <div className="rounded-xl border border-emerald-500/20 bg-emerald-950/20 p-4 text-center">
        <div className="mx-auto flex h-10 w-10 items-center justify-center rounded-full bg-emerald-500/10 text-emerald-400">
          <CheckCircle className="h-6 w-6" />
        </div>
        <h4 className="mt-2 text-sm font-semibold text-white">Thank You for Your Feedback!</h4>
        <p className="mt-1 text-xs text-slate-400">
          Your insights help continuously evaluate and calibrate our retention models.
        </p>
      </div>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="rounded-xl border border-slate-800 bg-slate-950/70 p-4">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-semibold text-white">Prediction Feedback</h4>
        <span className="text-xs text-slate-400">Stored securely in Supabase</span>
      </div>

      {error && (
        <div className="mt-3 flex items-center gap-2 rounded-lg border border-rose-500/30 bg-rose-950/30 p-2.5 text-xs text-rose-300">
          <AlertCircle className="h-4 w-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Star Rating */}
      <div className="mt-3 flex items-center gap-3">
        <span className="text-xs text-slate-400">Model Accuracy Rating:</span>
        <div className="flex items-center gap-1">
          {[1, 2, 3, 4, 5].map((star) => (
            <button
              type="button"
              key={star}
              onClick={() => setRating(star)}
              className="p-0.5 text-amber-400 transition-transform hover:scale-110"
            >
              <Star
                className={`h-4 w-4 ${
                  star <= rating ? "fill-amber-400 text-amber-400" : "text-slate-600"
                }`}
              />
            </button>
          ))}
        </div>
      </div>

      {/* Useful Toggle */}
      <div className="mt-3 flex items-center gap-4">
        <span className="text-xs text-slate-400">Was this prediction useful?</span>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setIsUseful(true)}
            className={`rounded px-2.5 py-1 text-xs font-medium transition-colors ${
              isUseful
                ? "bg-blue-600 text-white"
                : "bg-slate-800 text-slate-400 hover:bg-slate-700"
            }`}
          >
            Yes
          </button>
          <button
            type="button"
            onClick={() => setIsUseful(false)}
            className={`rounded px-2.5 py-1 text-xs font-medium transition-colors ${
              !isUseful
                ? "bg-rose-600 text-white"
                : "bg-slate-800 text-slate-400 hover:bg-slate-700"
            }`}
          >
            No
          </button>
        </div>
      </div>

      {/* Optional Comments */}
      <div className="mt-3">
        <textarea
          rows={2}
          value={comments}
          onChange={(e) => setComments(e.target.value)}
          placeholder="Optional notes or retention team observations..."
          className="w-full rounded-lg border border-slate-800 bg-slate-900 px-3 py-2 text-xs text-slate-200 placeholder-slate-500 focus:border-blue-500 focus:outline-none"
        />
      </div>

      <div className="mt-3 flex justify-end">
        <button
          type="submit"
          disabled={submitting}
          className="flex items-center gap-1.5 rounded-lg bg-blue-600 px-3 py-1.5 text-xs font-semibold text-white transition-colors hover:bg-blue-500 disabled:opacity-50"
        >
          <Send className="h-3.5 w-3.5" />
          <span>{submitting ? "Saving..." : "Submit Feedback"}</span>
        </button>
      </div>
    </form>
  );
}
