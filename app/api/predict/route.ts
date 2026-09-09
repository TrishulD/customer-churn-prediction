import { NextRequest, NextResponse } from "next/server";
import { getServiceSupabase } from "@/lib/supabase";

export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const modelName = req.nextUrl.searchParams.get("model") || "xgboost";

    // Resolve ML inference URL:
    //   1. ML_INFERENCE_URL env var (explicit override — highest priority)
    //   2. VERCEL_URL (auto-injected by Vercel on every deployment)
    //   3. Localhost (local development fallback)
    const vercelUrl = process.env.VERCEL_URL;
    const mlUrl =
      process.env.ML_INFERENCE_URL ||
      (vercelUrl ? `https://${vercelUrl}/api/py` : "http://127.0.0.1:8000");
    let predictionResult = null;

    try {
      const mlRes = await fetch(`${mlUrl}/predict?model_name=${modelName}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(10000), // 10 second timeout for model inference
      });

      if (!mlRes.ok) {
        const errData = await mlRes.json().catch(() => ({}));
        return NextResponse.json(
          {
            message:
              errData.detail ||
              "ML inference service returned an error during prediction.",
          },
          { status: mlRes.status || 500 }
        );
      }

      predictionResult = await mlRes.json();
    } catch (err: any) {
      console.error(
        `[ML API Connection Error]: Unable to reach inference service at ${mlUrl}. Error:`,
        err.message
      );
      return NextResponse.json(
        {
          message:
            "ML inference service unavailable. Please try again later.",
        },
        { status: 503 }
      );
    }

    if (!predictionResult) {
      return NextResponse.json(
        {
          message:
            "ML inference service unavailable. Please try again later.",
        },
        { status: 503 }
      );
    }

    // Optionally record verified prediction in Supabase
    const supabase = getServiceSupabase();
    if (supabase) {
      try {
        await supabase.from("predictions").insert({
          customer_id: predictionResult.customer_id,
          gender: body.gender,
          senior_citizen: Number(body.SeniorCitizen || 0),
          partner: body.Partner,
          dependents: body.Dependents,
          tenure: Number(body.tenure || 0),
          phone_service: body.PhoneService,
          multiple_lines: body.MultipleLines,
          internet_service: body.InternetService,
          online_security: body.OnlineSecurity,
          online_backup: body.OnlineBackup,
          device_protection: body.DeviceProtection,
          tech_support: body.TechSupport,
          streaming_tv: body.StreamingTV,
          streaming_movies: body.StreamingMovies,
          contract: body.Contract,
          paperless_billing: body.PaperlessBilling,
          payment_method: body.PaymentMethod,
          monthly_charges: Number(body.MonthlyCharges || 0),
          total_charges: Number(body.TotalCharges || 0),
          churn_probability: predictionResult.churn_probability,
          churn_prediction: predictionResult.churn_prediction,
          risk_tier: predictionResult.risk_tier,
          top_drivers: predictionResult.top_drivers,
          recommendation: predictionResult.recommendation,
          model_used: predictionResult.model_used,
        });
      } catch (dbErr) {
        console.warn("[Supabase] Failed to write prediction log:", dbErr);
      }
    }

    return NextResponse.json(predictionResult);
  } catch (error: any) {
    console.error("Prediction API route error:", error);
    return NextResponse.json(
      { message: "Internal server error during prediction request", error: error.message },
      { status: 500 }
    );
  }
}
