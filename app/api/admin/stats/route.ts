import { NextRequest, NextResponse } from "next/server";
import { getServiceSupabase } from "@/lib/supabase";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  try {
    const adminPassword = process.env.ADMIN_PASSWORD || "admin123";

    // Validate passkey
    const authHeader = req.headers.get("authorization");
    const customHeader = req.headers.get("x-admin-passkey");
    const urlPasskey = req.nextUrl.searchParams.get("passkey");

    const providedKey =
      customHeader ||
      urlPasskey ||
      (authHeader?.startsWith("Bearer ") ? authHeader.slice(7) : null);

    if (!providedKey || providedKey !== adminPassword) {
      return NextResponse.json(
        { message: "Unauthorized. Invalid admin passkey." },
        { status: 401 }
      );
    }

    const supabase = getServiceSupabase();

    if (supabase) {
      // 1. Fetch predictions
      const { data: predictions, error: predErr } = await supabase
        .from("predictions")
        .select("*")
        .order("created_at", { ascending: false })
        .limit(100);

      // 2. Fetch feedback
      const { data: feedbacks, error: feedErr } = await supabase
        .from("feedbacks")
        .select("*")
        .order("created_at", { ascending: false })
        .limit(100);

      const totalPredictions = predictions?.length || 0;
      const highRisk = predictions?.filter((p) => p.risk_tier === "High Risk").length || 0;
      const medRisk = predictions?.filter((p) => p.risk_tier === "Medium Risk").length || 0;
      const lowRisk = predictions?.filter((p) => p.risk_tier === "Low Risk").length || 0;

      const totalFeedback = feedbacks?.length || 0;
      const avgRating = totalFeedback > 0
        ? feedbacks!.reduce((acc, f) => acc + (f.rating || 5), 0) / totalFeedback
        : 5.0;

      return NextResponse.json({
        total_predictions: totalPredictions,
        high_risk_count: highRisk,
        medium_risk_count: medRisk,
        low_risk_count: lowRisk,
        feedback_count: totalFeedback,
        average_rating: Number(avgRating.toFixed(1)),
        recent_predictions: predictions || [],
        recent_feedbacks: feedbacks || [],
        source: "supabase",
      });
    }

    // Default baseline analytics if Supabase is not connected
    return NextResponse.json({
      total_predictions: 1420,
      high_risk_count: 284,
      medium_risk_count: 355,
      low_risk_count: 781,
      feedback_count: 48,
      average_rating: 4.8,
      recent_predictions: [
        {
          id: "mock-1",
          customer_id: "CUST-9821",
          churn_probability: 0.8389,
          churn_prediction: "Churn",
          risk_tier: "High Risk",
          model_used: "xgboost",
          contract: "Month-to-month",
          tenure: 1,
          monthly_charges: 95.65,
          created_at: new Date(Date.now() - 1000 * 60 * 12).toISOString(),
        },
        {
          id: "mock-2",
          customer_id: "CUST-4102",
          churn_probability: 0.0151,
          churn_prediction: "No Churn",
          risk_tier: "Low Risk",
          model_used: "xgboost",
          contract: "Two year",
          tenure: 60,
          monthly_charges: 65.0,
          created_at: new Date(Date.now() - 1000 * 60 * 45).toISOString(),
        },
        {
          id: "mock-3",
          customer_id: "CUST-7719",
          churn_probability: 0.521,
          churn_prediction: "Churn",
          risk_tier: "Medium Risk",
          model_used: "xgboost",
          contract: "One year",
          tenure: 14,
          monthly_charges: 78.4,
          created_at: new Date(Date.now() - 1000 * 60 * 90).toISOString(),
        },
      ],
      recent_feedbacks: [
        {
          id: "fb-1",
          customer_id: "CUST-9821",
          rating: 5,
          is_useful: true,
          churn_prediction: "Churn",
          risk_tier: "High Risk",
          comments: "Accurately flagged customer who was calling about billing cancel.",
          created_at: new Date(Date.now() - 1000 * 60 * 10).toISOString(),
        },
      ],
      source: "baseline_demo",
    });
  } catch (err: any) {
    console.error("Admin stats error:", err);
    return NextResponse.json(
      { message: "Failed to fetch admin stats", error: err.message },
      { status: 500 }
    );
  }
}
