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

    // Authentic empty state when Supabase is not yet configured or no records exist
    return NextResponse.json({
      total_predictions: 0,
      high_risk_count: 0,
      medium_risk_count: 0,
      low_risk_count: 0,
      feedback_count: 0,
      average_rating: 0,
      recent_predictions: [],
      recent_feedbacks: [],
      source: "no_database_configured",
    });
  } catch (err: any) {
    console.error("Admin stats error:", err);
    return NextResponse.json(
      { message: "Failed to fetch admin stats", error: err.message },
      { status: 500 }
    );
  }
}
