import { NextRequest, NextResponse } from "next/server";
import { getServiceSupabase } from "@/lib/supabase";

export const dynamic = "force-dynamic";

// In-memory fallback buffer for feedback
const inMemoryFeedbacks: any[] = [];

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();

    if (!body.prediction_id) {
      return NextResponse.json(
        { message: "prediction_id is required" },
        { status: 400 }
      );
    }

    const feedbackEntry = {
      prediction_id: String(body.prediction_id),
      customer_id: String(body.customer_id || body.prediction_id),
      rating: Number(body.rating || 5),
      is_useful: Boolean(body.is_useful),
      churn_prediction: body.churn_prediction || "Unknown",
      risk_tier: body.risk_tier || "Unknown",
      comments: body.comments || "",
      created_at: new Date().toISOString(),
    };

    const supabase = getServiceSupabase();
    if (supabase) {
      const { error } = await supabase.from("feedbacks").insert(feedbackEntry);
      if (error) {
        console.error("[Supabase Feedback Error]:", error);
        // Fallback to local memory buffer
        inMemoryFeedbacks.push(feedbackEntry);
      }
    } else {
      inMemoryFeedbacks.push(feedbackEntry);
    }

    return NextResponse.json({
      success: true,
      message: "Feedback submitted successfully.",
      stored_in: supabase ? "supabase" : "in-memory-buffer",
    });
  } catch (err: any) {
    console.error("Feedback route error:", err);
    return NextResponse.json(
      { message: "Failed to submit feedback", error: err.message },
      { status: 500 }
    );
  }
}

export async function GET() {
  const supabase = getServiceSupabase();
  if (supabase) {
    const { data } = await supabase
      .from("feedbacks")
      .select("*")
      .order("created_at", { ascending: false })
      .limit(50);
    return NextResponse.json({ feedbacks: data || [] });
  }

  return NextResponse.json({ feedbacks: inMemoryFeedbacks });
}
