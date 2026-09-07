-- ==============================================================================
-- Supabase Schema for Customer Churn Prediction & Retention Platform
-- ==============================================================================

-- 1. Predictions Table
CREATE TABLE IF NOT EXISTS public.predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id TEXT NOT NULL,
    gender TEXT,
    senior_citizen INTEGER DEFAULT 0,
    partner TEXT,
    dependents TEXT,
    tenure NUMERIC,
    phone_service TEXT,
    multiple_lines TEXT,
    internet_service TEXT,
    online_security TEXT,
    online_backup TEXT,
    device_protection TEXT,
    tech_support TEXT,
    streaming_tv TEXT,
    streaming_movies TEXT,
    contract TEXT,
    paperless_billing TEXT,
    payment_method TEXT,
    monthly_charges NUMERIC,
    total_charges NUMERIC,
    churn_probability NUMERIC NOT NULL,
    churn_prediction TEXT NOT NULL,
    risk_tier TEXT NOT NULL,
    top_drivers JSONB,
    recommendation TEXT,
    model_used TEXT DEFAULT 'xgboost',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 2. Feedbacks Table
CREATE TABLE IF NOT EXISTS public.feedbacks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_id TEXT,
    customer_id TEXT,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    is_useful BOOLEAN NOT NULL DEFAULT true,
    churn_prediction TEXT,
    risk_tier TEXT,
    comments TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 3. Indexes for fast dashboard query performance
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON public.predictions (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_risk_tier ON public.predictions (risk_tier);
CREATE INDEX IF NOT EXISTS idx_predictions_customer_id ON public.predictions (customer_id);
CREATE INDEX IF NOT EXISTS idx_feedbacks_created_at ON public.feedbacks (created_at DESC);

-- 4. Enable Row Level Security (RLS)
ALTER TABLE public.predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.feedbacks ENABLE ROW LEVEL SECURITY;

-- Allow public read & insert for web app users (or use service-role key for backend)
CREATE POLICY "Allow public insert to predictions" ON public.predictions
    FOR INSERT WITH CHECK (true);

CREATE POLICY "Allow public read predictions" ON public.predictions
    FOR SELECT USING (true);

CREATE POLICY "Allow public insert to feedbacks" ON public.feedbacks
    FOR INSERT WITH CHECK (true);

CREATE POLICY "Allow public read feedbacks" ON public.feedbacks
    FOR SELECT USING (true);
