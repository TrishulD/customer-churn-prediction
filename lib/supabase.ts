import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || "";
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || "";
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY || "";

export const isSupabaseConfigured = () => {
  return Boolean(
    supabaseUrl &&
    supabaseUrl !== "https://your-project-id.supabase.co" &&
    (supabaseAnonKey || supabaseServiceKey)
  );
};

// Public client for browser operations (uses anon key)
export const supabase = isSupabaseConfigured()
  ? createClient(supabaseUrl, supabaseAnonKey || supabaseServiceKey)
  : null;

// Server client for backend routes (prefers service-role key for administrative bypass)
export const getServiceSupabase = () => {
  if (!isSupabaseConfigured()) return null;
  return createClient(supabaseUrl, supabaseServiceKey || supabaseAnonKey, {
    auth: { persistSession: false },
  });
};
