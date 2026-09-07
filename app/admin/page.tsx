"use client";

import React, { useState, useEffect } from "react";
import {
  ShieldCheck,
  ShieldAlert,
  AlertTriangle,
  Lock,
  RefreshCw,
  Users,
  Star,
  Activity,
  LogOut,
  Search,
} from "lucide-react";

interface AdminStats {
  total_predictions: number;
  high_risk_count: number;
  medium_risk_count: number;
  low_risk_count: number;
  feedback_count: number;
  average_rating: number;
  recent_predictions: any[];
  recent_feedbacks: any[];
  source?: string;
}

export default function AdminPage() {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const [passkey, setPasskey] = useState<string>("");
  const [authError, setAuthError] = useState<string | null>(null);

  const [stats, setStats] = useState<AdminStats | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [searchTerm, setSearchTerm] = useState<string>("");

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthError(null);
    setLoading(true);

    try {
      const res = await fetch(`/api/admin/stats?passkey=${encodeURIComponent(passkey)}`);
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.message || "Invalid Admin Passkey");
      }

      setIsAuthenticated(true);
      setStats(data);
      sessionStorage.setItem("admin_passkey", passkey);
    } catch (err: any) {
      setAuthError(err.message || "Authentication failed");
    } finally {
      setLoading(false);
    }
  };

  const fetchStats = async () => {
    const savedKey = sessionStorage.getItem("admin_passkey") || passkey;
    if (!savedKey) return;
    setLoading(true);
    try {
      const res = await fetch(`/api/admin/stats?passkey=${encodeURIComponent(savedKey)}`);
      if (res.ok) {
        const data = await res.json();
        setStats(data);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const savedKey = sessionStorage.getItem("admin_passkey");
    if (savedKey) {
      setPasskey(savedKey);
      setIsAuthenticated(true);
      fetch(`/api/admin/stats?passkey=${encodeURIComponent(savedKey)}`)
        .then((r) => r.json())
        .then((d) => setStats(d))
        .catch(() => setIsAuthenticated(false));
    }
  }, []);

  const handleLogout = () => {
    sessionStorage.removeItem("admin_passkey");
    setIsAuthenticated(false);
    setPasskey("");
    setStats(null);
  };

  if (!isAuthenticated) {
    return (
      <div className="flex min-h-[75vh] items-center justify-center px-4">
        <div className="w-full max-w-md rounded-2xl border border-slate-800 bg-slate-900/80 p-8 shadow-2xl backdrop-blur-sm">
          <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-blue-500/10 text-blue-400">
            <Lock className="h-6 w-6" />
          </div>
          <h2 className="mt-4 text-center text-2xl font-bold text-white">
            Admin Analytics Portal
          </h2>
          <p className="mt-1 text-center text-xs text-slate-400">
            Protected dashboard. Please enter your administrator passkey.
          </p>

          <form onSubmit={handleLogin} className="mt-6">
            {authError && (
              <div className="mb-4 rounded-lg border border-rose-500/30 bg-rose-950/30 p-2.5 text-center text-xs text-rose-300">
                {authError}
              </div>
            )}

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400">
                Admin Passkey
              </label>
              <input
                type="password"
                required
                value={passkey}
                onChange={(e) => setPasskey(e.target.value)}
                placeholder="Default: admin123"
                className="mt-2 w-full rounded-lg border border-slate-800 bg-slate-950 px-3.5 py-2.5 text-sm text-white placeholder-slate-600 focus:border-blue-500 focus:outline-none"
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="mt-5 w-full rounded-lg bg-blue-600 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-blue-500 disabled:opacity-50"
            >
              {loading ? "Authenticating..." : "Unlock Dashboard"}
            </button>
          </form>
        </div>
      </div>
    );
  }

  const total = stats?.total_predictions || 0;
  const highPct = total > 0 ? Math.round(((stats?.high_risk_count || 0) / total) * 100) : 0;
  const medPct = total > 0 ? Math.round(((stats?.medium_risk_count || 0) / total) * 100) : 0;
  const lowPct = total > 0 ? Math.round(((stats?.low_risk_count || 0) / total) * 100) : 0;

  const filteredPredictions = stats?.recent_predictions.filter((p) =>
    (p.customer_id || p.CustomerID || "")
      .toLowerCase()
      .includes(searchTerm.toLowerCase())
  );

  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
      {/* Top Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-6">
        <div>
          <h1 className="text-2xl font-bold text-white sm:text-3xl">
            Executive Churn Analytics Dashboard
          </h1>
          <p className="mt-1 text-xs text-slate-400">
            Real-time customer risk segmentation &bull; Database source:{" "}
            <span className="font-semibold text-blue-400">{stats?.source || "Supabase"}</span>
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={fetchStats}
            disabled={loading}
            className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-300 hover:bg-slate-700"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
            <span>Refresh</span>
          </button>

          <button
            onClick={handleLogout}
            className="flex items-center gap-1.5 rounded-lg border border-rose-500/30 bg-rose-950/40 px-3 py-1.5 text-xs font-medium text-rose-300 hover:bg-rose-900/50"
          >
            <LogOut className="h-3.5 w-3.5" />
            <span>Logout</span>
          </button>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="mt-8 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-5">
        <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5">
          <div className="flex items-center gap-2 text-slate-400">
            <Users className="h-4 w-4" />
            <span className="text-xs uppercase font-semibold">Total Scored</span>
          </div>
          <p className="mt-3 text-3xl font-extrabold text-white">{stats?.total_predictions}</p>
          <span className="text-[11px] text-slate-500">Across all segments</span>
        </div>

        <div className="rounded-2xl border border-rose-500/20 bg-rose-950/10 p-5">
          <div className="flex items-center gap-2 text-rose-400">
            <ShieldAlert className="h-4 w-4" />
            <span className="text-xs uppercase font-semibold">High Risk</span>
          </div>
          <p className="mt-3 text-3xl font-extrabold text-rose-400">
            {stats?.high_risk_count}{" "}
            <span className="text-sm font-normal text-rose-400/80">({highPct}%)</span>
          </p>
          <span className="text-[11px] text-rose-400/60">Priority intervention</span>
        </div>

        <div className="rounded-2xl border border-amber-500/20 bg-amber-950/10 p-5">
          <div className="flex items-center gap-2 text-amber-400">
            <AlertTriangle className="h-4 w-4" />
            <span className="text-xs uppercase font-semibold">Medium Risk</span>
          </div>
          <p className="mt-3 text-3xl font-extrabold text-amber-400">
            {stats?.medium_risk_count}{" "}
            <span className="text-sm font-normal text-amber-400/80">({medPct}%)</span>
          </p>
          <span className="text-[11px] text-amber-400/60">Proactive outreach</span>
        </div>

        <div className="rounded-2xl border border-emerald-500/20 bg-emerald-950/10 p-5">
          <div className="flex items-center gap-2 text-emerald-400">
            <ShieldCheck className="h-4 w-4" />
            <span className="text-xs uppercase font-semibold">Low Risk</span>
          </div>
          <p className="mt-3 text-3xl font-extrabold text-emerald-400">
            {stats?.low_risk_count}{" "}
            <span className="text-sm font-normal text-emerald-400/80">({lowPct}%)</span>
          </p>
          <span className="text-[11px] text-emerald-400/60">Healthy subscribers</span>
        </div>

        <div className="rounded-2xl border border-blue-500/20 bg-blue-950/10 p-5">
          <div className="flex items-center gap-2 text-blue-400">
            <Star className="h-4 w-4 fill-blue-400" />
            <span className="text-xs uppercase font-semibold">Model Rating</span>
          </div>
          <p className="mt-3 text-3xl font-extrabold text-blue-400">
            {stats?.average_rating}{" "}
            <span className="text-sm font-normal text-slate-400">/ 5.0</span>
          </p>
          <span className="text-[11px] text-blue-400/60">
            {stats?.feedback_count} feedback logs
          </span>
        </div>
      </div>

      {/* Distribution Progress */}
      <div className="mt-8 rounded-2xl border border-slate-800 bg-slate-900/40 p-5">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          Portfolio Risk Tier Breakdown
        </h3>
        <div className="mt-3 flex h-4 w-full overflow-hidden rounded-full bg-slate-800">
          <div style={{ width: `${lowPct}%` }} className="bg-emerald-500" title={`Low: ${lowPct}%`} />
          <div style={{ width: `${medPct}%` }} className="bg-amber-500" title={`Medium: ${medPct}%`} />
          <div style={{ width: `${highPct}%` }} className="bg-rose-500" title={`High: ${highPct}%`} />
        </div>
        <div className="mt-3 flex flex-wrap items-center justify-between text-xs text-slate-400">
          <div className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full bg-emerald-500" />
            <span>Low Risk: {lowPct}%</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full bg-amber-500" />
            <span>Medium Risk: {medPct}%</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full bg-rose-500" />
            <span>High Risk: {highPct}%</span>
          </div>
        </div>
      </div>

      {/* Activity Tables */}
      <div className="mt-10 grid gap-8 lg:grid-cols-3">
        {/* Recent Predictions Table */}
        <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-6 lg:col-span-2">
          <div className="flex flex-wrap items-center justify-between gap-3 pb-4">
            <h3 className="text-base font-bold text-white">Recent Customer Scores</h3>
            <div className="relative">
              <Search className="absolute top-2.5 left-2.5 h-3.5 w-3.5 text-slate-500" />
              <input
                type="text"
                placeholder="Search Customer ID..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="rounded-lg border border-slate-800 bg-slate-950 py-1.5 pr-3 pl-8 text-xs text-white focus:border-blue-500 focus:outline-none"
              />
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="border-b border-slate-800 text-slate-400 uppercase">
                <tr>
                  <th className="py-2.5 pr-4">Customer</th>
                  <th className="py-2.5 px-3">Tenure</th>
                  <th className="py-2.5 px-3">Contract</th>
                  <th className="py-2.5 px-3">Probability</th>
                  <th className="py-2.5 px-3">Risk Tier</th>
                  <th className="py-2.5 pl-3">Date</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-slate-300">
                {filteredPredictions && filteredPredictions.length > 0 ? (
                  filteredPredictions.map((p, i) => (
                    <tr key={i} className="hover:bg-slate-800/40">
                      <td className="py-3 pr-4 font-mono font-medium text-white">
                        {p.customer_id || p.CustomerID}
                      </td>
                      <td className="py-3 px-3">{p.tenure || "—"} mo</td>
                      <td className="py-3 px-3">{p.contract || "M2M"}</td>
                      <td className="py-3 px-3 font-semibold">
                        {(Number(p.churn_probability) * 100).toFixed(1)}%
                      </td>
                      <td className="py-3 px-3">
                        <span
                          className={`rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase ${
                            p.risk_tier === "High Risk"
                              ? "border border-rose-500/30 bg-rose-950/40 text-rose-400"
                              : p.risk_tier === "Medium Risk"
                              ? "border border-amber-500/30 bg-amber-950/40 text-amber-400"
                              : "border border-emerald-500/30 bg-emerald-950/40 text-emerald-400"
                          }`}
                        >
                          {p.risk_tier}
                        </span>
                      </td>
                      <td className="py-3 pl-3 text-slate-500">
                        {new Date(p.created_at || Date.now()).toLocaleTimeString([], {
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={6} className="py-6 text-center text-slate-500">
                      No matching prediction logs found.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Feedback Stream */}
        <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-6">
          <h3 className="text-base font-bold text-white">Recent Feedback Stream</h3>
          <p className="mt-1 text-xs text-slate-400">Customer success & retention reviews</p>

          <div className="mt-4 divide-y divide-slate-800/60">
            {stats?.recent_feedbacks && stats.recent_feedbacks.length > 0 ? (
              stats.recent_feedbacks.map((f, i) => (
                <div key={i} className="py-3 text-xs">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-slate-300">{f.customer_id}</span>
                    <div className="flex items-center gap-0.5 text-amber-400">
                      {[...Array(f.rating || 5)].map((_, s) => (
                        <Star key={s} className="h-3 w-3 fill-amber-400" />
                      ))}
                    </div>
                  </div>
                  {f.comments && (
                    <p className="mt-1.5 text-slate-400 italic">"{f.comments}"</p>
                  )}
                  <div className="mt-1 flex items-center justify-between text-[10px] text-slate-500">
                    <span>Useful: {f.is_useful ? "Yes" : "No"}</span>
                    <span>{new Date(f.created_at).toLocaleDateString()}</span>
                  </div>
                </div>
              ))
            ) : (
              <p className="py-6 text-center text-xs text-slate-500">
                No feedback received yet.
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
