import Link from "next/link";
import { ShieldCheck, GitFork, BookOpen } from "lucide-react";

export default function Footer() {
  return (
    <footer className="mt-auto border-t border-slate-800 bg-slate-950 py-8 text-slate-400">
      <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-4 px-4 sm:flex-row sm:px-6 lg:px-8">
        <div className="flex items-center gap-2 text-sm">
          <ShieldCheck className="h-4 w-4 text-emerald-400" />
          <span>
            Customer Churn Forecasting Platform &bull; Production ML Pipeline
          </span>
        </div>

        <div className="flex items-center gap-6 text-xs">
          <Link
            href="/about"
            className="flex items-center gap-1.5 transition-colors hover:text-slate-200"
          >
            <BookOpen className="h-3.5 w-3.5" />
            <span>Methodology & Reports</span>
          </Link>
          <a
            href="https://github.com/TrishulD/customer-churn-prediction"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 transition-colors hover:text-slate-200"
          >
            <GitFork className="h-3.5 w-3.5" />
            <span>GitHub Repository</span>
          </a>
        </div>
      </div>
    </footer>
  );
}
