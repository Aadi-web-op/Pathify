import clsx from "clsx";
import type { ReactNode } from "react";

type Tone = "error" | "warn" | "info";

const tones: Record<Tone, string> = {
  error: "bg-danger-tint text-ink border-danger/40",
  warn: "bg-warn-tint text-ink border-warn/40",
  info: "bg-sign-tint text-ink border-sign/30",
};

export function Notice({ tone = "info", children, className }: { tone?: Tone; children: ReactNode; className?: string }) {
  return (
    <div
      role={tone === "error" ? "alert" : "status"}
      className={clsx("rounded-2xl border px-4 py-3 text-[0.95rem] leading-snug", tones[tone], className)}
    >
      {children}
    </div>
  );
}
