"use client";

import { useEffect, useState } from "react";

import { CAREERS } from "@/lib/careers";

/**
 * A departures card: careers as destinations, the entrance exam as
 * "via", when you board, and how many years the ride takes. Every value is read
 * from the career library, so nothing on the board is made up.
 */

const SLUGS = [
  "ux-designer",
  "doctor",
  "software-engineer",
  "lawyer",
  "architect",
  "chartered-accountant",
  "electrician",
  "civil-servant",
  "hotel-manager",
  "veterinarian",
  "armed-forces-officer",
  "journalist",
];

type Row = { dest: string; via: string; board: string; years: string; color: string };

const COLORS = ["var(--route-violet)", "var(--route-pink)", "var(--route-blue)", "var(--route-saffron)"];

function shortExam(label: string) {
  return label.split(/,| or /)[0].replace(/\(.*?\)/g, "").trim();
}

export const DEPARTURES: Row[] = SLUGS.flatMap((slug, i) => {
  const career = CAREERS.find((c) => c.slug === slug);
  if (!career) return [];
  const route = career.routes[0];
  const exam = route.steps.find((s) => s.kind === "exam");
  const years = route.steps.reduce((sum, s) => sum + (s.years ?? 0), 0);
  return [
    {
      dest: career.title.replace(/\s*\(.*\)$/, ""),
      via: exam ? shortExam(exam.label) : "Direct",
      board: route.after === "class_10" ? "Class 10" : route.after === "class_12" ? "Class 12" : "Graduate",
      years: years ? `${years} yr` : "—",
      color: COLORS[i % COLORS.length],
    },
  ];
});

export function DepartureBoard() {
  const [offset, setOffset] = useState(0);
  const visible = 4;

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const id = window.setInterval(() => setOffset((o) => (o + 1) % DEPARTURES.length), 3400);
    return () => window.clearInterval(id);
  }, []);

  const rows = Array.from({ length: visible }, (_, i) => DEPARTURES[(offset + i) % DEPARTURES.length]);

  return (
    <div
      role="img"
      aria-label={`Example routes from the career library, such as ${DEPARTURES.slice(0, 3)
        .map((r) => `${r.dest} via ${r.via}, after ${r.board}, about ${r.years.replace("yr", "years")}`)
        .join("; ")}.`}
      className="overflow-hidden rounded-3xl border border-rule bg-surface p-2 shadow-[0_30px_60px_-40px_rgba(20,18,40,0.35)]"
    >
      <div className="flex items-center justify-between gap-3 px-4 pb-2 pt-3">
        <span className="label-mono flex items-center gap-2 text-ink">
          <span className="size-2 rounded-full bg-pop" />
          Next departures
        </span>
        <span className="label-mono text-muted">From: you</span>
      </div>
      <ul className="grid">
        {rows.map((row, i) => (
          <li
            key={`${offset}-${i}`}
            className="rise grid grid-cols-[auto_1fr_auto] items-center gap-3 rounded-2xl px-4 py-3 odd:bg-paper"
            style={{ ["--rise-delay" as string]: `${i * 0.06}s` }}
          >
            <span className="h-8 w-1 rounded-full" style={{ background: row.color }} />
            <span className="grid min-w-0">
              <span className="truncate font-display text-[1.05rem] font-semibold tracking-[-0.02em]">{row.dest}</span>
              <span className="truncate font-mono text-xs text-muted">
                via {row.via} · after {row.board}
              </span>
            </span>
            <span className="font-mono text-sm tabular text-ink">{row.years}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
