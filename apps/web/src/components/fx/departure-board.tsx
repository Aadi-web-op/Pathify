"use client";

import { useEffect, useState } from "react";

import { CAREERS } from "@/lib/careers";

/**
 * A split-flap departures board: careers as destinations, the entrance exam as
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

const COLORS = ["var(--route-violet)", "var(--route-pink)", "var(--route-blue)", "var(--route-saffron)", "var(--pop)"];

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

const GLYPHS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/";

/** One line of flaps. When `text` changes, each cell spins through letters, then lands, left to right. */
function Flaps({ text, cells }: { text: string; cells: number }) {
  const target = text.toUpperCase().padEnd(cells, " ").slice(0, cells);
  const [shown, setShown] = useState(target);

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setShown(target);
      return;
    }
    const start = performance.now();
    let raf = 0;
    let lastStep = 0;
    const step = (now: number) => {
      if (now - lastStep > 45) {
        lastStep = now;
        const elapsed = now - start;
        let done = true;
        const next = target
          .split("")
          .map((ch, i) => {
            if (elapsed > 180 + i * 38) return ch;
            done = false;
            return ch === " " && elapsed > i * 38 ? " " : GLYPHS[Math.floor(Math.random() * GLYPHS.length)];
          })
          .join("");
        setShown(next);
        if (done) return;
      }
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [target]);

  return (
    <span className="flex gap-[2px]">
      {shown.split("").map((ch, i) => (
        <span
          key={i}
          className="relative grid h-[1.7em] w-[0.98em] place-items-center overflow-hidden rounded-[3px] bg-[#16161f] text-[#f3f1ff] shadow-[inset_0_-1px_0_rgba(255,255,255,0.04)] after:absolute after:inset-x-0 after:top-1/2 after:h-px after:bg-black/70"
        >
          {ch === " " ? " " : ch}
        </span>
      ))}
    </span>
  );
}

export function DepartureBoard() {
  const [offset, setOffset] = useState(0);
  const visible = 4;

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const id = window.setInterval(() => setOffset((o) => (o + 1) % DEPARTURES.length), 3200);
    return () => window.clearInterval(id);
  }, []);

  const rows = Array.from({ length: visible }, (_, i) => DEPARTURES[(offset + i) % DEPARTURES.length]);

  return (
    <div
      role="img"
      aria-label={`Departures board of example routes, such as ${DEPARTURES.slice(0, 3)
        .map((r) => `${r.dest} via ${r.via}, after ${r.board}, about ${r.years.replace("yr", "years")}`)
        .join("; ")}.`}
      className="overflow-hidden rounded-2xl border border-white/10 bg-[#0b0b11] p-3 font-mono text-[0.7rem] text-[#f3f1ff] shadow-[0_30px_80px_-30px_rgba(164,141,255,0.45)] sm:p-4 sm:text-[0.8rem]">
      <div className="mb-3 flex items-center justify-between gap-3 px-1">
        <span className="flex items-center gap-2 tracking-[0.18em] text-[#d7ff3c]">
          <span className="relative flex size-2">
            <span className="absolute inset-0 animate-pulse-ring rounded-full bg-[#d7ff3c]" />
            <span className="relative size-2 rounded-full bg-[#d7ff3c]" />
          </span>
          DEPARTURES
        </span>
        <span className="tracking-[0.18em] text-white/45">FROM: YOU</span>
      </div>
      <div className="flex justify-between gap-3 px-1 pb-1 pl-4 text-[0.62rem] tracking-[0.18em] text-white/40">
        <span>DESTINATION{" "}<span className="max-sm:hidden">/ VIA</span></span>
        <span>AFTER / RIDE</span>
      </div>
      <ul className="grid gap-1.5">
        {rows.map((row, i) => (
          <li key={i} className="flex items-center gap-2 overflow-hidden">
            <span className="h-[1.65em] w-1 shrink-0 rounded-full" style={{ background: row.color }} />
            <span className="flex min-w-0 flex-1 items-center justify-between gap-3">
              <span className="flex items-center gap-2">
                <Flaps text={row.dest} cells={13} />
                <span className="max-sm:hidden">
                  <Flaps text={row.via} cells={8} />
                </span>
              </span>
              <span className="grid shrink-0 justify-items-end whitespace-nowrap leading-tight">
                <span className="text-[0.85em] tracking-[0.12em] text-white/50">{row.board}</span>
                <span className="tracking-[0.1em] text-[#d7ff3c]">{row.years}</span>
              </span>
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
