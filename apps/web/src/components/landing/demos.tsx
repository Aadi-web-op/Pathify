"use client";

import clsx from "clsx";
import { AnimatePresence, motion, useInView, useReducedMotion } from "motion/react";
import { useEffect, useRef, useState } from "react";

/** Runs `tick` every `ms` while the element is on screen (and never with reduced motion). */
function useTicker(ms: number) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { margin: "-10% 0px" });
  const reduce = useReducedMotion();
  const [n, setN] = useState(0);
  useEffect(() => {
    if (!inView || reduce) return;
    const id = window.setInterval(() => setN((v) => v + 1), ms);
    return () => window.clearInterval(id);
  }, [inView, reduce, ms]);
  return { ref, n };
}

const AXES = ["Hands-on", "Curious", "Creative", "Helping", "Leading", "Organising"];
const PROFILES = [
  { name: "The builder-thinker", values: [0.85, 0.9, 0.35, 0.3, 0.4, 0.6] },
  { name: "The storyteller", values: [0.25, 0.5, 0.95, 0.7, 0.55, 0.2] },
  { name: "The people person", values: [0.3, 0.4, 0.5, 0.95, 0.75, 0.45] },
  { name: "The organiser", values: [0.4, 0.55, 0.2, 0.45, 0.6, 0.95] },
];

function hexPath(values: number[], r: number, c: number) {
  return (
    values
      .map((v, i) => {
        const a = (Math.PI / 3) * i - Math.PI / 2;
        return `${i ? "L" : "M"}${(c + Math.cos(a) * r * v).toFixed(1)} ${(c + Math.sin(a) * r * v).toFixed(1)}`;
      })
      .join(" ") + " Z"
  );
}

/** The six interest types as a radar that morphs between example profiles. */
export function RiasecDemo() {
  const { ref, n } = useTicker(2600);
  const profile = PROFILES[n % PROFILES.length];
  const c = 110;
  const r = 82;
  return (
    <div ref={ref} className="grid place-items-center gap-3">
      <svg viewBox="0 0 220 220" className="w-full max-w-[260px] overflow-visible" aria-hidden="true">
        <defs>
          <linearGradient id="hex-fill" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="var(--route-violet)" stopOpacity="0.75" />
            <stop offset="1" stopColor="var(--route-pink)" stopOpacity="0.55" />
          </linearGradient>
        </defs>
        {[1, 0.66, 0.33].map((s) => (
          <path key={s} d={hexPath(Array(6).fill(s), r, c)} fill="none" stroke="var(--rule-strong)" strokeWidth="1" />
        ))}
        {AXES.map((label, i) => {
          const a = (Math.PI / 3) * i - Math.PI / 2;
          return (
            <g key={label}>
              <line x1={c} y1={c} x2={c + Math.cos(a) * r} y2={c + Math.sin(a) * r} stroke="var(--rule)" />
              <text
                x={c + Math.cos(a) * (r + 18)}
                y={c + Math.sin(a) * (r + 18) + 3}
                textAnchor="middle"
                className="fill-muted font-mono text-[8.5px] uppercase tracking-[0.08em]"
              >
                {label}
              </text>
            </g>
          );
        })}
        <motion.path
          d={hexPath(profile.values, r, c)}
          animate={{ d: hexPath(profile.values, r, c) }}
          transition={{ type: "spring", stiffness: 120, damping: 14 }}
          fill="url(#hex-fill)"
          stroke="var(--pop)"
          strokeWidth="2"
          strokeLinejoin="round"
        />
      </svg>
      <AnimatePresence mode="wait">
        <motion.p
          key={profile.name}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          className="label-mono rounded-full border border-rule px-3 py-1 text-ink"
        >
          {profile.name}
        </motion.p>
      </AnimatePresence>
    </div>
  );
}

const CHAT = [
  { who: "you", text: "Which exam do I need to become a doctor?" },
  { who: "ksarth", text: "NEET-UG, after Class 12 with Physics, Chemistry and Biology. Then MBBS, about 5.5 years with the internship. [1]" },
];

/** Ksarth answering, with a typing indicator and a citation chip. */
export function ChatDemo() {
  const { ref, n } = useTicker(1700);
  const reduce = useReducedMotion();
  const step = reduce ? 3 : n % 5; // 0: question, 1: typing, 2-4: answer shown
  return (
    <div ref={ref} className="grid min-h-[190px] content-start gap-2.5 text-[0.9rem]" aria-hidden="true">
      <Bubble mine>{CHAT[0].text}</Bubble>
      <AnimatePresence mode="popLayout">
        {step === 1 && (
          <motion.div key="typing" initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}>
            <span className="inline-flex gap-1 rounded-2xl rounded-bl-md bg-sunk px-3.5 py-3">
              {[0, 1, 2].map((i) => (
                <motion.span
                  key={i}
                  className="size-1.5 rounded-full bg-muted"
                  animate={{ y: [0, -4, 0] }}
                  transition={{ repeat: Infinity, duration: 0.8, delay: i * 0.12 }}
                />
              ))}
            </span>
          </motion.div>
        )}
        {step >= 2 && (
          <motion.div key="answer" initial={{ opacity: 0, y: 10, scale: 0.96 }} animate={{ opacity: 1, y: 0, scale: 1 }}>
            <Bubble>
              {CHAT[1].text.replace(" [1]", "")}{" "}
              <span className="ml-0.5 inline-grid size-5 place-items-center rounded-md bg-sign align-[1px] font-mono text-[0.65rem] font-bold text-sign-ink">
                1
              </span>
            </Bubble>
            <p className="label-mono mt-2 pl-1 text-muted">Source: Pathify guide, Doctor</p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function Bubble({ mine, children }: { mine?: boolean; children: React.ReactNode }) {
  return (
    <p
      className={clsx(
        "max-w-[88%] px-3.5 py-2.5 leading-snug",
        mine ? "justify-self-end rounded-2xl rounded-br-md bg-sign text-sign-ink" : "rounded-2xl rounded-bl-md bg-sunk",
      )}
    >
      {children}
    </p>
  );
}

/** A tryout ticket: a ten-minute ring that runs down, then a "feedback ready" stamp. */
export function TryoutDemo() {
  const { ref, n } = useTicker(120);
  const reduce = useReducedMotion();
  const cycle = 70;
  const t = reduce ? cycle : n % (cycle + 25);
  const progress = Math.min(t / cycle, 1);
  const left = Math.ceil((1 - progress) * 600);
  const mm = String(Math.floor(left / 60)).padStart(2, "0");
  const ss = String(left % 60).padStart(2, "0");
  const C = 2 * Math.PI * 34;
  return (
    <div ref={ref} className="flex items-center gap-4" aria-hidden="true">
      <div className="relative grid size-[92px] shrink-0 place-items-center">
        <svg viewBox="0 0 80 80" className="absolute inset-0 -rotate-90">
          <circle cx="40" cy="40" r="34" fill="none" stroke="var(--rule)" strokeWidth="6" />
          <circle
            cx="40"
            cy="40"
            r="34"
            fill="none"
            stroke="var(--route-saffron)"
            strokeWidth="6"
            strokeLinecap="round"
            strokeDasharray={C}
            strokeDashoffset={C * progress}
            className="transition-[stroke-dashoffset] duration-150"
          />
        </svg>
        <span className="font-mono text-sm font-semibold tabular">
          {mm}:{ss}
        </span>
      </div>
      <div className="grid min-w-0 gap-1.5">
        <p className="label-mono text-muted">Doctor · 10 min</p>
        <p className="font-display text-lg font-bold leading-tight">Three patients, one doctor</p>
        <div className="h-7">
          {progress >= 1 && (
            <motion.span
              initial={{ scale: 2.2, rotate: -14, opacity: 0 }}
              animate={{ scale: 1, rotate: -4, opacity: 1 }}
              transition={{ type: "spring", stiffness: 420, damping: 14 }}
              className="label-mono inline-block rounded-md border-2 border-pop bg-pop px-2 py-0.5 font-bold text-pop-ink"
            >
              Feedback ready
            </motion.span>
          )}
        </div>
      </div>
    </div>
  );
}

const REPORT = {
  en: { title: "Career report for Asha", line: "Asha enjoys investigating and solving problems, and this career is full of it." },
  hi: { title: "आशा की करियर रिपोर्ट", line: "आशा को खोज करना और समस्याएँ सुलझाना पसंद है, और इस करियर में यह बहुत होता है।" },
};

/** The family report flipping between English and Hindi. */
export function ReportDemo() {
  const { ref, n } = useTicker(2800);
  const lang = n % 2 ? "hi" : "en";
  return (
    <div ref={ref} className="grid gap-3 [perspective:900px]" aria-hidden="true">
      <div className="flex gap-1 self-start rounded-full border border-rule p-1 font-mono text-xs">
        {(["en", "hi"] as const).map((l) => (
          <span key={l} className={clsx("rounded-full px-3 py-1 transition-colors", lang === l ? "bg-ink text-paper" : "text-muted")}>
            {l === "en" ? "English" : "हिंदी"}
          </span>
        ))}
      </div>
      <AnimatePresence mode="wait">
        <motion.div
          key={lang}
          initial={{ rotateX: -80, opacity: 0 }}
          animate={{ rotateX: 0, opacity: 1 }}
          exit={{ rotateX: 80, opacity: 0 }}
          transition={{ duration: 0.45, ease: [0.2, 0.7, 0.2, 1] }}
          className="grid gap-2 rounded-xl border border-rule bg-paper p-4 [transform-origin:50%_0%]"
        >
          <p className="font-display font-bold">{REPORT[lang].title}</p>
          <p className="text-sm leading-relaxed text-muted">{REPORT[lang].line}</p>
          <div className="mt-1 flex gap-1.5">
            {[0.9, 0.7, 0.45, 0.3, 0.55, 0.35].map((w, i) => (
              <span key={i} className="h-1.5 flex-1 overflow-hidden rounded-full bg-sunk">
                <span className="block h-full rounded-full bg-sign" style={{ width: `${w * 100}%` }} />
              </span>
            ))}
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

/** Three tiny routes drawing in, for the career-map tile. */
export function MiniMap() {
  const ref = useRef<SVGSVGElement>(null);
  const inView = useInView(ref, { once: true, margin: "-10% 0px" });
  const routes = [
    { d: "M14 70 H70 L100 40 H200 L230 70 H286", color: "var(--route-violet)" },
    { d: "M14 70 H286", color: "var(--route-pink)" },
    { d: "M14 70 H90 L120 100 H190 L220 70 H286", color: "var(--route-blue)" },
  ];
  return (
    <svg ref={ref} viewBox="0 0 300 130" className="w-full" aria-hidden="true">
      {routes.map((r, i) => (
        <motion.path
          key={i}
          d={r.d}
          fill="none"
          stroke={r.color}
          strokeWidth="6"
          strokeLinecap="round"
          strokeLinejoin="round"
          initial={{ pathLength: 0 }}
          animate={inView ? { pathLength: 1 } : undefined}
          transition={{ duration: 1.4, delay: i * 0.2, ease: [0.5, 0, 0.2, 1] }}
        />
      ))}
      {[
        [14, 70],
        [100, 40],
        [150, 70],
        [155, 100],
        [286, 70],
      ].map(([x, y], i) => (
        <motion.circle
          key={i}
          cx={x}
          cy={y}
          r={i === 0 || i === 4 ? 8 : 6}
          fill="var(--surface)"
          stroke={i === 0 ? "var(--pop)" : "var(--ink)"}
          strokeWidth="3"
          initial={{ scale: 0 }}
          animate={inView ? { scale: 1 } : undefined}
          transition={{ type: "spring", stiffness: 400, damping: 12, delay: 0.6 + i * 0.12 }}
        />
      ))}
    </svg>
  );
}
