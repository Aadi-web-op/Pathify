"use client";

import clsx from "clsx";
import { AnimatePresence, motion, useMotionValueEvent, useScroll, useSpring } from "motion/react";
import { useRef, useState } from "react";

export type Stop = { when: string; question: string; body: string; exams: string[] };

const COLORS = ["var(--route-violet)", "var(--route-pink)", "var(--route-saffron)"];
// Station positions along the drawn route (fractions of its length) and on the 400x600 canvas.
const STATIONS = [
  { at: 0.12, x: 80, y: 90 },
  { at: 0.5, x: 230, y: 300 },
  { at: 0.88, x: 320, y: 520 },
];
const PATH = "M80 20 V150 L230 300 V400 L320 490 V580";

/**
 * Scroll-pinned story: the section sticks while you scroll, the route draws
 * itself, and each station lights up with the decision you face there.
 */
export function Journey({ stops }: { stops: Stop[] }) {
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start start", "end end"] });
  const progress = useSpring(scrollYProgress, { stiffness: 120, damping: 28, mass: 0.3 });
  const [active, setActive] = useState(0);
  useMotionValueEvent(scrollYProgress, "change", (v) => setActive(v < 0.36 ? 0 : v < 0.7 ? 1 : 2));
  const stop = stops[active];

  return (
    <section ref={ref} aria-labelledby="journey" className="relative h-[300vh]">
      <div className="sticky top-0 flex h-svh items-center overflow-hidden">
        <div className="gutter relative mx-auto grid w-full max-w-6xl items-center gap-8 md:grid-cols-[1.1fr_1fr] [&>*]:min-w-0">
          <div className="grid gap-4 md:gap-6">
            <p className="label-mono text-muted">
              <span className="text-ink">0{active + 1}</span> / 03 · Three big decisions
            </p>
            <h2 id="journey" className="sr-only">
              Three big decisions, one guide
            </h2>
            <AnimatePresence mode="wait">
              <motion.div
                key={active}
                initial={{ opacity: 0, y: 30, filter: "blur(10px)" }}
                animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
                exit={{ opacity: 0, y: -30, filter: "blur(10px)" }}
                transition={{ duration: 0.45, ease: [0.2, 0.7, 0.2, 1] }}
                className="grid gap-4 md:gap-5"
              >
                <span className="label-mono flex w-fit items-center gap-2 rounded-full border border-rule-strong px-3 py-1 text-ink">
                  <span className="size-2 rounded-full" style={{ background: COLORS[active] }} />
                  Stop {active + 1}: {stop.when}
                </span>
                <p className="font-display text-[clamp(2rem,5.5vw,4rem)] font-semibold leading-[1] tracking-[-0.04em]">
                  {stop.question}
                </p>
                <p className="max-w-[48ch] text-muted md:text-lg">{stop.body}</p>
                <ul className="flex flex-wrap gap-2" aria-label="Examples">
                  {stop.exams.map((e) => (
                    <li key={e} className="label-mono rounded-full border border-rule-strong px-2.5 py-1 text-ink md:px-3 md:py-1.5">
                      {e}
                    </li>
                  ))}
                </ul>
              </motion.div>
            </AnimatePresence>
            {/* Every stop stays in the page for screen readers and search, whichever one is showing. */}
            <ol className="sr-only">
              {stops.map((s) => (
                <li key={s.when}>
                  {s.when}: {s.question} {s.body}
                </li>
              ))}
            </ol>
          </div>

          <svg viewBox="0 0 400 600" className="h-[26svh] w-full max-md:order-first md:h-[72svh]" aria-hidden="true">
            <path d={PATH} fill="none" stroke="var(--rule)" strokeWidth="8" strokeLinecap="round" strokeLinejoin="round" />
            <motion.path
              d={PATH}
              fill="none"
              stroke="url(#journey-grad)"
              strokeWidth="8"
              strokeLinecap="round"
              strokeLinejoin="round"
              style={{ pathLength: progress }}
            />
            <defs>
              <linearGradient id="journey-grad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0" stopColor="var(--route-violet)" />
                <stop offset="0.5" stopColor="var(--route-pink)" />
                <stop offset="1" stopColor="var(--route-saffron)" />
              </linearGradient>
            </defs>
            {STATIONS.map((s, i) => (
              <g key={i}>
                <circle
                  cx={s.x}
                  cy={s.y}
                  r={i <= active ? 16 : 12}
                  fill="var(--paper)"
                  stroke={i <= active ? COLORS[i] : "var(--rule-strong)"}
                  strokeWidth="4"
                  className="transition-all duration-500 ease-[cubic-bezier(0.34,1.56,0.64,1)]"
                />
                <circle
                  cx={s.x}
                  cy={s.y}
                  r={i === active ? 6 : 0}
                  fill="var(--pop)"
                  className="transition-all duration-500 ease-[cubic-bezier(0.34,1.56,0.64,1)]"
                />
                <text
                  x={s.x + (i === 2 ? -38 : 38)}
                  y={s.y + 5}
                  textAnchor={i === 2 ? "end" : "start"}
                  className={clsx("font-mono text-[15px] uppercase tracking-[0.1em]", i <= active ? "fill-ink" : "fill-muted")}
                >
                  {stops[i].when}
                </text>
              </g>
            ))}
          </svg>
        </div>
      </div>
    </section>
  );
}
