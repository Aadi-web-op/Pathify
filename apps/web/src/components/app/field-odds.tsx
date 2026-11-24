"use client";

import { useFieldOdds } from "@/lib/insights";

/** Pathfinder on the results page: what people with similar interests actually studied. */
export function FieldOddsSection() {
  const odds = useFieldOdds();
  if (!odds.data) return null;
  const top = odds.data.fields.slice(0, 5);
  const max = Math.max(...top.map((f) => f.probability));

  return (
    <section aria-labelledby="field-odds" className="spotlight grid gap-5 rounded-3xl border border-rule bg-surface p-5 sm:p-6">
      <div className="grid gap-1">
        <p className="eyebrow">Pathfinder</p>
        <h2 id="field-odds" className="text-xl font-semibold">
          What people like you studied
        </h2>
        <p className="max-w-[65ch] text-sm text-muted">
          Among {odds.data.trained_on.toLocaleString("en-IN")} people who took a similar interest test and told us their
          field, these were the most common for interests like yours.
        </p>
      </div>
      <ol className="grid gap-3">
        {top.map((field) => (
          <li key={field.key} className="grid gap-1.5">
            <div className="flex flex-wrap items-baseline justify-between gap-x-3">
              <span className="font-display font-semibold">{field.label}</span>
              <span className="font-mono text-xs text-muted">
                {Math.round(field.probability * 100)}%
                {field.lift >= 1.15 && <span className="text-sign"> · {field.lift.toFixed(1)}× the average</span>}
              </span>
            </div>
            <span className="h-2 overflow-hidden rounded-full bg-sunk">
              <span className="block h-full rounded-full bg-sign" style={{ width: `${(field.probability / max) * 100}%` }} />
            </span>
          </li>
        ))}
      </ol>
      <p className="border-t border-rule pt-4 text-xs text-muted">
        On people it hadn&apos;t seen, the field they studied was in its top three {Math.round(odds.data.test_top3 * 100)}%
        of the time. It describes what others did, not what you should do. The data comes from an international online
        sample, so treat it as one more clue.
      </p>
    </section>
  );
}
