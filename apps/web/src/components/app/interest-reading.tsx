"use client";

import { DIMENSIONS } from "@/lib/careers";
import { useInterestReading } from "@/lib/insights";

/** A live first guess from the free-text interests, before the questionnaire. */
export function InterestReading({ items }: { items: string[] }) {
  const reading = useInterestReading(items);
  if (!items.length || !reading.data) return null;
  const name = (key: string) => DIMENSIONS.find((d) => d.key === key)?.short ?? key;
  const [first, second] = reading.data.top;

  return (
    <div className="grid gap-3 rounded-2xl border border-rule bg-sunk/60 p-4" aria-live="polite">
      <p className="text-sm">
        From what you&apos;ve written, you sound <strong className="font-semibold text-sign">{name(first)}</strong> and{" "}
        <strong className="font-semibold text-sign">{name(second)}</strong>.
      </p>
      <ul className="grid grid-cols-6 gap-2" aria-label="Interest estimate from your words">
        {reading.data.top
          .slice()
          .sort((a, b) => "RIASEC".indexOf(a) - "RIASEC".indexOf(b))
          .map((key) => (
            <li key={key} className="grid gap-1">
              <span className="h-12 overflow-hidden rounded-md bg-surface">
                <span
                  className="block w-full rounded-md bg-sign transition-[height] duration-500"
                  style={{ height: `${reading.data.scores[key]}%`, marginTop: `${100 - reading.data.scores[key]}%` }}
                />
              </span>
              <span className="truncate text-center font-mono text-[0.65rem] text-muted">{name(key)}</span>
            </li>
          ))}
      </ul>
      <p className="text-xs text-muted">
        A quick guess by Pathify&apos;s Interest Reader, a model we fine-tuned to read interests in English, Hinglish and
        Hindi. The questionnaire later gives the real answer.
      </p>
    </div>
  );
}
