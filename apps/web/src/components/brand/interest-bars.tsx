import clsx from "clsx";

type Row = { key: string; name: string; short: string; score: number; margin?: number };

type Props = {
  rows: Row[];
  /** How many leading rows to emphasise. */
  highlight?: number;
  /** Largest possible value; scores are drawn as a share of it. */
  max?: number;
  label: string;
  showValues?: boolean;
};

/**
 * Horizontal bars, strongest first. When a margin is given, a thin bracket shows
 * the range the true score probably sits in, so near-ties look like near-ties.
 */
export function InterestBars({ rows, highlight = 3, max = 100, label, showValues = true }: Props) {
  const sorted = [...rows].sort((a, b) => b.score - a.score);
  return (
    <ol aria-label={label} className="grid gap-3.5">
      {sorted.map((row, index) => {
        const pct = (row.score / max) * 100;
        const low = row.margin !== undefined ? Math.max(0, row.score - row.margin) / max * 100 : null;
        const high = row.margin !== undefined ? Math.min(max, row.score + row.margin) / max * 100 : null;
        const strong = index < highlight;
        return (
          <li key={row.key} className="grid grid-cols-[minmax(0,8.5rem)_1fr_auto] items-center gap-3 sm:grid-cols-[minmax(0,10rem)_1fr_auto]">
            <span className="min-w-0 leading-tight">
              <span className={clsx("block font-display font-bold", !strong && "text-muted")}>{row.name}</span>
              <span className="block text-sm text-muted">{row.short}</span>
            </span>
            <span className="relative h-3 rounded-full bg-sunk" aria-hidden="true">
              <span
                className={clsx("absolute inset-y-0 left-0 rounded-full", strong ? "bg-sign" : "bg-rule-strong")}
                style={{ width: `${pct}%` }}
              />
              {low !== null && high !== null && (
                <span
                  className="absolute -top-1 -bottom-1 rounded-[3px] border-2 border-ink/45"
                  style={{ left: `${low}%`, width: `${Math.max(high - low, 1)}%` }}
                />
              )}
            </span>
            {showValues && (
              <span className="tabular w-8 text-right font-mono text-sm">
                <span className="sr-only">
                  {row.name}: {row.score}
                  {row.margin !== undefined ? `, give or take ${row.margin}` : ""}
                </span>
                <span aria-hidden="true">{row.score}</span>
              </span>
            )}
          </li>
        );
      })}
    </ol>
  );
}
