import clsx from "clsx";

import type { Station } from "@/lib/profile";

type Props = {
  stations: Station[];
  /** Index of the station the student is at. Earlier stations show as passed. */
  current?: number;
  className?: string;
  label: string;
};

/**
 * A single transit line. Horizontal on wide screens, vertical on phones, so labels never collide.
 * Rendered as an ordered list because the stations are a real sequence.
 */
export function RouteLine({ stations, current = 0, className, label }: Props) {
  return (
    <ol aria-label={label} className={clsx("grid gap-0 sm:flex sm:gap-0", className)}>
      {stations.map((station, index) => {
        const state = index < current ? "passed" : index === current ? "here" : "ahead";
        const last = index === stations.length - 1;
        return (
          <li
            key={station.label}
            aria-current={state === "here" ? "step" : undefined}
            className="relative grid grid-cols-[28px_1fr] gap-3 pb-5 last:pb-0 sm:flex-1 sm:grid-cols-1 sm:grid-rows-[28px_auto] sm:gap-2 sm:pb-0 sm:pr-3"
          >
            {!last && (
              <span
                aria-hidden="true"
                className={clsx(
                  "absolute left-[12px] top-[22px] h-[calc(100%-14px)] w-1 rounded-full sm:left-[22px] sm:top-[12px] sm:h-1 sm:w-[calc(100%-14px)]",
                  state === "ahead" ? "bg-rule-strong" : "bg-sign",
                )}
              />
            )}
            <span
              aria-hidden="true"
              className={clsx(
                "relative z-10 grid place-items-center rounded-full border-[3px] bg-surface",
                state === "here" ? "size-7 border-sign" : "ml-[3px] mt-[3px] size-[22px] sm:ml-0",
                state === "passed" && "border-sign bg-sign",
                state === "ahead" && "border-rule-strong",
              )}
            >
              {state === "here" && <span className="size-2.5 rounded-full bg-sign" />}
            </span>
            <span className="grid min-w-0 content-start gap-0.5">
              <span className={clsx("font-display text-[0.95rem] font-bold leading-tight", state === "ahead" && "text-muted")}>
                {station.label}
                {state === "here" && <span className="ml-2 align-middle text-xs font-bold uppercase tracking-wider text-sign">You</span>}
              </span>
              <span className="text-sm leading-snug text-muted">{station.detail}</span>
            </span>
          </li>
        );
      })}
    </ol>
  );
}
