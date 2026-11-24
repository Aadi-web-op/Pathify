"use client";

import clsx from "clsx";
import Link from "next/link";

import { usePlan, useProgress } from "@/lib/plan";

/** Level, XP, streak, badges and the next step of the plan, on Home. */
export function ProgressCard() {
  const progress = useProgress();
  const plan = usePlan();
  if (!progress.data) return null;
  const p = progress.data;
  const span = Math.max(1, p.next_level_at - p.level_floor);
  const pct = Math.min(100, Math.round(((p.xp - p.level_floor) / span) * 100));
  const next = plan.data?.milestones.find((m) => !m.done_at);
  const earned = p.badges.filter((b) => b.earned);

  return (
    <section
      aria-labelledby="progress"
      className="spotlight grid gap-5 rounded-3xl border border-rule bg-surface p-5 sm:p-7"
    >
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="grid gap-1">
          <p className="eyebrow">Your progress</p>
          <h2
            id="progress"
            className="text-2xl font-semibold tracking-[-0.03em]"
          >
            Level {p.level}
          </h2>
        </div>
        <dl className="flex gap-6 font-mono text-sm">
          <div>
            <dt className="text-muted">XP</dt>
            <dd className="text-lg font-semibold tabular">{p.xp}</dd>
          </div>
          <div>
            <dt className="text-muted">Streak</dt>
            <dd className="text-lg font-semibold tabular">
              {p.streak_days} {p.streak_days === 1 ? "day" : "days"}
            </dd>
          </div>
        </dl>
      </div>
      <div className="grid gap-1.5">
        <div
          className="h-2 overflow-hidden rounded-full bg-sunk"
          role="progressbar"
          aria-label="XP towards the next level"
          aria-valuenow={pct}
          aria-valuemin={0}
          aria-valuemax={100}
        >
          <div
            className="h-full rounded-full bg-sign transition-[width] duration-700"
            style={{ width: `${pct}%` }}
          />
        </div>
        <p className="text-xs text-muted">
          {p.next_level_at - p.xp} XP to level {p.level + 1}. You earn XP for
          real steps: the assessment, tryouts, reflections and your plan.
        </p>
      </div>

      <div className="grid gap-2">
        <p className="text-sm font-semibold">
          Badges{" "}
          <span className="font-normal text-muted">
            ({earned.length} of {p.badges.length})
          </span>
        </p>
        <ul className="flex flex-wrap gap-2">
          {p.badges.map((b) => (
            <li
              key={b.key}
              title={b.description}
              className={clsx(
                "rounded-full border px-3 py-1 text-xs font-semibold",
                b.earned
                  ? "border-sign bg-sign-tint text-sign"
                  : "border-dashed border-rule-strong text-muted",
              )}
            >
              {b.name}
              <span className="sr-only">
                {b.earned ? " (earned)" : " (not yet)"}: {b.description}
              </span>
            </li>
          ))}
        </ul>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-rule pt-4">
        {plan.data ? (
          <p className="text-sm">
            <span className="text-muted">Next on your plan:</span>{" "}
            <span className="font-semibold">
              {next ? next.label : "All done. Time for a new goal?"}
            </span>
          </p>
        ) : (
          <p className="text-sm text-muted">
            Pick a route and Pathify turns it into steps you can tick off.
          </p>
        )}
        <Link
          href="/plan"
          className="font-semibold text-sign underline underline-offset-4"
        >
          {plan.data ? "Open my plan" : "Make a plan"}
        </Link>
      </div>
    </section>
  );
}
