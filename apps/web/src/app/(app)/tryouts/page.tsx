"use client";

import Link from "next/link";

import { Spinner } from "@/components/ui/button";
import { Notice } from "@/components/ui/notice";
import { useMatches, useOverview } from "@/lib/assessment";
import { scoreText, useTryouts } from "@/lib/tryouts";

export default function TryoutsPage() {
  const tryouts = useTryouts();
  const overview = useOverview();
  const matches = useMatches(Boolean(overview.data?.latest));
  const topCareers = new Set((matches.data?.matches ?? []).slice(0, 6).map((m) => m.slug));

  if (tryouts.isPending) {
    return (
      <p className="flex items-center gap-2 text-muted">
        <Spinner /> Loading tryouts…
      </p>
    );
  }
  if (tryouts.error) return <Notice tone="error">{tryouts.error.message}</Notice>;

  const sorted = [...tryouts.data].sort((a, b) => Number(topCareers.has(b.career)) - Number(topCareers.has(a.career)));

  return (
    <div className="grid gap-10">
      <div className="grid max-w-3xl gap-3">
        <p className="eyebrow">Job tryouts</p>
        <h1 className="text-[2.4rem] font-extrabold leading-[1.05] tracking-[-0.015em]">Try the work before you choose it</h1>
        <p className="text-lg text-muted">
          Each tryout is a short, simplified version of a real task from the job. You&apos;ll get feedback, see how a
          professional would approach it, and then say how much you enjoyed it. That last part matters most.
        </p>
      </div>

      <ul className="grid gap-4 md:grid-cols-2">
        {sorted.map((t) => {
          const attempt = t.latest_attempt;
          const recommended = topCareers.has(t.career);
          return (
            <li key={t.slug}>
              <Link
                href={`/tryouts/${t.slug}`}
                className="group grid h-full content-start gap-3 rounded-[14px] border border-rule bg-surface p-5 transition-colors hover:border-sign sm:p-6"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <p className="font-display text-sm font-bold text-muted">
                    {t.career_title}, {t.minutes} min
                  </p>
                  {recommended && (
                    <span className="rounded-full bg-sign-tint px-2.5 py-0.5 font-display text-xs font-bold text-sign">
                      One of your matches
                    </span>
                  )}
                </div>
                <h2 className="text-xl font-extrabold group-hover:text-sign">{t.title}</h2>
                <p className="text-[0.95rem] text-muted">{t.role}</p>
                {attempt && (
                  <p className="border-t border-rule pt-3 text-sm">
                    <span className="font-display font-bold">Tried</span> · {scoreText(attempt.points, attempt.max_points, attempt.ungraded)}
                    {attempt.enjoyment ? ` · enjoyed it ${attempt.enjoyment} of 5` : ""}
                  </p>
                )}
              </Link>
            </li>
          );
        })}
      </ul>
      <p className="text-sm text-muted">Scenarios, people and numbers in tryouts are made up for practice.</p>
    </div>
  );
}
