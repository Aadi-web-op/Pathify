"use client";

import { useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { CareerMap } from "@/components/brand/career-map";
import { InterestBars } from "@/components/brand/interest-bars";
import { RouteLine } from "@/components/brand/route-line";
import { Button, ButtonLink, Spinner } from "@/components/ui/button";
import { Notice } from "@/components/ui/notice";
import { ApiError, type StudentProfile } from "@/lib/api";
import { extendAssessment, OVERVIEW_KEY, startAssessment, useMatches, useOverview } from "@/lib/assessment";
import { chosenRoute, LINE_COLORS, mapLine, startStation } from "@/lib/career-map-data";
import { clusterLabel, type Match, type Result, withArticle, yearsText } from "@/lib/careers";
import { useMe } from "@/lib/session";

const CONFIDENCE: Record<Result["confidence"], { title: string; body: string }> = {
  high: {
    title: "Your top matches are fairly settled.",
    body: "Small changes in your answers wouldn't reshuffle them much.",
  },
  medium: {
    title: "Your top matches are a good starting point.",
    body: "A few careers near the top are close together, so their order could change.",
  },
  low: {
    title: "Your top matches could still move around.",
    body: "Several of your interests are close together. Answering the remaining statements will sharpen the list.",
  },
};

export default function ResultsPage() {
  const { data: me } = useMe();
  const matches = useMatches();
  const overview = useOverview();

  if (matches.isPending) {
    return (
      <p className="flex items-center gap-2 text-muted">
        <Spinner /> Working out your matches…
      </p>
    );
  }
  if (matches.error?.status === 404) {
    return (
      <div className="grid max-w-xl gap-4">
        <h1 className="text-[2.2rem] font-extrabold">No results yet</h1>
        <p className="text-muted">Take the interest assessment and your career map appears here. It takes about 5 minutes.</p>
        <ButtonLink href="/assessment" className="justify-self-start">
          Start the assessment
        </ButtonLink>
      </div>
    );
  }
  if (matches.error || !me?.profile) return <Notice tone="error">{matches.error?.message ?? "Something went wrong."}</Notice>;

  const { result, matches: list, total_careers } = matches.data;
  const profile = me.profile;
  const top = list.slice(0, 3);
  const rest = list.slice(3);
  const names = result.ranked.slice(0, 3).map((k) => result.dimensions.find((d) => d.key === k)!.name);

  return (
    <div className="grid gap-12">
      <section className="grid gap-3">
        <p className="eyebrow">Your results</p>
        <h1 className="max-w-3xl text-[2.2rem] font-extrabold leading-[1.08] tracking-[-0.015em] sm:text-[2.6rem]">
          You&apos;re most drawn to {names[0]}, {names[1]} and {names[2]} work
        </h1>
        <p className="max-w-[62ch] text-muted">
          Here are the careers whose day-to-day best matches what you enjoy, out of {total_careers}, drawn as routes from
          where you are now. They&apos;re places to start exploring, not a verdict.
        </p>
      </section>

      <section aria-labelledby="map-title" className="grid gap-4 rounded-[14px] border border-rule bg-surface p-4 sm:p-6">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 id="map-title" className="text-xl font-bold">
            Your career map
          </h2>
          <p className="text-sm text-muted">Tap a line to jump to its details</p>
        </div>
        <CareerMap
          start={startStation(profile)}
          lines={top.map((m, i) => mapLine(m, profile, i))}
          label={`Routes from where you are to your top matches: ${top.map((m) => m.title).join(", ")}.`}
        />
        <p className="text-sm text-muted sm:hidden">Swipe the map sideways to see every stop.</p>
      </section>

      <div className="grid gap-6 lg:grid-cols-[3fr_2fr]">
        <section aria-labelledby="interests" className="grid content-start gap-5 rounded-[14px] border border-rule bg-surface p-5 sm:p-6">
          <div className="grid gap-1">
            <h2 id="interests" className="text-xl font-bold">
              Your interests
            </h2>
            <p className="text-sm text-muted">
              How much you enjoy each kind of activity, 0 to 100. The bracket shows where the score probably sits.
            </p>
          </div>
          <InterestBars
            label="Your interest scores, strongest first"
            rows={result.dimensions.map((d) => ({ key: d.key, name: d.name, short: d.short, score: d.score, margin: d.margin }))}
          />
        </section>
        <section aria-labelledby="sure" className="grid content-start gap-3 rounded-[14px] border border-rule p-5 sm:p-6">
          <h2 id="sure" className="text-xl font-bold">
            How sure is this?
          </h2>
          <p className="font-display font-bold">{CONFIDENCE[result.confidence].title}</p>
          <p className="text-muted">{CONFIDENCE[result.confidence].body}</p>
          {result.even_spread && (
            <p className="text-muted">
              Your interests are quite even across the board. That&apos;s common, and it means many paths could suit you,
              so the job tryouts we&apos;re building will help more than scores.
            </p>
          )}
          <p className="text-sm text-muted">Based on {result.answered} answers.</p>
          <ResultActions canExtend={Boolean(overview.data?.can_extend)} sessionId={overview.data?.latest_session_id ?? null} />
        </section>
      </div>

      <section aria-labelledby="top" className="grid gap-6">
        <h2 id="top" className="text-2xl font-extrabold">
          Your top three, up close
        </h2>
        {top.map((match, index) => (
          <MatchCard key={match.slug} match={match} index={index} profile={profile} />
        ))}
      </section>

      {rest.length > 0 && (
        <section aria-labelledby="more" className="grid gap-4">
          <h2 id="more" className="text-2xl font-extrabold">
            More to explore
          </h2>
          <ol className="grid border-t border-rule" start={4}>
            {rest.map((m, i) => (
              <li key={m.slug} className="grid grid-cols-[2rem_minmax(0,1fr)_auto] items-center gap-3 border-b border-rule py-3">
                <span className="font-mono text-sm text-muted">{i + 4}</span>
                <span className="min-w-0">
                  <Link href={`/careers/${m.slug}`} className="font-display font-bold underline-offset-4 hover:underline">
                    {m.title}
                  </Link>
                  <span className="block text-sm text-muted">
                    {clusterLabel(m.cluster)}
                    {!m.open_to_you && " · not open from your stream"}
                  </span>
                </span>
                <FitPill match={m} />
              </li>
            ))}
          </ol>
          <Link href="/careers" className="justify-self-start font-semibold text-sign underline underline-offset-4">
            Browse all careers
          </Link>
        </section>
      )}

      <details className="group rounded-[14px] border border-rule p-5 sm:p-6">
        <summary className="cursor-pointer font-display text-lg font-bold">How we work this out</summary>
        <div className="mt-4 grid max-w-[68ch] gap-3 text-muted">
          <p>
            Your answers give you a score for each of Holland&apos;s six interest types: Realistic, Investigative,
            Artistic, Social, Enterprising and Conventional. Each career in Pathify has its own six-part profile, which
            we estimated from what the work involves, guided by the U.S. Department of Labor&apos;s O*NET ratings for
            similar jobs.
          </p>
          <p>
            Fit compares the <em>shape</em> of your interests with each career&apos;s: which types you enjoy more than
            others, not how high your scores are overall. The reasons on each card show which types pushed the fit up
            or down.
          </p>
          <p>
            In simulations, 30 answers find about 7 in 10 of a student&apos;s true top-five careers, and answering all
            48 raises that to nearly 8 in 10. Interests also change as you try things, so use this to decide what to
            explore next.
          </p>
        </div>
      </details>
    </div>
  );
}

function ResultActions({ canExtend, sessionId }: { canExtend: boolean; sessionId: string | null }) {
  const router = useRouter();
  const queryClient = useQueryClient();
  const [busy, setBusy] = useState<"extend" | "retake" | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function run(kind: "extend" | "retake") {
    setBusy(kind);
    setError(null);
    try {
      if (kind === "extend" && sessionId) await extendAssessment(sessionId);
      else await startAssessment(true);
      await queryClient.invalidateQueries({ queryKey: OVERVIEW_KEY });
      router.push("/assessment");
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "That didn't work. Try again.");
      setBusy(null);
    }
  }

  return (
    <div className="grid gap-3 pt-2">
      {error && <Notice tone="error">{error}</Notice>}
      <div className="flex flex-wrap gap-2">
        {canExtend && sessionId && (
          <Button busy={busy === "extend"} disabled={busy !== null} onClick={() => run("extend")}>
            Answer 18 more
          </Button>
        )}
        <Button variant="secondary" busy={busy === "retake"} disabled={busy !== null} onClick={() => run("retake")}>
          Retake
        </Button>
      </div>
    </div>
  );
}

function FitPill({ match }: { match: Match }) {
  const strong = match.fit >= 70;
  return (
    <span
      className={clsx(
        "tabular whitespace-nowrap rounded-full px-2.5 py-1 font-display text-sm font-bold",
        strong ? "bg-sign-tint text-sign" : "bg-sunk text-muted",
      )}
    >
      {match.fit_label}
      <span className="ml-2 font-mono text-[0.8rem] opacity-80">{match.fit}</span>
    </span>
  );
}

function MatchCard({ match, index, profile }: { match: Match; index: number; profile: StudentProfile }) {
  const route = chosenRoute(match);
  const stations = route.steps.map((s) => ({ label: s.label, detail: s.detail ?? (s.years ? yearsText(s.years) : "") }));
  const others = match.routes.filter((r) => r.id !== route.id && r.open);
  const ask = `What is it really like to become ${withArticle(match.title)}, starting from where I am?`;
  return (
    <article id={`match-${match.slug}`} className="grid scroll-mt-24 gap-5 rounded-[14px] border border-rule bg-surface p-5 sm:p-7">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-center gap-3">
          <span
            aria-hidden="true"
            className="grid h-8 min-w-11 place-items-center rounded-md font-display font-extrabold text-[var(--surface)]"
            style={{ background: LINE_COLORS[index] }}
          >
            {index + 1}
          </span>
          <div>
            <h3 className="text-2xl font-extrabold">{match.title}</h3>
            <p className="text-sm text-muted">{clusterLabel(match.cluster)}</p>
          </div>
        </div>
        <FitPill match={match} />
      </header>
      <p className="max-w-[65ch]">{match.summary}</p>

      <div className="grid gap-5 sm:grid-cols-2">
        <div className="grid content-start gap-2">
          <h4 className="font-display font-bold">Why it fits</h4>
          <ul className="grid gap-1.5 text-muted">
            {match.reasons.length ? (
              match.reasons.map((r) => (
                <li key={r.dimension} className="flex gap-2">
                  <span aria-hidden="true" className="mt-2 size-1.5 shrink-0 rounded-full bg-sign" />
                  {r.text}
                </li>
              ))
            ) : (
              <li>Your interests are spread evenly, so no single type stands out for this one.</li>
            )}
          </ul>
        </div>
        {match.watch_outs.length > 0 && (
          <div className="grid content-start gap-2">
            <h4 className="font-display font-bold">Keep in mind</h4>
            <ul className="grid gap-1.5 text-muted">
              {match.watch_outs.map((r) => (
                <li key={r.dimension} className="flex gap-2">
                  <span aria-hidden="true" className="mt-2 size-1.5 shrink-0 rounded-full bg-saffron" />
                  {r.text}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      <div className="grid gap-3 border-t border-rule pt-5">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h4 className="font-display font-bold">{route.open ? `Route: ${route.label}` : "The usual route"}</h4>
          {route.years > 0 && <p className="text-sm text-muted">{yearsText(route.years)} of study and training</p>}
        </div>
        {!route.open && (
          <Notice tone="warn">
            {route.note ?? "This route isn't open from your current stream."}{" "}
            {profile.stage === "school_11_12" && "Ask Ksarth about other ways in, or talk to your school counsellor."}
          </Notice>
        )}
        {route.open && route.note && <p className="text-sm text-muted">{route.note}</p>}
        <RouteLine stations={stations} current={-1} label={`Steps to become ${withArticle(match.title)}`} />
        {others.length > 0 && (
          <p className="text-sm text-muted">
            Also possible: {others.map((r) => r.label.toLowerCase()).join("; ")}.
          </p>
        )}
      </div>

      <footer className="flex flex-wrap gap-x-6 gap-y-2 border-t border-rule pt-4">
        <Link href={`/careers/${match.slug}`} className="font-semibold text-sign underline underline-offset-4">
          Full career guide
        </Link>
        <Link href={`/ksarth?ask=${encodeURIComponent(ask)}`} className="font-semibold text-sign underline underline-offset-4">
          Ask Ksarth about it
        </Link>
      </footer>
    </article>
  );
}
