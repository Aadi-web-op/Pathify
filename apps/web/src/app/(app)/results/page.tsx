"use client";

import { useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { CareerMap } from "@/components/brand/career-map";
import { RouteField } from "@/components/fx/route-field";
import { InterestBars } from "@/components/brand/interest-bars";
import { RouteLine } from "@/components/brand/route-line";
import { Button, ButtonLink, Spinner } from "@/components/ui/button";
import { Notice } from "@/components/ui/notice";
import { ApiError, type StudentProfile } from "@/lib/api";
import { extendAssessment, OVERVIEW_KEY, startAssessment, useMatches, useOverview } from "@/lib/assessment";
import { chosenRoute, LINE_COLORS, mapLine, startStation } from "@/lib/career-map-data";
import { clusterLabel, type Match, type Result, withArticle, yearsText } from "@/lib/careers";
import { burstFrom } from "@/lib/burst";
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
        <h1 className="text-[2.5rem] font-semibold leading-[1]">No results yet</h1>
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
      <ResultsHero
        title={`You're most drawn to ${names[0]}, ${names[1]} and ${names[2]} work`}
        totalCareers={total_careers}
        top={top}
        celebrateKey={`${result.answered}:${result.ranked.join("")}`}
      />

      <section aria-labelledby="map-title" className="grid gap-4 spotlight rounded-3xl border border-rule bg-surface p-4 sm:p-6">
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
        <section aria-labelledby="interests" className="grid content-start gap-5 spotlight rounded-3xl border border-rule bg-surface p-5 sm:p-6">
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
        <section aria-labelledby="sure" className="grid content-start gap-3 rounded-3xl border border-rule p-5 sm:p-6">
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
        <h2 id="top" className="text-2xl font-bold">
          Your top three, up close
        </h2>
        {top.map((match, index) => (
          <MatchCard key={match.slug} match={match} index={index} profile={profile} />
        ))}
      </section>

      {rest.length > 0 && (
        <section aria-labelledby="more" className="grid gap-4">
          <h2 id="more" className="text-2xl font-bold">
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

      <section aria-labelledby="family" className="grid gap-3 rounded-3xl border-2 border-sign bg-surface p-5 sm:flex sm:items-center sm:justify-between sm:p-6">
        <div className="grid gap-1">
          <h2 id="family" className="text-xl font-bold">
            Talk it through at home
          </h2>
          <p className="text-muted">A one-page report for your parents or guardian, in English or Hindi, with questions to discuss.</p>
        </div>
        <ButtonLink href="/report" className="justify-self-start">
          Open the family report
        </ButtonLink>
      </section>

      <details className="group rounded-3xl border border-rule p-5 sm:p-6">
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
    <article id={`match-${match.slug}`} className="grid scroll-mt-24 gap-5 spotlight rounded-3xl border border-rule bg-surface p-5 sm:p-7">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-center gap-3">
          <span
            aria-hidden="true"
            className="grid h-8 min-w-11 place-items-center rounded-md font-display font-bold text-[var(--surface)]"
            style={{ background: LINE_COLORS[index] }}
          >
            {index + 1}
          </span>
          <div>
            <h3 className="text-2xl font-bold">{match.title}</h3>
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

      {match.tryout && (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg bg-sunk px-4 py-3">
          <p className="text-[0.95rem]">
            {match.tryout.enjoyment ? (
              <>
                You tried <span className="font-semibold">{match.tryout.title}</span> and rated it{" "}
                <span className="font-semibold">{match.tryout.enjoyment} of 5</span>.
              </>
            ) : (
              <>
                Feel the work: <span className="font-semibold">{match.tryout.title}</span>, about {match.tryout.minutes} minutes.
              </>
            )}
          </p>
          <ButtonLink href={`/tryouts/${match.tryout.slug}`} size="sm" variant={match.tryout.enjoyment ? "ghost" : "primary"}>
            {match.tryout.enjoyment ? "See it again" : "Try it"}
          </ButtonLink>
        </div>
      )}

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

/** The reveal: a night card with the live map behind it, and one burst of confetti the first time you see these results. */
function ResultsHero({ title, totalCareers, top, celebrateKey }: { title: string; totalCareers: number; top: Match[]; celebrateKey: string }) {
  const ref = useRef<HTMLElement>(null);
  useEffect(() => {
    const key = `pathify:celebrated:${celebrateKey}`;
    try {
      if (sessionStorage.getItem(key)) return;
      sessionStorage.setItem(key, "1");
    } catch {
      // Storage can be unavailable (private mode); celebrating twice is fine.
    }
    const id = window.setTimeout(() => burstFrom(ref.current?.querySelector("h1"), { power: 1, confetti: true }), 450);
    return () => window.clearTimeout(id);
  }, [celebrateKey]);

  return (
    <section ref={ref} className="relative isolate overflow-hidden rounded-[28px] border border-rule bg-surface p-6 sm:p-10">
      <RouteField className="absolute inset-0 -z-10 size-full opacity-50" />
      <div aria-hidden="true" className="absolute inset-0 -z-10 bg-[radial-gradient(ellipse_80%_90%_at_20%_40%,var(--surface)_35%,transparent_85%)]" />
      <div className="grid gap-5">
        <p className="label-mono flex w-fit items-center gap-2 rounded-full border border-rule-strong bg-surface px-3 py-1.5 text-ink">
          <span className="size-1.5 rounded-full bg-pop" /> Your results are in
        </p>
        <h1 className="rise max-w-3xl text-[2.4rem] font-semibold leading-[1] tracking-[-0.04em] sm:text-[3.1rem]">{title}</h1>
        <p className="max-w-[62ch] text-muted">
          These are the careers whose day-to-day best matches what you enjoy, out of {totalCareers}, drawn as routes from
          where you are now. They&apos;re places to start exploring, not a verdict.
        </p>
        <ol className="flex flex-wrap gap-2" aria-label="Your top three matches">
          {top.map((m, i) => (
            <li
              key={m.slug}
              className="rise flex items-center gap-2 rounded-full border border-rule bg-surface py-1.5 pl-1.5 pr-4"
              style={{ ["--rise-delay" as string]: `${0.3 + i * 0.12}s` }}
            >
              <span className="grid size-7 place-items-center rounded-full font-mono text-xs font-bold text-white" style={{ background: LINE_COLORS[i] }}>
                {i + 1}
              </span>
              <span className="font-display font-semibold">{m.title}</span>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
