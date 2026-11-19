"use client";

import { useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";

import { Button, ButtonLink, Spinner } from "@/components/ui/button";
import { Notice } from "@/components/ui/notice";
import { ApiError } from "@/lib/api";
import {
  answerQuestion,
  goBack,
  MATCHES_KEY,
  type Overview,
  OVERVIEW_KEY,
  type Session,
  startAssessment,
  useOverview,
} from "@/lib/assessment";

export default function AssessmentPage() {
  const overview = useOverview();
  const [session, setSession] = useState<Session | null>(null);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function begin(restart: boolean) {
    setStarting(true);
    setError(null);
    try {
      setSession(await startAssessment(restart));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Couldn't start the assessment. Try again.");
    } finally {
      setStarting(false);
    }
  }

  if (overview.isPending) {
    return (
      <p className="flex items-center gap-2 text-muted">
        <Spinner /> Loading…
      </p>
    );
  }
  if (overview.error) return <Notice tone="error">{overview.error.message}</Notice>;

  // Pick up an unfinished test (including one being extended) straight away.
  const current = session ?? overview.data.in_progress;
  if (current && current.status === "in_progress") {
    return <Questions session={current} onChange={setSession} />;
  }

  const latest = overview.data.latest;
  return (
    <div className="grid max-w-2xl gap-8">
      <div className="grid gap-3">
        <p className="eyebrow">Interest assessment</p>
        <h1 className="text-[2.4rem] font-extrabold leading-[1.05] tracking-[-0.015em]">What do you enjoy doing?</h1>
        <p className="text-lg text-muted">
          You&apos;ll see {overview.data.standard_length} short activities. For each one, say how much you&apos;d enjoy
          it. Not how good you are at it, and not what sounds impressive.
        </p>
      </div>
      <dl className="grid gap-4 border-y border-rule py-5 sm:grid-cols-3">
        <Fact term="Takes" value="About 5 minutes" />
        <Fact term="Answers" value="No right or wrong ones" />
        <Fact term="Based on" value="Holland's six interest types (RIASEC)" />
      </dl>
      {latest && (
        <Notice tone="info">
          You already have results from{" "}
          {new Date(latest.completed_at).toLocaleDateString("en-IN", { day: "numeric", month: "long" })}. Taking the
          test again replaces them.{" "}
          <Link href="/results" className="font-bold underline underline-offset-4">
            See your results
          </Link>
        </Notice>
      )}
      {error && <Notice tone="error">{error}</Notice>}
      <div className="flex flex-wrap gap-3">
        <Button size="lg" busy={starting} onClick={() => begin(Boolean(latest))}>
          {latest ? "Take it again" : "Start"}
        </Button>
        {latest && (
          <ButtonLink href="/results" size="lg" variant="secondary">
            Back to my results
          </ButtonLink>
        )}
      </div>
    </div>
  );
}

function Fact({ term, value }: { term: string; value: string }) {
  return (
    <div className="grid gap-0.5">
      <dt className="text-sm text-muted">{term}</dt>
      <dd className="font-display font-bold leading-snug">{value}</dd>
    </div>
  );
}

function Questions({ session, onChange }: { session: Session; onChange: (s: Session) => void }) {
  const router = useRouter();
  const queryClient = useQueryClient();
  const [chosen, setChosen] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const heading = useRef<HTMLHeadingElement>(null);
  const question = session.question;
  const extending = session.result !== null && session.result !== undefined;
  const base = extending ? (session.result?.answered ?? 0) : 0;
  const total = session.target - base;
  const done = session.answered - base;

  useEffect(() => {
    setChosen(null);
    heading.current?.focus({ preventScroll: true });
  }, [question?.id]);

  const answer = useCallback(
    async (value: number) => {
      if (!question || busy) return;
      setChosen(value);
      setBusy(true);
      setError(null);
      try {
        const next = await answerQuestion(session.id, question.id, value);
        if (next.status === "completed") {
          await queryClient.invalidateQueries({ queryKey: OVERVIEW_KEY });
          await queryClient.invalidateQueries({ queryKey: MATCHES_KEY });
          router.push("/results");
          return;
        }
        onChange(next);
        queryClient.setQueryData<Overview>(OVERVIEW_KEY, (old) => (old ? { ...old, in_progress: next } : old));
      } catch (e) {
        setChosen(null);
        if (e instanceof ApiError && e.status === 409) {
          // Another tab moved on; pick up from the server's state.
          onChange(await startAssessment(false));
        } else {
          setError(e instanceof ApiError ? e.message : "Your answer didn't save. Try again.");
        }
      } finally {
        setBusy(false);
      }
    },
    [busy, onChange, queryClient, question, router, session.id],
  );

  async function back() {
    setBusy(true);
    setError(null);
    try {
      onChange(await goBack(session.id));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Couldn't go back. Try again.");
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.target instanceof HTMLElement && ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) return;
      const n = Number(event.key);
      if (n >= 1 && n <= session.scale.length) {
        event.preventDefault();
        void answer(session.scale[n - 1].value);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [answer, session.scale]);

  if (!question) return null;
  const progress = Math.min(1, done / total);

  return (
    <div className="mx-auto grid max-w-3xl gap-8">
      <div className="grid gap-3">
        <div className="flex flex-wrap items-baseline justify-between gap-2 text-sm">
          <p className="font-display font-bold">
            {extending ? "Sharpening your results · " : ""}
            {done + 1} of {total}
          </p>
          <Link href="/home" className="text-muted underline underline-offset-4 hover:text-ink">
            Save and finish later
          </Link>
        </div>
        <div
          role="progressbar"
          aria-label="Progress"
          aria-valuemin={0}
          aria-valuemax={total}
          aria-valuenow={done}
          className="relative h-6"
        >
          <span className="absolute inset-x-0 top-1/2 h-1 -translate-y-1/2 rounded-full bg-rule-strong" />
          <span
            className="absolute left-0 top-1/2 h-1 -translate-y-1/2 rounded-full bg-sign transition-[width] duration-300 motion-reduce:transition-none"
            style={{ width: `${progress * 100}%` }}
          />
          <span
            className="absolute top-1/2 size-5 -translate-x-1/2 -translate-y-1/2 rounded-full border-[3px] border-sign bg-surface transition-[left] duration-300 motion-reduce:transition-none"
            style={{ left: `${progress * 100}%` }}
          />
        </div>
      </div>

      <section aria-labelledby="statement" className="grid gap-8">
        <div className="grid gap-2">
          <p className="text-muted">How much would you enjoy this?</p>
          <h1
            id="statement"
            ref={heading}
            tabIndex={-1}
            className="min-h-[2.4em] text-[1.9rem] font-extrabold leading-[1.15] tracking-[-0.01em] focus:outline-none sm:text-[2.3rem]"
          >
            {question.text}
          </h1>
        </div>

        <fieldset className="grid gap-2.5 sm:grid-cols-5" disabled={busy}>
          <legend className="sr-only">Your answer</legend>
          {session.scale.map((point, index) => (
            <button
              key={point.value}
              type="button"
              onClick={() => answer(point.value)}
              aria-pressed={chosen === point.value}
              className={clsx(
                "flex items-center gap-3 rounded-lg border-2 px-4 py-3.5 text-left font-display font-bold transition-colors sm:flex-col sm:gap-2 sm:px-2 sm:py-4 sm:text-center",
                chosen === point.value
                  ? "border-sign bg-sign text-sign-ink"
                  : "border-rule bg-surface hover:border-sign disabled:hover:border-rule",
              )}
            >
              <span
                aria-hidden="true"
                className={clsx(
                  "grid size-7 shrink-0 place-items-center rounded-full border-2 font-mono text-sm",
                  chosen === point.value ? "border-sign-ink" : "border-rule-strong text-muted",
                )}
              >
                {index + 1}
              </span>
              <span className="leading-tight">{point.label}</span>
            </button>
          ))}
        </fieldset>
        <p className="hidden text-sm text-muted sm:block">Tip: press 1 to 5 on your keyboard.</p>
      </section>

      {error && <Notice tone="error">{error}</Notice>}

      <div className="border-t border-rule pt-5">
        <Button variant="ghost" onClick={back} disabled={!session.can_go_back || busy}>
          Back
        </Button>
      </div>
    </div>
  );
}
