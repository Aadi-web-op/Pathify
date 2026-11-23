"use client";

import { useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";

import { Button, ButtonLink, Spinner } from "@/components/ui/button";
import { Notice } from "@/components/ui/notice";
import { burstFrom } from "@/lib/burst";
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

// From "really dislike" to "love it": cool to hot.
const SCALE_COLORS = ["var(--route-blue)", "var(--route-violet)", "var(--rule-strong)", "var(--route-pink)", "var(--pop)"];

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
        <h1 className="text-[2.6rem] font-bold tracking-[-0.045em]">What do you enjoy doing?</h1>
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
          <p className="label-mono font-semibold">
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
          <span className="absolute inset-x-0 top-1/2 h-2 -translate-y-1/2 rounded-full bg-sunk" />
          <span
            className="absolute left-0 top-1/2 h-2 -translate-y-1/2 rounded-full bg-[linear-gradient(90deg,var(--route-violet),var(--route-pink),var(--route-saffron))] shadow-[0_0_18px_color-mix(in_oklab,var(--route-pink)_55%,transparent)] transition-[width] duration-500 ease-[cubic-bezier(0.34,1.56,0.64,1)] motion-reduce:transition-none"
            style={{ width: `${progress * 100}%` }}
          />
          <span
            className="absolute top-1/2 size-6 -translate-x-1/2 -translate-y-1/2 rounded-full border-4 border-paper bg-pop shadow-[0_0_0_2px_var(--route-pink)] transition-[left] duration-500 ease-[cubic-bezier(0.34,1.56,0.64,1)] motion-reduce:transition-none"
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
            className="min-h-[2.4em] text-[2rem] font-bold leading-[1.05] tracking-[-0.045em] focus:outline-none sm:text-[2.9rem]"
          >
            <span key={question.id} className="rise block">
              {question.text}
            </span>
          </h1>
        </div>

        <fieldset className="grid gap-2.5 sm:grid-cols-5" disabled={busy}>
          <legend className="sr-only">Your answer</legend>
          {session.scale.map((point, index) => (
            <button
              key={point.value}
              type="button"
              onClick={(e) => {
                // A small celebration at every tenth answer: a checkpoint, not every click.
                if ((done + 1) % 10 === 0) burstFrom(e.currentTarget, { confetti: true });
                void answer(point.value);
              }}
              aria-pressed={chosen === point.value}
              style={{ ["--scale-color" as string]: SCALE_COLORS[index] ?? "var(--sign)" }}
              className={clsx(
                "flex items-center gap-3 rounded-2xl border-2 px-4 py-3.5 text-left font-display font-semibold sm:flex-col sm:gap-2.5 sm:px-2 sm:py-5 sm:text-center",
                "transition-[transform,border-color,background-color,box-shadow] duration-300 ease-[cubic-bezier(0.34,1.56,0.64,1)] active:scale-[0.94]",
                chosen === point.value
                  ? "border-sign bg-sign text-sign-ink shadow-[0_14px_40px_-16px_var(--sign)]"
                  : "border-rule bg-surface hover:-translate-y-1 hover:border-[var(--scale-color)] hover:shadow-[0_14px_40px_-22px_var(--scale-color)] disabled:hover:translate-y-0 disabled:hover:border-rule",
              )}
            >
              <span
                aria-hidden="true"
                className={clsx(
                  "grid size-8 shrink-0 place-items-center rounded-full font-mono text-sm font-bold",
                  chosen === point.value ? "bg-sign-ink text-sign" : "bg-[var(--scale-color)] text-[#0b0a12]",
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
