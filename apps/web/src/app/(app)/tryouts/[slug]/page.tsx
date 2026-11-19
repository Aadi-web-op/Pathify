"use client";

import { useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import Link from "next/link";
import { useParams } from "next/navigation";
import { type FormEvent, useMemo, useState } from "react";

import { Button, ButtonLink, Spinner } from "@/components/ui/button";
import { Choice } from "@/components/ui/choice";
import { Input } from "@/components/ui/field";
import { Notice } from "@/components/ui/notice";
import { ApiError } from "@/lib/api";
import { MATCHES_KEY } from "@/lib/assessment";
import {
  type AnswerValue,
  type Attempt,
  saveReflection,
  scoreText,
  submitTryout,
  type Tryout,
  TRYOUTS_KEY,
  useAttempt,
  useTryout,
  useTryouts,
} from "@/lib/tryouts";

type Task = Tryout["tasks"][number];

export default function TryoutPage() {
  const { slug } = useParams<{ slug: string }>();
  const tryout = useTryout(slug);
  const list = useTryouts();
  const previous = list.data?.find((t) => t.slug === slug)?.latest_attempt ?? null;
  const [attempt, setAttempt] = useState<Attempt | null>(null);
  const [showPrevious, setShowPrevious] = useState(false);
  const past = useAttempt(showPrevious && previous ? previous.id : null);

  if (tryout.isPending) {
    return (
      <p className="flex items-center gap-2 text-muted">
        <Spinner /> Loading the tryout…
      </p>
    );
  }
  if (tryout.error) {
    return <Notice tone="error">{tryout.error.message}</Notice>;
  }

  const shown = attempt ?? (showPrevious ? past.data : undefined);
  return (
    <div className="mx-auto grid max-w-3xl gap-10">
      <header className="grid gap-3">
        <Link href="/tryouts" className="text-sm text-muted underline underline-offset-4 hover:text-ink">
          All tryouts
        </Link>
        <p className="eyebrow">
          {tryout.data.career_title}, about {tryout.data.minutes} minutes
        </p>
        <h1 className="text-[2.2rem] font-extrabold leading-[1.08] tracking-[-0.015em] sm:text-[2.6rem]">{tryout.data.title}</h1>
        <p className="text-lg">{tryout.data.role}</p>
        <p className="text-muted">{tryout.data.brief}</p>
      </header>

      {previous && !attempt && !showPrevious && (
        <Notice tone="info">
          You tried this before: {scoreText(previous.points, previous.max_points, previous.ungraded)}.{" "}
          <button type="button" className="font-bold underline underline-offset-4" onClick={() => setShowPrevious(true)}>
            See that feedback
          </button>
        </Notice>
      )}

      <Materials tryout={tryout.data} />

      {shown ? (
        <Review attempt={shown} onRetry={() => (setAttempt(null), setShowPrevious(false))} />
      ) : showPrevious && past.isPending ? (
        <p className="flex items-center gap-2 text-muted">
          <Spinner /> Loading your feedback…
        </p>
      ) : (
        <Tasks tryout={tryout.data} onSubmitted={setAttempt} />
      )}
    </div>
  );
}

function Materials({ tryout }: { tryout: Tryout }) {
  return (
    <section aria-label="Materials" className="grid gap-4">
      {tryout.materials.map((m) => (
        <div key={m.title} className="grid gap-3 rounded-[14px] border border-rule bg-surface p-5">
          <h2 className="font-display text-lg font-bold">{m.title}</h2>
          {m.kind === "table" && (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[420px] border-collapse text-[0.95rem]">
                <thead>
                  <tr>
                    {m.columns?.map((c, i) => (
                      <th key={c} scope="col" className={clsx("border-b-2 border-ink px-3 py-2 font-display font-bold", i === 0 ? "text-left" : "text-right")}>
                        {c}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {m.rows?.map((row, r) => (
                    <tr key={r} className={clsx(r === (m.rows?.length ?? 0) - 1 && row[0] === "Total" && "font-bold")}>
                      {row.map((cell, i) =>
                        i === 0 ? (
                          <th key={i} scope="row" className="border-b border-rule px-3 py-2 text-left font-normal">
                            {cell}
                          </th>
                        ) : (
                          <td key={i} className="tabular border-b border-rule px-3 py-2 text-right font-mono">
                            {cell}
                          </td>
                        ),
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {m.kind === "list" && (
            <ul className="grid gap-2">
              {m.items?.map((item) => (
                <li key={item} className="flex gap-3">
                  <span aria-hidden="true" className="mt-2.5 size-1.5 shrink-0 rounded-full bg-ink" />
                  {item}
                </li>
              ))}
            </ul>
          )}
          {m.kind === "text" && <p className="whitespace-pre-line rounded-lg bg-sunk px-4 py-3 text-[1.02rem]">{m.body}</p>}
          {m.note && <p className="text-sm text-muted">{m.note}</p>}
        </div>
      ))}
    </section>
  );
}

function wordCount(text: string) {
  return (text.trim().match(/\S+/g) ?? []).length;
}

function answered(task: Task, value: AnswerValue | undefined) {
  if (value === undefined) return false;
  if (Array.isArray(value)) return value.length > 0;
  if (typeof value === "string") return value.trim().length > 0;
  return !Number.isNaN(value);
}

function Tasks({ tryout, onSubmitted }: { tryout: Tryout; onSubmitted: (a: Attempt) => void }) {
  const queryClient = useQueryClient();
  const [answers, setAnswers] = useState<Record<string, AnswerValue>>({});
  const [numberText, setNumberText] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const done = tryout.tasks.filter((t) => answered(t, answers[t.id])).length;
  const hasWritten = tryout.tasks.some((t) => t.kind === "written");

  const set = (id: string, value: AnswerValue) => setAnswers((a) => ({ ...a, [id]: value }));

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const attempt = await submitTryout(tryout.slug, answers);
      await queryClient.invalidateQueries({ queryKey: TRYOUTS_KEY });
      onSubmitted(attempt);
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Your answers didn't send. Try again.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="grid gap-8" noValidate>
      <ol className="grid gap-8">
        {tryout.tasks.map((task, index) => (
          <li key={task.id} className="grid gap-3">
            <fieldset className="grid gap-3">
              <legend className="mb-1 grid gap-1">
                <span className="font-mono text-sm text-muted">
                  Task {index + 1} of {tryout.tasks.length}
                </span>
                <span className="font-display text-xl font-bold">{task.prompt}</span>
              </legend>
              {task.kind === "choice" &&
                task.options?.map((o) => (
                  <Choice key={o.id} name={task.id} value={o.id} label={o.text} checked={answers[task.id] === o.id} onChange={() => set(task.id, o.id)} />
                ))}
              {task.kind === "multi" &&
                task.options?.map((o) => {
                  const chosen = (answers[task.id] as string[] | undefined) ?? [];
                  const checked = chosen.includes(o.id);
                  return (
                    <Choice
                      key={o.id}
                      type="checkbox"
                      name={task.id}
                      value={o.id}
                      label={o.text}
                      checked={checked}
                      onChange={() => set(task.id, checked ? chosen.filter((x) => x !== o.id) : [...chosen, o.id])}
                    />
                  );
                })}
              {task.kind === "number" && (
                <div className="flex max-w-xs items-center gap-3">
                  <label htmlFor={`task-${task.id}`} className="sr-only">
                    Your answer
                  </label>
                  <Input
                    id={`task-${task.id}`}
                    inputMode="decimal"
                    value={numberText[task.id] ?? ""}
                    onChange={(e) => {
                      const text = e.target.value.replace(/[^\d.,-]/g, "");
                      setNumberText((n) => ({ ...n, [task.id]: text }));
                      const value = Number(text.replaceAll(",", ""));
                      if (text.trim() === "" || Number.isNaN(value)) {
                        setAnswers((a) => {
                          const next = { ...a };
                          delete next[task.id];
                          return next;
                        });
                      } else set(task.id, value);
                    }}
                  />
                  {task.unit && <span className="whitespace-nowrap text-muted">{task.unit}</span>}
                </div>
              )}
              {task.kind === "written" && (
                <div className="grid gap-1.5">
                  <label htmlFor={`task-${task.id}`} className="sr-only">
                    Your answer
                  </label>
                  <textarea
                    id={`task-${task.id}`}
                    rows={6}
                    maxLength={2000}
                    value={(answers[task.id] as string | undefined) ?? ""}
                    onChange={(e) => set(task.id, e.target.value)}
                    className="w-full rounded-lg border-2 border-rule-strong bg-surface px-3.5 py-3 focus:border-sign focus:outline-none"
                  />
                  <p className={clsx("text-sm", wordCount((answers[task.id] as string) ?? "") > (task.max_words ?? 150) ? "text-warn" : "text-muted")}>
                    {wordCount((answers[task.id] as string) ?? "")} words · aim for under {task.max_words}
                  </p>
                </div>
              )}
            </fieldset>
          </li>
        ))}
      </ol>
      {error && <Notice tone="error">{error}</Notice>}
      <div className="flex flex-wrap items-center gap-4 border-t border-rule pt-6">
        <Button type="submit" size="lg" busy={busy} disabled={done < tryout.tasks.length}>
          {busy ? (hasWritten ? "Reading your answers…" : "Checking…") : "Submit"}
        </Button>
        <p className="text-sm text-muted" aria-live="polite">
          {done} of {tryout.tasks.length} answered
        </p>
      </div>
    </form>
  );
}

const VERDICT: Record<string, { label: string; className: string }> = {
  met: { label: "Done well", className: "bg-sign-tint text-sign" },
  partly: { label: "Partly", className: "bg-warn-tint text-warn" },
  missed: { label: "Missing", className: "bg-danger-tint text-danger" },
};

function formatAnswer(task: Attempt["reviews"][number], tryout: Tryout, value: unknown): string {
  const spec = tryout.tasks.find((t) => t.id === task.result.task_id);
  const label = (id: string) => spec?.options?.find((o) => o.id === id)?.text ?? id;
  if (value === null || value === undefined || value === "") return "No answer";
  if (Array.isArray(value)) return value.length ? value.map((v) => label(String(v))).join("; ") : "No answer";
  if (typeof value === "number") return `${value.toLocaleString("en-IN")}${task.unit ? ` ${task.unit}` : ""}`;
  return label(String(value));
}

function Review({ attempt, onRetry }: { attempt: Attempt; onRetry: () => void }) {
  return (
    <section aria-labelledby="review-title" className="grid gap-8">
      <div className="grid gap-2 rounded-[14px] border-2 border-sign bg-surface p-5 sm:p-6">
        <p className="eyebrow">Your result</p>
        <h2 id="review-title" className="text-2xl font-extrabold">
          {scoreText(attempt.points, attempt.max_points, attempt.ungraded)}
        </h2>
        <p className="text-muted">
          The score is only part of it. What matters most is whether you enjoyed the work, which you&apos;ll rate below.
        </p>
      </div>

      <ol className="grid gap-6">
        {attempt.reviews.map((review, index) => {
          const r = review.result;
          return (
            <li key={r.task_id} className="grid gap-3 rounded-[14px] border border-rule bg-surface p-5 sm:p-6">
              <p className="font-mono text-sm text-muted">Task {index + 1}</p>
              <h3 className="font-display text-lg font-bold">{review.prompt}</h3>
              {r.kind !== "written" ? (
                <>
                  <p className={clsx("font-display font-bold", r.correct ? "text-sign" : r.points ? "text-warn" : "text-danger")}>
                    {r.correct ? "Correct" : r.points ? "Partly right" : "Not quite"}
                  </p>
                  <dl className="grid gap-1 text-[0.95rem]">
                    <div className="flex flex-wrap gap-x-2">
                      <dt className="text-muted">Your answer:</dt>
                      <dd>{formatAnswer(review, attempt.tryout, review.your_answer)}</dd>
                    </div>
                    {!r.correct && (
                      <div className="flex flex-wrap gap-x-2">
                        <dt className="text-muted">Answer:</dt>
                        <dd>{formatAnswer(review, attempt.tryout, review.correct_answer)}</dd>
                      </div>
                    )}
                  </dl>
                  {review.explanation && <p className="text-muted">{review.explanation}</p>}
                </>
              ) : (
                <>
                  {r.status === "ungraded" ? (
                    <Notice tone="warn">{r.note}</Notice>
                  ) : (
                    <>
                      {r.feedback && <p className="text-[1.05rem]">{r.feedback}</p>}
                      <ul className="grid gap-2">
                        {r.criteria?.map((c) => (
                          <li key={c.id} className="grid gap-1 border-t border-rule pt-2 sm:grid-cols-[7rem_1fr] sm:gap-3">
                            <span className={clsx("justify-self-start rounded-full px-2.5 py-0.5 font-display text-xs font-bold", VERDICT[c.verdict].className)}>
                              {VERDICT[c.verdict].label}
                            </span>
                            <span>
                              <span className="font-semibold">{c.text}.</span> <span className="text-muted">{c.reason}</span>
                            </span>
                          </li>
                        ))}
                      </ul>
                    </>
                  )}
                  <details className="rounded-lg bg-sunk px-4 py-3">
                    <summary className="cursor-pointer font-display font-bold">How a professional might write it</summary>
                    <p className="mt-2">{review.model_answer}</p>
                  </details>
                </>
              )}
            </li>
          );
        })}
      </ol>

      <div className="grid gap-3 rounded-[14px] border border-rule p-5 sm:p-6">
        <h2 className="text-xl font-extrabold">In the real job</h2>
        <p>{attempt.debrief.real_work}</p>
        <p className="text-muted">You might enjoy this work if: {attempt.debrief.enjoy_if.charAt(0).toLowerCase() + attempt.debrief.enjoy_if.slice(1)}</p>
      </div>

      <ReflectionForm attempt={attempt} />

      <div className="flex flex-wrap gap-3">
        <Button variant="secondary" onClick={onRetry}>
          Try it again
        </Button>
        <ButtonLink href="/tryouts" variant="ghost">
          Other tryouts
        </ButtonLink>
      </div>
    </section>
  );
}

function ReflectionForm({ attempt }: { attempt: Attempt }) {
  const queryClient = useQueryClient();
  const [enjoyment, setEnjoyment] = useState<number | null>(attempt.reflection?.enjoyment ?? null);
  const [see, setSee] = useState<"yes" | "maybe" | "no" | null>(attempt.reflection?.see_myself ?? null);
  const [state, setState] = useState<"idle" | "saving" | "saved">(attempt.reflection ? "saved" : "idle");
  const [error, setError] = useState<string | null>(null);
  const enjoyLabels = useMemo(() => ["Not at all", "Not much", "It was okay", "Quite a lot", "Loved it"], []);

  async function save(event: FormEvent) {
    event.preventDefault();
    if (!enjoyment || !see) return;
    setState("saving");
    setError(null);
    try {
      await saveReflection(attempt.id, enjoyment, see);
      await queryClient.invalidateQueries({ queryKey: TRYOUTS_KEY });
      await queryClient.invalidateQueries({ queryKey: MATCHES_KEY });
      setState("saved");
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Couldn't save. Try again.");
      setState("idle");
    }
  }

  return (
    <form onSubmit={save} className="grid gap-5 rounded-[14px] border-2 border-sign bg-surface p-5 sm:p-6">
      <h2 className="text-xl font-extrabold">How did that feel?</h2>
      <fieldset className="grid gap-2">
        <legend className="mb-2 font-display font-bold">How much did you enjoy the work?</legend>
        <div className="grid gap-2 sm:grid-cols-5">
          {enjoyLabels.map((label, i) => (
            <Choice key={label} name="enjoyment" value={String(i + 1)} label={label} checked={enjoyment === i + 1} onChange={() => (setEnjoyment(i + 1), setState("idle"))} />
          ))}
        </div>
      </fieldset>
      <fieldset className="grid gap-2">
        <legend className="mb-2 font-display font-bold">Could you see yourself doing this kind of work every day?</legend>
        <div className="grid gap-2 sm:grid-cols-3">
          {(
            [
              ["yes", "Yes"],
              ["maybe", "Maybe"],
              ["no", "Not really"],
            ] as const
          ).map(([value, label]) => (
            <Choice key={value} name="see" value={value} label={label} checked={see === value} onChange={() => (setSee(value), setState("idle"))} />
          ))}
        </div>
      </fieldset>
      {error && <Notice tone="error">{error}</Notice>}
      <div className="flex flex-wrap items-center gap-4">
        <Button type="submit" busy={state === "saving"} disabled={!enjoyment || !see || state === "saved"}>
          {state === "saved" ? "Saved" : "Save"}
        </Button>
        {state === "saved" && (
          <p className="text-sm text-muted" role="status">
            Saved. It now shows next to this career in{" "}
            <Link href="/results" className="font-semibold text-sign underline underline-offset-4">
              your results
            </Link>{" "}
            and in your family report.
          </p>
        )}
      </div>
    </form>
  );
}
