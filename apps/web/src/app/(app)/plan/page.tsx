"use client";

import clsx from "clsx";
import Link from "next/link";
import { type FormEvent, useState } from "react";

import { Button, Spinner } from "@/components/ui/button";
import { Field, Input, Select } from "@/components/ui/field";
import { Notice } from "@/components/ui/notice";
import { useMatches, useOverview } from "@/lib/assessment";
import { burstFrom } from "@/lib/burst";
import {
  CAREERS,
  getCareer,
  streamsText,
  totalYears,
  yearsText,
} from "@/lib/careers";
import {
  type Plan,
  useAddStep,
  useChooseRoute,
  usePlan,
  useRemoveStep,
  useTick,
} from "@/lib/plan";

const KIND_LABEL: Record<string, string> = {
  prep: "Get ready",
  exam: "Entrance",
  degree: "Degree",
  diploma: "Diploma",
  course: "Course",
  work: "Work",
  role: "First role",
  custom: "Your step",
};

export default function PlanPage() {
  const plan = usePlan();
  const [choosing, setChoosing] = useState(false);

  if (plan.isPending) {
    return (
      <p className="flex items-center gap-2 text-muted">
        <Spinner /> Loading your plan…
      </p>
    );
  }
  if (plan.error) return <Notice tone="error">{plan.error.message}</Notice>;

  return (
    <div className="grid gap-10">
      <section className="grid max-w-3xl gap-3">
        <p className="eyebrow">My plan</p>
        <h1 className="text-[2.5rem] font-semibold leading-[1] tracking-[-0.045em] sm:text-[3.1rem]">
          {plan.data ? (
            <>
              Your route to{" "}
              <span className="serif-accent text-sign">
                {plan.data.career_title.toLowerCase()}
              </span>
            </>
          ) : (
            "Pick a route and make it yours"
          )}
        </h1>
        <p className="text-muted">
          A plan turns a career route into steps you can tick off. You can
          switch routes any time; your own steps come with you.
        </p>
      </section>

      {plan.data && !choosing ? (
        <Checklist plan={plan.data} onSwitch={() => setChoosing(true)} />
      ) : (
        <RoutePicker
          current={plan.data ?? null}
          onDone={() => setChoosing(false)}
        />
      )}
    </div>
  );
}

function Checklist({ plan, onSwitch }: { plan: Plan; onSwitch: () => void }) {
  const tick = useTick();
  const remove = useRemoveStep();
  const pct = plan.total ? Math.round((plan.done / plan.total) * 100) : 0;
  const nextIndex = plan.milestones.findIndex((m) => !m.done_at);

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_18rem] lg:items-start">
      <section
        aria-labelledby="steps"
        className="spotlight grid gap-5 rounded-3xl border border-rule bg-surface p-5 sm:p-7"
      >
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div className="grid gap-1">
            <h2 id="steps" className="text-xl font-semibold">
              {plan.route_label}
            </h2>
            <p className="font-mono text-xs text-muted">
              {plan.done} of {plan.total} done
            </p>
          </div>
          <Button variant="secondary" size="sm" onClick={onSwitch}>
            Switch route
          </Button>
        </div>
        <div
          className="h-2 overflow-hidden rounded-full bg-sunk"
          role="progressbar"
          aria-label="Plan progress"
          aria-valuenow={pct}
          aria-valuemin={0}
          aria-valuemax={100}
        >
          <div
            className="h-full rounded-full bg-sign transition-[width] duration-500"
            style={{ width: `${pct}%` }}
          />
        </div>
        {!plan.route_open && (
          <Notice tone="warn">
            This route isn&apos;t open from your current stream.{" "}
            {plan.requirement ??
              "Check the eligibility rules before you plan around it."}
          </Notice>
        )}

        <ol className="relative grid gap-1" aria-label="Steps on your route">
          {plan.milestones.map((m, i) => {
            const done = Boolean(m.done_at);
            return (
              <li key={m.id} className="relative flex gap-4 pb-4 last:pb-0">
                {i < plan.milestones.length - 1 && (
                  <span
                    aria-hidden="true"
                    className={clsx(
                      "absolute left-[13px] top-8 h-[calc(100%-1.5rem)] w-0.5",
                      done ? "bg-sign" : "bg-rule",
                    )}
                  />
                )}
                <button
                  type="button"
                  onClick={(e) => {
                    if (!done) burstFrom(e.currentTarget, { power: 0.5 });
                    tick.mutate({ id: m.id, done: !done });
                  }}
                  aria-pressed={done}
                  aria-label={`${done ? "Untick" : "Tick off"}: ${m.label}`}
                  className={clsx(
                    "relative z-10 mt-0.5 grid size-7 shrink-0 place-items-center rounded-full border-2 transition-[transform,background-color] duration-300 ease-[cubic-bezier(0.34,1.56,0.64,1)] active:scale-90",
                    done
                      ? "border-sign bg-sign text-sign-ink"
                      : i === nextIndex
                        ? "border-sign bg-surface"
                        : "border-rule-strong bg-surface",
                  )}
                >
                  {done && (
                    <svg
                      viewBox="0 0 12 12"
                      className="size-3.5"
                      aria-hidden="true"
                    >
                      <path
                        d="M2.5 6.5l2.5 2.5 4.5-5.5"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                  )}
                </button>
                <div className="grid min-w-0 flex-1 gap-0.5">
                  <p className="label-mono text-muted">
                    {KIND_LABEL[m.kind] ?? m.kind}
                    {m.years ? ` · ${yearsText(m.years)}` : ""}
                    {i === nextIndex && (
                      <span className="text-sign"> · next up</span>
                    )}
                  </p>
                  <p
                    className={clsx(
                      "font-display font-semibold",
                      done && "text-muted line-through decoration-rule-strong",
                    )}
                  >
                    {m.label}
                  </p>
                  {m.detail && <p className="text-sm text-muted">{m.detail}</p>}
                </div>
                {m.custom && (
                  <button
                    type="button"
                    onClick={() => remove.mutate(m.id)}
                    className="self-start text-sm text-muted underline underline-offset-4 hover:text-ink"
                  >
                    Remove
                  </button>
                )}
              </li>
            );
          })}
        </ol>
        <AddStep />
      </section>

      <aside className="grid gap-3 rounded-3xl border border-rule p-5 text-sm">
        <p className="eyebrow">Tips</p>
        <p className="text-muted">
          Exam dates and rules change every year. Tick “check this year&apos;s
          eligibility” only after you&apos;ve read the official notice.
        </p>
        <Link
          href={`/careers/${plan.career_slug}`}
          className="font-semibold text-sign underline underline-offset-4"
        >
          Open the full guide
        </Link>
        <Link
          href={`/ksarth?ask=${encodeURIComponent(`What should I do this month for my plan to become a ${plan.career_title.toLowerCase()}?`)}`}
          className="font-semibold text-sign underline underline-offset-4"
        >
          Ask Ksarth what to do this month
        </Link>
      </aside>
    </div>
  );
}

function AddStep() {
  const add = useAddStep();
  const [label, setLabel] = useState("");
  async function submit(event: FormEvent) {
    event.preventDefault();
    if (label.trim().length < 2) return;
    await add.mutateAsync(label);
    setLabel("");
  }
  return (
    <form
      onSubmit={submit}
      className="flex flex-wrap items-end gap-2 border-t border-rule pt-5"
    >
      <Field
        id="custom-step"
        label="Add your own step"
        className="min-w-[14rem] flex-1"
      >
        <Input
          id="custom-step"
          value={label}
          maxLength={120}
          placeholder="For example: finish NCERT Biology by March"
          onChange={(e) => setLabel(e.target.value)}
        />
      </Field>
      <Button
        type="submit"
        variant="secondary"
        busy={add.isPending}
        disabled={label.trim().length < 2}
      >
        Add
      </Button>
      {add.error && (
        <Notice tone="error" className="w-full">
          {add.error.message}
        </Notice>
      )}
    </form>
  );
}

function RoutePicker({
  current,
  onDone,
}: {
  current: Plan | null;
  onDone: () => void;
}) {
  const overview = useOverview();
  const matches = useMatches(Boolean(overview.data?.latest));
  const choose = useChooseRoute();
  const [slug, setSlug] = useState(current?.career_slug ?? "");
  const suggested = (matches.data?.matches ?? []).slice(0, 3);
  const career = slug ? getCareer(slug) : undefined;

  return (
    <section
      aria-labelledby="pick"
      className="grid gap-6 rounded-3xl border border-rule bg-surface p-5 sm:p-7"
    >
      <h2 id="pick" className="text-xl font-semibold">
        Choose a career
      </h2>
      {suggested.length > 0 && (
        <div className="grid gap-2">
          <p className="text-sm text-muted">Your top matches</p>
          <div className="flex flex-wrap gap-2">
            {suggested.map((m) => (
              <button
                key={m.slug}
                type="button"
                onClick={() => setSlug(m.slug)}
                className={clsx(
                  "rounded-full border px-4 py-2 font-display font-semibold transition-colors",
                  slug === m.slug
                    ? "border-sign bg-sign-tint text-sign"
                    : "border-rule-strong hover:border-ink",
                )}
              >
                {m.title}
              </button>
            ))}
          </div>
        </div>
      )}
      <Field
        id="career"
        label={suggested.length ? "Or any career in the library" : "Career"}
      >
        <Select
          id="career"
          value={slug}
          onChange={(e) => setSlug(e.target.value)}
        >
          <option value="">Choose…</option>
          {CAREERS.map((c) => (
            <option key={c.slug} value={c.slug}>
              {c.title}
            </option>
          ))}
        </Select>
      </Field>

      {career && (
        <div className="grid gap-3">
          <p className="text-sm text-muted">
            Routes to {career.title.toLowerCase()}
          </p>
          <ul className="grid gap-3 md:grid-cols-2">
            {career.routes.map((route) => (
              <li key={route.id}>
                <button
                  type="button"
                  disabled={choose.isPending}
                  onClick={async () => {
                    await choose.mutateAsync({
                      career_slug: career.slug,
                      route_id: route.id,
                    });
                    onDone();
                  }}
                  className="grid h-full w-full content-start gap-2 rounded-2xl border border-rule p-4 text-left transition-[transform,border-color] duration-300 hover:-translate-y-0.5 hover:border-sign"
                >
                  <span className="font-display font-semibold">
                    {route.label}
                  </span>
                  <span className="text-sm text-muted">
                    {streamsText(route.streams)} · about{" "}
                    {yearsText(totalYears(route.steps))}
                  </span>
                  <span className="text-sm">
                    {route.steps.map((s) => s.label).join(" → ")}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
      {choose.error && <Notice tone="error">{choose.error.message}</Notice>}
      {current && (
        <Button variant="ghost" onClick={onDone} className="justify-self-start">
          Keep my current plan
        </Button>
      )}
    </section>
  );
}
