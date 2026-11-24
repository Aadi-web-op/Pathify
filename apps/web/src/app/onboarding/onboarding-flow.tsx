"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useRouter, useSearchParams } from "next/navigation";
import { type FormEvent, type KeyboardEvent, useEffect, useRef, useState } from "react";

import { InterestReading } from "@/components/app/interest-reading";
import { RouteLine } from "@/components/brand/route-line";
import { Button, Spinner } from "@/components/ui/button";
import { Choice } from "@/components/ui/choice";
import { Field, Input, Select } from "@/components/ui/field";
import { Notice } from "@/components/ui/notice";
import { api, ApiError, type Board, type Goal, type Stage, type StudentProfile, type Stream, unwrap } from "@/lib/api";
import {
  BOARDS,
  DEGREE_SUGGESTIONS,
  GOALS,
  INDIAN_STATES,
  INTEREST_SUGGESTIONS,
  isSchool,
  STAGES,
  STREAMS,
  SUGGESTED_GOALS,
} from "@/lib/profile";
import { ME_KEY, useMe } from "@/lib/session";

type Draft = {
  stage: Stage | null;
  grade: number | null;
  board: Board | null;
  stream: Stream | null;
  degree: string;
  year_of_study: number | null;
  goals: Goal[];
  interests: string[];
  state: string;
  language: "en" | "hi";
  guardian_aware: boolean;
};

const EMPTY: Draft = {
  stage: null,
  grade: null,
  board: null,
  stream: null,
  degree: "",
  year_of_study: null,
  goals: [],
  interests: [],
  state: "",
  language: "en",
  guardian_aware: false,
};

const STEPS = [
  { label: "Where you are", detail: "Class or college" },
  { label: "Details", detail: "Board, stream, degree" },
  { label: "What you need", detail: "Up to three goals" },
  { label: "About you", detail: "Interests and language" },
];

function fromProfile(profile: StudentProfile): Draft {
  return {
    stage: profile.stage,
    grade: profile.grade ?? null,
    board: profile.board ?? null,
    stream: profile.stream ?? null,
    degree: profile.degree ?? "",
    year_of_study: profile.year_of_study ?? null,
    goals: profile.goals,
    interests: profile.interests ?? [],
    state: profile.state ?? "",
    language: profile.language ?? "en",
    guardian_aware: profile.guardian_aware ?? false,
  };
}

function toProfile(draft: Draft): StudentProfile {
  const school = draft.stage !== null && isSchool(draft.stage);
  return {
    stage: draft.stage as Stage,
    grade: school ? draft.grade : null,
    board: school ? draft.board : null,
    stream: draft.stage === "school_11_12" ? draft.stream : null,
    degree: school ? null : draft.degree.trim() || null,
    year_of_study: draft.stage === "college" ? draft.year_of_study : null,
    goals: draft.goals,
    interests: draft.interests,
    state: draft.state || null,
    language: draft.language,
    guardian_aware: school ? draft.guardian_aware : false,
  };
}

/** Returns the first thing stopping this step from being finished, or null. */
function blocker(step: number, d: Draft): string | null {
  if (step === 0) return d.stage ? null : "Choose where you are now.";
  if (step === 1 && d.stage) {
    if (isSchool(d.stage)) {
      if (!d.grade) return "Choose your class.";
      if (d.stage === "school_11_12" && !d.stream) return "Choose your stream.";
      if (!d.guardian_aware) return "Confirm that a parent or guardian knows you're using Pathify.";
    } else if (!d.degree.trim()) {
      return "Tell us your degree or course.";
    }
  }
  if (step === 2 && d.goals.length === 0) return "Pick at least one thing you want help with.";
  return null;
}

export function OnboardingFlow() {
  const router = useRouter();
  const params = useSearchParams();
  const queryClient = useQueryClient();
  const { data: me, isPending } = useMe();
  const editing = params.get("edit") === "1";
  const [step, setStep] = useState(0);
  const [draft, setDraft] = useState<Draft>(EMPTY);
  const [loaded, setLoaded] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const heading = useRef<HTMLHeadingElement>(null);

  useEffect(() => {
    if (me && !loaded) {
      if (me.profile) setDraft(fromProfile(me.profile));
      setLoaded(true);
    }
  }, [me, loaded]);

  useEffect(() => {
    heading.current?.focus();
  }, [step]);

  const update = (patch: Partial<Draft>) => setDraft((d) => ({ ...d, ...patch }));

  function chooseStage(stage: Stage) {
    setDraft((d) => {
      const grades = stage === "school_9_10" ? [9, 10] : stage === "school_11_12" ? [11, 12] : [];
      const usingSuggestion = d.goals.length === 0 || (d.stage && d.goals.join() === SUGGESTED_GOALS[d.stage].join());
      return {
        ...d,
        stage,
        grade: d.grade && grades.includes(d.grade) ? d.grade : (grades[0] ?? null),
        goals: usingSuggestion ? SUGGESTED_GOALS[stage] : d.goals,
      };
    });
  }

  const problem = blocker(step, draft);
  const last = step === STEPS.length - 1;

  async function next(event: FormEvent) {
    event.preventDefault();
    if (problem) return;
    if (!last) {
      setStep(step + 1);
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const user = await unwrap(api.PUT("/api/users/me/profile", { body: toProfile(draft) }));
      queryClient.setQueryData(ME_KEY, user);
      router.replace(editing ? "/settings" : "/home");
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "We couldn't save your profile. Try again.");
      setSaving(false);
    }
  }

  if (isPending || !loaded) {
    return (
      <p className="flex items-center gap-2 text-muted">
        <Spinner /> Loading your profile…
      </p>
    );
  }

  const firstName = me?.name.split(" ")[0];

  return (
    <form onSubmit={next} className="grid gap-8" noValidate>
      <div className="grid gap-3">
        <p className="eyebrow">{editing ? "Edit your profile" : `Welcome, ${firstName}`}</p>
        <RouteLine stations={STEPS} current={step} label="Profile setup progress" />
      </div>

      <section aria-labelledby="step-title" className="grid gap-6">
        {step === 0 && (
          <>
            <StepTitle ref={heading} title="Where are you right now?" intro="This decides which questions come next." />
            <fieldset className="grid gap-3 sm:grid-cols-2">
              <legend className="sr-only">Your stage</legend>
              {STAGES.map((s) => (
                <Choice
                  key={s.value}
                  name="stage"
                  value={s.value}
                  label={s.label}
                  detail={s.detail}
                  checked={draft.stage === s.value}
                  onChange={() => chooseStage(s.value)}
                />
              ))}
            </fieldset>
          </>
        )}

        {step === 1 && draft.stage && isSchool(draft.stage) && (
          <>
            <StepTitle ref={heading} title="Tell us about school" intro="Boards and streams change which routes are open to you." />
            <fieldset className="grid gap-3">
              <legend className="mb-2 font-display font-bold">Class</legend>
              <div className="grid grid-cols-2 gap-3">
                {(draft.stage === "school_9_10" ? [9, 10] : [11, 12]).map((grade) => (
                  <Choice
                    key={grade}
                    name="grade"
                    value={String(grade)}
                    label={`Class ${grade}`}
                    checked={draft.grade === grade}
                    onChange={() => update({ grade })}
                  />
                ))}
              </div>
            </fieldset>
            <Field id="board" label="Board" hint="Optional">
              <Select
                id="board"
                value={draft.board ?? ""}
                onChange={(e) => update({ board: (e.target.value || null) as Board | null })}
              >
                <option value="">Choose your board</option>
                {BOARDS.map((b) => (
                  <option key={b.value} value={b.value}>
                    {b.label}
                  </option>
                ))}
              </Select>
            </Field>
            {draft.stage === "school_11_12" && (
              <fieldset className="grid gap-3">
                <legend className="mb-2 font-display font-bold">Stream</legend>
                <div className="grid gap-3 sm:grid-cols-2">
                  {STREAMS.map((s) => (
                    <Choice
                      key={s.value}
                      name="stream"
                      value={s.value}
                      label={s.label}
                      detail={s.detail}
                      checked={draft.stream === s.value}
                      onChange={() => update({ stream: s.value })}
                    />
                  ))}
                </div>
              </fieldset>
            )}
            <div className="rounded-2xl border border-rule bg-surface p-4">
              <Choice
                type="checkbox"
                name="guardian"
                value="yes"
                label="A parent or guardian knows I'm using Pathify"
                detail="We ask everyone in school. Big decisions go better when your family is part of them."
                checked={draft.guardian_aware}
                onChange={() => update({ guardian_aware: !draft.guardian_aware })}
              />
            </div>
          </>
        )}

        {step === 1 && draft.stage && !isSchool(draft.stage) && (
          <>
            <StepTitle
              ref={heading}
              title={draft.stage === "college" ? "Tell us about college" : "Tell us what you studied"}
              intro="Your degree shapes which roles are a short step away."
            />
            <Field id="degree" label="Degree or course" hint="For example B.Tech CSE, B.Com, BA Psychology.">
              <Input
                id="degree"
                list="degree-options"
                maxLength={80}
                value={draft.degree}
                onChange={(e) => update({ degree: e.target.value })}
              />
              <datalist id="degree-options">
                {DEGREE_SUGGESTIONS.map((d) => (
                  <option key={d} value={d} />
                ))}
              </datalist>
            </Field>
            {draft.stage === "college" && (
              <Field id="year" label="Year of study" hint="Optional">
                <Select
                  id="year"
                  value={draft.year_of_study ?? ""}
                  onChange={(e) => update({ year_of_study: e.target.value ? Number(e.target.value) : null })}
                >
                  <option value="">Choose your year</option>
                  {[1, 2, 3, 4, 5].map((y) => (
                    <option key={y} value={y}>
                      Year {y}
                    </option>
                  ))}
                </Select>
              </Field>
            )}
          </>
        )}

        {step === 2 && (
          <>
            <StepTitle ref={heading} title="What do you want help with?" intro="Pick up to three. You can change this later." />
            <fieldset className="grid gap-3">
              <legend className="sr-only">Goals</legend>
              {GOALS.map((g) => {
                const checked = draft.goals.includes(g.value);
                return (
                  <Choice
                    key={g.value}
                    type="checkbox"
                    name="goals"
                    value={g.value}
                    label={g.label}
                    detail={g.detail}
                    checked={checked}
                    disabled={!checked && draft.goals.length >= 3}
                    onChange={() =>
                      update({ goals: checked ? draft.goals.filter((x) => x !== g.value) : [...draft.goals, g.value] })
                    }
                  />
                );
              })}
            </fieldset>
          </>
        )}

        {step === 3 && (
          <>
            <StepTitle ref={heading} title="A little about you" intro="All optional. Ksarth uses this to give advice that fits you." />
            <InterestPicker value={draft.interests} onChange={(interests) => update({ interests })} />
            <Field id="state" label="State or union territory">
              <Select id="state" value={draft.state} onChange={(e) => update({ state: e.target.value })}>
                <option value="">Choose where you live</option>
                {INDIAN_STATES.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </Select>
            </Field>
            <fieldset className="grid gap-3">
              <legend className="mb-2 font-display font-bold">Language for Ksarth&apos;s replies</legend>
              <div className="grid grid-cols-2 gap-3">
                <Choice name="language" value="en" label="English" checked={draft.language === "en"} onChange={() => update({ language: "en" })} />
                <Choice name="language" value="hi" label="हिन्दी" detail="Hindi" checked={draft.language === "hi"} onChange={() => update({ language: "hi" })} />
              </div>
            </fieldset>
          </>
        )}
      </section>

      {error && <Notice tone="error">{error}</Notice>}

      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-rule pt-6">
        <Button
          type="button"
          variant="ghost"
          onClick={() => (step === 0 ? router.push(editing ? "/settings" : "/home") : setStep(step - 1))}
          className={step === 0 && !editing ? "invisible" : undefined}
        >
          {step === 0 ? "Cancel" : "Back"}
        </Button>
        <div className="flex items-center gap-4">
          {problem && <p className="hidden text-sm text-muted sm:block">{problem}</p>}
          <Button type="submit" busy={saving} disabled={Boolean(problem)}>
            {last ? (saving ? "Saving…" : editing ? "Save changes" : "Finish setup") : "Continue"}
          </Button>
        </div>
      </div>
    </form>
  );
}

function StepTitle({ ref, title, intro }: { ref: React.Ref<HTMLHeadingElement>; title: string; intro: string }) {
  return (
    <div className="grid gap-1.5">
      <h1 id="step-title" ref={ref} tabIndex={-1} className="text-[1.9rem] font-bold tracking-[-0.045em] focus:outline-none">
        {title}
      </h1>
      <p className="text-muted">{intro}</p>
    </div>
  );
}

function InterestPicker({ value, onChange }: { value: string[]; onChange: (next: string[]) => void }) {
  const [text, setText] = useState("");
  const full = value.length >= 8;

  function add(raw: string) {
    const item = raw.trim().replace(/\s+/g, " ").slice(0, 40);
    if (!item || full || value.some((v) => v.toLowerCase() === item.toLowerCase())) return;
    onChange([...value, item]);
  }

  function onKey(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "Enter" || event.key === ",") {
      event.preventDefault();
      add(text);
      setText("");
    } else if (event.key === "Backspace" && !text && value.length) {
      onChange(value.slice(0, -1));
    }
  }

  return (
    <div className="grid gap-3">
      <Field id="interests" label="Things you enjoy" hint={`Type and press Enter. ${value.length} of 8 added.`}>
        <Input
          id="interests"
          value={text}
          maxLength={40}
          disabled={full}
          placeholder={full ? "You've added 8" : "For example: robotics"}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKey}
          onBlur={() => {
            add(text);
            setText("");
          }}
        />
      </Field>
      {value.length > 0 && (
        <ul className="flex flex-wrap gap-2" aria-label="Your interests">
          {value.map((item) => (
            <li key={item}>
              <button
                type="button"
                onClick={() => onChange(value.filter((v) => v !== item))}
                className="inline-flex items-center gap-1.5 rounded-full bg-sign px-3 py-1 font-display text-sm font-bold text-sign-ink"
                aria-label={`Remove ${item}`}
              >
                {item}
                <span aria-hidden="true">×</span>
              </button>
            </li>
          ))}
        </ul>
      )}
      <div className="flex flex-wrap gap-2" aria-label="Suggestions">
        {INTEREST_SUGGESTIONS.filter((s) => !value.some((v) => v.toLowerCase() === s.toLowerCase())).map((s) => (
          <button
            key={s}
            type="button"
            disabled={full}
            onClick={() => add(s)}
            className="rounded-full border border-rule-strong px-3 py-1 text-sm hover:border-ink disabled:opacity-40"
          >
            + {s}
          </button>
        ))}
      </div>
      <InterestReading items={value} />
    </div>
  );
}
