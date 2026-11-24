"use client";

import clsx from "clsx";
import { type FormEvent, useState } from "react";

import { Button, Spinner } from "@/components/ui/button";
import { Field, Input } from "@/components/ui/field";
import { Notice } from "@/components/ui/notice";
import { DIMENSIONS } from "@/lib/careers";
import {
  type ClassDetail,
  useClass,
  useCreateClass,
  useMyClasses,
} from "@/lib/classes";

export default function CounsellorPage() {
  const classes = useMyClasses();
  const [selected, setSelected] = useState<string | null>(null);
  const active = selected ?? classes.data?.[0]?.id ?? null;

  return (
    <div className="grid gap-10">
      <section className="grid max-w-3xl gap-3">
        <p className="eyebrow">For school counsellors</p>
        <h1 className="text-[2.5rem] font-semibold leading-[1] tracking-[-0.045em] sm:text-[3.1rem]">
          See who needs a{" "}
          <span className="serif-accent text-sign">conversation</span>
        </h1>
        <p className="text-muted">
          Create a class and share its code. Students join with the code and
          agree to share a summary: their interest type, top matches, tryouts
          and plan progress. You never see their chats or answers, and they can
          leave at any time.
        </p>
      </section>

      {classes.isPending ? (
        <p className="flex items-center gap-2 text-muted">
          <Spinner /> Loading classes…
        </p>
      ) : (
        <div className="grid gap-6">
          {classes.data && classes.data.length > 0 && (
            <div
              className="flex flex-wrap gap-2"
              role="tablist"
              aria-label="Your classes"
            >
              {classes.data.map((c) => (
                <button
                  key={c.id}
                  role="tab"
                  aria-selected={c.id === active}
                  onClick={() => setSelected(c.id)}
                  className={clsx(
                    "rounded-full border px-4 py-2 font-display font-semibold",
                    c.id === active
                      ? "border-sign bg-sign-tint text-sign"
                      : "border-rule-strong hover:border-ink",
                  )}
                >
                  {c.name}{" "}
                  <span className="font-mono text-xs font-normal text-muted">
                    · {c.students}
                  </span>
                </button>
              ))}
            </div>
          )}
          {active && <ClassView id={active} />}
          <CreateClass onCreated={setSelected} />
        </div>
      )}
    </div>
  );
}

function ClassView({ id }: { id: string }) {
  const detail = useClass(id);
  if (detail.isPending) return <Spinner />;
  if (detail.error) return <Notice tone="error">{detail.error.message}</Notice>;
  const c = detail.data;
  const flagged = c.roster.filter((s) => s.flags.length > 0).length;

  return (
    <div className="grid gap-6">
      <div className="grid gap-4 sm:grid-cols-3">
        <Stat
          label="Join code"
          value={c.code}
          mono
          hint={`${c.school}. Share it in class.`}
        />
        <Stat
          label="Took the assessment"
          value={`${Math.round(c.assessed_share * 100)}%`}
          hint={`${c.students} students in the class`}
        />
        <Stat
          label="Might need a conversation"
          value={String(flagged)}
          hint="Flags are listed against each student below"
        />
      </div>

      {c.students > 0 && <ClassInterests detail={c} />}

      <section
        aria-labelledby="roster"
        className="grid gap-4 overflow-hidden rounded-3xl border border-rule bg-surface"
      >
        <h2 id="roster" className="px-5 pt-5 text-xl font-semibold sm:px-6">
          Students
        </h2>
        {c.roster.length === 0 ? (
          <p className="px-5 pb-6 text-muted sm:px-6">
            No one has joined yet. Share the code {c.code} with your class.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[720px] text-left text-sm">
              <thead className="border-y border-rule bg-sunk/60 font-mono text-xs uppercase tracking-wider text-muted">
                <tr>
                  <th className="px-5 py-3 font-medium sm:px-6">Student</th>
                  <th className="px-3 py-3 font-medium">Interests</th>
                  <th className="px-3 py-3 font-medium">Top matches</th>
                  <th className="px-3 py-3 font-medium">Tryouts</th>
                  <th className="px-3 py-3 font-medium">Plan</th>
                  <th className="px-3 py-3 font-medium">Flags</th>
                </tr>
              </thead>
              <tbody>
                {c.roster.map((s) => (
                  <tr
                    key={s.id}
                    className="border-b border-rule align-top last:border-0"
                  >
                    <td className="px-5 py-3 sm:px-6">
                      <p className="font-semibold">{s.name}</p>
                      <p className="text-xs text-muted">
                        {s.grade ? `Class ${s.grade}` : (s.stage ?? "")}
                        {s.last_active ? ` · active ${s.last_active}` : ""}
                      </p>
                    </td>
                    <td className="px-3 py-3 font-mono">
                      {s.interest_code ?? "—"}
                    </td>
                    <td className="px-3 py-3">
                      {s.top_matches.join(", ") || "—"}
                    </td>
                    <td className="px-3 py-3 tabular">{s.tryouts_tried}</td>
                    <td className="px-3 py-3">
                      {s.plan_career
                        ? `${s.plan_career} (${Math.round((s.plan_progress ?? 0) * 100)}%)`
                        : "—"}
                    </td>
                    <td className="px-3 py-3">
                      {s.flags.length ? (
                        <ul className="grid gap-1">
                          {s.flags.map((f) => (
                            <li
                              key={f}
                              className="rounded-md bg-warn-tint px-2 py-0.5 text-xs"
                            >
                              {f}
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <span className="text-muted">None</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

function ClassInterests({ detail }: { detail: ClassDetail }) {
  const averages = detail.interest_averages;
  if (!Object.keys(averages).length) return null;
  return (
    <section
      aria-labelledby="class-interests"
      className="grid gap-5 rounded-3xl border border-rule bg-surface p-5 sm:grid-cols-2 sm:p-6"
    >
      <div className="grid content-start gap-3">
        <h2 id="class-interests" className="text-xl font-semibold">
          What the class enjoys
        </h2>
        <ul className="grid gap-2.5">
          {DIMENSIONS.map((d) => (
            <li
              key={d.key}
              className="grid grid-cols-[6.5rem_1fr_2.5rem] items-center gap-3 text-sm"
            >
              <span>{d.short}</span>
              <span className="h-2 overflow-hidden rounded-full bg-sunk">
                <span
                  className="block h-full rounded-full bg-sign"
                  style={{ width: `${averages[d.key] ?? 0}%` }}
                />
              </span>
              <span className="text-right font-mono text-xs tabular">
                {Math.round(averages[d.key] ?? 0)}
              </span>
            </li>
          ))}
        </ul>
      </div>
      <div className="grid content-start gap-3">
        <h3 className="text-lg font-semibold">Careers that come up most</h3>
        <ol className="grid gap-1.5 text-sm">
          {detail.popular_careers.map((p) => (
            <li
              key={p.title}
              className="flex justify-between gap-3 border-b border-rule pb-1.5"
            >
              <span>{p.title}</span>
              <span className="font-mono text-xs text-muted">{p.count}</span>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}

function Stat({
  label,
  value,
  hint,
  mono,
}: {
  label: string;
  value: string;
  hint: string;
  mono?: boolean;
}) {
  return (
    <div className="grid gap-1 rounded-3xl border border-rule bg-surface p-5">
      <p className="label-mono text-muted">{label}</p>
      <p
        className={clsx(
          "text-3xl font-semibold tracking-[-0.03em]",
          mono && "font-mono tracking-[0.12em] text-sign",
        )}
      >
        {value}
      </p>
      <p className="text-xs text-muted">{hint}</p>
    </div>
  );
}

function CreateClass({ onCreated }: { onCreated: (id: string) => void }) {
  const create = useCreateClass();
  const [name, setName] = useState("");
  const [school, setSchool] = useState("");
  async function submit(event: FormEvent) {
    event.preventDefault();
    const room = await create.mutateAsync({ name, school });
    onCreated(room.id);
    setName("");
  }
  return (
    <form
      onSubmit={submit}
      className="grid gap-4 rounded-3xl border border-dashed border-rule-strong p-5 sm:grid-cols-[1fr_1fr_auto] sm:items-end sm:p-6"
    >
      <Field id="class-name" label="New class">
        <Input
          id="class-name"
          value={name}
          placeholder="Class 10 B"
          maxLength={60}
          onChange={(e) => setName(e.target.value)}
        />
      </Field>
      <Field id="school" label="School">
        <Input
          id="school"
          value={school}
          placeholder="Your school's name"
          maxLength={100}
          onChange={(e) => setSchool(e.target.value)}
        />
      </Field>
      <Button
        type="submit"
        busy={create.isPending}
        disabled={name.trim().length < 2 || school.trim().length < 2}
      >
        Create class
      </Button>
      {create.error && (
        <Notice tone="error" className="sm:col-span-3">
          {create.error.message}
        </Notice>
      )}
    </form>
  );
}
