"use client";

import Link from "next/link";
import { type FormEvent, useState } from "react";

import { Button } from "@/components/ui/button";
import { Field, Input } from "@/components/ui/field";
import { Notice } from "@/components/ui/notice";
import { useJoinClass, useJoinedClasses, useLeaveClass } from "@/lib/classes";

/** In Settings: join your school's class with a code, see what's shared, and leave any time. */
export function JoinClass({ isCounsellor }: { isCounsellor: boolean }) {
  const joined = useJoinedClasses();
  const join = useJoinClass();
  const leave = useLeaveClass();
  const [code, setCode] = useState("");
  const [agree, setAgree] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    await join.mutateAsync({ code: code.trim().toUpperCase(), agree });
    setCode("");
    setAgree(false);
  }

  return (
    <div className="grid gap-4">
      {joined.data?.map((c) => (
        <div
          key={c.id}
          className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-rule p-4"
        >
          <div>
            <p className="font-semibold">
              {c.name}, {c.school}
            </p>
            <p className="text-sm text-muted">
              Counsellor: {c.counsellor}. They see your summary, never your
              chats or answers.
            </p>
          </div>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => leave.mutate(c.id)}
          >
            Leave
          </Button>
        </div>
      ))}
      <form onSubmit={submit} className="grid gap-3">
        <Field
          id="class-code"
          label="Join your school's class"
          hint="Your counsellor or teacher gives you a 6-character code."
        >
          <Input
            id="class-code"
            value={code}
            maxLength={6}
            autoCapitalize="characters"
            className="font-mono uppercase tracking-[0.3em]"
            onChange={(e) => setCode(e.target.value)}
          />
        </Field>
        <label className="flex items-start gap-3 text-sm">
          <input
            type="checkbox"
            checked={agree}
            onChange={(e) => setAgree(e.target.checked)}
            className="mt-1 size-4 accent-[var(--sign)]"
          />
          <span>
            I agree to share a summary with the counsellor: my interest type,
            top matches, which tryouts I tried, my plan&apos;s progress and when
            I was last active. Not my chats with Ksarth or my answers.
          </span>
        </label>
        <Button
          type="submit"
          variant="secondary"
          busy={join.isPending}
          disabled={code.trim().length !== 6 || !agree}
          className="justify-self-start"
        >
          Join class
        </Button>
        {join.error && <Notice tone="error">{join.error.message}</Notice>}
      </form>
      <p className="text-sm text-muted">
        {isCounsellor
          ? "You run classes on Pathify. "
          : "Are you a school counsellor or teacher? "}
        <Link
          href="/counsellor"
          className="font-semibold text-sign underline underline-offset-4"
        >
          {isCounsellor
            ? "Open your classes"
            : "Set up a class for your students"}
        </Link>
      </p>
    </div>
  );
}
