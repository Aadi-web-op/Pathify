"use client";

import { type FormEvent, useState } from "react";

import { Button } from "@/components/ui/button";
import { Field, Input } from "@/components/ui/field";
import { Notice } from "@/components/ui/notice";
import { api, ApiError, unwrap } from "@/lib/api";

export function ForgotPasswordForm() {
  const [email, setEmail] = useState("");
  const [busy, setBusy] = useState(false);
  const [sent, setSent] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const body = await unwrap(
        api.POST("/api/auth/password/forgot", { body: { email } }),
      );
      setSent(
        (body as { detail?: string }).detail ?? "Check your email for a link.",
      );
    } catch (e) {
      setError(
        e instanceof ApiError ? e.message : "That didn't work. Try again.",
      );
    } finally {
      setBusy(false);
    }
  }

  if (sent) {
    return (
      <Notice>
        {sent} The link works for 30 minutes. Check your spam folder if it
        hasn&apos;t arrived in a few minutes.
      </Notice>
    );
  }
  return (
    <form onSubmit={submit} className="grid gap-4" noValidate>
      {error && <Notice tone="error">{error}</Notice>}
      <Field id="email" label="Email">
        <Input
          id="email"
          type="email"
          autoComplete="email"
          inputMode="email"
          required
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
      </Field>
      <Button
        type="submit"
        busy={busy}
        disabled={!email}
        className="mt-1 w-full"
      >
        {busy ? "Sending…" : "Send me a link"}
      </Button>
    </form>
  );
}
