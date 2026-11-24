"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { type FormEvent, useState } from "react";

import { Button } from "@/components/ui/button";
import { Field, Input } from "@/components/ui/field";
import { Notice } from "@/components/ui/notice";
import { api, ApiError, unwrap } from "@/lib/api";
import { useSignedIn } from "@/lib/session";

export function ResetPasswordForm() {
  const token = useSearchParams().get("token") ?? "";
  const signedIn = useSignedIn();
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const user = await unwrap(
        api.POST("/api/auth/password/reset", { body: { token, password } }),
      );
      signedIn(user, "/home");
    } catch (e) {
      setError(
        e instanceof ApiError ? e.message : "That didn't work. Try again.",
      );
      setBusy(false);
    }
  }

  if (!token) {
    return (
      <Notice tone="error">
        This link is incomplete.{" "}
        <Link href="/forgot-password" className="font-semibold underline">
          Ask for a new one
        </Link>
        .
      </Notice>
    );
  }
  return (
    <form onSubmit={submit} className="grid gap-4" noValidate>
      {error && (
        <Notice tone="error">
          {error}{" "}
          <Link href="/forgot-password" className="font-semibold underline">
            Get a new link
          </Link>
        </Notice>
      )}
      <Field id="password" label="New password" hint="At least 8 characters.">
        <Input
          id="password"
          type="password"
          autoComplete="new-password"
          required
          minLength={8}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
      </Field>
      <Button
        type="submit"
        busy={busy}
        disabled={password.length < 8}
        className="mt-1 w-full"
      >
        {busy ? "Saving…" : "Save and sign in"}
      </Button>
    </form>
  );
}
