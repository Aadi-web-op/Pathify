"use client";

import { type FormEvent, useState } from "react";

import { SocialSignIn } from "@/components/app/social-sign-in";
import { Button } from "@/components/ui/button";
import { Field, Input } from "@/components/ui/field";
import { Notice } from "@/components/ui/notice";
import { api, ApiError, unwrap } from "@/lib/api";
import { useSignedIn } from "@/lib/session";

export function SignUpForm() {
  const signedIn = useSignedIn();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const shortPassword = password.length > 0 && password.length < 8;

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const user = await unwrap(api.POST("/api/auth/register", { body: { name, email, password } }));
      signedIn(user);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "We couldn't create your account. Try again.");
      setBusy(false);
    }
  }

  return (
    <div className="grid gap-5">
      <form onSubmit={submit} className="grid gap-4" noValidate>
        {error && <Notice tone="error">{error}</Notice>}
        <Field id="name" label="Your name">
          <Input id="name" autoComplete="name" required maxLength={80} value={name} onChange={(e) => setName(e.target.value)} />
        </Field>
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
        <Field
          id="password"
          label="Password"
          hint="At least 8 characters."
          error={shortPassword ? "Use at least 8 characters." : null}
        >
          <Input
            id="password"
            type="password"
            autoComplete="new-password"
            required
            minLength={8}
            maxLength={128}
            aria-invalid={shortPassword || undefined}
            aria-describedby={shortPassword ? "password-error" : "password-hint"}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </Field>
        <Button type="submit" busy={busy} disabled={!name.trim() || !email || password.length < 8} className="mt-1 w-full">
          {busy ? "Creating your account…" : "Create account"}
        </Button>
        <p className="text-sm text-muted">
          If you&apos;re under 18, check with a parent or guardian before signing up. We&apos;ll ask you to confirm this
          during setup.
        </p>
      </form>
      <SocialSignIn onSignedIn={signedIn} />
    </div>
  );
}
