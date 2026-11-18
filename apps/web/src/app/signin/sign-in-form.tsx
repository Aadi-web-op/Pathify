"use client";

import { useSearchParams } from "next/navigation";
import { type FormEvent, useState } from "react";

import { SocialSignIn } from "@/components/app/social-sign-in";
import { Button } from "@/components/ui/button";
import { Field, Input } from "@/components/ui/field";
import { Notice } from "@/components/ui/notice";
import { api, ApiError, unwrap } from "@/lib/api";
import { useSignedIn } from "@/lib/session";

const PROVIDER_ERRORS: Record<string, string> = {
  github: "GitHub sign-in didn't finish. Try again, or use your email and password.",
  linkedin: "LinkedIn sign-in didn't finish. Try again, or use your email and password.",
};

export function SignInForm() {
  const params = useSearchParams();
  const signedIn = useSignedIn();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(PROVIDER_ERRORS[params.get("error") ?? ""] ?? null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const user = await unwrap(api.POST("/api/auth/login", { body: { email, password } }));
      signedIn(user, params.get("next"));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Sign-in didn't work. Try again.");
      setBusy(false);
    }
  }

  return (
    <div className="grid gap-5">
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
        <Field id="password" label="Password">
          <Input
            id="password"
            type="password"
            autoComplete="current-password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </Field>
        <Button type="submit" busy={busy} disabled={!email || !password} className="mt-1 w-full">
          {busy ? "Signing in…" : "Sign in"}
        </Button>
      </form>
      <SocialSignIn onSignedIn={(user) => signedIn(user, params.get("next"))} />
    </div>
  );
}
