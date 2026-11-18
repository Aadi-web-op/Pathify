"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { type FormEvent, useState } from "react";

import { Button, ButtonLink } from "@/components/ui/button";
import { Choice } from "@/components/ui/choice";
import { Field, Input } from "@/components/ui/field";
import { Notice } from "@/components/ui/notice";
import { api, ApiError, unwrap } from "@/lib/api";
import { stageLabel } from "@/lib/profile";
import { ME_KEY, useMe, useSignOut } from "@/lib/session";
import { applyTheme, type Theme } from "@/lib/theme";

const THEMES: { value: Theme; label: string; detail: string }[] = [
  { value: "system", label: "Match my device", detail: "Light or dark, following your system" },
  { value: "light", label: "Light", detail: "Paper background" },
  { value: "dark", label: "Dark", detail: "Easier at night" },
];

export default function SettingsPage() {
  const { data: me } = useMe();
  const queryClient = useQueryClient();
  const router = useRouter();
  const signOut = useSignOut();
  const [name, setName] = useState(me?.name ?? "");
  const [nameState, setNameState] = useState<"idle" | "saving" | "saved">("idle");
  const [error, setError] = useState<string | null>(null);
  const [confirmEmail, setConfirmEmail] = useState("");
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  if (!me) return null;

  async function saveName(event: FormEvent) {
    event.preventDefault();
    setNameState("saving");
    setError(null);
    try {
      const user = await unwrap(api.PATCH("/api/users/me", { body: { name } }));
      queryClient.setQueryData(ME_KEY, user);
      setNameState("saved");
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Couldn't save your name.");
      setNameState("idle");
    }
  }

  async function chooseTheme(theme: Theme) {
    applyTheme(theme);
    try {
      queryClient.setQueryData(ME_KEY, await unwrap(api.PATCH("/api/users/me", { body: { theme } })));
    } catch {
      // The theme already applied in this browser; saving it to the account can be retried next time.
    }
  }

  async function deleteAccount(event: FormEvent) {
    event.preventDefault();
    setDeleting(true);
    setDeleteError(null);
    try {
      await unwrap(api.POST("/api/users/me/delete", { body: { email: confirmEmail } }));
      queryClient.clear();
      router.replace("/");
      router.refresh();
    } catch (e) {
      setDeleteError(e instanceof ApiError ? e.message : "Couldn't delete your account. Try again.");
      setDeleting(false);
    }
  }

  const theme = me.preferences.theme ?? "system";
  const signInMethods = me.auth_providers.map((p) => (p === "password" ? "Email and password" : p[0].toUpperCase() + p.slice(1)));

  return (
    <div className="grid max-w-2xl gap-12">
      <h1 className="text-[2.2rem] font-extrabold tracking-[-0.015em]">Settings</h1>

      <Section title="Account">
        <form onSubmit={saveName} className="grid gap-4">
          {error && <Notice tone="error">{error}</Notice>}
          <Field id="name" label="Name">
            <div className="flex gap-2">
              <Input
                id="name"
                autoComplete="name"
                maxLength={80}
                value={name}
                onChange={(e) => {
                  setName(e.target.value);
                  setNameState("idle");
                }}
              />
              <Button type="submit" variant="secondary" busy={nameState === "saving"} disabled={!name.trim() || name.trim() === me.name}>
                Save
              </Button>
            </div>
          </Field>
          {nameState === "saved" && <p className="text-sm text-sign" role="status">Name saved.</p>}
        </form>
        <dl className="grid gap-2 text-[0.95rem]">
          <div className="grid grid-cols-[8rem_1fr] gap-2">
            <dt className="text-muted">Email</dt>
            <dd className="min-w-0 break-words">{me.email}</dd>
          </div>
          <div className="grid grid-cols-[8rem_1fr] gap-2">
            <dt className="text-muted">Signs in with</dt>
            <dd>{signInMethods.join(", ") || "Email and password"}</dd>
          </div>
        </dl>
      </Section>

      <Section title="Profile">
        <p className="text-muted">{me.profile ? stageLabel(me.profile) : "Not set up yet."}</p>
        <ButtonLink href="/onboarding?edit=1" variant="secondary" className="justify-self-start">
          Edit profile
        </ButtonLink>
      </Section>

      <Section title="Appearance">
        <fieldset className="grid gap-3 sm:grid-cols-3">
          <legend className="sr-only">Theme</legend>
          {THEMES.map((t) => (
            <Choice key={t.value} name="theme" value={t.value} label={t.label} detail={t.detail} checked={theme === t.value} onChange={() => chooseTheme(t.value)} />
          ))}
        </fieldset>
      </Section>

      <Section title="Sign out">
        <p className="text-muted">Signs you out on this device and ends this session.</p>
        <Button variant="secondary" onClick={signOut} className="justify-self-start">
          Sign out
        </Button>
      </Section>

      <Section title="Delete account" danger>
        <p className="text-muted">
          This permanently deletes your account, your profile and your conversations with Ksarth. It can&apos;t be undone.
        </p>
        <form onSubmit={deleteAccount} className="grid gap-4">
          {deleteError && <Notice tone="error">{deleteError}</Notice>}
          <Field id="confirm-email" label={`Type ${me.email} to confirm`}>
            <Input id="confirm-email" type="email" autoComplete="off" value={confirmEmail} onChange={(e) => setConfirmEmail(e.target.value)} />
          </Field>
          <Button
            type="submit"
            variant="danger"
            busy={deleting}
            disabled={confirmEmail.trim().toLowerCase() !== me.email}
            className="justify-self-start"
          >
            Delete my account
          </Button>
        </form>
      </Section>
    </div>
  );
}

function Section({ title, danger, children }: { title: string; danger?: boolean; children: React.ReactNode }) {
  return (
    <section className="grid gap-4 border-t-2 pt-5" style={{ borderColor: danger ? "var(--danger)" : "var(--ink)" }}>
      <h2 className={danger ? "text-xl font-bold text-danger" : "text-xl font-bold"}>{title}</h2>
      {children}
    </section>
  );
}
