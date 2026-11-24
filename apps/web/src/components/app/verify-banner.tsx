"use client";

import { useState } from "react";

import { Notice } from "@/components/ui/notice";
import { api, ApiError, unwrap } from "@/lib/api";
import { useMe } from "@/lib/session";

/** Asks students to confirm their email, with a resend button. Hidden once confirmed. */
export function VerifyEmailBanner() {
  const { data: me } = useMe();
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  if (!me || me.email_verified) return null;

  async function resend() {
    setBusy(true);
    try {
      const body = await unwrap(api.POST("/api/auth/email/resend"));
      setMessage((body as { detail?: string }).detail ?? "Sent.");
    } catch (e) {
      setMessage(
        e instanceof ApiError
          ? e.message
          : "Couldn't send it. Try again later.",
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <Notice className="flex flex-wrap items-center justify-between gap-3">
      <span>
        {message ?? (
          <>
            Confirm your email (
            <strong className="font-semibold">{me.email}</strong>) so you can
            reset your password if you ever need to.
          </>
        )}
      </span>
      {!message && (
        <button
          type="button"
          onClick={resend}
          disabled={busy}
          className="font-semibold text-sign underline underline-offset-4"
        >
          {busy ? "Sending…" : "Send the link again"}
        </button>
      )}
    </Notice>
  );
}
