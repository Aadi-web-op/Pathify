"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { FamilyReport } from "@/components/app/family-report";
import { type Lang, LanguageSwitch } from "@/components/app/language-switch";
import { Button, Spinner } from "@/components/ui/button";
import { Notice } from "@/components/ui/notice";
import { api, ApiError, unwrap } from "@/lib/api";
import type { Share } from "@/lib/tryouts";

const SHARE_KEY = ["report", "share"] as const;

export default function ReportPage() {
  const [lang, setLang] = useState<Lang>("en");
  const report = useQuery({
    queryKey: ["report", lang],
    queryFn: () => unwrap(api.GET("/api/report", { params: { query: { lang } } })),
    staleTime: 0,
  });

  return (
    <div className="grid gap-8">
      <div className="grid gap-4 print:hidden">
        <div className="grid max-w-3xl gap-2">
          <p className="eyebrow">For your family</p>
          <p className="text-muted">
            A summary of your results to read with your parents or guardian. Choose a language, then print it, save it
            as a PDF, or send them a private link.
          </p>
        </div>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <LanguageSwitch value={lang} onChange={setLang} />
          <Button variant="secondary" onClick={() => window.print()}>
            Print or save as PDF
          </Button>
        </div>
        <SharePanel lang={lang} />
      </div>

      {report.isPending ? (
        <p className="flex items-center gap-2 text-muted">
          <Spinner /> Preparing your report…
        </p>
      ) : report.error ? (
        <Notice tone="error">{report.error instanceof ApiError ? report.error.message : "Couldn't load the report."}</Notice>
      ) : (
        <FamilyReport report={report.data} />
      )}
    </div>
  );
}

function SharePanel({ lang }: { lang: Lang }) {
  const queryClient = useQueryClient();
  const share = useQuery({ queryKey: SHARE_KEY, queryFn: () => unwrap(api.GET("/api/report/share")) });
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const active = share.data as Share | null | undefined;
  const origin = typeof window === "undefined" ? "" : window.location.origin;
  const url = active ? `${origin}/r/${active.token}` : "";

  async function create() {
    setBusy(true);
    setError(null);
    try {
      queryClient.setQueryData(SHARE_KEY, await unwrap(api.POST("/api/report/share", { body: { language: lang } })));
      setCopied(false);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Couldn't create a link.");
    } finally {
      setBusy(false);
    }
  }

  async function revoke() {
    setBusy(true);
    try {
      await unwrap(api.DELETE("/api/report/share"));
      queryClient.setQueryData(SHARE_KEY, null);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Couldn't turn off the link.");
    } finally {
      setBusy(false);
    }
  }

  async function copy() {
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
    } catch {
      document.getElementById("share-url")?.focus();
      (document.getElementById("share-url") as HTMLInputElement | null)?.select();
    }
  }

  return (
    <section aria-labelledby="share-title" className="grid gap-3 rounded-[14px] border border-rule bg-surface p-5">
      <h2 id="share-title" className="font-display text-lg font-bold">
        Send a private link
      </h2>
      {error && <Notice tone="error">{error}</Notice>}
      {active ? (
        <>
          <p className="text-sm text-muted">
            Anyone with this link can read your report until{" "}
            {new Date(active.expires_at).toLocaleDateString("en-IN", { day: "numeric", month: "long" })}. It shows your
            first name only.
          </p>
          <div className="flex flex-wrap gap-2">
            <label htmlFor="share-url" className="sr-only">
              Report link
            </label>
            <input
              id="share-url"
              readOnly
              value={url}
              onFocus={(e) => e.currentTarget.select()}
              className="h-11 min-w-0 flex-1 rounded-lg border-2 border-rule-strong bg-paper px-3 font-mono text-sm"
            />
            <Button onClick={copy}>{copied ? "Copied" : "Copy link"}</Button>
            <Button variant="ghost" onClick={revoke} busy={busy}>
              Turn off link
            </Button>
          </div>
          <p className="text-sm text-muted">
            Opens in {active.language === "hi" ? "Hindi" : "English"}.{" "}
            {active.language !== lang && (
              <button type="button" onClick={create} className="font-semibold text-sign underline underline-offset-4">
                Make a new {lang === "hi" ? "Hindi" : "English"} link instead
              </button>
            )}
          </p>
        </>
      ) : (
        <>
          <p className="text-sm text-muted">
            Make a link you can send on WhatsApp or email. It works for 30 days, shows only your first name, and you can
            turn it off at any time.
          </p>
          <Button onClick={create} busy={busy} className="justify-self-start">
            {lang === "hi" ? "Create a Hindi link" : "Create an English link"}
          </Button>
        </>
      )}
    </section>
  );
}
