"use client";

import { useQuery } from "@tanstack/react-query";
import { useParams } from "next/navigation";
import { useState } from "react";

import { FamilyReport } from "@/components/app/family-report";
import { type Lang, LanguageSwitch } from "@/components/app/language-switch";
import { Logo } from "@/components/brand/logo";
import { Button, ButtonLink, Spinner } from "@/components/ui/button";
import { ApiError, api, unwrap } from "@/lib/api";

export default function SharedReportPage() {
  const { token } = useParams<{ token: string }>();
  const [lang, setLang] = useState<Lang | null>(null);
  const report = useQuery({
    queryKey: ["shared-report", token, lang],
    queryFn: () =>
      unwrap(
        api.GET("/api/report/shared/{token}", {
          params: { path: { token }, query: lang ? { lang } : {} },
        }),
      ),
    retry: false,
  });

  return (
    <div className="gutter min-h-dvh pb-16">
      <header className="mx-auto flex h-18 max-w-4xl items-center justify-between gap-3 print:hidden">
        <Logo />
        {report.data && (
          <div className="flex flex-wrap items-center gap-2">
            <LanguageSwitch value={lang ?? report.data.language} onChange={setLang} />
            <Button variant="secondary" size="sm" onClick={() => window.print()} className="max-sm:hidden">
              Print
            </Button>
          </div>
        )}
      </header>
      <main id="main" className="mx-auto max-w-4xl pt-4">
        {report.isPending ? (
          <p className="flex items-center gap-2 text-muted">
            <Spinner /> Opening the report…
          </p>
        ) : report.error ? (
          <div className="grid max-w-xl gap-4 pt-10">
            <h1 className="text-[2rem] font-extrabold">This link isn&apos;t working</h1>
            <p className="text-muted">
              {report.error instanceof ApiError && report.error.status === 404
                ? report.error.message
                : "We couldn't open the report. Check your connection and try again."}
            </p>
            <ButtonLink href="/" variant="secondary" className="justify-self-start">
              What is Pathify?
            </ButtonLink>
          </div>
        ) : (
          <FamilyReport report={report.data} />
        )}
      </main>
    </div>
  );
}
