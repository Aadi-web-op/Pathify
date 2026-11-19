import type { Report } from "@/lib/tryouts";

/** The family report. Used for the student's own view and for the shared link; prints cleanly as a PDF. */
export function FamilyReport({ report }: { report: Report }) {
  const t = report.text;
  return (
    <article lang={report.language} className="report grid gap-10">
      <header className="grid gap-3 border-b-2 border-ink pb-6">
        <p className="eyebrow">Pathify</p>
        <h1 className="text-[2.2rem] font-extrabold leading-[1.1] tracking-[-0.01em] sm:text-[2.6rem]">{t.title}</h1>
        <p className="font-display font-bold">{report.stage}</p>
        <p className="max-w-[65ch] text-lg text-muted">{t.intro}</p>
        <p className="text-sm text-muted">{t.generated}</p>
      </header>

      <section className="report-block grid gap-4">
        <h2 className="text-2xl font-extrabold">{t.interests}</h2>
        {report.interests.length === 0 ? (
          <p className="text-muted">{t.interests_none}</p>
        ) : (
          <>
            <ol className="grid gap-3 sm:grid-cols-3">
              {report.interests.map((i, index) => (
                <li key={i.key} className="grid content-start gap-1 rounded-[12px] border border-rule bg-surface p-4">
                  <span className="font-mono text-sm text-muted">{index + 1}</span>
                  <span className="font-display text-lg font-bold">{i.name}</span>
                  <span className="text-[0.95rem] text-muted">{i.description}</span>
                </li>
              ))}
            </ol>
            {report.confidence && <p className="text-muted">{report.confidence}</p>}
          </>
        )}
      </section>

      <section className="grid gap-5">
        <h2 className="text-2xl font-extrabold">{t.careers}</h2>
        {report.careers.length === 0 && <p className="text-muted">{t.careers_none}</p>}
        {report.careers.map((c, index) => (
          <div key={c.slug} className="report-block grid gap-4 rounded-[14px] border border-rule bg-surface p-5 sm:p-6">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <h3 className="text-xl font-extrabold">
                <span className="mr-2 font-mono text-base text-muted">{index + 1}</span>
                {c.title}
              </h3>
              <span className="rounded-full bg-sign-tint px-2.5 py-0.5 font-display text-sm font-bold text-sign">{c.fit_label}</span>
            </div>
            <p>{c.summary}</p>
            <div className="grid gap-4 sm:grid-cols-2">
              {c.reasons.length > 0 && (
                <div className="grid content-start gap-1.5">
                  <h4 className="font-display font-bold">{t.why}</h4>
                  <ul className="grid gap-1 text-[0.95rem] text-muted">
                    {c.reasons.map((r) => (
                      <li key={r}>{r}</li>
                    ))}
                  </ul>
                </div>
              )}
              {c.watch_outs.length > 0 && (
                <div className="grid content-start gap-1.5">
                  <h4 className="font-display font-bold">{t.keep_in_mind}</h4>
                  <ul className="grid gap-1 text-[0.95rem] text-muted">
                    {c.watch_outs.map((r) => (
                      <li key={r}>{r}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
            <div className="grid gap-2 border-t border-rule pt-4">
              <h4 className="font-display font-bold">{t.route}</h4>
              {c.route_note && <p className="text-sm font-semibold text-warn">{c.route_note}</p>}
              <ol className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[0.95rem]">
                {c.steps.map((s, i) => (
                  <li key={`${s.label}-${i}`} className="flex items-center gap-2">
                    <span lang="en">
                      {s.label}
                      {s.years ? <span className="text-muted"> ({s.years}y)</span> : null}
                    </span>
                    {i < c.steps.length - 1 && (
                      <span aria-hidden="true" className="text-muted">
                        →
                      </span>
                    )}
                  </li>
                ))}
              </ol>
              {c.years_text && <p className="text-sm text-muted">{c.years_text}</p>}
              {c.check_with.length > 0 && (
                <p className="text-sm text-muted">
                  {t.check_with}: <span lang="en">{c.check_with.join("; ")}</span>
                </p>
              )}
            </div>
            {c.tryout && <p className="rounded-lg bg-sunk px-4 py-2.5 text-[0.95rem]">{c.tryout}</p>}
          </div>
        ))}
        {t.names_note && <p className="text-sm text-muted">{t.names_note}</p>}
      </section>

      <section className="report-block grid gap-3">
        <h2 className="text-2xl font-extrabold">{t.tryouts}</h2>
        {report.tryouts.length === 0 ? (
          <p className="text-muted">{t.tryouts_none}</p>
        ) : (
          <ul className="grid gap-2">
            {report.tryouts.map((tr) => (
              <li key={tr.title} className="grid gap-0.5 border-t border-rule pt-2">
                <span className="font-display font-bold">
                  {tr.career_title}: <span lang="en">{tr.title}</span>
                </span>
                <span className="text-muted">{tr.summary}</span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="report-block grid gap-3 rounded-[14px] border-2 border-sign p-5 sm:p-6">
        <h2 className="text-2xl font-extrabold">{t.questions}</h2>
        <ol className="grid list-decimal gap-2 pl-5 marker:font-display marker:font-bold marker:text-sign">
          {report.questions.map((q) => (
            <li key={q}>{q}</li>
          ))}
        </ol>
      </section>

      <footer className="report-block grid gap-2 border-t border-rule pt-5 text-sm text-muted">
        <h2 className="font-display text-base font-bold text-ink">{t.about_heading}</h2>
        <p className="max-w-[75ch]">{t.about}</p>
      </footer>
    </article>
  );
}
