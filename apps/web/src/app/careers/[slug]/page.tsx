import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";

import { PublicFooter, PublicHeader } from "@/components/app/public-header";
import { InterestBars } from "@/components/brand/interest-bars";
import { RouteLine } from "@/components/brand/route-line";
import { ButtonLink } from "@/components/ui/button";
import {
  AFTER_LABEL,
  CAREERS,
  type Career,
  clusterLabel,
  DIMENSIONS,
  getCareer,
  type Step,
  streamsText,
  withArticle,
  totalYears,
  yearsText,
} from "@/lib/careers";

type Params = { slug: string };

export function generateStaticParams(): Params[] {
  return CAREERS.map((c) => ({ slug: c.slug }));
}

export const dynamicParams = false;

export async function generateMetadata({ params }: { params: Promise<Params> }): Promise<Metadata> {
  const career = getCareer((await params).slug);
  if (!career) return {};
  return {
    title: `How to become ${withArticle(career.title)} in India`,
    description: `${career.summary} Routes, entrance exams and degrees from each stream.`,
  };
}

function stations(steps: Step[]) {
  return steps.map((s) => ({
    label: s.label,
    detail: [s.detail, s.years && !/year/i.test(s.detail ?? "") ? yearsText(s.years) : null].filter(Boolean).join(" · "),
  }));
}

export default async function CareerPage({ params }: { params: Promise<Params> }) {
  const career = getCareer((await params).slug);
  if (!career) notFound();
  const related = (career.related ?? []).map(getCareer).filter((c): c is Career => Boolean(c));

  return (
    <>
      <PublicHeader />
      <main id="main" className="gutter pb-20 pt-6">
        <article className="mx-auto grid max-w-5xl gap-12">
          <nav aria-label="Breadcrumb" className="text-sm text-muted">
            <Link href="/careers" className="underline underline-offset-4 hover:text-ink">
              Careers
            </Link>{" "}
            / {clusterLabel(career.cluster)}
          </nav>

          <header className="grid gap-4">
            <p className="eyebrow">{clusterLabel(career.cluster)}</p>
            <h1 className="text-[2.6rem] font-bold tracking-[-0.045em] sm:text-[3.2rem]">{career.title}</h1>
            <p className="max-w-[60ch] text-xl text-muted">{career.summary}</p>
          </header>

          <section aria-labelledby="work" className="grid gap-6 md:grid-cols-[3fr_2fr]">
            <div className="grid content-start gap-3">
              <h2 id="work" className="text-2xl font-bold">
                What the work is like
              </h2>
              <ul className="grid gap-2.5">
                {career.work.map((w) => (
                  <li key={w} className="flex gap-3">
                    <span aria-hidden="true" className="mt-2.5 size-2 shrink-0 rounded-full bg-sign" />
                    {w}
                  </li>
                ))}
              </ul>
            </div>
            <dl className="grid content-start gap-4 spotlight rounded-3xl border border-rule bg-surface p-5">
              <div className="grid gap-1">
                <dt className="font-display text-sm font-bold text-muted">Skills that help</dt>
                <dd>{career.skills.join(", ")}</dd>
              </div>
              <div className="grid gap-1">
                <dt className="font-display text-sm font-bold text-muted">Useful school subjects</dt>
                <dd>{career.subjects.join(", ")}</dd>
              </div>
            </dl>
          </section>

          <section aria-labelledby="routes" className="grid gap-6">
            <div className="grid gap-1">
              <h2 id="routes" className="text-2xl font-bold">
                Routes in India
              </h2>
              <p className="text-muted">Durations are typical. Exams and rules change, so check with the official body each year.</p>
            </div>
            {career.routes.map((route) => (
              <div key={route.id} className="grid gap-4 spotlight rounded-3xl border border-rule bg-surface p-5 sm:p-6">
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <h3 className="text-xl font-bold">{route.label}</h3>
                  <p className="flex flex-wrap gap-2 text-sm">
                    <span className="rounded-full bg-sunk px-2.5 py-0.5 font-display font-bold">{AFTER_LABEL[route.after ?? "class_12"]}</span>
                    <span className="rounded-full bg-sign-tint px-2.5 py-0.5 font-display font-bold text-sign">
                      {streamsText(route.streams ?? ["any"])}
                    </span>
                  </p>
                </div>
                {route.requirement && <p className="text-sm text-muted">{route.requirement}</p>}
                <RouteLine stations={stations(route.steps)} current={-1} label={`${route.label}: steps`} />
                {totalYears(route.steps) > 0 && (
                  <p className="text-sm text-muted">In total: {yearsText(totalYears(route.steps))} of study and training.</p>
                )}
              </div>
            ))}
            {career.graduate_route && (
              <div className="grid gap-4 rounded-3xl border border-dashed border-rule-strong p-5 sm:p-6">
                <h3 className="text-xl font-bold">{career.graduate_route.label}</h3>
                <RouteLine stations={stations(career.graduate_route.steps)} current={-1} label={`${career.graduate_route.label}: steps`} />
              </div>
            )}
            {career.note && (
              <p className="max-w-[65ch] border-l-4 border-saffron pl-4 text-[1.05rem]">{career.note}</p>
            )}
          </section>

          <section aria-labelledby="fit" className="grid gap-6 md:grid-cols-2">
            <div className="grid content-start gap-3">
              <h2 id="fit" className="text-2xl font-bold">
                Who tends to enjoy it
              </h2>
              <p className="text-muted">
                This career&apos;s interest profile on Holland&apos;s six types, from 1 (hardly used) to 7 (used all the
                time). It&apos;s Pathify&apos;s estimate, guided by O*NET&apos;s ratings for {career.similar_onet}.
              </p>
              <ButtonLink href="/assessment" className="mt-2 justify-self-start">
                See how well it fits you
              </ButtonLink>
            </div>
            <InterestBars
              label={`Interest profile for ${career.title}`}
              max={7}
              rows={DIMENSIONS.map((d) => ({ key: d.key, name: d.name, short: d.short, score: career.riasec[d.key] }))}
            />
          </section>

          {(career.check_with ?? []).length > 0 && (
            <section aria-labelledby="check" className="grid gap-3">
              <h2 id="check" className="text-xl font-bold">
                Check the latest rules with
              </h2>
              <ul className="grid gap-2">
                {(career.check_with ?? []).map((body) => (
                  <li key={body.name}>
                    {body.url ? (
                      <a href={body.url} className="font-semibold text-sign underline underline-offset-4" rel="noopener noreferrer" target="_blank">
                        {body.name}
                      </a>
                    ) : (
                      <span className="font-semibold">{body.name}</span>
                    )}
                  </li>
                ))}
              </ul>
            </section>
          )}

          {related.length > 0 && (
            <section aria-labelledby="related" className="grid gap-4 border-t-2 border-ink pt-5">
              <h2 id="related" className="text-xl font-bold">
                Related careers
              </h2>
              <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                {related.map((r) => (
                  <li key={r.slug} className="grid gap-1">
                    <Link href={`/careers/${r.slug}`} className="font-display font-bold underline-offset-4 hover:text-sign hover:underline">
                      {r.title}
                    </Link>
                    <span className="text-sm leading-snug text-muted">{r.summary}</span>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </article>
      </main>
      <PublicFooter />
    </>
  );
}
