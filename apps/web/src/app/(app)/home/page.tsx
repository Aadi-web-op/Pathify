"use client";

import Link from "next/link";

import { RouteLine } from "@/components/brand/route-line";
import { ButtonLink, Spinner } from "@/components/ui/button";
import { useMatches, useOverview } from "@/lib/assessment";
import type { StudentProfile } from "@/lib/api";
import { BOARDS, GOALS, routeAhead, stageLabel } from "@/lib/profile";
import { useMe } from "@/lib/session";

const STARTERS: Record<StudentProfile["stage"], string[]> = {
  school_9_10: [
    "How do I choose between Science, Commerce and Humanities?",
    "What can I do after Class 10 if I like drawing?",
    "Do I need Maths in Class 11 for a design career?",
  ],
  school_11_12: [
    "Which entrance exams fit my stream?",
    "Is a drop year a good idea?",
    "What degrees can I take with CUET?",
  ],
  college: [
    "What skills do I need for my first internship?",
    "How do I build projects that recruiters notice?",
    "Should I do a master's straight after my degree?",
  ],
  graduate: [
    "How do I move into a different field?",
    "Which short courses are worth the money?",
    "How do I explain a career switch in interviews?",
  ],
};

export default function HomePage() {
  const { data: me } = useMe();
  if (!me?.profile) return null;
  const profile = me.profile;
  const firstName = me.name.split(" ")[0];
  const stations = routeAhead(profile);
  const goals = GOALS.filter((g) => profile.goals.includes(g.value));
  const board = BOARDS.find((b) => b.value === profile.board)?.label;

  return (
    <div className="grid gap-10">
      <section className="grid gap-2">
        <p className="eyebrow">{stageLabel(profile)}</p>
        <h1 className="text-[2.6rem] font-bold leading-[0.95] tracking-[-0.05em] sm:text-[3.8rem]">
          Hi <span className="serif-accent text-sign">{firstName}</span>, here&apos;s your road ahead
        </h1>
      </section>

      <section aria-labelledby="route" className="grid gap-5 spotlight rounded-3xl border border-rule bg-surface p-5 sm:p-7">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 id="route" className="text-xl font-bold">
            Your route
          </h2>
          <p className="text-sm text-muted">The big decisions from where you are now</p>
        </div>
        <RouteLine stations={stations} current={0} label="Decisions ahead of you" />
      </section>

      <div className="grid gap-6 md:grid-cols-[3fr_2fr]">
        <section aria-labelledby="ask" className="grid content-start gap-4 rounded-3xl border-2 border-sign shadow-[0_24px_60px_-32px_color-mix(in_oklab,var(--sign)_75%,transparent)] bg-surface p-5 sm:p-7">
          <p className="eyebrow">Next step</p>
          <h2 id="ask" className="text-2xl font-bold">
            Talk it through with Ksarth
          </h2>
          <p className="text-muted">
            Ksarth knows your stage and goals. Ask the question you&apos;d ask an older cousin who works in the field.
          </p>
          <ul className="grid gap-2">
            {STARTERS[profile.stage].map((question) => (
              <li key={question}>
                <Link
                  href={`/ksarth?ask=${encodeURIComponent(question)}`}
                  className="group flex items-center justify-between gap-3 rounded-2xl border border-rule px-4 py-3 hover:border-sign"
                >
                  <span>{question}</span>
                  <span aria-hidden="true" className="font-display font-bold text-sign transition-transform group-hover:translate-x-0.5">
                    →
                  </span>
                </Link>
              </li>
            ))}
          </ul>
          <ButtonLink href="/ksarth" variant="secondary" className="justify-self-start">
            Ask your own question
          </ButtonLink>
        </section>

        <div className="grid content-start gap-6">
          <AssessmentCard />

          <section aria-labelledby="profile" className="grid gap-3 spotlight rounded-3xl border border-rule bg-surface p-5 sm:p-6">
            <div className="flex items-baseline justify-between gap-2">
              <h2 id="profile" className="text-lg font-bold">
                Your profile
              </h2>
              <Link href="/onboarding?edit=1" className="text-sm font-semibold text-sign underline underline-offset-4">
                Edit
              </Link>
            </div>
            <dl className="grid gap-2 text-[0.95rem]">
              <Row term="Now" value={stageLabel(profile)} />
              {board && <Row term="Board" value={board} />}
              <Row term="Help with" value={goals.map((g) => g.label).join(", ")} />
              {profile.interests && profile.interests.length > 0 && <Row term="Interests" value={profile.interests.join(", ")} />}
              {profile.state && <Row term="State" value={profile.state} />}
            </dl>
          </section>
        </div>
      </div>
    </div>
  );
}

function Row({ term, value }: { term: string; value: string }) {
  return (
    <div className="grid grid-cols-[6rem_1fr] gap-2">
      <dt className="text-muted">{term}</dt>
      <dd className="min-w-0">{value}</dd>
    </div>
  );
}

function AssessmentCard() {
  const overview = useOverview();
  const hasResult = Boolean(overview.data?.latest);
  const matches = useMatches(hasResult);

  if (overview.isPending || (hasResult && matches.isPending)) {
    return (
      <section className="grid gap-3 rounded-3xl border border-rule p-5 sm:p-6">
        <p className="flex items-center gap-2 text-muted">
          <Spinner /> Loading your results…
        </p>
      </section>
    );
  }

  if (hasResult && matches.data) {
    return (
      <section aria-labelledby="matches" className="grid gap-4 spotlight rounded-3xl border border-rule bg-surface p-5 sm:p-6">
        <div className="flex items-baseline justify-between gap-2">
          <h2 id="matches" className="text-lg font-bold">
            Your top matches
          </h2>
          <Link href="/results" className="text-sm font-semibold text-sign underline underline-offset-4">
            See your map
          </Link>
        </div>
        <ol className="grid gap-2">
          {matches.data.matches.slice(0, 3).map((m, i) => (
            <li key={m.slug} className="flex items-center justify-between gap-3">
              <span className="flex min-w-0 items-center gap-2.5">
                <span className="font-mono text-sm text-muted">{i + 1}</span>
                <Link href={`/careers/${m.slug}`} className="truncate font-display font-bold hover:underline">
                  {m.title}
                </Link>
              </span>
              <span className="tabular shrink-0 text-sm text-muted">{m.fit_label}</span>
            </li>
          ))}
        </ol>
        <div className="flex flex-wrap gap-x-5 gap-y-1 border-t border-rule pt-3 text-sm">
          <Link href="/tryouts" className="font-semibold text-sign underline underline-offset-4">
            Try a job
          </Link>
          <Link href="/report" className="font-semibold text-sign underline underline-offset-4">
            Family report
          </Link>
        </div>
      </section>
    );
  }

  const inProgress = overview.data?.in_progress;
  return (
    <section aria-labelledby="assessment" className="grid gap-3 spotlight rounded-3xl border border-rule bg-surface p-5 sm:p-6">
      <p className="eyebrow">{inProgress ? "Halfway there" : "5 minutes"}</p>
      <h2 id="assessment" className="text-lg font-bold">
        Find careers that fit what you enjoy
      </h2>
      <p className="text-sm text-muted">
        Rate {overview.data?.standard_length ?? 30} everyday activities. You&apos;ll get your best-fit careers drawn as
        routes from where you are now.
      </p>
      <ButtonLink href="/assessment" className="justify-self-start">
        {inProgress ? `Continue (${inProgress.answered} of ${inProgress.target})` : "Take the interest assessment"}
      </ButtonLink>
    </section>
  );
}
