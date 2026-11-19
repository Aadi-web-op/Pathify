import { PublicFooter, PublicHeader } from "@/components/app/public-header";
import { HeroMap } from "@/components/brand/hero-map";
import { ButtonLink } from "@/components/ui/button";
import { CAREERS } from "@/lib/careers";

const DECISIONS = [
  {
    when: "Class 9–10",
    question: "Which stream should I take?",
    body: "Compare Science, Commerce and Humanities by what you enjoy, what each one opens up, and what it asks of you in Class 11 and 12.",
  },
  {
    when: "Class 11–12",
    question: "Which exams and which degree?",
    body: "See how JEE, NEET, CUET, CLAT, NID DAT and others connect to degrees, and plan around the ones that fit you.",
  },
  {
    when: "College",
    question: "How do I get my first role?",
    body: "Turn your degree into a target role, then build the skills and projects that role asks for.",
  },
];

const AVAILABLE = [
  { title: "Your profile in two minutes", body: "Tell Pathify your class, board, stream or degree, and what you want help with." },
  {
    title: "Interest assessment",
    body: "Rate 30 everyday activities. Based on Holland's six interest types, with an honest note on how sure the result is.",
  },
  {
    title: "Your career map",
    body: "Your best-fit careers drawn as routes from where you are, with why each one fits and which routes your stream allows.",
  },
  {
    title: `A library of ${CAREERS.length} careers`,
    body: "What the work is like and the exams, degrees and steps to get there. Open to everyone, no account needed.",
  },
  {
    title: "Job tryouts",
    body: "Ten-minute tasks from real jobs, with feedback on your answers, so you can feel the work before you choose it.",
  },
  {
    title: "Ksarth, your guide",
    body: "Ask anything about streams, exams, degrees or jobs. Answers point to the career guides they come from.",
  },
  {
    title: "Family report",
    body: "A summary in English or Hindi to talk the decision through at home, with a private link you control.",
  },
];

const NEXT = [
  { title: "Hindi interface", body: "The whole app in Hindi, not just Ksarth's replies and the family report." },
  { title: "Progress on your route", body: "Track the steps on your chosen route and get reminders for exam dates." },
  { title: "More job tryouts", body: "Tryouts for every career in the library, written with people who do the work." },
  { title: "School counsellor view", body: "A class-level picture for counsellors, so they know who needs a conversation." },
]

const RULES = [
  { title: "No made-up numbers", body: "Salaries, cut-offs and fees come with a source and a date, or we don't show them." },
  { title: "Rankings aren't for sale", body: "Colleges and coaching centres can't pay to appear in your results." },
  { title: "Your data is yours", body: "Delete your account and everything in it from Settings, in one step." },
  { title: "Families included", body: "Students under 18 confirm that a parent or guardian knows they use Pathify." },
];

export default function LandingPage() {
  return (
    <>
      <PublicHeader />

      <main id="main">
        <section className="gutter pb-10 pt-10 sm:pt-16">
          <div className="mx-auto grid max-w-6xl gap-8 [&>*]:min-w-0">
            <div className="grid max-w-3xl gap-5">
              <p className="eyebrow">For students from Class 9 to college</p>
              <h1 className="text-[2.6rem] font-extrabold leading-[1.02] tracking-[-0.02em] sm:text-[4rem]">
                Find your route from school to your first job.
              </h1>
              <p className="max-w-[58ch] text-lg text-muted sm:text-xl">
                Pathify helps you choose a stream, pick exams and a degree, and get ready for work. Every suggestion
                comes with reasons you can explain to your parents.
              </p>
              <div className="flex flex-wrap items-center gap-3 pt-1">
                <ButtonLink href="/signup" size="lg">
                  Create your free account
                </ButtonLink>
                <ButtonLink href="/careers" size="lg" variant="secondary">
                  Explore careers
                </ButtonLink>
              </div>
              <p className="text-sm text-muted">Free for students. Built around CBSE, ICSE and state boards.</p>
            </div>

            <div className="min-w-0 rounded-[14px] border border-rule bg-surface px-3 pb-3 pt-5 sm:px-6">
              <HeroMap />
              <p className="px-1 pt-2 text-sm text-muted">
                Example routes. Many careers can be reached from more than one stream.
                <span className="sm:hidden"> Swipe the map to see all of it.</span>
              </p>
            </div>
          </div>
        </section>

        <section className="gutter border-t border-rule py-16" aria-labelledby="decisions">
          <div className="mx-auto grid max-w-6xl gap-10">
            <div className="grid max-w-2xl gap-3">
              <p className="eyebrow">Three big decisions</p>
              <h2 id="decisions" className="text-3xl font-extrabold sm:text-4xl">
                One guide for each turn in the road
              </h2>
            </div>
            <ol className="grid gap-8 md:grid-cols-3 md:gap-6">
              {DECISIONS.map((step, index) => (
                <li key={step.when} className="relative grid content-start gap-3">
                  <div className="flex items-center gap-3" aria-hidden="true">
                    <span className="grid size-9 place-items-center rounded-full border-[3px] border-sign bg-surface font-display font-extrabold text-sign">
                      {index + 1}
                    </span>
                    <span className="h-1 flex-1 rounded-full bg-sign/25" />
                  </div>
                  <p className="font-mono text-sm text-muted">{step.when}</p>
                  <h3 className="text-xl font-bold">{step.question}</h3>
                  <p className="text-muted">{step.body}</p>
                </li>
              ))}
            </ol>
          </div>
        </section>

        <section className="gutter border-t border-rule bg-surface py-16" aria-labelledby="status">
          <div className="mx-auto grid max-w-6xl gap-10 lg:grid-cols-[1fr_2fr]">
            <div className="grid content-start gap-3">
              <p className="eyebrow">Built in the open</p>
              <h2 id="status" className="text-3xl font-extrabold sm:text-4xl">
                What works today, and what&apos;s next
              </h2>
              <p className="text-muted">
                Pathify is new. We&apos;d rather tell you what&apos;s finished than pretend it all is.
              </p>
            </div>
            <div className="grid gap-8 sm:grid-cols-2">
              <div className="grid content-start gap-4">
                <h3 className="flex items-center gap-2 text-lg font-bold">
                  <span className="size-2.5 rounded-full bg-sign" aria-hidden="true" /> Available now
                </h3>
                <ul className="grid gap-4">
                  {AVAILABLE.map((item) => (
                    <li key={item.title} className="grid gap-1 border-l-4 border-sign pl-4">
                      <span className="font-display font-bold">{item.title}</span>
                      <span className="text-muted">{item.body}</span>
                    </li>
                  ))}
                </ul>
              </div>
              <div className="grid content-start gap-4">
                <h3 className="flex items-center gap-2 text-lg font-bold">
                  <span className="size-2.5 rounded-full border-2 border-muted" aria-hidden="true" /> Coming next
                </h3>
                <ul className="grid gap-4">
                  {NEXT.map((item) => (
                    <li key={item.title} className="grid gap-1 border-l-4 border-rule-strong pl-4">
                      <span className="font-display font-bold">{item.title}</span>
                      <span className="text-muted">{item.body}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </section>

        <section className="gutter border-t border-rule py-16" aria-labelledby="rules">
          <div className="mx-auto grid max-w-6xl gap-10">
            <div className="grid max-w-2xl gap-3">
              <p className="eyebrow">Our rules</p>
              <h2 id="rules" className="text-3xl font-extrabold sm:text-4xl">
                Advice you can trust
              </h2>
            </div>
            <dl className="grid gap-x-8 gap-y-6 sm:grid-cols-2 lg:grid-cols-4">
              {RULES.map((rule) => (
                <div key={rule.title} className="grid content-start gap-1.5 border-t-2 border-ink pt-4">
                  <dt className="font-display text-lg font-bold">{rule.title}</dt>
                  <dd className="text-muted">{rule.body}</dd>
                </div>
              ))}
            </dl>
          </div>
        </section>

        <section className="gutter border-t border-rule py-16">
          <div className="mx-auto flex max-w-6xl flex-col items-start justify-between gap-6 md:flex-row md:items-center">
            <h2 className="max-w-xl text-3xl font-extrabold">Start with where you are. It takes two minutes.</h2>
            <ButtonLink href="/signup" size="lg">
              Create your free account
            </ButtonLink>
          </div>
        </section>
      </main>

      <PublicFooter />
    </>
  );
}
