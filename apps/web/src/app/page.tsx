import { PublicFooter, PublicHeader } from "@/components/app/public-header";
import { BurstLink } from "@/components/fx/burst-link";
import { CountUp, Magnetic, Marquee, Reveal, ScrollRuler, SpotlightGroup } from "@/components/fx/bits";
import { DepartureBoard } from "@/components/fx/departure-board";
import { NeonMetro } from "@/components/fx/neon-metro";
import { ChatDemo, MiniMap, ReportDemo, RiasecDemo, TryoutDemo } from "@/components/landing/demos";
import { Journey, type Stop } from "@/components/landing/journey";
import { ButtonLink } from "@/components/ui/button";
import { CAREERS, CLUSTERS } from "@/lib/careers";
import tryoutsJson from "@data/tryouts.json";

const TRYOUT_COUNT = (Array.isArray(tryoutsJson) ? tryoutsJson : (tryoutsJson as { tryouts: unknown[] }).tryouts).length;

const STOPS: Stop[] = [
  {
    when: "Class 9–10",
    question: "Which stream should I take?",
    body: "Compare Science, Commerce and Humanities by what you enjoy, what each one opens up, and what it asks of you in Class 11 and 12.",
    exams: ["Science (PCM)", "Science (PCB)", "Commerce", "Humanities"],
  },
  {
    when: "Class 11–12",
    question: "Which exams, which degree?",
    body: "See how entrance exams connect to degrees, and plan around the ones that fit you, with a plan B.",
    exams: ["JEE", "NEET-UG", "CUET-UG", "CLAT", "NID DAT"],
  },
  {
    when: "College",
    question: "How do I land my first role?",
    body: "Turn your degree into a target role, then build the skills and projects that role asks for.",
    exams: ["Skills", "Projects", "First job"],
  },
];

const RULES = [
  { title: "No made-up numbers", body: "Salaries, cut-offs and fees come with a source and a date, or we don't show them." },
  { title: "Rankings aren't for sale", body: "Colleges and coaching centres can't pay to appear in your results." },
  { title: "Your data is yours", body: "Delete your account and everything in it from Settings, in one step." },
  { title: "Families included", body: "Students under 18 confirm that a parent or guardian knows they use Pathify." },
];

const ROUTE_DOTS = ["bg-route-violet", "bg-route-pink", "bg-route-blue", "bg-saffron", "bg-pop"];

export default function LandingPage() {
  const titles = CAREERS.map((c) => c.title.replace(/\s*\(.*\)$/, ""));
  const half = Math.ceil(titles.length / 2);

  return (
    <div className="night min-h-dvh overflow-x-clip">
      <ScrollRuler />
      <PublicHeader />

      <main id="main">
        {/* Hero: the night map, the headline, the departures board. */}
        <section className="relative -mt-[4.5rem] flex min-h-[100svh] items-center pb-16 pt-28">
          <NeonMetro className="absolute inset-0 size-full" />
          <div
            aria-hidden="true"
            className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_70%_60%_at_30%_45%,var(--paper)_20%,transparent_75%)] opacity-90"
          />
          <div aria-hidden="true" className="pointer-events-none absolute inset-x-0 bottom-0 h-40 bg-gradient-to-b from-transparent to-paper" />

          <div className="gutter relative mx-auto grid w-full max-w-6xl items-center gap-12 lg:grid-cols-[1.2fr_1fr] [&>*]:min-w-0">
            <div className="grid gap-7">
              <p className="rise label-mono flex w-fit items-center gap-2 rounded-full border border-rule-strong bg-paper/60 px-3 py-1.5 text-ink backdrop-blur">
                <span className="size-1.5 rounded-full bg-pop" />
                Class 9 → college · free for students
              </p>
              <h1
                className="rise font-display text-[clamp(3rem,10.5vw,6.4rem)] font-bold leading-[0.9] tracking-[-0.055em]"
                style={{ ["--rise-delay" as string]: "0.08s" }}
              >
                Don&apos;t guess your future.{" "}
                <span className="serif-accent whitespace-nowrap pr-2 text-[1.08em] text-pop">Map it.</span>
              </h1>
              <p className="rise max-w-[46ch] text-lg text-muted sm:text-xl" style={{ ["--rise-delay" as string]: "0.16s" }}>
                Pick a stream, crack the right exams, land your first role. Pathify turns what you enjoy into routes you
                can actually explain to your parents.
              </p>
              <div className="rise flex flex-wrap items-center gap-3" style={{ ["--rise-delay" as string]: "0.24s" }}>
                <Magnetic>
                  <BurstLink href="/signup" variant="pop" size="lg">
                    Start free, takes 2 min
                    <span aria-hidden="true">→</span>
                  </BurstLink>
                </Magnetic>
                <ButtonLink href="/careers" variant="secondary" size="lg">
                  Browse {CAREERS.length} careers
                </ButtonLink>
              </div>
              <dl className="rise grid max-w-lg grid-cols-3 gap-4 border-t border-rule pt-5" style={{ ["--rise-delay" as string]: "0.32s" }}>
                {[
                  { n: 30, label: "quick questions" },
                  { n: CAREERS.length, label: "careers mapped" },
                  { n: TRYOUT_COUNT, label: "job tryouts" },
                ].map((s) => (
                  <div key={s.label} className="grid gap-0.5">
                    <dt className="label-mono order-2 text-muted">{s.label}</dt>
                    <dd className="order-1 font-display text-3xl font-bold tracking-[-0.04em] sm:text-4xl">
                      <CountUp value={s.n} />
                    </dd>
                  </div>
                ))}
              </dl>
            </div>

            <div className="rise relative lg:rotate-[2deg]" style={{ ["--rise-delay" as string]: "0.4s" }}>
              <div className="animate-float" style={{ ["--tilt" as string]: "0deg" }}>
                <DepartureBoard />
              </div>
              <p className="label-mono mt-3 text-center text-muted">Real routes from the Pathify career library</p>
            </div>
          </div>

          <a
            href="#careers-ticker"
            className="label-mono absolute bottom-6 left-1/2 hidden -translate-x-1/2 items-center gap-2 text-muted hover:text-ink sm:flex"
          >
            Scroll <span className="animate-bounce motion-reduce:animate-none">↓</span>
          </a>
        </section>

        {/* Ticker: every career in the library. */}
        <section id="careers-ticker" aria-label="Careers in the library" className="grid gap-3 border-y border-rule py-8">
          <Marquee duration={55}>
            {titles.slice(0, half).map((t, i) => (
              <span key={t} className="flex items-center gap-5 pr-5 font-display text-[clamp(1.8rem,5vw,3.4rem)] font-bold tracking-[-0.04em]">
                {t}
                <span className={`size-3 rounded-full ${ROUTE_DOTS[i % ROUTE_DOTS.length]}`} aria-hidden="true" />
              </span>
            ))}
          </Marquee>
          <Marquee duration={60} reverse>
            {titles.slice(half).map((t, i) => (
              <span
                key={t}
                className="flex items-center gap-5 pr-5 font-display text-[clamp(1.8rem,5vw,3.4rem)] font-bold tracking-[-0.04em] text-transparent [-webkit-text-stroke:1.5px_var(--muted)]"
              >
                {t}
                <span className={`size-3 rounded-full ${ROUTE_DOTS[(i + 2) % ROUTE_DOTS.length]}`} aria-hidden="true" />
              </span>
            ))}
          </Marquee>
        </section>

        <Journey stops={STOPS} />

        {/* Bento: what's inside, each tile a live demo. */}
        <section aria-labelledby="inside" className="gutter py-24">
          <div className="mx-auto grid max-w-6xl gap-10">
            <Reveal className="grid gap-4">
              <p className="label-mono text-muted">
                <span className="text-ink">02</span> / What&apos;s inside
              </p>
              <h2 id="inside" className="max-w-3xl text-[clamp(2.4rem,6vw,4.4rem)] font-bold leading-[0.95] tracking-[-0.045em]">
                Everything you need to pick a path, <span className="serif-accent text-aurora pr-1">none of the noise.</span>
              </h2>
            </Reveal>

            <SpotlightGroup className="grid gap-4 md:grid-cols-6">
              <Tile className="md:col-span-3 md:row-span-2" index="A" title="Interest quiz" body="30 everyday activities. Based on Holland's six interest types, with an honest note on how sure the result is.">
                <RiasecDemo />
              </Tile>
              <Tile className="md:col-span-3" index="B" title="Your career map" body="Your best-fit careers drawn as routes from where you are, with why each one fits.">
                <MiniMap />
              </Tile>
              <Tile className="md:col-span-3" index="C" title="Ksarth, your guide" body="Ask anything about streams, exams and jobs. Answers cite the guides they come from.">
                <ChatDemo />
              </Tile>
              <Tile className="md:col-span-3" index="D" title="Job tryouts" body="Ten-minute tasks from real jobs, with feedback, so you can feel the work before you choose it.">
                <TryoutDemo />
              </Tile>
              <Tile className="md:col-span-3" index="E" title="Family report" body="A summary in English or Hindi to talk it through at home, with a private link you control.">
                <ReportDemo />
              </Tile>
            </SpotlightGroup>

            <Reveal>
              <div className="flex flex-wrap items-end justify-between gap-6 rounded-3xl border border-rule bg-surface p-6 sm:p-8">
                <div className="grid gap-1">
                  <p className="font-display text-[clamp(3.5rem,10vw,7rem)] font-bold leading-none tracking-[-0.06em]">
                    <CountUp value={CAREERS.length} />
                    <span className="text-sign">+</span>
                  </p>
                  <p className="label-mono text-muted">careers across {CLUSTERS.length} fields, open to everyone</p>
                </div>
                <ButtonLink href="/careers" variant="secondary">
                  Open the career library →
                </ButtonLink>
              </div>
            </Reveal>
          </div>
        </section>

        {/* House rules on a lime band. */}
        <section aria-labelledby="rules" className="gutter bg-pop py-20 text-pop-ink">
          <div className="mx-auto grid max-w-6xl gap-10">
            <div className="flex flex-wrap items-end justify-between gap-4">
              <h2 id="rules" className="text-[clamp(2.4rem,6vw,4.4rem)] font-bold leading-[0.95] tracking-[-0.045em]">
                Advice you can <span className="serif-accent">trust.</span>
              </h2>
              <p className="label-mono rotate-[-3deg] rounded-full border-2 border-current px-4 py-1.5 font-bold">No paid rankings, ever</p>
            </div>
            <ol className="grid border-t-2 border-current md:grid-cols-4">
              {RULES.map((rule, i) => (
                <li
                  key={rule.title}
                  className="group grid content-start gap-3 border-b-2 border-current py-6 md:border-b-0 md:border-r-2 md:px-5 md:first:pl-0 md:last:border-r-0"
                >
                  <span className="font-mono text-sm font-bold">0{i + 1}</span>
                  <h3 className="text-2xl font-bold tracking-[-0.03em] transition-transform duration-300 group-hover:translate-x-1">
                    {rule.title}
                  </h3>
                  <p className="text-[0.98rem] opacity-80">{rule.body}</p>
                </li>
              ))}
            </ol>
          </div>
        </section>

        {/* Final call, and a wordmark big enough to sit on. */}
        <section aria-labelledby="start" className="gutter relative overflow-hidden pt-24">
          <div className="mx-auto grid max-w-6xl justify-items-center gap-8 text-center">
            <Reveal className="grid justify-items-center gap-6">
              <h2 id="start" className="max-w-4xl text-[clamp(2.6rem,7vw,5.4rem)] font-bold leading-[0.92] tracking-[-0.05em]">
                Start with where you are. <span className="serif-accent text-aurora pr-1">It takes two minutes.</span>
              </h2>
              <Magnetic>
                <BurstLink href="/signup" size="lg">
                  Create your free account <span aria-hidden="true">→</span>
                </BurstLink>
              </Magnetic>
            </Reveal>
          </div>
          <p
            aria-hidden="true"
            className="pointer-events-none mt-16 select-none text-center font-display text-[clamp(5rem,27vw,24rem)] font-bold leading-[0.78] tracking-[-0.08em] text-transparent [-webkit-text-stroke:1.5px_var(--rule-strong)] [mask-image:linear-gradient(#000_40%,transparent)]"
          >
            pathify
          </p>
        </section>
      </main>

      <PublicFooter />
    </div>
  );
}

function Tile({
  index,
  title,
  body,
  className,
  children,
}: {
  index: string;
  title: string;
  body: string;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <article
      className={`spotlight group grid grid-rows-[auto_1fr] gap-8 overflow-hidden rounded-3xl border border-rule bg-surface p-6 transition-transform duration-500 ease-[cubic-bezier(0.34,1.56,0.64,1)] hover:-translate-y-1 sm:p-7 ${className ?? ""}`}
    >
      <div className="grid gap-2">
        <div className="flex items-center justify-between">
          <h3 className="text-2xl font-bold tracking-[-0.03em]">{title}</h3>
          <span className="label-mono text-muted">{index}</span>
        </div>
        <p className="max-w-[44ch] text-muted">{body}</p>
      </div>
      <div className="grid min-w-0 items-center">{children}</div>
    </article>
  );
}
