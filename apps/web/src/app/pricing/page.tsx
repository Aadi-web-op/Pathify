import type { Metadata } from "next";

import { PublicFooter, PublicHeader } from "@/components/app/public-header";
import { ButtonLink } from "@/components/ui/button";

export const metadata: Metadata = {
  title: "Pricing",
  description: "Pathify is free for students. Families and schools can pay for extras.",
};

const TIERS = [
  {
    name: "Students",
    price: "Free",
    note: "Always",
    perks: [
      "The interest assessment and your career map",
      "Every career guide and job tryout",
      "Ksarth, your guide (30 questions an hour)",
      "Your plan, XP and badges",
      "A family report in English or Hindi (30-day link)",
    ],
    cta: { href: "/signup", label: "Create your free account" },
  },
  {
    name: "Pathify Plus",
    price: "₹299",
    note: "a year, for families",
    perks: ["Family report links that last a year", "Up to 100 questions an hour to Ksarth", "Keeps Pathify free and ad-free for students"],
    cta: { href: "/signup?next=/settings", label: "Get Plus" },
    highlight: true,
  },
  {
    name: "Schools",
    price: "₹4,999",
    note: "a year, per counsellor",
    perks: ["Up to 20 classes (free: 1)", "Up to 400 students (free: 40 per class)", "Class summaries with conversation flags"],
    cta: { href: "/signup?next=/counsellor", label: "Start with one free class" },
  },
];

export default function PricingPage() {
  return (
    <div className="min-h-dvh">
      <PublicHeader />
      <main id="main" className="gutter py-16">
        <div className="mx-auto grid max-w-6xl gap-12">
          <div className="grid max-w-3xl gap-4">
            <p className="eyebrow">Pricing</p>
            <h1 className="text-[clamp(2.4rem,6vw,4rem)] font-semibold leading-[1] tracking-[-0.045em]">
              Free for students. <span className="serif-accent text-sign">Always.</span>
            </h1>
            <p className="text-lg text-muted">
              Families and schools can pay for extras. Colleges and coaching centres can&apos;t pay to appear in anyone&apos;s
              results, at any price.
            </p>
          </div>
          <ul className="grid gap-4 md:grid-cols-3">
            {TIERS.map((t) => (
              <li
                key={t.name}
                className={`grid content-between gap-6 rounded-3xl border p-6 ${t.highlight ? "border-sign bg-surface" : "border-rule bg-surface"}`}
              >
                <div className="grid gap-3">
                  <p className="label-mono text-muted">{t.name}</p>
                  <p className="font-display text-4xl font-semibold tracking-[-0.04em]">
                    {t.price} <span className="text-base font-normal text-muted">{t.note}</span>
                  </p>
                  <ul className="grid gap-2 text-[0.95rem]">
                    {t.perks.map((p) => (
                      <li key={p} className="flex gap-2">
                        <span aria-hidden="true" className="text-sign">
                          ✓
                        </span>
                        {p}
                      </li>
                    ))}
                  </ul>
                </div>
                <ButtonLink href={t.cta.href} variant={t.highlight ? "primary" : "secondary"}>
                  {t.cta.label}
                </ButtonLink>
              </li>
            ))}
          </ul>
          <p className="text-sm text-muted">Payments are handled by Razorpay. Prices include GST where applicable.</p>
        </div>
      </main>
      <PublicFooter />
    </div>
  );
}
