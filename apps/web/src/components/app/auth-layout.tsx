import type { ReactNode } from "react";

import { Logo } from "@/components/brand/logo";
import { RouteLine } from "@/components/brand/route-line";

const PREVIEW = [
  { label: "Your profile", detail: "Two minutes" },
  { label: "Ask Ksarth", detail: "Streams, exams, jobs" },
  { label: "Your career map", detail: "Coming next" },
];

export function AuthLayout({ title, intro, children }: { title: string; intro: ReactNode; children: ReactNode }) {
  return (
    <div className="gutter grid min-h-dvh grid-rows-[auto_1fr] pb-10">
      <header className="mx-auto flex h-18 w-full max-w-5xl items-center">
        <Logo />
      </header>
      <main id="main" className="mx-auto grid w-full max-w-5xl items-start gap-12 pt-4 md:grid-cols-[minmax(0,26rem)_1fr] md:pt-10">
        <div className="grid gap-6">
          <div className="grid gap-2">
            <h1 className="text-[2rem] font-extrabold tracking-[-0.01em]">{title}</h1>
            <p className="text-muted">{intro}</p>
          </div>
          {children}
        </div>
        <aside className="hidden rounded-[14px] border border-rule bg-surface p-8 md:grid md:gap-6" aria-label="What happens next">
          <p className="eyebrow">What happens next</p>
          <RouteLine stations={PREVIEW} current={0} label="Your first steps on Pathify" className="sm:grid sm:gap-0" />
          <p className="border-t border-rule pt-5 text-sm text-muted">
            Pathify is free for students. We never sell your data or let anyone pay to change your results.
          </p>
        </aside>
      </main>
    </div>
  );
}
