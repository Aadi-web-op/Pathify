import type { ReactNode } from "react";

import { Logo } from "@/components/brand/logo";
import { DepartureBoard } from "@/components/fx/departure-board";
import { RouteField } from "@/components/fx/route-field";

export function AuthLayout({ title, intro, children }: { title: string; intro: ReactNode; children: ReactNode }) {
  return (
    <div className="grid min-h-dvh lg:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)]">
      <div className="gutter grid grid-rows-[auto_1fr] pb-10">
        <header className="mx-auto flex h-18 w-full max-w-md items-center">
          <Logo />
        </header>
        <main id="main" className="mx-auto grid w-full max-w-md content-start gap-7 pt-6 md:pt-14">
          <div className="grid gap-3">
            <h1 className="text-[clamp(2.1rem,5vw,2.8rem)] font-semibold leading-[1] tracking-[-0.04em]">{title}</h1>
            <p className="text-muted">{intro}</p>
          </div>
          {children}
        </main>
      </div>

      {/* The night map: a brand moment beside the form on wide screens. */}
      <aside
        aria-label="What Pathify does"
        className="relative m-3 hidden overflow-hidden rounded-[28px] border border-rule bg-sunk lg:grid lg:content-end lg:gap-8 lg:p-10"
      >
        <RouteField className="absolute inset-0 size-full opacity-70" />
        <div aria-hidden="true" className="absolute inset-0 bg-gradient-to-t from-sunk via-sunk/50 to-transparent" />
        <div className="relative grid gap-8">
          <div>
            <DepartureBoard />
          </div>
          <p className="font-display text-[2.4rem] font-semibold leading-[1] tracking-[-0.04em]">
            Your route starts <span className="serif-accent text-sign">here.</span>
          </p>
          <p className="max-w-[44ch] text-muted">
            Free for students. We never sell your data or let anyone pay to change your results.
          </p>
        </div>
      </aside>
    </div>
  );
}
