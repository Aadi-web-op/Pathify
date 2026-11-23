import type { ReactNode } from "react";

import { Logo } from "@/components/brand/logo";
import { DepartureBoard } from "@/components/fx/departure-board";
import { NeonMetro } from "@/components/fx/neon-metro";

export function AuthLayout({ title, intro, children }: { title: string; intro: ReactNode; children: ReactNode }) {
  return (
    <div className="grid min-h-dvh lg:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)]">
      <div className="gutter grid grid-rows-[auto_1fr] pb-10">
        <header className="mx-auto flex h-18 w-full max-w-md items-center">
          <Logo />
        </header>
        <main id="main" className="mx-auto grid w-full max-w-md content-start gap-7 pt-6 md:pt-14">
          <div className="grid gap-3">
            <h1 className="text-[clamp(2.2rem,5vw,3rem)] font-bold leading-[0.95] tracking-[-0.045em]">{title}</h1>
            <p className="text-muted">{intro}</p>
          </div>
          {children}
        </main>
      </div>

      {/* The night map: a brand moment beside the form on wide screens. */}
      <aside
        aria-label="What Pathify does"
        className="night relative m-3 hidden overflow-hidden rounded-[28px] border border-rule lg:grid lg:content-end lg:gap-8 lg:p-10"
      >
        <NeonMetro className="absolute inset-0 size-full opacity-80" />
        <div aria-hidden="true" className="absolute inset-0 bg-gradient-to-t from-paper via-paper/40 to-transparent" />
        <div className="relative grid gap-8">
          <div className="rotate-[-1.5deg]">
            <DepartureBoard />
          </div>
          <p className="font-display text-[2.6rem] font-bold leading-[0.95] tracking-[-0.045em]">
            Your route starts <span className="serif-accent text-pop">here.</span>
          </p>
          <p className="max-w-[44ch] text-muted">
            Free for students. We never sell your data or let anyone pay to change your results.
          </p>
        </div>
      </aside>
    </div>
  );
}
