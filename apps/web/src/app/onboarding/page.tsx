import type { Metadata } from "next";
import { Suspense } from "react";

import { Logo } from "@/components/brand/logo";

import { OnboardingFlow } from "./onboarding-flow";

export const metadata: Metadata = { title: "Set up your profile" };

export default function OnboardingPage() {
  return (
    <div className="gutter min-h-dvh pb-16">
      <header className="mx-auto flex h-18 max-w-3xl items-center">
        <Logo href="/home" />
      </header>
      <main id="main" className="mx-auto max-w-3xl pt-2">
        <Suspense>
          <OnboardingFlow />
        </Suspense>
      </main>
    </div>
  );
}
