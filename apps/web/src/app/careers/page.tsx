import type { Metadata } from "next";

import { PublicFooter, PublicHeader } from "@/components/app/public-header";
import { CAREERS, CLUSTERS } from "@/lib/careers";

import { CareerFinder } from "./career-finder";

export const metadata: Metadata = {
  title: "Careers in India, route by route",
  description:
    "What the work is like, and the exams, degrees and steps to get there from your stream: medicine, engineering, design, law, business, the civil services and more.",
};

export default function CareersPage() {
  const careers = CAREERS.map((c) => ({
    slug: c.slug,
    title: c.title,
    cluster: c.cluster,
    summary: c.summary,
    streams: Array.from(new Set(c.routes.flatMap((r) => r.streams ?? []))),
    afterClass10: c.routes.some((r) => r.after === "class_10"),
  }));
  return (
    <>
      <PublicHeader />
      <main id="main" className="gutter pb-20 pt-8">
        <div className="mx-auto grid max-w-6xl gap-10">
          <div className="grid max-w-3xl gap-3">
            <p className="eyebrow">Career library</p>
            <h1 className="text-[2.6rem] font-bold tracking-[-0.045em] sm:text-[3rem]">
              Careers in India, route by route
            </h1>
            <p className="text-lg text-muted">
              What each job is really like, and the exams, degrees and steps that lead to it from your stream.
            </p>
          </div>
          <CareerFinder careers={careers} clusters={CLUSTERS} />
        </div>
      </main>
      <PublicFooter />
    </>
  );
}
