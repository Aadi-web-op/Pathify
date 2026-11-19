import type { Metadata } from "next";

// Shared reports are private to whoever has the link: keep them out of search engines.
export const metadata: Metadata = {
  title: "Career report",
  robots: { index: false, follow: false },
  referrer: "no-referrer",
};

export default function SharedReportLayout({ children }: { children: React.ReactNode }) {
  return children;
}
