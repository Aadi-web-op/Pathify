import "@fontsource/overpass/400.css";
import "@fontsource/overpass/600.css";
import "@fontsource/overpass/700.css";
import "@fontsource/overpass/800.css";
import "@fontsource/overpass-mono/400.css";
import "@fontsource/overpass-mono/600.css";
import "@fontsource/source-sans-3/400.css";
import "@fontsource/source-sans-3/400-italic.css";
import "@fontsource/source-sans-3/600.css";
import "@fontsource/source-sans-3/700.css";
import "@fontsource/noto-sans-devanagari/devanagari-400.css";
import "@fontsource/noto-sans-devanagari/devanagari-700.css";
import "./globals.css";

import type { Metadata, Viewport } from "next";

import { themeBootScript } from "@/lib/theme";

import { Providers } from "./providers";

export const metadata: Metadata = {
  title: { default: "Pathify: career guidance for Class 9 to college", template: "%s · Pathify" },
  description:
    "Pathify helps Indian students choose a stream, pick entrance exams and a degree, and get ready for a first job, with advice they can explain to their parents.",
  icons: { icon: "/icon.svg" },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f3f5f1" },
    { media: "(prefers-color-scheme: dark)", color: "#111613" },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeBootScript }} />
      </head>
      <body className="min-h-dvh">
        <a
          href="#main"
          className="sr-only z-50 rounded-md bg-sign px-4 py-2 font-display font-bold text-sign-ink focus:not-sr-only focus:fixed focus:left-4 focus:top-4"
        >
          Skip to content
        </a>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
