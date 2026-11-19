"use client";

import clsx from "clsx";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { type ReactNode, useEffect } from "react";

import { Logo } from "@/components/brand/logo";
import { Button, Spinner } from "@/components/ui/button";
import { Notice } from "@/components/ui/notice";
import { applyTheme } from "@/lib/theme";
import { useMe, useSignOut } from "@/lib/session";

const ICONS: Record<string, ReactNode> = {
  home: <path d="M4 11.5 12 5l8 6.5V20h-5v-5h-6v5H4z" />,
  results: (
    <>
      <circle cx="6" cy="17" r="2.2" />
      <circle cx="18" cy="7" r="2.2" />
      <path d="M8 16.2 11 13h3l2.4-4.2" />
    </>
  ),
  tryouts: <path d="M9 4h6M10 4v5l-5 9a1.5 1.5 0 0 0 1.3 2h11.4a1.5 1.5 0 0 0 1.3-2l-5-9V4" />,
  ksarth: <path d="M5 6h14v9H10l-4 3.5V15H5z" />,
  settings: (
    <>
      <circle cx="12" cy="12" r="3" />
      <path d="M12 3.5v2.5M12 18v2.5M3.5 12H6M18 12h2.5M6 6l1.8 1.8M16.2 16.2 18 18M6 18l1.8-1.8M16.2 7.8 18 6" />
    </>
  ),
};

const NAV = [
  { href: "/home", label: "Home", icon: "home" },
  { href: "/results", label: "Results", icon: "results" },
  { href: "/tryouts", label: "Tryouts", icon: "tryouts" },
  { href: "/ksarth", label: "Ksarth", icon: "ksarth" },
  { href: "/settings", label: "Settings", icon: "settings" },
];

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const signOut = useSignOut();
  const { data: me, error, isPending, refetch } = useMe();

  useEffect(() => {
    if (error?.status === 401) router.replace(`/signin?next=${encodeURIComponent(pathname)}`);
  }, [error, pathname, router]);

  useEffect(() => {
    if (me && !me.onboarded) router.replace("/onboarding");
  }, [me, router]);

  useEffect(() => {
    // Keep this browser in step with the theme saved on the account.
    if (me) applyTheme(me.preferences.theme ?? "system");
  }, [me]);

  return (
    <div className="grid min-h-dvh grid-rows-[auto_1fr]">
      <header className="gutter sticky top-0 z-20 border-b border-rule bg-paper/95 backdrop-blur-[2px] [top:env(safe-area-inset-top,0px)]">
        <div className="mx-auto flex h-16 max-w-5xl items-center justify-between gap-4">
          <Logo href="/home" compactOnPhone />
          <nav aria-label="Main" className="flex items-center gap-1 max-sm:hidden">
            {NAV.map((item) => {
              const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  aria-current={active ? "page" : undefined}
                  className={clsx(
                    "relative rounded-md px-2 py-2 font-display text-[0.95rem] font-bold sm:px-3",
                    active ? "text-ink" : "text-muted hover:text-ink",
                  )}
                >
                  {item.label}
                  {active && <span className="absolute inset-x-2 -bottom-[13px] h-1 rounded-full bg-sign sm:inset-x-3" />}
                </Link>
              );
            })}
            <Button variant="ghost" size="sm" onClick={signOut} className="ml-1 text-muted max-sm:hidden">
              Sign out
            </Button>
          </nav>
        </div>
      </header>
      {/* On phones the five sections live in a bottom tab bar, so they fit even 360px screens. */}
      <nav
        aria-label="Main"
        className="fixed inset-x-0 bottom-0 z-30 grid grid-cols-5 border-t border-rule bg-paper/95 pb-[env(safe-area-inset-bottom,0px)] backdrop-blur-[2px] sm:hidden"
      >
        {NAV.map((item) => {
          const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
          return (
            <Link
              key={item.href}
              href={item.href}
              aria-current={active ? "page" : undefined}
              className={clsx(
                "relative grid justify-items-center gap-0.5 pb-2 pt-2.5 font-display text-[0.72rem] font-bold",
                active ? "text-sign" : "text-muted",
              )}
            >
              {active && <span className="absolute inset-x-4 top-0 h-[3px] rounded-b-full bg-sign" />}
              <svg viewBox="0 0 24 24" className="size-6" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                {ICONS[item.icon]}
              </svg>
              {item.label}
            </Link>
          );
        })}
      </nav>
      <main id="main" className="gutter pb-16 pt-8 max-sm:pb-28">
        <div className="mx-auto max-w-5xl">
          {isPending || (me && !me.onboarded) ? (
            <p className="flex items-center gap-2 text-muted">
              <Spinner /> Loading…
            </p>
          ) : error && error.status !== 401 ? (
            <Notice tone="error">
              {error.message}{" "}
              <button type="button" className="font-bold underline" onClick={() => refetch()}>
                Try again
              </button>
            </Notice>
          ) : me ? (
            children
          ) : null}
        </div>
      </main>
    </div>
  );
}
