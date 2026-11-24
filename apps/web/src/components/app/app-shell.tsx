"use client";

import clsx from "clsx";
import { motion } from "motion/react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { type ReactNode, useEffect } from "react";

import { Logo } from "@/components/brand/logo";
import { SpotlightGroup } from "@/components/fx/bits";
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
  tryouts: (
    <path d="M9 4h6M10 4v5l-5 9a1.5 1.5 0 0 0 1.3 2h11.4a1.5 1.5 0 0 0 1.3-2l-5-9V4" />
  ),
  ksarth: <path d="M5 6h14v9H10l-4 3.5V15H5z" />,
  plan: (
    <>
      <path d="M6 20V5" />
      <path d="M6 5h11l-2.5 3.5L17 12H6" />
    </>
  ),
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
  { href: "/plan", label: "Plan", icon: "plan" },
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
    if (error?.status === 401)
      router.replace(`/signin?next=${encodeURIComponent(pathname)}`);
  }, [error, pathname, router]);

  useEffect(() => {
    if (me && !me.onboarded) router.replace("/onboarding");
  }, [me, router]);

  useEffect(() => {
    // Keep this browser in step with the theme saved on the account.
    if (me) applyTheme(me.preferences.theme ?? "system");
  }, [me]);

  return (
    <div className="relative grid min-h-dvh grid-rows-[auto_1fr]">
      {/* Ambient light at the top of every app page. */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-x-0 top-0 -z-10 h-[520px] bg-[radial-gradient(60%_80%_at_15%_0%,color-mix(in_oklab,var(--route-violet)_9%,transparent),transparent_70%),radial-gradient(45%_70%_at_90%_0%,color-mix(in_oklab,var(--route-pink)_6%,transparent),transparent_70%)]"
      />
      <header className="gutter sticky top-0 z-20 pt-3 [top:env(safe-area-inset-top,0px)]">
        <div className="mx-auto flex h-15 max-w-5xl items-center justify-between gap-4 rounded-full border border-rule bg-paper/75 pl-4 pr-2 shadow-[0_10px_40px_-24px_rgba(0,0,0,0.45)] backdrop-blur-xl">
          <Logo href="/home" />
          <nav
            aria-label="Main"
            className="flex items-center gap-1 max-sm:hidden"
          >
            {NAV.map((item) => {
              const active =
                pathname === item.href || pathname.startsWith(`${item.href}/`);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  aria-current={active ? "page" : undefined}
                  className={clsx(
                    "relative isolate rounded-full px-3 py-2 font-display text-[0.95rem] font-semibold transition-colors",
                    active ? "text-sign-ink" : "text-muted hover:text-ink",
                  )}
                >
                  {active && (
                    <motion.span
                      layoutId="nav-pill"
                      className="absolute inset-0 -z-10 rounded-full bg-sign"
                      transition={{
                        type: "spring",
                        stiffness: 420,
                        damping: 32,
                      }}
                    />
                  )}
                  {item.label}
                </Link>
              );
            })}
            <Button
              variant="ghost"
              size="sm"
              onClick={signOut}
              className="ml-1 text-muted"
            >
              Sign out
            </Button>
          </nav>
        </div>
      </header>
      {/* On phones the five sections live in a floating dock, so they fit even 360px screens. */}
      <nav
        aria-label="Main"
        className="fixed inset-x-3 bottom-[max(12px,env(safe-area-inset-bottom,0px))] z-30 grid grid-cols-6 rounded-[22px] border border-rule bg-paper/80 p-1 shadow-[0_18px_50px_-20px_rgba(0,0,0,0.55)] backdrop-blur-xl sm:hidden"
      >
        {NAV.map((item) => {
          const active =
            pathname === item.href || pathname.startsWith(`${item.href}/`);
          return (
            <Link
              key={item.href}
              href={item.href}
              aria-current={active ? "page" : undefined}
              className={clsx(
                "relative isolate grid justify-items-center gap-0.5 rounded-[18px] pb-1.5 pt-2 font-display text-[0.62rem] font-semibold transition-colors",
                active ? "text-sign-ink" : "text-muted",
              )}
            >
              {active && (
                <motion.span
                  layoutId="dock-pill"
                  className="absolute inset-0 -z-10 rounded-[18px] bg-sign"
                  transition={{ type: "spring", stiffness: 420, damping: 32 }}
                />
              )}
              <svg
                viewBox="0 0 24 24"
                className="size-[22px]"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                {ICONS[item.icon]}
              </svg>
              {item.label}
            </Link>
          );
        })}
      </nav>
      <main id="main" className="gutter isolate pb-16 pt-10 max-sm:pb-32">
        <SpotlightGroup className="mx-auto max-w-5xl">
          {isPending || (me && !me.onboarded) ? (
            <p className="flex items-center gap-2 text-muted">
              <Spinner /> Loading…
            </p>
          ) : error && error.status !== 401 ? (
            <Notice tone="error">
              {error.message}{" "}
              <button
                type="button"
                className="font-bold underline"
                onClick={() => refetch()}
              >
                Try again
              </button>
            </Notice>
          ) : me ? (
            children
          ) : null}
        </SpotlightGroup>
      </main>
    </div>
  );
}
