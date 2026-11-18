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

const NAV = [
  { href: "/home", label: "Home" },
  { href: "/ksarth", label: "Ksarth" },
  { href: "/settings", label: "Settings" },
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
          <Logo href="/home" />
          <nav aria-label="Main" className="flex items-center gap-1">
            {NAV.map((item) => {
              const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  aria-current={active ? "page" : undefined}
                  className={clsx(
                    "relative rounded-md px-2.5 py-2 font-display text-[0.95rem] font-bold sm:px-3",
                    active ? "text-ink" : "text-muted hover:text-ink",
                  )}
                >
                  {item.label}
                  {active && <span className="absolute inset-x-2.5 -bottom-[13px] h-1 rounded-full bg-sign sm:inset-x-3" />}
                </Link>
              );
            })}
            <Button variant="ghost" size="sm" onClick={signOut} className="ml-1 text-muted max-sm:hidden">
              Sign out
            </Button>
          </nav>
        </div>
      </header>
      <main id="main" className="gutter pb-16 pt-8">
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
