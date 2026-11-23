import Link from "next/link";

import { Logo } from "@/components/brand/logo";
import { ButtonLink } from "@/components/ui/button";

export function PublicHeader() {
  return (
    <header className="gutter sticky top-0 z-50 pt-3">
      <div className="mx-auto flex h-15 max-w-6xl items-center justify-between gap-3 rounded-full border border-rule bg-paper/70 pl-4 pr-2 shadow-[0_10px_40px_-20px_rgba(0,0,0,0.4)] backdrop-blur-xl">
        <Logo />
        <nav className="flex items-center gap-1 sm:gap-2" aria-label="Main">
          <Link href="/careers" className="rounded-full px-2 py-2 font-display text-[0.95rem] font-semibold text-muted transition-colors hover:text-ink sm:px-3">
            Careers
          </Link>
          <ButtonLink href="/signin" variant="ghost" size="sm" className="max-sm:hidden sm:h-11 sm:px-5">
            Sign in
          </ButtonLink>
          <ButtonLink href="/signup" size="sm" className="sm:h-11 sm:px-6">
            <span className="max-[399px]:hidden">Create account</span>
            <span className="min-[400px]:hidden">Join free</span>
          </ButtonLink>
        </nav>
      </div>
    </header>
  );
}

export function PublicFooter() {
  return (
    <footer className="gutter border-t border-rule py-8 font-mono text-xs uppercase tracking-[0.12em] text-muted">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-4">
        <span>Pathify, career guidance for Indian students</span>
        <nav className="flex gap-5" aria-label="Footer">
          <Link href="/careers" className="hover:text-ink">
            Careers
          </Link>
          <Link href="/signin" className="hover:text-ink">
            Sign in
          </Link>
          <Link href="/signup" className="hover:text-ink">
            Create account
          </Link>
        </nav>
      </div>
    </footer>
  );
}
