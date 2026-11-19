import Link from "next/link";

import { Logo } from "@/components/brand/logo";
import { ButtonLink } from "@/components/ui/button";

export function PublicHeader() {
  return (
    <header className="gutter">
      <div className="mx-auto flex h-18 max-w-6xl items-center justify-between gap-3">
        <Logo />
        <nav className="flex items-center gap-1 sm:gap-2" aria-label="Main">
          <Link href="/careers" className="rounded-md px-2 py-2 font-display text-[0.95rem] font-bold text-muted hover:text-ink sm:px-3">
            Careers
          </Link>
          <ButtonLink href="/signin" variant="ghost" size="sm" className="max-sm:hidden sm:h-11 sm:px-4">
            Sign in
          </ButtonLink>
          <ButtonLink href="/signup" size="sm" className="sm:h-11 sm:px-5">
            Create account
          </ButtonLink>
        </nav>
      </div>
    </header>
  );
}

export function PublicFooter() {
  return (
    <footer className="gutter border-t border-rule py-8 text-sm text-muted">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-4">
        <span>Pathify · Career guidance for Indian students</span>
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
