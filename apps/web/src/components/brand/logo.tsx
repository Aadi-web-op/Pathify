import clsx from "clsx";
import Link from "next/link";

/** The mark: a road turning onto a route, drawn like a highway direction sign. */
export function Mark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={clsx("shrink-0", className)} aria-hidden="true">
      <rect x="1" y="1" width="30" height="30" rx="7" fill="var(--sign)" />
      <rect x="3.5" y="3.5" width="25" height="25" rx="5" fill="none" stroke="var(--sign-ink)" strokeWidth="1.5" />
      <path d="M10 24V15.5a5 5 0 0 1 5-5h7" fill="none" stroke="var(--sign-ink)" strokeWidth="3" strokeLinecap="round" />
      <path d="M19 6.5l4.5 4-4.5 4" fill="none" stroke="var(--sign-ink)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function Logo({ href = "/", className }: { href?: string; className?: string }) {
  return (
    <Link href={href} className={clsx("inline-flex items-center gap-2.5 rounded-md", className)} aria-label="Pathify home">
      <Mark className="size-8" />
      <span className="font-display text-[1.35rem] font-extrabold tracking-[-0.01em]">Pathify</span>
    </Link>
  );
}
