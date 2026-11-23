import clsx from "clsx";
import Link from "next/link";

/** The mark: one metro line with a 45° change, from your station (lime) to a destination. */
export function Mark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={clsx("shrink-0", className)} aria-hidden="true">
      <rect x="1" y="1" width="30" height="30" rx="9" fill="var(--sign)" />
      <path d="M7 22.5h5.5l8-8H25" fill="none" stroke="var(--sign-ink)" strokeWidth="3.2" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="7.5" cy="22.5" r="3.4" fill="var(--pop)" stroke="var(--sign-ink)" strokeWidth="1.6" />
      <circle cx="24.5" cy="14.5" r="2.9" fill="var(--sign)" stroke="var(--sign-ink)" strokeWidth="2" />
    </svg>
  );
}

export function Logo({ href = "/", className, compactOnPhone }: { href?: string; className?: string; compactOnPhone?: boolean }) {
  return (
    <Link href={href} className={clsx("group inline-flex items-center gap-2 rounded-md", className)} aria-label="Pathify home">
      <Mark className="size-8 transition-transform duration-500 ease-[cubic-bezier(0.34,1.56,0.64,1)] group-hover:-rotate-6 group-hover:scale-105" />
      <span className={clsx("font-display text-[1.4rem] font-bold tracking-[-0.04em]", compactOnPhone && "max-sm:hidden")}>
        pathify<span className="text-sign">.</span>
      </span>
    </Link>
  );
}
