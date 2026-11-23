import clsx from "clsx";
import Link from "next/link";
import type { ComponentProps } from "react";

type Variant = "primary" | "pop" | "secondary" | "ghost" | "danger";
type Size = "md" | "lg" | "sm";

// Pill buttons with a springy press: a quick squash on :active, an overshoot on release.
const base =
  "inline-flex items-center justify-center gap-2 rounded-full font-display font-semibold tracking-[-0.005em] " +
  "transition-[transform,background-color,color,box-shadow,border-color] duration-300 ease-[cubic-bezier(0.34,1.56,0.64,1)] " +
  "active:scale-[0.94] active:duration-100 disabled:cursor-not-allowed disabled:opacity-55 disabled:active:scale-100 " +
  "select-none whitespace-nowrap";

const variants: Record<Variant, string> = {
  primary:
    "bg-sign text-sign-ink shadow-[0_8px_30px_-8px_color-mix(in_oklab,var(--sign)_70%,transparent)] hover:bg-sign-hover hover:-translate-y-0.5",
  pop: "bg-pop text-pop-ink shadow-[0_8px_30px_-8px_color-mix(in_oklab,var(--pop)_70%,transparent)] hover:-translate-y-0.5 hover:rotate-[-1deg]",
  secondary: "border border-rule-strong bg-surface/40 text-ink backdrop-blur hover:border-ink hover:-translate-y-0.5",
  ghost: "text-ink hover:bg-sunk",
  danger: "bg-danger text-paper hover:opacity-90",
};

const sizes: Record<Size, string> = {
  sm: "h-9 px-4 text-sm",
  md: "h-11 px-6 text-[0.95rem]",
  lg: "h-14 px-7 text-[1.05rem]",
};

export function buttonClass(variant: Variant = "primary", size: Size = "md", className?: string) {
  return clsx(base, variants[variant], sizes[size], className);
}

type ButtonProps = ComponentProps<"button"> & { variant?: Variant; size?: Size; busy?: boolean };

export function Button({ variant, size, busy, className, children, disabled, ...rest }: ButtonProps) {
  return (
    <button
      className={buttonClass(variant, size, className)}
      disabled={disabled || busy}
      aria-busy={busy || undefined}
      {...rest}
    >
      {busy && <Spinner />}
      {children}
    </button>
  );
}

type LinkProps = ComponentProps<typeof Link> & { variant?: Variant; size?: Size };

export function ButtonLink({ variant, size, className, ...rest }: LinkProps) {
  return <Link className={buttonClass(variant, size, className)} {...rest} />;
}

export function Spinner({ className }: { className?: string }) {
  return (
    <svg
      className={clsx("size-4 animate-spin motion-reduce:animate-none", className)}
      viewBox="0 0 24 24"
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" strokeOpacity="0.3" strokeWidth="3" />
      <path d="M21 12a9 9 0 0 0-9-9" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}
