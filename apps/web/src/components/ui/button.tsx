import clsx from "clsx";
import Link from "next/link";
import type { ComponentProps } from "react";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "md" | "lg" | "sm";

const base =
  "inline-flex items-center justify-center gap-2 rounded-lg font-display font-bold tracking-[0.01em] " +
  "transition-colors disabled:cursor-not-allowed disabled:opacity-55 select-none whitespace-nowrap";

const variants: Record<Variant, string> = {
  primary: "bg-sign text-sign-ink hover:bg-sign-hover",
  secondary: "border-2 border-ink/80 text-ink hover:bg-ink hover:text-paper",
  ghost: "text-ink hover:bg-sunk",
  danger: "bg-danger text-paper hover:opacity-90",
};

const sizes: Record<Size, string> = {
  sm: "h-9 px-3 text-sm",
  md: "h-11 px-5 text-[0.95rem]",
  lg: "h-13 px-6 text-base",
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
