import clsx from "clsx";
import type { ComponentProps, ReactNode } from "react";

export const controlClass =
  "w-full rounded-lg border-2 border-rule-strong bg-surface px-3.5 h-12 text-ink placeholder:text-muted/70 " +
  "focus:border-sign focus:outline-none focus-visible:outline-none transition-colors " +
  "aria-[invalid=true]:border-danger";

type FieldProps = {
  id: string;
  label: string;
  hint?: ReactNode;
  error?: string | null;
  children: ReactNode;
  className?: string;
};

export function Field({ id, label, hint, error, children, className }: FieldProps) {
  return (
    <div className={clsx("grid min-w-0 gap-1.5", className)}>
      <label htmlFor={id} className="font-display text-[0.95rem] font-bold [overflow-wrap:anywhere]">
        {label}
      </label>
      {children}
      {error ? (
        <p id={`${id}-error`} className="text-sm text-danger" role="alert">
          {error}
        </p>
      ) : hint ? (
        <p id={`${id}-hint`} className="text-sm text-muted">
          {hint}
        </p>
      ) : null}
    </div>
  );
}

export function Input({ className, ...rest }: ComponentProps<"input">) {
  return <input className={clsx(controlClass, className)} {...rest} />;
}

export function Select({ className, children, ...rest }: ComponentProps<"select">) {
  return (
    <div className="relative">
      <select className={clsx(controlClass, "appearance-none pr-10", className)} {...rest}>
        {children}
      </select>
      <svg
        className="pointer-events-none absolute right-3.5 top-1/2 size-4 -translate-y-1/2 text-muted"
        viewBox="0 0 16 16"
        aria-hidden="true"
      >
        <path d="M3 6l5 5 5-5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      </svg>
    </div>
  );
}
