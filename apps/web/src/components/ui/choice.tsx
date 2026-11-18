import clsx from "clsx";

type ChoiceProps = {
  name: string;
  value: string;
  label: string;
  detail?: string;
  checked: boolean;
  onChange: (value: string) => void;
  type?: "radio" | "checkbox";
  disabled?: boolean;
};

/** A large radio or checkbox that reads like a card. The native input keeps keyboard and screen-reader support. */
export function Choice({ name, value, label, detail, checked, onChange, type = "radio", disabled }: ChoiceProps) {
  const id = `${name}-${value}`;
  return (
    <label
      htmlFor={id}
      className={clsx(
        "group relative flex cursor-pointer items-start gap-3 rounded-lg border-2 bg-surface px-4 py-3.5 transition-colors",
        "has-[:focus-visible]:outline-3 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-[var(--focus)]",
        checked ? "border-sign bg-sign-tint" : "border-rule hover:border-rule-strong",
        disabled && !checked && "cursor-not-allowed opacity-50",
      )}
    >
      <input
        id={id}
        type={type}
        name={name}
        value={value}
        checked={checked}
        disabled={disabled && !checked}
        onChange={() => onChange(value)}
        className="peer sr-only"
      />
      <span
        aria-hidden="true"
        className={clsx(
          "mt-0.5 grid size-5 shrink-0 place-items-center border-2 transition-colors",
          type === "radio" ? "rounded-full" : "rounded-[5px]",
          checked ? "border-sign bg-sign" : "border-rule-strong bg-surface",
        )}
      >
        {checked &&
          (type === "radio" ? (
            <span className="size-2 rounded-full bg-sign-ink" />
          ) : (
            <svg viewBox="0 0 12 12" className="size-3 text-sign-ink">
              <path d="M2.5 6.5l2.5 2.5 4.5-5.5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          ))}
      </span>
      <span className="grid gap-0.5">
        <span className="font-display font-bold leading-tight">{label}</span>
        {detail && <span className="text-sm leading-snug text-muted">{detail}</span>}
      </span>
    </label>
  );
}
