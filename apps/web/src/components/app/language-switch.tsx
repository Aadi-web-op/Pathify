"use client";

import clsx from "clsx";

export type Lang = "en" | "hi";

export function LanguageSwitch({ value, onChange }: { value: Lang; onChange: (lang: Lang) => void }) {
  return (
    <div role="radiogroup" aria-label="Report language" className="inline-flex rounded-2xl border-2 border-ink p-0.5">
      {(
        [
          ["en", "English"],
          ["hi", "हिन्दी"],
        ] as const
      ).map(([key, label]) => (
        <button
          key={key}
          type="button"
          role="radio"
          aria-checked={value === key}
          lang={key}
          onClick={() => onChange(key)}
          className={clsx("rounded-md px-4 py-1.5 font-display font-bold", value === key ? "bg-ink text-paper" : "hover:bg-sunk")}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
