"use client";

import clsx from "clsx";
import Link from "next/link";
import { useMemo, useState } from "react";

import { Input } from "@/components/ui/field";

type Item = {
  slug: string;
  title: string;
  cluster: string;
  summary: string;
  streams: string[];
  afterClass10: boolean;
};

const FILTERS = [
  { key: "all", label: "Any stream" },
  { key: "science_pcm", label: "Science (PCM)" },
  { key: "science_pcb", label: "Science (PCB)" },
  { key: "commerce", label: "Commerce" },
  { key: "humanities", label: "Humanities" },
  { key: "class_10", label: "Right after Class 10" },
];

function openFrom(item: Item, filter: string) {
  if (filter === "all") return true;
  if (filter === "class_10") return item.afterClass10;
  return item.streams.includes("any") || item.streams.includes(filter);
}

export function CareerFinder({ careers, clusters }: { careers: Item[]; clusters: { key: string; label: string }[] }) {
  const [filter, setFilter] = useState("all");
  const [query, setQuery] = useState("");

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    return careers.filter(
      (c) => openFrom(c, filter) && (!q || c.title.toLowerCase().includes(q) || c.summary.toLowerCase().includes(q)),
    );
  }, [careers, filter, query]);

  return (
    <div className="grid gap-8">
      <div className="grid gap-4">
        <div className="max-w-md">
          <label htmlFor="career-search" className="sr-only">
            Search careers
          </label>
          <Input id="career-search" type="search" placeholder="Search, for example: doctor, design, law" value={query} onChange={(e) => setQuery(e.target.value)} />
        </div>
        <fieldset className="flex flex-wrap gap-2">
          <legend className="mb-2 w-full font-display text-sm font-bold">Show careers open from</legend>
          {FILTERS.map((f) => (
            <label
              key={f.key}
              className={clsx(
                "cursor-pointer rounded-full border-2 px-3.5 py-1.5 font-display text-sm font-bold transition-colors has-[:focus-visible]:outline-3 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-[var(--focus)]",
                filter === f.key ? "border-sign bg-sign text-sign-ink" : "border-rule-strong hover:border-ink",
              )}
            >
              <input type="radio" name="stream-filter" value={f.key} checked={filter === f.key} onChange={() => setFilter(f.key)} className="sr-only" />
              {f.label}
            </label>
          ))}
        </fieldset>
        <p className="text-sm text-muted" role="status">
          {visible.length} of {careers.length} careers
        </p>
      </div>

      {clusters.map((cluster) => {
        const items = visible.filter((c) => c.cluster === cluster.key);
        if (!items.length) return null;
        return (
          <section key={cluster.key} aria-labelledby={`cluster-${cluster.key}`} className="grid gap-4">
            <h2 id={`cluster-${cluster.key}`} className="border-b-2 border-ink pb-2 text-xl font-bold">
              {cluster.label}
            </h2>
            <ul className="grid gap-x-8 gap-y-5 sm:grid-cols-2 lg:grid-cols-3">
              {items.map((c) => (
                <li key={c.slug} className="grid content-start gap-1">
                  <Link href={`/careers/${c.slug}`} className="font-display text-lg font-bold underline-offset-4 hover:text-sign hover:underline">
                    {c.title}
                  </Link>
                  <p className="text-[0.95rem] leading-snug text-muted">{c.summary}</p>
                </li>
              ))}
            </ul>
          </section>
        );
      })}
      {visible.length === 0 && (
        <p className="text-muted">No careers match that search. Try a broader word, or choose &ldquo;Any stream&rdquo;.</p>
      )}
    </div>
  );
}
