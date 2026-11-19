import catalogJson from "@data/careers.json";
import itemBankJson from "@data/assessment.json";

import type { components } from "./api-schema";

export type Career = components["schemas"]["Career"];
export type Route = components["schemas"]["Route"];
export type Step = components["schemas"]["Step"];
export type Dimension = "R" | "I" | "A" | "S" | "E" | "C";
export type Match = components["schemas"]["MatchOut"];
export type Result = components["schemas"]["ResultOut"];

// The API validates this file on startup and in CI; here it is typed with the API's own schema.
const catalog = catalogJson as unknown as { version: string; about: string; careers: Career[] };

export const CAREERS: Career[] = catalog.careers;
export const CATALOG_ABOUT = catalog.about;

export const DIMENSIONS = itemBankJson.dimensions as {
  key: Dimension;
  name: string;
  short: string;
  description: string;
}[];

export function getCareer(slug: string): Career | undefined {
  return CAREERS.find((c) => c.slug === slug);
}

export const CLUSTERS: { key: string; label: string }[] = [
  { key: "health", label: "Health and medicine" },
  { key: "technology", label: "Technology and data" },
  { key: "engineering", label: "Engineering" },
  { key: "design", label: "Design and creative" },
  { key: "science", label: "Science and research" },
  { key: "business", label: "Business and finance" },
  { key: "law-government", label: "Law and government" },
  { key: "education-social", label: "Education and social work" },
  { key: "media", label: "Media" },
  { key: "hospitality", label: "Hospitality" },
  { key: "defence-aviation", label: "Defence and aviation" },
  { key: "skilled-trades", label: "Skilled trades" },
];

export const clusterLabel = (key: string) => CLUSTERS.find((c) => c.key === key)?.label ?? key;

export const STREAM_LABEL: Record<string, string> = {
  science_pcm: "Science (PCM)",
  science_pcb: "Science (PCB)",
  science_pcmb: "Science (PCMB)",
  commerce: "Commerce",
  humanities: "Humanities",
  vocational: "Vocational",
  any: "Any stream",
};

export function streamsText(streams: string[]): string {
  if (streams.includes("any")) return "Any stream";
  // PCMB is implied wherever PCM or PCB is accepted, so leave it out to keep the label short.
  const shown = streams.filter((s) => s !== "science_pcmb" || streams.length === 1);
  return shown.map((s) => STREAM_LABEL[s] ?? s).join(" or ");
}

export const AFTER_LABEL: Record<string, string> = {
  class_10: "After Class 10",
  class_12: "After Class 12",
  graduation: "After graduation",
};

export function totalYears(steps: Step[]): number {
  return steps.reduce((sum, s) => sum + (s.years ?? 0), 0);
}

export function yearsText(years: number): string {
  if (years <= 0) return "";
  const rounded = Math.round(years * 2) / 2;
  return `about ${rounded} year${rounded === 1 ? "" : "s"}`;
}

export const STEP_LABEL: Record<Step["kind"], string> = {
  exam: "Entrance",
  degree: "Degree",
  diploma: "Diploma",
  course: "Course",
  work: "Experience",
  role: "First role",
};

/** "Doctor" -> "doctor", but keep acronyms such as "UX designer" as they are. */
export function inSentence(title: string): string {
  return /^[A-Z][a-z]/.test(title) ? title.charAt(0).toLowerCase() + title.slice(1) : title;
}

/** "a doctor", "an architect", "a UX designer". */
export function withArticle(title: string): string {
  const phrase = inSentence(title);
  const vowelSound = /^[aeio]/i.test(phrase) || /^u[^s]/.test(phrase);
  return `${vowelSound ? "an" : "a"} ${phrase}`;
}
