import type { MapLine, MapStation } from "@/components/brand/career-map";

import type { StudentProfile } from "./api";
import { getCareer, type Match, STEP_LABEL, streamsText } from "./careers";
import { stageLabel } from "./profile";

export const LINE_COLORS = ["var(--sign)", "var(--route-saffron)", "var(--route-blue)"];

export function startStation(profile: StudentProfile): MapStation {
  return { label: "You", sub: stageLabel(profile) };
}

/** The route a student would most likely take: the first open one, else the first listed. */
export function chosenRoute(match: Match) {
  return match.routes.find((r) => r.open) ?? match.routes[0];
}

export function mapLine(match: Match, profile: StudentProfile, index: number): MapLine {
  const option = chosenRoute(match);
  const catalogRoute = getCareer(match.slug)?.routes.find((r) => r.id === option.id);
  const stations: MapStation[] = [];

  // Class 9-10 students still pick a stream first, so show it as the first stop.
  if (profile.stage === "school_9_10" && catalogRoute && catalogRoute.after !== "class_10") {
    stations.push({ label: streamsText(catalogRoute.streams ?? ["any"]), sub: "Class 11–12" });
  }
  for (const step of option.steps.slice(0, -1)) {
    stations.push({ label: step.label, sub: step.years ? `${step.years} yr` : STEP_LABEL[step.kind] });
  }

  const closed = !option.open;
  return {
    id: match.slug,
    title: match.title,
    color: LINE_COLORS[index % LINE_COLORS.length],
    stations: stations.slice(0, 5),
    caption: `${match.fit_label} · ${match.fit}`,
    closed,
    closedNote: closed && catalogRoute ? `Needs ${streamsText(catalogRoute.streams ?? [])}` : undefined,
  };
}
