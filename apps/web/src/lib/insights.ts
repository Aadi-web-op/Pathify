import { useQuery } from "@tanstack/react-query";

import { api, ApiError, unwrap } from "./api";
import type { components } from "./api-schema";

export type Reading = components["schemas"]["ReadOut"];
export type FieldOdds = components["schemas"]["FieldOddsOut"];

/** Pathify's Interest Reader on what the student typed. Quietly absent if the server has no model. */
export function useInterestReading(items: string[]) {
  const text = items.join(", ");
  return useQuery<Reading, ApiError>({
    queryKey: ["insights", "read", text],
    queryFn: () => unwrap(api.POST("/api/insights/read-interests", { body: { text } })),
    enabled: text.length >= 3,
    retry: false,
    staleTime: Infinity,
    placeholderData: (previous) => previous,
  });
}

/** What people with similar interests went on to study (Pathfinder). */
export function useFieldOdds(enabled = true) {
  return useQuery<FieldOdds, ApiError>({
    queryKey: ["insights", "field-odds"],
    queryFn: () => unwrap(api.GET("/api/insights/field-odds")),
    enabled,
    retry: false,
  });
}
