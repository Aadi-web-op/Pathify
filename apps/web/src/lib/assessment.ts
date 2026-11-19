"use client";

import { useQuery } from "@tanstack/react-query";

import { api, type ApiError, unwrap } from "./api";
import type { components } from "./api-schema";

export type Session = components["schemas"]["SessionOut"];
export type Overview = components["schemas"]["AssessmentOverview"];
export type Matches = components["schemas"]["MatchesOut"];

export const OVERVIEW_KEY = ["assessment"] as const;
export const MATCHES_KEY = ["careers", "matches"] as const;

export function useOverview() {
  return useQuery<Overview, ApiError>({
    queryKey: OVERVIEW_KEY,
    queryFn: () => unwrap(api.GET("/api/assessment")),
    // Progress changes on another page, so always check again when a page opens.
    staleTime: 0,
  });
}

export function useMatches(enabled = true) {
  return useQuery<Matches, ApiError>({
    queryKey: MATCHES_KEY,
    queryFn: () => unwrap(api.GET("/api/careers/matches")),
    enabled,
    staleTime: 0,
  });
}

export const startAssessment = (restart: boolean) =>
  unwrap(api.POST("/api/assessment/start", { body: { restart } }));

export const answerQuestion = (sessionId: string, itemId: string, value: number) =>
  unwrap(
    api.POST("/api/assessment/{session_id}/answer", {
      params: { path: { session_id: sessionId } },
      body: { item_id: itemId, value },
    }),
  );

export const goBack = (sessionId: string) =>
  unwrap(api.POST("/api/assessment/{session_id}/back", { params: { path: { session_id: sessionId } } }));

export const extendAssessment = (sessionId: string) =>
  unwrap(api.POST("/api/assessment/{session_id}/extend", { params: { path: { session_id: sessionId } } }));
