"use client";

import { useQuery } from "@tanstack/react-query";

import { api, type ApiError, unwrap } from "./api";
import type { components } from "./api-schema";

export type TryoutSummary = components["schemas"]["TryoutSummary"];
export type Tryout = components["schemas"]["TryoutOut"];
export type Attempt = components["schemas"]["AttemptOut"];
export type Report = components["schemas"]["ReportOut"];
export type Share = components["schemas"]["ShareOut"];
export type AnswerValue = string | number | string[];

export const TRYOUTS_KEY = ["tryouts"] as const;

export function useTryouts() {
  return useQuery<TryoutSummary[], ApiError>({
    queryKey: TRYOUTS_KEY,
    queryFn: () => unwrap(api.GET("/api/tryouts")),
    staleTime: 0,
  });
}

export function useTryout(slug: string) {
  return useQuery<Tryout, ApiError>({
    queryKey: ["tryouts", slug],
    queryFn: () => unwrap(api.GET("/api/tryouts/{slug}", { params: { path: { slug } } })),
  });
}

export function useAttempt(id: string | null) {
  return useQuery<Attempt, ApiError>({
    queryKey: ["tryouts", "attempt", id],
    queryFn: () => unwrap(api.GET("/api/tryouts/attempts/{attempt_id}", { params: { path: { attempt_id: id! } } })),
    enabled: Boolean(id),
  });
}

export const submitTryout = (slug: string, answers: Record<string, AnswerValue>) =>
  unwrap(api.POST("/api/tryouts/{slug}/attempts", { params: { path: { slug } }, body: { answers } }));

export const saveReflection = (attemptId: string, enjoyment: number, seeMyself: "yes" | "maybe" | "no") =>
  unwrap(
    api.PUT("/api/tryouts/attempts/{attempt_id}/reflection", {
      params: { path: { attempt_id: attemptId } },
      body: { enjoyment, see_myself: seeMyself },
    }),
  );

export function scoreText(points: number, max: number, ungraded: number): string {
  const shown = Number.isInteger(points) ? String(points) : points.toFixed(1);
  const base = max > 0 ? `${shown} of ${max}` : "Not scored";
  return ungraded > 0 ? `${base} (${ungraded} written ${ungraded === 1 ? "answer" : "answers"} not graded)` : base;
}
