import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api, ApiError, unwrap } from "./api";
import type { components } from "./api-schema";

export type Plan = components["schemas"]["PlanOut"];
export type Progress = components["schemas"]["ProgressOut"];

export const PLAN_KEY = ["plan"] as const;
export const PROGRESS_KEY = ["progress"] as const;

export function usePlan() {
  return useQuery<Plan | null, ApiError>({
    queryKey: PLAN_KEY,
    queryFn: async () => {
      try {
        return await unwrap(api.GET("/api/plan"));
      } catch (e) {
        if (e instanceof ApiError && e.status === 404) return null;
        throw e;
      }
    },
  });
}

export function useProgress() {
  return useQuery<Progress, ApiError>({
    queryKey: PROGRESS_KEY,
    queryFn: () => unwrap(api.GET("/api/progress")),
    staleTime: 0,
  });
}

/** Every plan change can earn XP, so refresh progress alongside the plan. */
function usePlanMutation<T>(fn: (input: T) => Promise<Plan | void>) {
  const queryClient = useQueryClient();
  return useMutation<Plan | void, ApiError, T>({
    mutationFn: fn,
    onSuccess: (plan) => {
      if (plan) queryClient.setQueryData(PLAN_KEY, plan);
      else void queryClient.invalidateQueries({ queryKey: PLAN_KEY });
      void queryClient.invalidateQueries({ queryKey: PROGRESS_KEY });
    },
  });
}

export function useChooseRoute() {
  return usePlanMutation((body: { career_slug: string; route_id: string }) =>
    unwrap(api.PUT("/api/plan", { body })),
  );
}

export function useTick() {
  return usePlanMutation(({ id, done }: { id: string; done: boolean }) =>
    unwrap(
      api.PATCH("/api/plan/milestones/{milestone_id}", {
        params: { path: { milestone_id: id } },
        body: { done },
      }),
    ),
  );
}

export function useAddStep() {
  return usePlanMutation((label: string) =>
    unwrap(api.POST("/api/plan/milestones", { body: { label } })),
  );
}

export function useRemoveStep() {
  return usePlanMutation(async (id: string) => {
    await unwrap(
      api.DELETE("/api/plan/milestones/{milestone_id}", {
        params: { path: { milestone_id: id } },
      }),
    );
  });
}
