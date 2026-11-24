import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api, ApiError, unwrap } from "./api";
import type { components } from "./api-schema";

export type ClassSummary = components["schemas"]["ClassOut"];
export type ClassDetail = components["schemas"]["ClassDetail"];
export type Joined = components["schemas"]["JoinedOut"];

export function useMyClasses(enabled = true) {
  return useQuery<ClassSummary[], ApiError>({ queryKey: ["classes"], queryFn: () => unwrap(api.GET("/api/classes")), enabled });
}

export function useClass(id: string | null) {
  return useQuery<ClassDetail, ApiError>({
    queryKey: ["classes", id],
    queryFn: () => unwrap(api.GET("/api/classes/{class_id}", { params: { path: { class_id: id! } } })),
    enabled: Boolean(id),
  });
}

export function useCreateClass() {
  const queryClient = useQueryClient();
  return useMutation<ClassSummary, ApiError, { name: string; school: string }>({
    mutationFn: (body) => unwrap(api.POST("/api/classes", { body })),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["classes"] });
      void queryClient.invalidateQueries({ queryKey: ["me"] });
    },
  });
}

export function useJoinedClasses() {
  return useQuery<Joined[], ApiError>({ queryKey: ["joined-classes"], queryFn: () => unwrap(api.GET("/api/me/classes")) });
}

export function useJoinClass() {
  const queryClient = useQueryClient();
  return useMutation<Joined, ApiError, { code: string; agree: boolean }>({
    mutationFn: (body) => unwrap(api.POST("/api/me/classes", { body })),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ["joined-classes"] }),
  });
}

export function useLeaveClass() {
  const queryClient = useQueryClient();
  return useMutation<void, ApiError, string>({
    mutationFn: async (id) => {
      await unwrap(api.DELETE("/api/me/classes/{class_id}", { params: { path: { class_id: id } } }));
    },
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ["joined-classes"] }),
  });
}
