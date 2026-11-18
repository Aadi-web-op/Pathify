"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { useCallback } from "react";

import { api, ApiError, unwrap, type User } from "./api";

export const ME_KEY = ["me"] as const;

export function useMe() {
  return useQuery<User, ApiError>({
    queryKey: ME_KEY,
    queryFn: () => unwrap(api.GET("/api/users/me")),
  });
}

/** Only follow `next` if it points inside this app, so a link can't bounce users to another site. */
export function safeNext(next: string | null, fallback: string): string {
  if (!next || !next.startsWith("/") || next.startsWith("//") || next.startsWith("/\\")) return fallback;
  return next;
}

export function useSignedIn() {
  const router = useRouter();
  const queryClient = useQueryClient();
  return useCallback(
    (user: User, next?: string | null) => {
      queryClient.setQueryData(ME_KEY, user);
      router.replace(user.onboarded ? safeNext(next ?? null, "/home") : "/onboarding");
      router.refresh();
    },
    [queryClient, router],
  );
}

export function useSignOut() {
  const router = useRouter();
  const queryClient = useQueryClient();
  return useCallback(async () => {
    await fetch("/api/auth/logout", { method: "POST", credentials: "same-origin" }).catch(() => undefined);
    queryClient.clear();
    router.replace("/signin");
    router.refresh();
  }, [queryClient, router]);
}
