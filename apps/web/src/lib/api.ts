import createClient, { type Middleware } from "openapi-fetch";

import type { components, paths } from "./api-schema";

export type User = components["schemas"]["UserOut"];
export type StudentProfile = components["schemas"]["StudentProfile"];
export type Stage = components["schemas"]["Stage"];
export type Stream = components["schemas"]["Stream"];
export type Board = components["schemas"]["Board"];
export type Goal = components["schemas"]["Goal"];
export type KsarthMessage = components["schemas"]["KsarthMessageOut"];
export type Providers = components["schemas"]["ProvidersOut"];

/** An error the API explained. `message` is written for people and safe to show. */
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

type Detail = string | { msg: string; loc?: (string | number)[] }[] | undefined;

/** Turn FastAPI's error body into one readable sentence. */
export function describeError(status: number, body: unknown): string {
  const detail = (body as { detail?: Detail } | undefined)?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail.length > 0) {
    const first = detail[0];
    const field = first.loc?.filter((part) => part !== "body").join(" ");
    const msg = first.msg.replace(/^Value error, /, "");
    return field ? `${capitalise(String(field).replaceAll("_", " "))}: ${msg}` : capitalise(msg);
  }
  if (status === 0) return "Can't reach Pathify. Check your connection and try again.";
  if (status >= 500) return "Something went wrong on our side. Try again in a moment.";
  return "That didn't work. Try again.";
}

function capitalise(text: string) {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

let refreshing: Promise<boolean> | null = null;

/** Exchange the refresh cookie for new tokens. Concurrent callers share one request. */
export function refreshSession(): Promise<boolean> {
  refreshing ??= fetch("/api/auth/refresh", { method: "POST", credentials: "same-origin" })
    .then((response) => response.ok)
    .catch(() => false)
    .finally(() => {
      refreshing = null;
    });
  return refreshing;
}

const NO_RETRY = ["/api/auth/login", "/api/auth/register", "/api/auth/refresh", "/api/auth/logout"];

// A request body can only be read once, so keep an untouched copy for a possible retry.
const pending = new Map<string, Request>();

const retryAfterRefresh: Middleware = {
  onRequest({ request, id }) {
    pending.set(id, request.clone());
    return request;
  },
  async onResponse({ request, response, id }) {
    const copy = pending.get(id);
    pending.delete(id);
    const path = new URL(request.url).pathname;
    if (response.status !== 401 || NO_RETRY.includes(path) || !copy) return response;
    if (!(await refreshSession())) return response;
    return fetch(copy);
  },
  onError({ id }) {
    pending.delete(id);
  },
};

export const api = createClient<paths>({ baseUrl: "", credentials: "same-origin" });
api.use(retryAfterRefresh);

/** Unwrap an openapi-fetch result, throwing ApiError with a readable message. */
export async function unwrap<T>(
  call: Promise<{ data?: T; error?: unknown; response: Response }>,
): Promise<T> {
  let result;
  try {
    result = await call;
  } catch {
    throw new ApiError(0, describeError(0, undefined));
  }
  if (result.response.ok) return result.data as T;
  throw new ApiError(result.response.status, describeError(result.response.status, result.error));
}
