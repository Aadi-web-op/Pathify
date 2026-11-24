"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { ButtonLink, Spinner } from "@/components/ui/button";
import { Notice } from "@/components/ui/notice";
import { api, ApiError, unwrap } from "@/lib/api";
import { ME_KEY } from "@/lib/session";

export function VerifyEmail() {
  const token = useSearchParams().get("token") ?? "";
  const queryClient = useQueryClient();
  const [state, setState] = useState<"checking" | "done" | { error: string }>(
    "checking",
  );
  const started = useRef(false);

  useEffect(() => {
    // Links work once, so never send the same one twice (React may run effects twice in development).
    if (started.current) return;
    started.current = true;
    if (!token) {
      setState({ error: "This link is incomplete." });
      return;
    }
    unwrap(api.POST("/api/auth/email/verify", { body: { token } }))
      .then(() => {
        void queryClient.invalidateQueries({ queryKey: ME_KEY });
        setState("done");
      })
      .catch((e) =>
        setState({
          error:
            e instanceof ApiError ? e.message : "We couldn't check this link.",
        }),
      );
  }, [token, queryClient]);

  if (state === "checking") {
    return (
      <p className="flex items-center gap-2 text-muted">
        <Spinner /> Checking…
      </p>
    );
  }
  if (state === "done") {
    return (
      <div className="grid gap-4">
        <Notice>Your email is confirmed. Thanks!</Notice>
        <ButtonLink href="/home" className="justify-self-start">
          Go to Pathify
        </ButtonLink>
      </div>
    );
  }
  return (
    <div className="grid gap-4">
      <Notice tone="error">
        {state.error} You can ask for a new link from your Home page.
      </Notice>
      <ButtonLink
        href="/home"
        variant="secondary"
        className="justify-self-start"
      >
        Go to Home
      </ButtonLink>
    </div>
  );
}
