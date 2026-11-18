"use client";

import { useQuery } from "@tanstack/react-query";
import Script from "next/script";
import { useCallback, useEffect, useRef, useState } from "react";

import { buttonClass } from "@/components/ui/button";
import { Notice } from "@/components/ui/notice";
import { api, ApiError, unwrap, type User } from "@/lib/api";

type GoogleId = {
  accounts: {
    id: {
      initialize: (options: { client_id: string; callback: (response: { credential: string }) => void }) => void;
      renderButton: (element: HTMLElement, options: Record<string, unknown>) => void;
    };
  };
};

declare global {
  interface Window {
    google?: GoogleId;
  }
}

/** Shows only the providers the API has configured, so every button on screen works. */
export function SocialSignIn({ onSignedIn }: { onSignedIn: (user: User) => void }) {
  const { data: providers } = useQuery({
    queryKey: ["auth-providers"],
    queryFn: () => unwrap(api.GET("/api/auth/providers")),
    staleTime: Infinity,
  });
  const [error, setError] = useState<string | null>(null);
  const googleButton = useRef<HTMLDivElement>(null);
  const [googleReady, setGoogleReady] = useState(false);

  const handleCredential = useCallback(
    async ({ credential }: { credential: string }) => {
      setError(null);
      try {
        onSignedIn(await unwrap(api.POST("/api/auth/google", { body: { credential } })));
      } catch (e) {
        setError(e instanceof ApiError ? e.message : "Google sign-in didn't work. Try again.");
      }
    },
    [onSignedIn],
  );

  useEffect(() => {
    const clientId = providers?.google_client_id;
    if (!googleReady || !clientId || !googleButton.current || !window.google) return;
    window.google.accounts.id.initialize({ client_id: clientId, callback: handleCredential });
    window.google.accounts.id.renderButton(googleButton.current, {
      theme: "outline",
      size: "large",
      shape: "rectangular",
      text: "continue_with",
      width: googleButton.current.offsetWidth,
    });
  }, [googleReady, providers?.google_client_id, handleCredential]);

  if (!providers || (!providers.google_client_id && !providers.github && !providers.linkedin)) return null;

  return (
    <div className="grid gap-3">
      <div className="flex items-center gap-3 text-sm text-muted" aria-hidden="true">
        <span className="h-px flex-1 bg-rule" />
        or
        <span className="h-px flex-1 bg-rule" />
      </div>
      {error && <Notice tone="error">{error}</Notice>}
      {providers.google_client_id && (
        <>
          <Script src="https://accounts.google.com/gsi/client" strategy="afterInteractive" onReady={() => setGoogleReady(true)} />
          <div ref={googleButton} className="min-h-11 w-full" />
        </>
      )}
      {providers.github && (
        <a href="/api/auth/github/start" className={buttonClass("secondary", "md", "w-full")}>
          Continue with GitHub
        </a>
      )}
      {providers.linkedin && (
        <a href="/api/auth/linkedin/start" className={buttonClass("secondary", "md", "w-full")}>
          Continue with LinkedIn
        </a>
      )}
    </div>
  );
}
