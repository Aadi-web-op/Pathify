"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { type FormEvent, type KeyboardEvent, useCallback, useEffect, useRef, useState } from "react";
import Markdown from "react-markdown";

import { Button, Spinner } from "@/components/ui/button";
import { Notice } from "@/components/ui/notice";
import { api, ApiError, type KsarthMessage, unwrap } from "@/lib/api";
import { sendToKsarth } from "@/lib/ksarth-stream";

const HISTORY_KEY = ["ksarth", "messages"] as const;
const MAX_LENGTH = 2000;

type Pending = { question: string; answer: string };

export function KsarthChat() {
  const queryClient = useQueryClient();
  const router = useRouter();
  const params = useSearchParams();
  const history = useQuery({ queryKey: HISTORY_KEY, queryFn: () => unwrap(api.GET("/api/ksarth/messages")) });
  const status = useQuery({ queryKey: ["ksarth", "status"], queryFn: () => unwrap(api.GET("/api/ksarth/status")) });

  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState<Pending | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirmClear, setConfirmClear] = useState(false);
  const bottom = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  const abort = useRef<AbortController | null>(null);
  const askedFromLink = useRef(false);

  const messages = history.data ?? [];

  const append = useCallback(
    (message: KsarthMessage) =>
      queryClient.setQueryData<KsarthMessage[]>(HISTORY_KEY, (old = []) => [...old, message]),
    [queryClient],
  );

  const send = useCallback(
    async (text: string) => {
      const content = text.trim();
      if (!content || pending) return;
      setError(null);
      setDraft("");
      setPending({ question: content, answer: "" });
      abort.current = new AbortController();
      await sendToKsarth(
        content,
        {
          onQuestion: append,
          onDelta: (delta) => setPending((p) => (p ? { ...p, answer: p.answer + delta } : p)),
          onDone: (message) => {
            append(message);
            setPending(null);
          },
          onError: (detail) => {
            // The server drops an unanswered question, so drop it here too and give the text back.
            queryClient.setQueryData<KsarthMessage[]>(HISTORY_KEY, (old = []) =>
              old.filter((m, i) => !(i === old.length - 1 && m.role === "user" && m.content === content)),
            );
            setPending(null);
            setDraft(content);
            setError(detail);
          },
        },
        abort.current.signal,
      );
      input.current?.focus();
    },
    [append, pending, queryClient],
  );

  useEffect(() => () => abort.current?.abort(), []);

  useEffect(() => {
    const ask = params.get("ask");
    if (ask && history.isSuccess && !askedFromLink.current) {
      askedFromLink.current = true;
      router.replace("/ksarth");
      void send(ask.slice(0, MAX_LENGTH));
    }
  }, [params, history.isSuccess, router, send]);

  const streamedText = pending?.answer;
  useEffect(() => {
    // Jump while text streams in; glide when a finished message lands.
    bottom.current?.scrollIntoView({ block: "end", behavior: streamedText === undefined ? "smooth" : "auto" });
  }, [messages.length, streamedText]);

  async function clear() {
    try {
      await unwrap(api.DELETE("/api/ksarth/messages"));
      queryClient.setQueryData(HISTORY_KEY, []);
      setConfirmClear(false);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Couldn't clear the conversation.");
    }
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    void send(draft);
  }

  function onKey(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      void send(draft);
    }
  }

  // Hide the optimistic copy of the question once the server has echoed it into history.
  const showPendingQuestion = pending && messages.at(-1)?.content !== pending.question;

  return (
    <div className="grid gap-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div className="grid gap-1">
          <p className="eyebrow">Your guide</p>
          <h1 className="text-[2.5rem] font-semibold leading-[1] tracking-[-0.045em]">Ksarth</h1>
        </div>
        {messages.length > 0 &&
          (confirmClear ? (
            <div className="flex items-center gap-2 text-sm">
              <span>Delete this conversation?</span>
              <Button size="sm" variant="danger" onClick={clear}>
                Delete
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setConfirmClear(false)}>
                Keep
              </Button>
            </div>
          ) : (
            <Button size="sm" variant="ghost" className="text-muted" onClick={() => setConfirmClear(true)}>
              Clear conversation
            </Button>
          ))}
      </div>

      {status.data?.offline && (
        <Notice tone="warn">
          Ksarth is in offline mode: no AI key is set on the server, so replies are placeholders. Add{" "}
          <code className="font-mono text-sm">GEMINI_API_KEY</code> to the API&apos;s environment to turn it on.
        </Notice>
      )}

      <section aria-label="Conversation" aria-live="polite" className="grid gap-5">
        {history.isPending && (
          <p className="flex items-center gap-2 text-muted">
            <Spinner /> Loading your conversation…
          </p>
        )}
        {history.isSuccess && messages.length === 0 && !pending && (
          <div className="rounded-3xl border border-dashed border-rule-strong p-6 text-muted">
            <p className="font-display text-lg font-bold text-ink">Ask about streams, exams, degrees or jobs.</p>
            <p className="mt-1">
              Ksarth knows your class and goals from your profile. It will tell you when it isn&apos;t sure, and point you
              to the official source for dates, fees and cut-offs.
            </p>
          </div>
        )}
        {messages.map((message) => (
          <Bubble key={message.id} role={message.role} content={message.content} sources={message.sources} />
        ))}
        {showPendingQuestion && <Bubble role="user" content={pending.question} />}
        {pending && (
          <Bubble role="assistant" content={pending.answer} thinking={pending.answer.length === 0} streaming />
        )}
        <div ref={bottom} />
      </section>

      {error && <Notice tone="error">{error}</Notice>}

      <form
        onSubmit={submit}
        className="sticky bottom-0 -mx-4 grid gap-2 border-t border-rule bg-paper px-4 pb-[max(16px,env(safe-area-inset-bottom,0px))] pt-4 max-sm:bottom-[calc(62px+env(safe-area-inset-bottom,0px))] max-sm:pb-4 sm:mx-0 sm:px-0"
      >
        <label htmlFor="ksarth-input" className="sr-only">
          Message Ksarth
        </label>
        <div className="flex items-end gap-2">
          <textarea
            id="ksarth-input"
            ref={input}
            rows={2}
            maxLength={MAX_LENGTH}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={onKey}
            placeholder="Ask Ksarth anything about your next step"
            className="max-h-48 min-h-14 w-full resize-y rounded-2xl border-2 border-rule-strong bg-surface px-3.5 py-3 focus:border-sign focus:outline-none"
          />
          <Button type="submit" disabled={!draft.trim() || Boolean(pending)} className="h-14 shrink-0">
            Send
          </Button>
        </div>
        <p className="flex justify-between gap-3 text-xs text-muted">
          <span>Enter to send · Shift + Enter for a new line</span>
          {draft.length > MAX_LENGTH - 200 && (
            <span className="tabular">
              {draft.length} / {MAX_LENGTH}
            </span>
          )}
        </p>
      </form>
    </div>
  );
}

function Bubble({
  role,
  content,
  thinking,
  streaming,
  sources,
}: {
  role: "user" | "assistant";
  content: string;
  thinking?: boolean;
  streaming?: boolean;
  sources?: KsarthMessage["sources"];
}) {
  if (role === "user") {
    return (
      <div className="flex justify-end">
        <p className="max-w-[85%] whitespace-pre-wrap rounded-3xl rounded-br-[4px] bg-ink px-4 py-3 text-paper">{content}</p>
      </div>
    );
  }
  return (
    <div className="grid grid-cols-[2rem_1fr] gap-3">
      <span
        aria-hidden="true"
        className="grid size-8 place-items-center rounded-full bg-sign font-display text-sm font-bold text-sign-ink"
      >
        K
      </span>
      <div className={clsx("min-w-0 rounded-3xl rounded-tl-[4px] border border-rule bg-surface px-4 py-3", streaming && "border-sign/40")}>
        <span className="sr-only">Ksarth said:</span>
        {thinking ? (
          <span className="flex items-center gap-2 text-muted">
            <Spinner /> Ksarth is thinking…
          </span>
        ) : (
          <div className="prose-ksarth">
            <Markdown>{content}</Markdown>
          </div>
        )}
        {sources && sources.length > 0 && (
          <div className="mt-3 border-t border-rule pt-2.5">
            <p className="font-display text-xs font-bold uppercase tracking-[0.08em] text-muted">From Pathify&apos;s career guides</p>
            <ul className="mt-1.5 flex flex-wrap gap-2">
              {sources.map((s) => (
                <li key={s.n}>
                  <Link
                    href={`/careers/${s.slug}`}
                    className="inline-flex items-center gap-1.5 rounded-full border border-rule-strong px-2.5 py-0.5 text-sm hover:border-sign hover:text-sign"
                  >
                    <span className="font-mono text-xs text-muted">[{s.n}]</span>
                    {s.title}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
