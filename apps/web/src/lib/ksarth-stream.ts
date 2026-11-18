import { describeError, type KsarthMessage, refreshSession } from "./api";

export type StreamHandlers = {
  onQuestion: (message: KsarthMessage) => void;
  onDelta: (text: string) => void;
  onDone: (message: KsarthMessage, offline: boolean) => void;
  onError: (detail: string) => void;
};

/** Send a message and read Ksarth's reply as server-sent events. */
export async function sendToKsarth(content: string, handlers: StreamHandlers, signal?: AbortSignal) {
  const post = () =>
    fetch("/api/ksarth/messages", {
      method: "POST",
      credentials: "same-origin",
      headers: { "content-type": "application/json", accept: "text/event-stream" },
      body: JSON.stringify({ content }),
      signal,
    });

  let response: Response;
  try {
    response = await post();
    if (response.status === 401 && (await refreshSession())) response = await post();
  } catch (error) {
    if ((error as Error).name !== "AbortError") handlers.onError(describeError(0, undefined));
    return;
  }

  if (!response.ok || !response.body) {
    const body = await response.json().catch(() => undefined);
    handlers.onError(describeError(response.status, body));
    return;
  }

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  let finished = false;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += value;
      let boundary: number;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const event = /^event: (.+)$/m.exec(block)?.[1];
        const data = /^data: (.+)$/m.exec(block)?.[1];
        if (!event || !data) continue;
        const payload = JSON.parse(data);
        if (event === "question") handlers.onQuestion(payload.message);
        else if (event === "delta") handlers.onDelta(payload.text);
        else if (event === "done") {
          finished = true;
          handlers.onDone(payload.message, payload.offline);
        } else if (event === "error") {
          finished = true;
          handlers.onError(payload.detail);
        }
      }
    }
  } catch (error) {
    if ((error as Error).name === "AbortError") return;
  }
  if (!finished) handlers.onError("The reply was cut off. Check your connection and ask again.");
}
