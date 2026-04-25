import { JobEvent } from "../types";

const BASE_WS = import.meta.env.VITE_API_URL?.replace(/^http/, "ws") ?? "ws://localhost:8000";

export function connectJob(
  jobId: string,
  onEvent: (ev: JobEvent) => void,
  onClose?: () => void
): () => void {
  const ws = new WebSocket(`${BASE_WS}/ws/jobs/${jobId}`);
  ws.onmessage = (e) => {
    try {
      onEvent(JSON.parse(e.data) as JobEvent);
    } catch {
      /* ignore malformed */
    }
  };
  ws.onclose = () => onClose?.();
  return () => ws.close();
}
