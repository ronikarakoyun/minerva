import { useEffect, useRef, useState } from "react";
import { connectJob } from "../lib/ws";
import { EvaluateResult, JobEvent } from "../types";

export interface JobState {
  progress: number;
  logs: string[];
  result: EvaluateResult | null;
  error: string | null;
  done: boolean;
}

export function useJob(jobId: string | null): JobState {
  const [state, setState] = useState<JobState>({
    progress: 0,
    logs: [],
    result: null,
    error: null,
    done: false,
  });
  const disconnectRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    if (!jobId) return;
    setState({ progress: 0, logs: [], result: null, error: null, done: false });

    const disconnect = connectJob(
      jobId,
      (ev: JobEvent) => {
        setState((prev) => {
          if (ev.type === "progress") return { ...prev, progress: ev.value ?? prev.progress };
          if (ev.type === "log") return { ...prev, logs: [...prev.logs, ev.line ?? ""] };
          if (ev.type === "result") return { ...prev, result: ev.data ?? null, done: true, progress: 1 };
          if (ev.type === "error") return { ...prev, error: ev.message ?? "Hata", done: true };
          return prev;
        });
      },
      () => setState((prev) => ({ ...prev, done: true }))
    );
    disconnectRef.current = disconnect;
    return () => disconnect();
  }, [jobId]);

  return state;
}
