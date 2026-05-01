import { useCallback, useEffect, useRef, useState } from "react";
import { connectJob } from "../lib/ws";
import { apiFetch } from "../lib/api";
import { EvaluateResult, JobEvent } from "../types";

export interface JobState {
  progress: number;
  logs: string[];
  result: EvaluateResult | null;
  error: string | null;
  done: boolean;
  /** Bağlantı kesildi, yeniden bağlanmaya çalışılıyor */
  reconnecting: boolean;
  cancel: () => Promise<void>;
}

export function useJob(jobId: string | null): JobState {
  const [state, setState] = useState<Omit<JobState, "cancel">>({
    progress: 0,
    logs: [],
    result: null,
    error: null,
    done: false,
    reconnecting: false,
  });
  const disconnectRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    if (!jobId) return;
    setState({ progress: 0, logs: [], result: null, error: null, done: false, reconnecting: false });

    const disconnect = connectJob(
      jobId,
      (ev: JobEvent) => {
        setState((prev) => {
          // Bağlantı yeniden kuruldu — reconnecting bayrağını kaldır
          const base = prev.reconnecting ? { ...prev, reconnecting: false } : prev;
          if (ev.type === "progress") return { ...base, progress: ev.value ?? base.progress };
          if (ev.type === "log")      return { ...base, logs: [...base.logs, ev.line ?? ""] };
          if (ev.type === "result")   return { ...base, result: ev.data ?? null, done: true, progress: 1 };
          if (ev.type === "error")    return { ...base, error: ev.message ?? "Hata", done: true };
          return base;
        });
      },
      // onClose: tüm retry'lar tükendi veya job tamamlandı
      () => setState((prev) => ({ ...prev, done: true, reconnecting: false })),
      {
        maxRetries: 3,
        onReconnecting: (attempt, delayMs) => {
          setState((prev) => ({
            ...prev,
            reconnecting: true,
            logs: [
              ...prev.logs,
              `⚠ Bağlantı kesildi — ${attempt}. deneme ${delayMs / 1000}s sonra…`,
            ],
          }));
        },
      },
    );

    disconnectRef.current = disconnect;
    return () => disconnect();
  }, [jobId]);

  const cancel = useCallback(async () => {
    if (!jobId) return;
    try {
      await apiFetch(`/api/jobs/${jobId}/cancel`, { method: "POST" });
    } catch {
      // hata olsa da local state'i done yap
    }
    setState((prev) => ({ ...prev, done: true, error: "İptal edildi", reconnecting: false }));
    disconnectRef.current?.();
  }, [jobId]);

  return { ...state, cancel };
}
