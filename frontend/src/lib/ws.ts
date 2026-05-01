import { JobEvent } from "../types";

const BASE_WS =
  import.meta.env.VITE_API_URL?.replace(/^http/, "ws") ?? "ws://localhost:8000";

/** Heartbeat aralığı: 25 sn (sunucu proxy'leri genelde 30 sn timeout kullanır). */
const HEARTBEAT_INTERVAL_MS = 25_000;

export interface ConnectJobOptions {
  /** Maksimum yeniden bağlanma denemesi (varsayılan: 3) */
  maxRetries?: number;
  /** Her yeniden bağlanma girişiminde çağrılır */
  onReconnecting?: (attempt: number, delayMs: number) => void;
  /** Heartbeat aralığı (ms). 0 → kapalı. Varsayılan: 25000 */
  heartbeatMs?: number;
}

/**
 * Bir job ID'sine WebSocket ile bağlanır.
 * - Beklenmedik kapanmada exponential backoff ile yeniden dener (1s → 2s → 4s).
 * - "result" veya "error" event'i gelince terminal kabul edilir; reconnect yapılmaz.
 * - Her 25 sn'de bir ping frame gönderir; pong gelmezse bağlantı yenilenir.
 * - Dönen cleanup fonksiyonu çağrılınca bağlantı kalıcı olarak kapatılır.
 */
export function connectJob(
  jobId: string,
  onEvent: (ev: JobEvent) => void,
  onClose?: () => void,
  options?: ConnectJobOptions,
): () => void {
  const {
    maxRetries = 3,
    onReconnecting,
    heartbeatMs = HEARTBEAT_INTERVAL_MS,
  } = options ?? {};
  let retries = 0;
  let manuallyClosed = false;
  let ws: WebSocket | null = null;
  let heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  let pongReceived = true; // ilk ping öncesi "received" sayılır

  const clearHeartbeat = () => {
    if (heartbeatTimer !== null) {
      clearInterval(heartbeatTimer);
      heartbeatTimer = null;
    }
  };

  const connect = () => {
    if (manuallyClosed) return;

    // N30: WS auth — api_key query param (VITE_API_KEY env var'dan)
    const apiKey = import.meta.env.VITE_API_KEY ?? "";
    const wsUrl = apiKey
      ? `${BASE_WS}/ws/jobs/${jobId}?api_key=${encodeURIComponent(apiKey)}`
      : `${BASE_WS}/ws/jobs/${jobId}`;
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      retries = 0;
      pongReceived = true;

      if (heartbeatMs > 0) {
        clearHeartbeat();
        heartbeatTimer = setInterval(() => {
          if (!pongReceived) {
            // Pong gelmedi → bağlantı koptu sayılır
            console.warn("[Minerva WS] Heartbeat timeout — reconnecting…");
            ws?.close();
            return;
          }
          pongReceived = false;
          try {
            // Tarayıcı WebSocket API'si binary ping desteklemez;
            // uygulama-seviyesi "ping" JSON mesajı gönderiyoruz.
            ws?.send(JSON.stringify({ type: "ping" }));
          } catch {
            /* bağlantı zaten kapanmış olabilir */
          }
        }, heartbeatMs);
      }
    };

    ws.onmessage = (e) => {
      let ev: JobEvent;
      try {
        ev = JSON.parse(e.data) as JobEvent;
      } catch (err) {
        console.error("[Minerva WS] Malformed message:", e.data, err);
        return;
      }

      // Sunucu pong yanıtı → sadece bayrak güncelle, onEvent'e iletme
      if ((ev as { type: string }).type === "pong") {
        pongReceived = true;
        return;
      }

      onEvent(ev);

      // Job terminale ulaştı — sonraki kapanmada reconnect yapma
      if (ev.type === "result" || ev.type === "error") {
        manuallyClosed = true;
      }
    };

    ws.onclose = () => {
      clearHeartbeat();
      if (manuallyClosed) {
        onClose?.();
        return;
      }

      if (retries < maxRetries) {
        retries++;
        const delayMs = Math.min(1000 * Math.pow(2, retries - 1), 8000); // 1s, 2s, 4s, 8s cap
        onReconnecting?.(retries, delayMs);
        setTimeout(connect, delayMs);
      } else {
        // Tüm denemeler tükendi
        onClose?.();
      }
    };

    // onerror her zaman onclose ile devam eder; ayrıca işleme gerek yok
    ws.onerror = () => {};
  };

  connect();

  return () => {
    manuallyClosed = true;
    clearHeartbeat();
    ws?.close();
  };
}
