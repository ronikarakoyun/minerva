export type Window = "test" | "train" | "all";

export const WINDOW_MAP: Record<string, Window> = {
  "TEST · oos": "test",
  "TRAIN · is": "train",
  "TAM": "all",
  "TEST": "test",
  "TRAIN": "train",
};

export function windowToLabel(w: Window): string {
  return w === "test" ? "TEST · oos" : w === "train" ? "TRAIN · is" : "TAM";
}
