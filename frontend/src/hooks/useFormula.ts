import { useMutation } from "@tanstack/react-query";
import { apiFetch } from "../lib/api";
import { EvaluateResult } from "../types";

export function useFormula() {
  return useMutation<EvaluateResult, Error, { formula: string; window: "test" | "train" | "all" }>({
    mutationFn: (body) =>
      apiFetch<EvaluateResult>("/api/formulas/evaluate", {
        method: "POST",
        body: JSON.stringify(body),
      }),
  });
}
