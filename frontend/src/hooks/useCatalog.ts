import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiFetch, apiUrl } from "../lib/api";
import { CatalogRecord } from "../types";

export function useCatalog() {
  const qc = useQueryClient();

  const query = useQuery<CatalogRecord[]>({
    queryKey: ["catalog"],
    queryFn: () => apiFetch<CatalogRecord[]>("/api/catalog"),
  });

  const deleteOne = useMutation({
    mutationFn: (formula: string) =>
      apiFetch(`/api/catalog/${encodeURIComponent(formula)}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["catalog"] }),
  });

  const deleteAll = useMutation({
    mutationFn: () => apiFetch("/api/catalog", { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["catalog"] }),
  });

  const exportCsv = () => {
    window.open(apiUrl("/api/catalog/export.csv"), "_blank");
  };

  return { ...query, deleteOne, deleteAll, exportCsv };
}
