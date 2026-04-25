import { useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { CChrome } from "../components/chrome/CChrome";
import { Panel } from "../components/chrome/Panel";
import { Btn } from "../components/atoms/Btn";
import { Pill } from "../components/atoms/Pill";
import { Check } from "../components/atoms/Check";
import { Select } from "../components/inputs/Select";
import { useCatalog } from "../hooks/useCatalog";
import { CatalogRecord } from "../types";

type SortKey = "ic" | "rank_ic" | "adj_ic" | "sharpe";

export default function Catalog() {
  const { data: records = [], isLoading, refetch, deleteOne, deleteAll, exportCsv } = useCatalog();
  const navigate = useNavigate();

  const [overfitFilter, setOverfitFilter] = useState("Tümü");
  const [sourceFilter, setSourceFilter] = useState("Tümü");
  const [onlyBacktest, setOnlyBacktest] = useState(false);
  const [regex, setRegex] = useState("");
  const [sortKey] = useState<SortKey>("rank_ic");
  const [selected, setSelected] = useState<Set<string>>(new Set());

  const filtered = useMemo(() => {
    let rs = records;
    if (overfitFilter !== "Tümü") {
      rs = rs.filter((r) => r.overfit_score != null);
    }
    if (sourceFilter !== "Tümü") {
      rs = rs.filter((r) => r.source === sourceFilter.toLowerCase());
    }
    if (onlyBacktest) {
      rs = rs.filter((r) => r.has_backtest);
    }
    if (regex) {
      try {
        const re = new RegExp(regex, "i");
        rs = rs.filter((r) => re.test(r.formula));
      } catch {
        /* invalid regex — ignore */
      }
    }
    return [...rs].sort((a, b) => (b[sortKey] ?? -Infinity) - (a[sortKey] ?? -Infinity));
  }, [records, overfitFilter, sourceFilter, onlyBacktest, regex, sortKey]);

  const toggleSelect = (formula: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      next.has(formula) ? next.delete(formula) : next.add(formula);
      return next;
    });
  };

  const handleDeleteSelected = async () => {
    for (const f of selected) {
      await deleteOne.mutateAsync(f);
    }
    setSelected(new Set());
  };

  const handleDeleteAll = async () => {
    if (!confirm("Katalogdaki TÜM kayıtları silmek istediğinizden emin misiniz?")) return;
    await deleteAll.mutateAsync();
    setSelected(new Set());
  };

  const sources = ["Tümü", ...Array.from(new Set(records.map((r) => r.source ?? "—").filter(Boolean)))];

  return (
    <CChrome
      title="alpha kataloğu"
      sub={`${records.length} kayıt`}
      statusExtra={
        <>
          <span>
            <span style={{ color: "var(--fg-3)" }}>filtered</span> {filtered.length}/{records.length}
          </span>
          <span>
            <span style={{ color: "var(--fg-3)" }}>selected</span> {selected.size}
          </span>
        </>
      }
      top={
        <>
          <Btn variant="ghost" onClick={exportCsv}>↓ Katalog CSV indir</Btn>
          <Btn variant="danger" onClick={handleDeleteAll}>Kataloğu temizle</Btn>
        </>
      }
      width="100%"
      height="100vh"
    >
      <Panel
        num="A"
        title="Alpha Kataloğu"
        sub={`${filtered.length} / ${records.length} kayıt gösteriliyor`}
        right={<Pill mono tone="ghost">sort: rank_ic ↓</Pill>}
        flex={1}
      >
        {/* Filter row */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr auto 1fr",
            gap: 14,
            marginBottom: 14,
            alignItems: "end",
          }}
        >
          <Select
            label="Overfit filtresi"
            value={overfitFilter}
            options={["Tümü", "low", "mid", "high"]}
            onChange={setOverfitFilter}
          />
          <Select
            label="Kaynak"
            value={sourceFilter}
            options={sources}
            hint="evolution · llm · mcts"
            onChange={setSourceFilter}
          />
          <Check
            label="Sadece backtest yapılmışlar"
            checked={onlyBacktest}
            onChange={setOnlyBacktest}
          />
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <input
              value={regex}
              onChange={(e) => setRegex(e.target.value)}
              placeholder="⌕  filter regex…"
              style={{
                flex: 1,
                padding: "5px 9px",
                border: "1px solid var(--line)",
                borderRadius: 2,
                background: "var(--bg-2)",
                fontFamily: "var(--mono)",
                fontSize: 11,
                color: "var(--fg-0)",
                outline: "none",
              }}
            />
          </div>
        </div>

        {/* Table */}
        {isLoading ? (
          <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--fg-3)", padding: 16 }}>
            Yükleniyor…
          </div>
        ) : (
          <div
            style={{
              border: "1px solid var(--line-soft)",
              borderRadius: 3,
              overflow: "auto",
              flex: 1,
            }}
          >
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "24px 32px 1fr 70px 70px 70px 160px 70px 70px 70px 80px",
                gap: 10,
                padding: "8px 12px",
                background: "var(--bg-2)",
                fontFamily: "var(--mono)",
                fontSize: 9.5,
                color: "var(--fg-3)",
                textTransform: "uppercase",
                letterSpacing: 0.5,
                borderBottom: "1px solid var(--line-soft)",
                position: "sticky",
                top: 0,
              }}
            >
              <span />
              <span>#</span>
              <span>formül</span>
              <span style={{ textAlign: "right" }}>ic</span>
              <span style={{ textAlign: "right" }}>rank ic</span>
              <span style={{ textAlign: "right" }}>adj ic</span>
              <span>wf</span>
              <span style={{ textAlign: "right" }}>sharpe</span>
              <span>overfit</span>
              <span>best</span>
              <span>kaynak</span>
            </div>

            {filtered.map((r: CatalogRecord, i: number) => {
              const isSelected = selected.has(r.formula);
              return (
                <div
                  key={r.formula}
                  style={{
                    display: "grid",
                    gridTemplateColumns: "24px 32px 1fr 70px 70px 70px 160px 70px 70px 70px 80px",
                    gap: 10,
                    padding: "8px 12px",
                    alignItems: "center",
                    background: isSelected ? "var(--bg-1)" : "transparent",
                    borderBottom: "1px dotted var(--line-soft)",
                    fontSize: 11.5,
                    cursor: "pointer",
                  }}
                  onClick={() => navigate(`/workbench?id=${encodeURIComponent(r.formula)}`)}
                >
                  <span
                    onClick={(e) => { e.stopPropagation(); toggleSelect(r.formula); }}
                    style={{
                      width: 11,
                      height: 11,
                      border: `1px solid ${isSelected ? "var(--accent)" : "var(--line)"}`,
                      background: isSelected ? "var(--accent)" : "transparent",
                      borderRadius: 2,
                      cursor: "pointer",
                      display: "inline-block",
                    }}
                  />
                  <span style={{ fontFamily: "var(--mono)", fontSize: 10.5, color: "var(--fg-2)" }}>{i + 1}</span>
                  <code
                    style={{
                      fontFamily: "var(--mono)",
                      fontSize: 11,
                      color: "var(--fg-0)",
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                    }}
                  >
                    {r.formula}
                  </code>
                  <span style={{ fontFamily: "var(--mono)", textAlign: "right", color: "var(--fg-1)" }}>
                    {r.ic != null ? r.ic.toFixed(4) : "—"}
                  </span>
                  <span
                    style={{
                      fontFamily: "var(--mono)",
                      textAlign: "right",
                      color: (r.rank_ic ?? 0) > 0.02 ? "var(--pos)" : "var(--fg-1)",
                    }}
                  >
                    {r.rank_ic != null ? r.rank_ic.toFixed(4) : "—"}
                  </span>
                  <span style={{ fontFamily: "var(--mono)", textAlign: "right", color: "var(--fg-2)" }}>
                    {r.adj_ic != null ? r.adj_ic.toFixed(4) : "—"}
                  </span>
                  <span style={{ fontFamily: "var(--mono)", fontSize: 10.5, color: "var(--fg-1)" }}>
                    {r.wf_score != null ? `fit=${r.wf_score.toFixed(4)}` : "—"}
                  </span>
                  <span
                    style={{
                      fontFamily: "var(--mono)",
                      textAlign: "right",
                      color: r.sharpe != null ? "var(--pos)" : "var(--fg-3)",
                    }}
                  >
                    {r.sharpe != null ? r.sharpe.toFixed(2) : "—"}
                  </span>
                  <span
                    style={{
                      fontFamily: "var(--mono)",
                      fontSize: 10,
                      color:
                        r.overfit_score != null && r.overfit_score < 0.3
                          ? "var(--pos)"
                          : r.overfit_score != null && r.overfit_score < 0.6
                          ? "var(--warn)"
                          : "var(--fg-3)",
                    }}
                  >
                    {r.overfit_score != null
                      ? r.overfit_score < 0.3
                        ? "low"
                        : r.overfit_score < 0.6
                        ? "mid"
                        : "high"
                      : "—"}
                  </span>
                  <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--fg-3)" }}>
                    {r.best_window ?? "—"}
                  </span>
                  <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--fg-2)" }}>
                    {r.source ?? "—"}
                  </span>
                </div>
              );
            })}
          </div>
        )}

        {/* Bulk actions */}
        {selected.size > 0 && (
          <div
            style={{
              marginTop: 14,
              padding: 12,
              background: "var(--bg-1)",
              border: "1px solid var(--line-soft)",
              borderRadius: 3,
              display: "grid",
              gridTemplateColumns: "1fr auto",
              gap: 14,
              alignItems: "end",
            }}
          >
            <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--fg-1)" }}>
              {selected.size} kayıt seçili
            </span>
            <Btn variant="danger" onClick={handleDeleteSelected}>
              ✕ Seçilenleri sil
            </Btn>
          </div>
        )}

        <div
          style={{
            marginTop: 12,
            padding: 12,
            background: "var(--bg-1)",
            border: "1px solid var(--warn)",
            borderRadius: 3,
            display: "flex",
            alignItems: "center",
            gap: 14,
          }}
        >
          <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--warn)" }}>⚠</span>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 11.5, color: "var(--fg-1)" }}>Katalogdaki TÜM kayıtları sil</div>
            <div style={{ fontFamily: "var(--mono)", fontSize: 9.5, color: "var(--fg-3)" }}>
              sadece JSON — buffer ve ağırlıklar korunur
            </div>
          </div>
          <Btn variant="danger" onClick={handleDeleteAll}>Kataloğu temizle</Btn>
        </div>

        <div style={{ marginTop: 12 }}>
          <Btn variant="ghost" onClick={() => refetch()}>↺ Yenile</Btn>
        </div>
      </Panel>
    </CChrome>
  );
}
