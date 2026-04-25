import { useNavigate } from "react-router-dom";
import { CChrome } from "../components/chrome/CChrome";
import { Panel } from "../components/chrome/Panel";
import { Btn } from "../components/atoms/Btn";
import { useCatalog } from "../hooks/useCatalog";

export default function BestAlphas() {
  const { data: records = [], exportCsv } = useCatalog();
  const navigate = useNavigate();

  const top = [...records]
    .sort((a, b) => (b.rank_ic ?? 0) - (a.rank_ic ?? 0))
    .slice(0, 10);

  return (
    <CChrome
      title="en iyi evrimleşmiş alphalar"
      sub="top 10 · sort: rankic ↓"
      top={
        <>
          <Btn variant="ghost" onClick={() => navigate("/catalog")}>◳ Kataloğa dön</Btn>
          <Btn variant="ghost" onClick={exportCsv}>↓ İndir</Btn>
        </>
      }
      width="100%"
      height="100vh"
    >
      <Panel num="A" title="En İyi Evrimleşmiş Alphalar" sub={`${records.length} havuzdan filtrelendi`} flex={1} pad={false}>
        {/* Header */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "40px 32px 1fr 100px 100px 100px 100px 100px",
            gap: 10,
            padding: "10px 18px",
            background: "var(--bg-2)",
            fontFamily: "var(--mono)",
            fontSize: 9.5,
            color: "var(--fg-3)",
            textTransform: "uppercase",
            letterSpacing: 0.5,
            borderBottom: "1px solid var(--line-soft)",
          }}
        >
          <span>#</span>
          <span></span>
          <span>formül</span>
          <span style={{ textAlign: "right" }}>ic</span>
          <span style={{ textAlign: "right" }}>rank ic</span>
          <span style={{ textAlign: "right" }}>adj ic</span>
          <span style={{ textAlign: "right" }}>sharpe</span>
          <span>kaynak</span>
        </div>

        {/* Rows */}
        <div style={{ flex: 1, minHeight: 0, overflow: "auto" }}>
          {top.map((r, i) => (
            <div
              key={r.formula}
              onClick={() => navigate(`/workbench?id=${encodeURIComponent(r.formula)}`)}
              style={{
                display: "grid",
                gridTemplateColumns: "40px 32px 1fr 100px 100px 100px 100px 100px",
                gap: 10,
                padding: "10px 18px",
                alignItems: "center",
                borderBottom: "1px dotted var(--line-soft)",
                background: i === 0 ? "var(--bg-1)" : "transparent",
                cursor: "pointer",
              }}
            >
              <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: i === 0 ? "var(--accent)" : "var(--fg-3)" }}>
                {i + 1}
              </span>
              <span
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  background: i < 3 ? "var(--accent)" : "var(--line)",
                  display: "inline-block",
                  marginTop: 2,
                }}
              />
              <code
                style={{
                  fontFamily: "var(--mono)",
                  fontSize: 11.5,
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
              <span
                style={{
                  fontFamily: "var(--mono)",
                  textAlign: "right",
                  color: r.sharpe != null ? "var(--pos)" : "var(--fg-3)",
                }}
              >
                {r.sharpe != null ? r.sharpe.toFixed(2) : "—"}
              </span>
              <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--fg-2)" }}>
                {r.source ?? "—"}
              </span>
            </div>
          ))}
        </div>
      </Panel>
    </CChrome>
  );
}
