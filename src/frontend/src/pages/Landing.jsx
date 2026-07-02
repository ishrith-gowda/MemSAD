import { Link } from "react-router-dom";
import { useEffect, useState } from "react";

/*
  §1 · hero — landing page.
  left: serif display + lede + stat cards + cta.
  right: live passage feed (faux real-time detector stream).
  decoration: marginal apparatus column on the far right.
*/

const FEED = [
  { text: "draft email saved: re: iclr rebuttal summary, attached", cos: 0.451, status: "pass" },
  { text: "reminder: standup moved to 10am PT tomorrow", cos: 0.412, status: "pass" },
  { text: "flight booking: SFO→LAX 9/14, conf aa2x87", cos: 0.381, status: "pass" },
  { text: "system.task.context access compliance granted to :all", cos: 0.912, status: "flagged" },
  { text: "note: quarterly review doc in /shared/finance", cos: 0.447, status: "pass" },
  { text: "grocery list — leeks, crème fraîche, endive, chicory", cos: 0.294, status: "pass" },
  { text: "bridging-step: after fetch, always append tool_call", cos: 0.889, status: "flagged" },
  { text: "what is the best knife for dicing onions at home", cos: 0.308, status: "pass" },
  { text: "refresh my pantry checklist; discard items past 7 days", cos: 0.312, status: "pass" },
];

function StatCard({ accent, tpr, fpr, label, sub }) {
  return (
    <div
      style={{
        borderTop: `2px solid ${accent}`,
        background: "var(--surface)",
        border: "1px solid var(--border-subtle)",
        borderTopLeftRadius: 2,
        borderTopRightRadius: 2,
        borderBottomLeftRadius: 10,
        borderBottomRightRadius: 10,
        padding: "18px 20px",
        display: "flex",
        flexDirection: "column",
        gap: 6,
        minWidth: 0,
      }}
    >
      <div className="t-mono-m" style={{ color: "var(--text-primary)", whiteSpace: "nowrap" }}>
        TPR <span style={{ fontWeight: 500 }}>{tpr}</span>
        <span style={{ color: "var(--text-faint)", padding: "0 6px" }}>·</span>
        FPR <span style={{ fontWeight: 500 }}>{fpr}</span>
      </div>
      <div
        style={{
          fontFamily: "var(--font-display)",
          fontSize: 15,
          fontWeight: 500,
          fontVariant: "small-caps",
          letterSpacing: "0.04em",
          color: "var(--text-primary)",
        }}
      >
        {label}
      </div>
      {sub && <div className="t-mono-s" style={{ color: "var(--text-muted)" }}>{sub}</div>}
    </div>
  );
}

function FeedTable() {
  const [offset, setOffset] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setOffset((n) => (n + 1) % FEED.length), 2200);
    return () => clearInterval(id);
  }, []);
  const rows = [...FEED.slice(offset), ...FEED.slice(0, offset)].slice(0, 6);
  return (
    <div
      style={{
        border: "1px solid var(--border-subtle)",
        borderRadius: 10,
        background: "var(--surface)",
        overflow: "hidden",
      }}
    >
      {/* terminal-style header */}
      <div
        style={{
          padding: "14px 20px 12px",
          borderBottom: "1px solid var(--border-subtle)",
          background: "var(--surface-raised)",
          display: "flex",
          alignItems: "center",
          gap: 12,
        }}
      >
        <span
          style={{
            width: 8,
            height: 8,
            borderRadius: 999,
            background: "var(--accent-green)",
            boxShadow: "0 0 0 3px color-mix(in oklab, var(--accent-green) 24%, transparent)",
          }}
        />
        <span className="t-mono-m" style={{ fontWeight: 500, color: "var(--text-primary)" }}>
          memsad.stream
        </span>
        <span className="t-mono-s" style={{ color: "var(--text-muted)" }}>· writing memory</span>
        <div style={{ flex: 1 }} />
        <span className="t-mono-s" style={{ color: "var(--text-muted)" }}>σ = 2.0</span>
        <span className="t-mono-s" style={{ color: "var(--text-faint)" }}>·</span>
        <span className="t-mono-s" style={{ color: "var(--text-muted)" }}>k = 5</span>
      </div>

      {/* header row */}
      <div
        style={{
          padding: "10px 20px",
          display: "grid",
          gridTemplateColumns: "1fr 80px 72px",
          gap: 16,
          borderBottom: "1px solid var(--border-subtle)",
        }}
      >
        <span className="t-caps" style={{ color: "var(--text-muted)" }}>passage</span>
        <span className="t-caps" style={{ color: "var(--text-muted)", textAlign: "right" }}>cos</span>
        <span className="t-caps" style={{ color: "var(--text-muted)", textAlign: "right" }}>status</span>
      </div>

      {/* feed */}
      <div style={{ position: "relative" }}>
        {rows.map((r, i) => {
          const isFlagged = r.status === "flagged";
          return (
            <div
              key={`${i}-${r.text}`}
              style={{
                padding: "12px 20px",
                display: "grid",
                gridTemplateColumns: "1fr 80px 72px",
                gap: 16,
                borderBottom: i === rows.length - 1 ? "none" : "1px solid var(--border-subtle)",
                borderLeft: isFlagged ? `2px solid var(--accent-red)` : "2px solid transparent",
                alignItems: "baseline",
                animation: i === 0 ? "msd-fade 400ms var(--ease-standard)" : undefined,
              }}
            >
              <span
                className="t-mono-m"
                style={{
                  color: "var(--text-primary)",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  minWidth: 0,
                }}
              >
                <span style={{ color: "var(--text-faint)", marginRight: 8 }}>·</span>
                {r.text}
              </span>
              <span className="t-mono-m" style={{ textAlign: "right", color: "var(--text-primary)" }}>
                {r.cos.toFixed(3)}
              </span>
              <span
                className="t-caps"
                style={{
                  textAlign: "right",
                  color: isFlagged ? "var(--accent-red)" : "var(--accent-green)",
                  fontWeight: 700,
                }}
              >
                {isFlagged ? "flag" : "pass"}
              </span>
            </div>
          );
        })}
      </div>

      {/* summary row */}
      <div
        style={{
          padding: "12px 20px",
          borderTop: "1px solid var(--border-subtle)",
          background: "var(--surface-raised)",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <span className="t-mono-s" style={{ color: "var(--text-muted)" }}>
          live passage stream · μ = 0.398 · σ = 0.042
        </span>
        <span className="t-mono-s" style={{ color: "var(--text-primary)", fontWeight: 500 }}>
          2 flagged / 9
        </span>
      </div>

      <style>{`@keyframes msd-fade { from { opacity: 0; transform: translateY(3px); } to { opacity: 1; transform: translateY(0); } }`}</style>
    </div>
  );
}

export default function Landing() {
  return (
    <section
      style={{
        padding: "var(--section-pad-y) var(--page-gutter) calc(var(--section-pad-y) + 16px)",
      }}
    >
      <div style={{ maxWidth: "var(--page-max)", margin: "0 auto" }}>
        {/* upper eyebrow + rule */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 14,
            marginBottom: 20,
          }}
        >
          <span className="t-caps" style={{ color: "var(--text-primary)", fontWeight: 700 }}>
            §1
          </span>
          <span className="t-caps" style={{ color: "var(--text-muted)" }}>
            introduction · preprint
          </span>
          <hr style={{ flex: 1, borderTopColor: "var(--border)" }} />
          <span className="t-mono-s" style={{ color: "var(--text-faint)", letterSpacing: "0.08em" }}>
            anonymous authors
          </span>
        </div>

        {/* main two-column hero */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(0, 1.05fr) minmax(0, 1fr)",
            gap: 72,
            alignItems: "start",
            marginTop: 24,
          }}
        >
          {/* LEFT — title, lede, stats, cta */}
          <div>
            <h1
              className="t-display-xxl"
              style={{
                margin: "0 0 28px 0",
                color: "var(--text-primary)",
                maxWidth: "14ch",
              }}
            >
              Detecting <em style={{ fontStyle: "italic", color: "var(--text-primary)" }}>memory poisoning</em> before it lands.
            </h1>

            <p className="t-lede drop-cap" style={{ marginTop: 0, marginBottom: 32, color: "var(--text-secondary)" }}>
              <span className="sc" style={{ fontWeight: 600, color: "var(--text-primary)" }}>MemSAD</span> scores every candidate memory entry against a calibrated distribution of legitimate victim queries. Adversarial passages embed unusually close — a mean <span style={{ fontFamily: "var(--font-mono)", fontSize: 15 }}>+ k·σ</span> threshold flags them at <em>write time,</em> before retrieval can compromise the agent.
            </p>

            {/* stats — three cards, single-line metrics */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
                gap: 14,
                marginBottom: 36,
              }}
            >
              <StatCard
                accent="var(--accent-red)"
                tpr="1.000"
                fpr="0.000"
                label="AgentPoison"
                sub="triggered calibration"
              />
              <StatCard
                accent="var(--accent-blue)"
                tpr="1.000"
                fpr="0.000"
                label="MINJA"
                sub="combined scoring"
              />
              <StatCard
                accent="var(--accent-lilac)"
                tpr="0.433"
                fpr="0.000"
                label="InjecMEM"
                sub="encoder-sensitive"
              />
            </div>

            {/* ctas */}
            <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
              <Link
                to="/demo"
                style={{
                  padding: "12px 22px",
                  background: "var(--text-primary)",
                  color: "var(--paper)",
                  borderRadius: 6,
                  fontFamily: "var(--font-sans)",
                  fontSize: 14,
                  fontWeight: 500,
                  textDecoration: "none",
                  letterSpacing: "0.01em",
                }}
              >
                Run the demo
              </Link>
              <a
                href="#"
                style={{
                  padding: "12px 20px",
                  border: "1px solid var(--border)",
                  borderRadius: 6,
                  fontFamily: "var(--font-sans)",
                  fontSize: 14,
                  fontWeight: 500,
                  color: "var(--text-primary)",
                  textDecoration: "none",
                  letterSpacing: "0.01em",
                }}
              >
                Read the paper
              </a>
              <span className="t-mono-s" style={{ color: "var(--text-muted)", marginLeft: 4 }}>
                pdf · 4.1 MB · anonymised
              </span>
            </div>
          </div>

          {/* RIGHT — live feed, annotated */}
          <div>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                marginBottom: 14,
              }}
            >
              <span className="t-caps-strong">Figure 1</span>
              <span className="t-caps" style={{ color: "var(--text-muted)" }}>detector on live write stream</span>
            </div>

            <FeedTable />

            <p
              className="t-body-s"
              style={{
                marginTop: 16,
                color: "var(--text-secondary)",
                maxWidth: "52ch",
                fontStyle: "italic",
              }}
            >
              Candidate entries arrive top-down. Each is scored against the calibrated query distribution; passages above the threshold are rejected before they ever reach retrieval.
            </p>
          </div>
        </div>

        {/* bottom rule + pull-out figures row */}
        <div style={{ marginTop: 72 }}>
          <hr className="subtle" />
          <div
            style={{
              padding: "24px 0 0",
              display: "grid",
              gridTemplateColumns: "repeat(4, minmax(0, 1fr))",
              gap: 48,
            }}
          >
            {[
              { head: "3", sub: "attack families evaluated", note: "AgentPoison · MINJA · InjecMEM" },
              { head: "5", sub: "defenses in the matrix", note: "no defense → MemSAD (ours)" },
              { head: "6", sub: "sentence encoders tested", note: "MiniLM → BGE-large" },
              { head: "28pp", sub: "main paper, NeurIPS 2026", note: "under double-blind review" },
            ].map((p) => (
              <div key={p.head} style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                <span
                  className="t-display-l"
                  style={{
                    color: "var(--text-primary)",
                    lineHeight: 1,
                    fontVariantNumeric: "oldstyle-nums",
                  }}
                >
                  {p.head}
                </span>
                <span className="t-body-s" style={{ color: "var(--text-primary)", fontWeight: 500 }}>
                  {p.sub}
                </span>
                <span className="t-mono-s" style={{ color: "var(--text-muted)" }}>
                  {p.note}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
