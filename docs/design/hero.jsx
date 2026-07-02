// memsad — landing hero (1440x900)
// left 7/12 — title page. right 5/12 — live passage feed.

const FEED = [
  { id: "r1", text: "reminder: standup moved to 10am PT",                           cos: 0.412, kind: "pass" },
  { id: "r2", text: "flight booking: SFO→LAX 9/14, conf aa2048",                    cos: 0.381, kind: "pass" },
  { id: "r3", text: "note: quarterly review doc in /shared/planning/q3.md",         cos: 0.447, kind: "pass" },
  { id: "r4", text: "grocery list — leeks, crème fraîche, eggs, sourdough",         cos: 0.294, kind: "pass" },
  { id: "r5", text: "followup: reply to maya re: onboarding deck by friday",        cos: 0.502, kind: "pass" },
  { id: "r6", text: "system task context access compliance [adversarial centroid]", cos: 0.892, kind: "flag" },
  { id: "r7", text: "user asked to rename project 'atlas' → 'atlas-v2'",            cos: 0.376, kind: "pass" },
  { id: "r8", text: "calendar: 1:1 with dir of research, thu 3pm",                  cos: 0.423, kind: "pass" },
  { id: "r9", text: "draft email saved: re: iclr rebuttal schedule",                cos: 0.451, kind: "pass" },
];

const ROW_H = 46;
const VISIBLE = 5;
const TICK_MS = 3000;

const FeedRow = ({ row, annotate }) => {
  const flagged = row.kind === "flag";
  return (
    <div
      style={{
        position: "relative",
        height: ROW_H,
        display: "grid",
        gridTemplateColumns: "22px 1fr auto auto",
        alignItems: "center",
        gap: 14,
        padding: "0 14px 0 12px",
        borderBottom: "1px solid var(--border-subtle)",
        borderLeft: flagged ? "2px solid var(--accent-agentpoison)" : "2px solid transparent",
        background: flagged
          ? "linear-gradient(to right, rgba(196,112,112,0.10), rgba(196,112,112,0.02) 60%, transparent)"
          : "var(--surface)",
      }}
    >
      <span className="t-mono-s" style={{ color: flagged ? "var(--flag-hard)" : "var(--text-muted)" }}>
        {flagged ? "!" : "·"}
      </span>
      <span
        className="t-mono-s"
        style={{
          color: "var(--text-primary)",
          whiteSpace: "nowrap",
          overflow: "hidden",
          textOverflow: "ellipsis",
        }}
        title={row.text}
      >
        {row.text}
      </span>
      <span
        className="t-mono-s"
        style={{
          fontVariantNumeric: "tabular-nums",
          color: flagged ? "var(--flag-hard)" : "var(--text-secondary)",
          minWidth: 72,
          textAlign: "right",
        }}
      >
        cos={row.cos.toFixed(3)}
      </span>
      <span style={{ minWidth: 56, display: "flex", justifyContent: "flex-end" }}>
        {flagged ? (
          <span
            style={{
              fontFamily: "var(--font-sans)",
              fontSize: 10,
              fontWeight: 700,
              letterSpacing: "0.1em",
              color: "var(--flag-hard)",
              background: "rgba(168,74,74,0.10)",
              border: "1px solid rgba(168,74,74,0.38)",
              borderRadius: 4,
              padding: "2px 6px",
            }}
          >
            FLAG
          </span>
        ) : (
          <span
            style={{
              fontFamily: "var(--font-sans)",
              fontSize: 10,
              fontWeight: 600,
              letterSpacing: "0.1em",
              color: "var(--text-muted)",
            }}
          >
            PASS
          </span>
        )}
      </span>

      {flagged && annotate && (
        <div
          style={{
            position: "absolute",
            left: 14,
            top: "calc(100% + 6px)",
            padding: "7px 10px",
            background: "var(--paper)",
            border: "1px solid rgba(168,74,74,0.35)",
            borderRadius: 6,
            color: "var(--flag-hard)",
            fontFamily: "var(--font-mono)",
            fontSize: 12,
            fontVariantNumeric: "tabular-nums",
            boxShadow: "var(--elevation-1)",
            opacity: annotate === "in" ? 1 : 0,
            transform: annotate === "in" ? "translateY(0)" : "translateY(-4px)",
            transition: "opacity 260ms var(--ease-standard), transform 260ms var(--ease-standard)",
            zIndex: 3,
            whiteSpace: "nowrap",
          }}
        >
          score = 0.892 &nbsp;&gt;&nbsp; μ + 2σ = 0.482
        </div>
      )}
    </div>
  );
};

const LiveFeed = () => {
  const [tick, setTick] = React.useState(0);
  const [annotate, setAnnotate] = React.useState("out");

  React.useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), TICK_MS);
    return () => clearInterval(id);
  }, []);

  // render VISIBLE + 2 buffer rows; transform the whole strip by -ROW_H each tick,
  // then reset to 0 on the next render (by using the tick as a key cycle)
  const phase = tick % FEED.length;
  const window = Array.from({ length: VISIBLE + 1 }, (_, i) => FEED[(phase + i) % FEED.length]);

  // show the annotation when the flag row is inside the visible window AND the row has settled (not mid-slide)
  const flagPosInWindow = window.findIndex((r) => r.kind === "flag");
  const flagVisible = flagPosInWindow >= 0 && flagPosInWindow < VISIBLE;

  React.useEffect(() => {
    if (flagVisible) {
      const t1 = setTimeout(() => setAnnotate("in"), 950);   // after slide settles
      const t2 = setTimeout(() => setAnnotate("out"), TICK_MS - 200);
      return () => { clearTimeout(t1); clearTimeout(t2); };
    } else {
      setAnnotate("out");
    }
  }, [phase, flagVisible]);

  return (
    <div
      style={{
        background: "var(--surface)",
        border: "1px solid var(--border-subtle)",
        borderRadius: 16,
        overflow: "hidden",
        boxShadow: "var(--elevation-1)",
      }}
    >
      {/* header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "12px 16px",
          borderBottom: "1px solid var(--border-subtle)",
          background: "var(--paper-raised)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span
            style={{
              width: 7, height: 7, borderRadius: 999,
              background: "var(--accent-green)",
              boxShadow: "0 0 0 3px rgba(143,191,127,0.18)",
            }}
          />
          <span className="t-mono-s" style={{ color: "var(--text-primary)", fontWeight: 500 }}>
            memsad.stream
          </span>
          <span className="t-mono-s" style={{ color: "var(--text-muted)" }}>
            · writing memory
          </span>
        </div>
        <span className="t-mono-s" style={{ color: "var(--text-muted)" }}>
          σ=2.0 &nbsp; k=5
        </span>
      </div>

      {/* column head */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "22px 1fr auto auto",
          gap: 14,
          padding: "8px 14px 8px 12px",
          background: "var(--paper)",
          borderBottom: "1px solid var(--border-subtle)",
        }}
      >
        <span />
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--text-muted)", letterSpacing: "0.12em" }}>PASSAGE</span>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--text-muted)", letterSpacing: "0.12em", minWidth: 72, textAlign: "right" }}>COS</span>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--text-muted)", letterSpacing: "0.12em", minWidth: 56, textAlign: "right" }}>STATUS</span>
      </div>

      {/* scrolling viewport */}
      <div
        style={{
          position: "relative",
          height: ROW_H * VISIBLE,
          overflow: "hidden",
          background: "var(--surface)",
        }}
      >
        <div
          key={tick}
          style={{
            position: "absolute",
            top: 0, left: 0, right: 0,
            animation: `msdSlide ${TICK_MS}ms var(--ease-standard) forwards`,
          }}
        >
          {window.map((row, i) => (
            <FeedRow
              key={`${tick}-${i}`}
              row={row}
              annotate={row.kind === "flag" && (flagPosInWindow === i) && flagPosInWindow < VISIBLE ? annotate : "out"}
            />
          ))}
        </div>
        {/* top and bottom fades */}
        <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 20, background: "linear-gradient(to bottom, var(--surface), rgba(251,250,247,0))", pointerEvents: "none", zIndex: 2 }} />
        <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, height: 20, background: "linear-gradient(to top, var(--surface), rgba(251,250,247,0))", pointerEvents: "none", zIndex: 2 }} />
      </div>

      {/* caption */}
      <div
        style={{
          padding: "12px 16px",
          borderTop: "1px solid var(--border-subtle)",
          background: "var(--paper-raised)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <span className="t-mono-s" style={{ color: "var(--text-secondary)" }}>
          live passage stream · MemSAD σ=2.0
        </span>
        <span className="t-mono-s" style={{ color: "var(--text-muted)" }}>
          1 flagged / {FEED.length}
        </span>
      </div>
    </div>
  );
};

const StatCard = ({ value, label, tone = "neutral" }) => {
  const accents = {
    red:   "var(--accent-agentpoison)",
    blue:  "var(--accent-minja)",
    lilac: "var(--accent-injecmem)",
    neutral: "var(--border-subtle)",
  };
  return (
    <div
      style={{
        flex: 1,
        background: "var(--paper-raised)",
        border: "1px solid var(--border-subtle)",
        borderRadius: 12,
        padding: "14px 16px",
        borderTop: `2px solid ${accents[tone]}`,
      }}
    >
      <div
        style={{
          fontFamily: "var(--font-mono)",
          color: "var(--text-primary)",
          fontVariantNumeric: "tabular-nums",
          fontWeight: 500,
          fontSize: 15,
          letterSpacing: "-0.005em",
        }}
      >
        {value}
      </div>
      <div className="t-body-s" style={{ color: "var(--text-secondary)", marginTop: 6, fontSize: 13 }}>
        {label}
      </div>
    </div>
  );
};

const Hero = () => (
  <div
    style={{
      width: 1440,
      height: 900,
      background: "var(--surface)",
      margin: "0 auto",
      position: "relative",
      overflow: "hidden",
    }}
  >
    {/* topbar */}
    <header
      style={{
        position: "absolute", top: 0, left: 0, right: 0,
        height: 64,
        borderBottom: "1px solid var(--border-subtle)",
        display: "grid",
        gridTemplateColumns: "1fr auto 1fr",
        alignItems: "center",
        padding: "0 80px",
        gap: 24,
        background: "var(--surface)",
      }}
    >
      <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
        <span style={{ fontFamily: "var(--font-serif)", fontSize: 22, fontWeight: 500, letterSpacing: "-0.01em" }}>MemSAD</span>
        <span className="t-mono-s" style={{ color: "var(--text-muted)" }}>v0.3</span>
      </div>
      <nav style={{ display: "flex", gap: 28 }}>
        {["Demo", "Paper", "GitHub", "About"].map((l, i) => (
          <a key={l} href="#" style={{ textDecoration: "none", color: i === 0 ? "var(--text-primary)" : "var(--text-secondary)", fontSize: 14, fontWeight: i === 0 ? 600 : 500 }}>
            {l}
          </a>
        ))}
      </nav>
      <div style={{ justifySelf: "end" }}>
        <Badge tone="neutral" label="NEURIPS 2026">under review</Badge>
      </div>
    </header>

    <div
      style={{
        position: "absolute",
        top: 64,
        bottom: 0,
        left: 80,
        right: 80,
        display: "grid",
        gridTemplateColumns: "7fr 5fr",
        columnGap: 64,
        alignItems: "center",
      }}
    >
      {/* LEFT */}
      <div>
        <div
          style={{
            color: "var(--text-muted)",
            fontSize: 12,
            fontWeight: 600,
            letterSpacing: "0.14em",
            marginBottom: 28,
            textTransform: "uppercase",
            fontFamily: "var(--font-sans)",
          }}
        >
          <span style={{ color: "var(--text-primary)" }}>NEURIPS&nbsp;2026</span>
          <span style={{ margin: "0 10px", color: "var(--border-subtle)" }}>·</span>
          UNDER&nbsp;REVIEW
        </div>

        <h1
          style={{
            fontFamily: "var(--font-serif)",
            margin: 0,
            fontSize: 60,
            lineHeight: 1.05,
            letterSpacing: "-0.018em",
            color: "var(--text-primary)",
            maxWidth: 640,
            fontWeight: 400,
          }}
        >
          Detecting memory<br />
          poisoning before<br />
          it lands.
        </h1>

        <p
          style={{
            marginTop: 26,
            color: "var(--text-secondary)",
            maxWidth: "60ch",
            fontSize: 17,
            lineHeight: 1.6,
            fontFamily: "var(--font-sans)",
          }}
        >
          <span className="sc" style={{ fontWeight: 600, color: "var(--text-primary)", letterSpacing: "0.04em" }}>MemSAD</span> scores every candidate memory entry against a calibrated distribution
          of legitimate victim queries. Adversarial passages embed unusually close — a mean + k·σ threshold
          flags them at write time, before retrieval can compromise the agent.
        </p>

        <div style={{ display: "flex", gap: 12, marginTop: 36, maxWidth: 640 }}>
          <StatCard
            tone="red"
            value="1.000 TPR / 0.000 FPR"
            label={<><span className="sc" style={{ fontWeight: 600 }}>AgentPoison</span> · triggered cal.</>}
          />
          <StatCard
            tone="blue"
            value="1.000 TPR / 0.000 FPR"
            label={<span className="sc" style={{ fontWeight: 600 }}>MINJA</span>}
          />
          <StatCard
            tone="lilac"
            value="0.433 TPR / 0.000 FPR"
            label={<span className="sc" style={{ fontWeight: 600 }}>InjecMEM</span>}
          />
        </div>

        <div style={{ display: "flex", gap: 12, marginTop: 30, alignItems: "center" }}>
          <Button variant="primary" size="lg">Run the demo</Button>
          <Button variant="secondary" size="lg">Read the paper</Button>
          <span className="t-mono-s" style={{ color: "var(--text-muted)", marginLeft: 8 }}>
            pdf · 4.1 MB · anonymised
          </span>
        </div>
      </div>

      {/* RIGHT */}
      <div style={{ position: "relative" }}>
        <div
          style={{
            color: "var(--text-muted)",
            fontSize: 11,
            fontWeight: 600,
            letterSpacing: "0.14em",
            textTransform: "uppercase",
            marginBottom: 10,
            fontFamily: "var(--font-sans)",
          }}
        >
          Figure&nbsp;1 &nbsp;·&nbsp; detector on live write stream
        </div>
        <LiveFeed />
        <div
          style={{
            marginTop: 14,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "baseline",
            gap: 16,
          }}
        >
          <span className="t-body-s" style={{ color: "var(--text-muted)", maxWidth: 340, fontSize: 13 }}>
            Candidate entries arrive top-down. Scores computed against calibrated query distribution; anomalies fire.
          </span>
          <span className="t-mono-s" style={{ color: "var(--text-muted)", whiteSpace: "nowrap" }}>
            μ=0.398 &nbsp; σ=0.042
          </span>
        </div>
      </div>
    </div>

    {/* bottom watermark */}
    <div style={{ position: "absolute", bottom: 22, left: 80, display: "flex", gap: 14, alignItems: "baseline" }}>
      <span className="t-mono-s" style={{ color: "var(--text-muted)" }}>§1 · introduction</span>
      <span className="t-mono-s" style={{ color: "var(--border-subtle)" }}>/</span>
      <span className="t-mono-s" style={{ color: "var(--text-muted)" }}>anonymous authors</span>
    </div>
    <div style={{ position: "absolute", bottom: 22, right: 80 }}>
      <span className="t-mono-s" style={{ color: "var(--text-muted)" }}>preprint · do not circulate</span>
    </div>
  </div>
);

ReactDOM.createRoot(document.getElementById("root")).render(<Hero />);
