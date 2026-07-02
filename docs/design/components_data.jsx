// memsad — data components
// scorebar, passagerow, slider, matrixcell

const fmt3 = (n) => (n == null ? "—" : n.toFixed(3));

const ScoreBar = ({ value = 0, threshold, poisoned = false, animated = true, label, width = 280 }) => {
  const [v, setV] = React.useState(animated ? 0 : value);
  const pulseRef = React.useRef(false);

  React.useEffect(() => {
    if (!animated) { setV(value); return; }
    // animate 0 -> value over 400ms
    const t0 = performance.now();
    const from = 0;
    const to = value;
    let raf;
    const step = (now) => {
      const p = Math.min(1, (now - t0) / 400);
      // cubic-bezier approx via easeOutCubic
      const e = 1 - Math.pow(1 - p, 3);
      setV(from + (to - from) * e);
      if (p < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [value, animated]);

  const pct = Math.max(0, Math.min(1, v));
  const over = threshold != null && value >= threshold;
  const fill = poisoned && over
    ? "var(--flag-hard)"
    : poisoned
      ? "var(--accent-agentpoison)"
      : "var(--accent-minja)";

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, width }}>
      {label && (
        <span className="t-body-s" style={{ minWidth: 80, color: "var(--text-muted)" }}>
          {label}
        </span>
      )}
      <div
        style={{
          position: "relative",
          flex: 1,
          height: 8,
          background: "var(--paper-raised)",
          borderRadius: 999,
          overflow: "visible",
        }}
      >
        <div
          style={{
            position: "absolute",
            inset: 0,
            width: `${pct * 100}%`,
            background: fill,
            borderRadius: 999,
            transition: `background var(--duration-base) var(--ease-standard)`,
          }}
        />
        {threshold != null && (
          <div
            title={`threshold ${fmt3(threshold)}`}
            style={{
              position: "absolute",
              top: -3,
              bottom: -3,
              left: `${Math.max(0, Math.min(1, threshold)) * 100}%`,
              width: 1,
              background: "var(--ink-700)",
              opacity: 0.6,
            }}
          />
        )}
      </div>
      <span
        className="t-mono-s"
        style={{
          minWidth: 52,
          textAlign: "right",
          color: over ? "var(--flag-hard)" : "var(--text-primary)",
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {fmt3(value)}
      </span>
    </div>
  );
};

const PassageRow = ({ rank, text, cosine, poisoned = false, flag = null, threshold = 0.72 }) => {
  // flag: null | "soft" | "hard"
  const flagTone = flag === "hard" ? "red" : flag === "soft" ? "amber" : null;
  const flagLabel = flag === "hard" ? "ANOMALY" : flag === "soft" ? "NEAR-THRESH" : null;

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "28px 1fr auto auto",
        alignItems: "center",
        gap: 16,
        padding: "12px 16px 12px 14px",
        borderBottom: "1px solid var(--border-subtle)",
        borderLeft: poisoned ? "2px solid var(--accent-agentpoison)" : "2px solid transparent",
        background: "var(--surface)",
      }}
    >
      <span className="t-mono-s" style={{ color: "var(--text-muted)" }}>
        #{rank}
      </span>
      <span
        className="t-mono-m"
        style={{
          color: "var(--text-primary)",
          whiteSpace: "nowrap",
          overflow: "hidden",
          textOverflow: "ellipsis",
        }}
        title={text}
      >
        {text}
      </span>
      <span
        className="t-mono-s"
        style={{
          color: cosine >= threshold ? "var(--text-primary)" : "var(--text-secondary)",
          fontVariantNumeric: "tabular-nums",
          minWidth: 56,
          textAlign: "right",
        }}
      >
        {fmt3(cosine)}
      </span>
      <span style={{ minWidth: 108, display: "flex", justifyContent: "flex-end" }}>
        {flagTone ? <Badge tone={flagTone}>{flagLabel}</Badge> : <span style={{ opacity: 0 }}>—</span>}
      </span>
    </div>
  );
};

const Slider = ({ min = 0, max = 5, step = 0.1, value, onChange, label, unit = "σ", width = 280 }) => {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, width }}>
      {label && (
        <span className="t-body-s" style={{ minWidth: 60, color: "var(--text-muted)" }}>
          {label}
        </span>
      )}
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="msd-slider"
        style={{ flex: 1 }}
      />
      <span
        className="t-mono-s"
        style={{
          minWidth: 56,
          textAlign: "right",
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {value.toFixed(2)}{unit}
      </span>
    </div>
  );
};

const MatrixCell = ({ value, row, col }) => {
  // value 0..1 — weight a pastel, warm background
  const v = Math.max(0, Math.min(1, value));
  // low value -> paper; high value -> warm red-ish tint
  // linear mix between paper-raised and a soft red
  const mix = (a, b, t) => Math.round(a + (b - a) * t);
  const r1 = 0xF4, g1 = 0xF1, b1 = 0xEA;
  const r2 = 0xC4, g2 = 0x70, b2 = 0x70;
  const bg = `rgba(${mix(r1,r2,v)}, ${mix(g1,g2,v)}, ${mix(b1,b2,v)}, ${0.25 + 0.55 * v})`;
  const fg = v > 0.55 ? "#3A1E1E" : "var(--text-primary)";

  const [hover, setHover] = React.useState(false);

  return (
    <div
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        position: "relative",
        background: bg,
        border: "1px solid var(--border-subtle)",
        borderRadius: 8,
        padding: "14px 12px",
        textAlign: "center",
        color: fg,
        fontFamily: "var(--font-mono)",
        fontSize: 13,
        fontVariantNumeric: "tabular-nums",
        cursor: "default",
        transition: `transform var(--duration-base) var(--ease-standard)`,
      }}
    >
      {value.toFixed(3)}
      {hover && (row || col) && (
        <div
          style={{
            position: "absolute",
            left: "50%",
            top: "calc(100% + 6px)",
            transform: "translateX(-50%)",
            background: "var(--ink-900)",
            color: "var(--paper)",
            padding: "6px 10px",
            borderRadius: 6,
            whiteSpace: "nowrap",
            fontSize: 12,
            fontFamily: "var(--font-sans)",
            zIndex: 5,
            boxShadow: "var(--elevation-1)",
          }}
        >
          <span className="sc" style={{ opacity: 0.8, marginRight: 6 }}>{col}</span>
          vs <span className="sc" style={{ opacity: 0.8, marginLeft: 4 }}>{row}</span>
          <span style={{ marginLeft: 8, fontFamily: "var(--font-mono)" }}>ASR-R {value.toFixed(3)}</span>
        </div>
      )}
    </div>
  );
};

Object.assign(window, { ScoreBar, PassageRow, Slider, MatrixCell });
