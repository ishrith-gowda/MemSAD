// memsad — §3 threshold sweep
// left: roc plot (hand-rolled svg). right: vertical σ slider + live operating-point table.

const SWEEP = [
  { sigma: 0.5, ap: { tpr: 1.000, fpr: 0.185 }, mj: { tpr: 1.000, fpr: 0.140 }, im: { tpr: 0.633, fpr: 0.155 } },
  { sigma: 1.0, ap: { tpr: 1.000, fpr: 0.095 }, mj: { tpr: 1.000, fpr: 0.075 }, im: { tpr: 0.567, fpr: 0.090 } },
  { sigma: 1.5, ap: { tpr: 1.000, fpr: 0.035 }, mj: { tpr: 1.000, fpr: 0.020 }, im: { tpr: 0.500, fpr: 0.040 } },
  { sigma: 2.0, ap: { tpr: 1.000, fpr: 0.000 }, mj: { tpr: 1.000, fpr: 0.000 }, im: { tpr: 0.433, fpr: 0.000 } },
  { sigma: 2.5, ap: { tpr: 0.900, fpr: 0.000 }, mj: { tpr: 0.900, fpr: 0.000 }, im: { tpr: 0.333, fpr: 0.000 } },
  { sigma: 3.0, ap: { tpr: 0.800, fpr: 0.000 }, mj: { tpr: 0.800, fpr: 0.000 }, im: { tpr: 0.233, fpr: 0.000 } },
];

const CURVES = [
  { key: "ap", label: "AgentPoison", color: "var(--accent-agentpoison)", rawColor: "#C47070" },
  { key: "mj", label: "MINJA",       color: "var(--accent-minja)",        rawColor: "#7EA6CF" },
  { key: "im", label: "InjecMEM",    color: "var(--accent-injecmem)",     rawColor: "#B5A8C9" },
];

/* ---------- ROC plot ---------- */

const RocPlot = ({ sigma, hover, setHover }) => {
  // svg coords: 0..500 logical, padding for axes
  const W = 560, H = 560;
  const padL = 62, padR = 28, padT = 24, padB = 52;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  // x domain [0, 0.2], y domain [0, 1]
  const x = (fpr) => padL + (Math.min(0.2, Math.max(0, fpr)) / 0.2) * plotW;
  const y = (tpr) => padT + (1 - Math.min(1, Math.max(0, tpr))) * plotH;

  const xTicks = [0, 0.05, 0.10, 0.15, 0.20];
  const yTicks = [0, 0.2, 0.4, 0.6, 0.8, 1.0];

  // gridlines every 0.05 x and 0.2 y (already ticks)
  const fineX = [0.025, 0.075, 0.125, 0.175];
  const fineY = [0.1, 0.3, 0.5, 0.7, 0.9];

  const currentRow = SWEEP.find((s) => Math.abs(s.sigma - sigma) < 1e-6) || SWEEP[3];

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label="ROC curves for AgentPoison, MINJA, and InjecMEM across σ from 0.5 to 3.0"
      style={{ width: "100%", height: "auto", display: "block" }}
    >
      {/* outer plot frame */}
      <rect
        x={padL} y={padT} width={plotW} height={plotH}
        fill="none" stroke="var(--rule)" strokeWidth="1"
      />

      {/* fine gridlines */}
      {fineX.map((f) => (
        <line key={`fx-${f}`} x1={x(f)} x2={x(f)} y1={padT} y2={padT + plotH} stroke="var(--rule)" strokeWidth="0.5" opacity="0.55" />
      ))}
      {fineY.map((f) => (
        <line key={`fy-${f}`} y1={y(f)} y2={y(f)} x1={padL} x2={padL + plotW} stroke="var(--rule)" strokeWidth="0.5" opacity="0.55" />
      ))}
      {/* major gridlines */}
      {xTicks.slice(1).map((t) => (
        <line key={`gx-${t}`} x1={x(t)} x2={x(t)} y1={padT} y2={padT + plotH} stroke="var(--rule)" strokeWidth="1" />
      ))}
      {yTicks.slice(1, -1).map((t) => (
        <line key={`gy-${t}`} y1={y(t)} y2={y(t)} x1={padL} x2={padL + plotW} stroke="var(--rule)" strokeWidth="1" />
      ))}

      {/* chance diagonal y = x, from (0,0) to (0.2, 0.2) */}
      <line
        x1={x(0)} y1={y(0)}
        x2={x(0.2)} y2={y(0.2)}
        stroke="var(--ink-500)"
        strokeWidth="1"
        strokeDasharray="3 4"
        opacity="0.6"
      />
      <text x={x(0.2) - 4} y={y(0.2) - 6} textAnchor="end" fontFamily="var(--font-mono)" fontSize="10" fill="var(--ink-500)" letterSpacing="0.05em">
        y = x
      </text>

      {/* x ticks + labels */}
      {xTicks.map((t) => (
        <g key={`xt-${t}`}>
          <line x1={x(t)} x2={x(t)} y1={padT + plotH} y2={padT + plotH + 5} stroke="var(--ink-500)" strokeWidth="1" />
          <text x={x(t)} y={padT + plotH + 18} textAnchor="middle" fontFamily="var(--font-mono)" fontSize="11" fill="var(--ink-500)" style={{ fontVariantNumeric: "tabular-nums" }}>
            {t.toFixed(2)}
          </text>
        </g>
      ))}
      {/* y ticks + labels */}
      {yTicks.map((t) => (
        <g key={`yt-${t}`}>
          <line x1={padL - 5} x2={padL} y1={y(t)} y2={y(t)} stroke="var(--ink-500)" strokeWidth="1" />
          <text x={padL - 10} y={y(t) + 4} textAnchor="end" fontFamily="var(--font-mono)" fontSize="11" fill="var(--ink-500)" style={{ fontVariantNumeric: "tabular-nums" }}>
            {t.toFixed(1)}
          </text>
        </g>
      ))}

      {/* axis labels */}
      <text x={padL + plotW / 2} y={H - 14} textAnchor="middle" fontFamily="var(--font-sans)" fontSize="11" fontWeight="600" letterSpacing="0.12em" fill="var(--ink-500)">
        FALSE POSITIVE RATE
      </text>
      <text
        x={16}
        y={padT + plotH / 2}
        textAnchor="middle"
        fontFamily="var(--font-sans)" fontSize="11" fontWeight="600" letterSpacing="0.12em" fill="var(--ink-500)"
        transform={`rotate(-90, 16, ${padT + plotH / 2})`}
      >
        TRUE POSITIVE RATE
      </text>

      {/* curves */}
      {CURVES.map((c) => {
        const pts = SWEEP.map((s) => ({ fpr: s[c.key].fpr, tpr: s[c.key].tpr, sigma: s.sigma }));
        const d = pts.map((p, i) => `${i === 0 ? "M" : "L"} ${x(p.fpr)} ${y(p.tpr)}`).join(" ");
        return (
          <g key={c.key}>
            <path d={d} fill="none" stroke={c.rawColor} strokeWidth="1.6" strokeLinejoin="round" strokeLinecap="round" />
            {pts.map((p) => {
              const isActive = Math.abs(p.sigma - sigma) < 1e-6;
              const isHover = hover && hover.key === c.key && Math.abs(hover.sigma - p.sigma) < 1e-6;
              return (
                <g key={`${c.key}-${p.sigma}`}>
                  <circle cx={x(p.fpr)} cy={y(p.tpr)} r={3} fill={c.rawColor} />
                  {isActive && (
                    <circle cx={x(p.fpr)} cy={y(p.tpr)} r={7} fill="none" stroke={c.rawColor} strokeWidth="1.5" />
                  )}
                  {/* large invisible hover target */}
                  <circle
                    cx={x(p.fpr)} cy={y(p.tpr)} r={10}
                    fill="transparent"
                    style={{ cursor: "pointer" }}
                    onMouseEnter={() => setHover({ key: c.key, label: c.label, sigma: p.sigma, tpr: p.tpr, fpr: p.fpr, color: c.rawColor, px: x(p.fpr), py: y(p.tpr) })}
                    onMouseLeave={() => setHover(null)}
                  />
                  {isHover && (
                    <circle cx={x(p.fpr)} cy={y(p.tpr)} r={5} fill={c.rawColor} stroke="var(--paper)" strokeWidth="1" />
                  )}
                </g>
              );
            })}
            {/* inline curve label at the right end (σ=0.5 is the rightmost FPR point) */}
            <text
              x={x(pts[0].fpr) + 6}
              y={y(pts[0].tpr) + 3}
              fontFamily="var(--font-mono)"
              fontSize="10"
              fill={c.rawColor}
              letterSpacing="0.08em"
              style={{ fontVariant: "small-caps", fontWeight: 600 }}
            >
              {c.label}
            </text>
          </g>
        );
      })}

      {/* active σ vertical rule for each curve — dotted line from plot bottom up to point */}
      {CURVES.map((c) => {
        const p = SWEEP.find((s) => Math.abs(s.sigma - sigma) < 1e-6);
        if (!p) return null;
        const pt = p[c.key];
        return (
          <line
            key={`act-${c.key}`}
            x1={x(pt.fpr)} x2={x(pt.fpr)}
            y1={y(pt.tpr)} y2={padT + plotH}
            stroke={c.rawColor}
            strokeWidth="0.8"
            strokeDasharray="2 3"
            opacity="0.55"
          />
        );
      })}
    </svg>
  );
};

/* ---------- vertical σ slider ---------- */

const VerticalSigmaSlider = ({ value, onChange, ariaText }) => {
  const steps = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
  const min = 0.5, max = 3.0;
  const track = 320;
  // value mapped to vertical position: high σ at top
  const norm = (value - min) / (max - min);
  const thumbTop = (1 - norm) * track;

  const ref = React.useRef(null);
  const onKey = (e) => {
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") { e.preventDefault(); onChange(Math.max(min, Math.round((value - 0.5) * 2) / 2)); }
    if (e.key === "ArrowRight" || e.key === "ArrowUp")  { e.preventDefault(); onChange(Math.min(max, Math.round((value + 0.5) * 2) / 2)); }
    if (e.key === "Home") { e.preventDefault(); onChange(min); }
    if (e.key === "End")  { e.preventDefault(); onChange(max); }
  };

  // convert mouse y within track into a stepped value
  const fromY = (clientY) => {
    const rect = ref.current.getBoundingClientRect();
    const yPx = Math.max(0, Math.min(track, clientY - rect.top));
    const n = 1 - yPx / track;
    const raw = min + n * (max - min);
    const snap = Math.round(raw * 2) / 2;
    return Math.max(min, Math.min(max, snap));
  };

  const onPointer = (e) => {
    if (e.buttons !== 1 && e.type !== "click") return;
    onChange(fromY(e.clientY));
  };

  return (
    <div style={{ display: "flex", gap: 14, alignItems: "stretch" }}>
      {/* ticks */}
      <div style={{ display: "flex", flexDirection: "column", justifyContent: "space-between", height: track, paddingTop: 0 }}>
        {steps.slice().reverse().map((s) => (
          <span key={s} style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-muted)", fontVariantNumeric: "tabular-nums", lineHeight: 1 }}>
            {s.toFixed(1)}σ
          </span>
        ))}
      </div>

      {/* track */}
      <div
        ref={ref}
        role="slider"
        tabIndex={0}
        aria-valuemin={min}
        aria-valuemax={max}
        aria-valuenow={value}
        aria-valuetext={ariaText}
        aria-orientation="vertical"
        onKeyDown={onKey}
        onMouseDown={onPointer}
        onMouseMove={onPointer}
        onClick={onPointer}
        style={{
          position: "relative",
          width: 10,
          height: track,
          background: "var(--paper-raised)",
          border: "1px solid var(--border-subtle)",
          borderRadius: 999,
          cursor: "pointer",
          outline: "none",
        }}
      >
        {/* filled portion from current thumb down */}
        <div
          style={{
            position: "absolute",
            left: -1, right: -1,
            top: thumbTop,
            bottom: -1,
            background: "var(--ink-900)",
            borderRadius: 999,
            transition: "top 180ms var(--ease-standard)",
          }}
        />
        {/* step markers */}
        {steps.map((s) => {
          const n = (s - min) / (max - min);
          return (
            <span
              key={`m-${s}`}
              style={{
                position: "absolute",
                left: -3, right: -3,
                top: (1 - n) * track - 0.5,
                height: 1,
                background: Math.abs(s - value) < 1e-6 ? "transparent" : "var(--border-subtle)",
              }}
            />
          );
        })}
        {/* thumb */}
        <div
          style={{
            position: "absolute",
            top: thumbTop - 10,
            left: -7,
            width: 24, height: 24,
            borderRadius: 999,
            background: "var(--ink-900)",
            border: "2px solid var(--paper)",
            boxShadow: "0 0 0 1px var(--ink-900), var(--elevation-1)",
            transition: "top 180ms var(--ease-standard)",
          }}
        />
        {/* floating readout next to thumb */}
        <div
          style={{
            position: "absolute",
            left: 28,
            top: thumbTop - 12,
            fontFamily: "var(--font-mono)",
            fontSize: 14,
            fontWeight: 500,
            color: "var(--text-primary)",
            fontVariantNumeric: "tabular-nums",
            whiteSpace: "nowrap",
            transition: "top 180ms var(--ease-standard)",
          }}
        >
          σ = {value.toFixed(2)}
        </div>
      </div>
    </div>
  );
};

/* ---------- small num-tween ---------- */

const useTween = (target, ms = 200) => {
  const [val, setVal] = React.useState(target);
  const rafRef = React.useRef(null);
  const fromRef = React.useRef(target);
  const startRef = React.useRef(0);
  React.useEffect(() => {
    cancelAnimationFrame(rafRef.current);
    fromRef.current = val;
    startRef.current = performance.now();
    const step = (now) => {
      const p = Math.min(1, (now - startRef.current) / ms);
      const e = 1 - Math.pow(1 - p, 3);
      setVal(fromRef.current + (target - fromRef.current) * e);
      if (p < 1) rafRef.current = requestAnimationFrame(step);
    };
    rafRef.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(rafRef.current);
    // eslint-disable-next-line
  }, [target]);
  return val;
};

/* ---------- operating-point row ---------- */

const OpRow = ({ curve, tpr, fpr }) => {
  const tprT = useTween(tpr, 200);
  const fprT = useTween(fpr, 200);
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "18px 130px 1fr 1fr",
        gap: 14,
        alignItems: "center",
        padding: "10px 0",
        borderBottom: "1px solid var(--border-subtle)",
      }}
    >
      <span style={{ width: 12, height: 12, borderRadius: 2, background: curve.rawColor, display: "inline-block" }} />
      <span style={{ fontFamily: "var(--font-sans)", fontSize: 13, color: "var(--text-primary)", fontVariant: "small-caps", fontWeight: 600, letterSpacing: "0.04em" }}>
        {curve.label}
      </span>
      <OpMetric label="TPR" value={tprT} fill={curve.rawColor} />
      <OpMetric label="FPR" value={fprT} fill="var(--ink-500)" />
    </div>
  );
};

const OpMetric = ({ label, value, fill }) => {
  const pct = Math.max(0, Math.min(1, value));
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <span style={{ fontFamily: "var(--font-sans)", fontSize: 10, fontWeight: 700, letterSpacing: "0.12em", color: "var(--text-muted)", minWidth: 26 }}>
        {label}
      </span>
      <div style={{ position: "relative", width: 60, height: 6, background: "var(--rule)", borderRadius: 999 }}>
        <div
          style={{
            position: "absolute", inset: 0,
            width: `${pct * 100}%`,
            background: fill,
            borderRadius: 999,
          }}
        />
      </div>
      <span
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 14,
          fontVariantNumeric: "tabular-nums",
          color: "var(--text-primary)",
          minWidth: 54,
        }}
      >
        {value.toFixed(3)}
      </span>
    </div>
  );
};

/* ---------- section ---------- */

const ThresholdSweep = () => {
  const [sigma, setSigma] = React.useState(2.0);
  const [hover, setHover] = React.useState(null);

  const row = SWEEP.find((s) => Math.abs(s.sigma - sigma) < 1e-6) || SWEEP[3];

  const aria = `σ = ${sigma.toFixed(1)}, AgentPoison TPR ${row.ap.tpr.toFixed(3)} FPR ${row.ap.fpr.toFixed(3)}, MINJA TPR ${row.mj.tpr.toFixed(3)} FPR ${row.mj.fpr.toFixed(3)}, InjecMEM TPR ${row.im.tpr.toFixed(3)} FPR ${row.im.fpr.toFixed(3)}`;

  return (
    <section
      id="threshold-sweep"
      style={{
        background: "var(--surface)",
        padding: "96px 80px",
        borderTop: "1px solid var(--border-subtle)",
      }}
    >
      <div style={{ maxWidth: 1280, margin: "0 auto" }}>
        {/* eyebrow */}
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
          §3 · THRESHOLD
        </div>

        <h2
          style={{
            fontFamily: "var(--font-serif)",
            fontSize: 32,
            lineHeight: 1.15,
            margin: "0 0 14px 0",
            fontWeight: 400,
            letterSpacing: "-0.005em",
            color: "var(--text-primary)",
          }}
        >
          Threshold sweep across σ
        </h2>

        <p
          className="t-body-l"
          style={{
            margin: 0,
            color: "var(--text-secondary)",
            maxWidth: "62ch",
            fontSize: 16,
          }}
        >
          <span className="sc" style={{ fontWeight: 600, color: "var(--text-primary)" }}>MemSAD</span>'s only knob is k — the number of standard deviations above the calibrated mean at which a passage fires.
          Drag the slider to trace TPR and FPR across attacks. Operating points in the table stream live.
        </p>

        <div
          className="sweep-grid"
          style={{
            marginTop: 48,
            display: "grid",
            gridTemplateColumns: "60fr 40fr",
            columnGap: 48,
            alignItems: "start",
          }}
        >
          {/* LEFT — plot */}
          <div>
            <div
              style={{
                border: "1px solid var(--border-subtle)",
                borderRadius: 10,
                padding: 24,
                background: "var(--surface)",
                position: "relative",
                maxWidth: 620,
                aspectRatio: "1 / 1",
              }}
            >
              <RocPlot sigma={sigma} hover={hover} setHover={setHover} />

              {/* hover tooltip */}
              {hover && (
                <div
                  style={{
                    position: "absolute",
                    left: `calc(24px + ${(hover.px / 560) * 100}% * (1 - 48px/560))`,
                    top: 24 + (hover.py / 560) * (620 - 48) - 56,
                    transform: "translateX(10px)",
                    background: "var(--surface)",
                    border: "1px solid var(--border-subtle)",
                    borderLeft: `3px solid ${hover.color}`,
                    borderRadius: 8,
                    padding: "8px 10px",
                    boxShadow: "var(--elevation-1)",
                    pointerEvents: "none",
                    zIndex: 5,
                    minWidth: 160,
                  }}
                >
                  <div style={{ fontFamily: "var(--font-sans)", fontSize: 11, fontWeight: 700, letterSpacing: "0.08em", color: hover.color, fontVariant: "small-caps", marginBottom: 4 }}>
                    {hover.label}
                  </div>
                  <div style={{ fontFamily: "var(--font-mono)", fontSize: 12, color: "var(--text-primary)", fontVariantNumeric: "tabular-nums", lineHeight: 1.5 }}>
                    σ = {hover.sigma.toFixed(1)} · TPR = {hover.tpr.toFixed(3)}<br />
                    FPR = {hover.fpr.toFixed(3)}
                  </div>
                </div>
              )}
            </div>

            <div
              style={{
                marginTop: 10,
                fontFamily: "var(--font-mono)",
                fontSize: 12,
                color: "var(--text-muted)",
              }}
            >
              fpr clipped to [0, 0.2]. operating points at σ ∈ {"{"}0.5, 1.0, 1.5, 2.0, 2.5, 3.0{"}"}.
            </div>

            {/* legend */}
            <div style={{ marginTop: 18, display: "flex", gap: 20, flexWrap: "wrap" }}>
              <LegendPill color="#C47070" label="AgentPoison (triggered cal.)" />
              <LegendPill color="#7EA6CF" label="MINJA" />
              <LegendPill color="#B5A8C9" label="InjecMEM" />
              <LegendPill color="var(--ink-500)" label="Chance (y = x)" dashed />
            </div>

            {/* visually-hidden data table for a11y */}
            <table style={{ position: "absolute", width: 1, height: 1, overflow: "hidden", clip: "rect(0 0 0 0)", clipPath: "inset(50%)", whiteSpace: "nowrap" }}>
              <thead>
                <tr><th>σ</th><th>AgentPoison TPR</th><th>AgentPoison FPR</th><th>MINJA TPR</th><th>MINJA FPR</th><th>InjecMEM TPR</th><th>InjecMEM FPR</th></tr>
              </thead>
              <tbody>
                {SWEEP.map((s) => (
                  <tr key={s.sigma}>
                    <td>{s.sigma}</td>
                    <td>{s.ap.tpr}</td><td>{s.ap.fpr}</td>
                    <td>{s.mj.tpr}</td><td>{s.mj.fpr}</td>
                    <td>{s.im.tpr}</td><td>{s.im.fpr}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* RIGHT — slider + table */}
          <div>
            <div
              style={{
                fontFamily: "var(--font-sans)",
                fontSize: 11,
                fontWeight: 700,
                letterSpacing: "0.12em",
                color: "var(--text-muted)",
                textTransform: "uppercase",
                marginBottom: 20,
              }}
            >
              SCORE MODE: combined
            </div>

            <div style={{ paddingLeft: 4, marginBottom: 36 }}>
              <VerticalSigmaSlider value={sigma} onChange={setSigma} ariaText={aria} />
            </div>

            {/* operating-point card */}
            <div
              style={{
                border: "1px solid var(--border-subtle)",
                borderRadius: 10,
                padding: 20,
                background: "var(--surface)",
              }}
            >
              <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 8 }}>
                <span
                  style={{
                    fontFamily: "var(--font-sans)",
                    fontSize: 11,
                    fontWeight: 700,
                    letterSpacing: "0.12em",
                    color: "var(--text-muted)",
                    textTransform: "uppercase",
                  }}
                >
                  OPERATING POINT
                </span>
                <span
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: 14,
                    fontVariantNumeric: "tabular-nums",
                    color: "var(--text-primary)",
                  }}
                >
                  σ = {sigma.toFixed(2)}
                </span>
              </div>

              <div>
                <OpRow curve={CURVES[0]} tpr={row.ap.tpr} fpr={row.ap.fpr} />
                <OpRow curve={CURVES[1]} tpr={row.mj.tpr} fpr={row.mj.fpr} />
                <div style={{ borderBottom: 0 }}>
                  <OpRow curve={CURVES[2]} tpr={row.im.tpr} fpr={row.im.fpr} />
                </div>
              </div>
            </div>

            <div
              style={{
                marginTop: 12,
                fontFamily: "var(--font-mono)",
                fontSize: 12,
                color: "var(--text-muted)",
                lineHeight: 1.6,
              }}
            >
              threshold μ + k·σ computed against calibrated victim-query distribution · n_queries = 20 · scoring mode = combined
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

const LegendPill = ({ color, label, dashed }) => (
  <span style={{ display: "inline-flex", alignItems: "center", gap: 8, fontFamily: "var(--font-mono)", fontSize: 12, color: "var(--text-secondary)" }}>
    <span
      style={{
        width: 16, height: 16, borderRadius: 3,
        background: dashed ? "transparent" : color,
        border: dashed ? `1px dashed ${color}` : "none",
        display: "inline-block",
      }}
    />
    {label}
  </span>
);

ReactDOM.createRoot(document.getElementById("root")).render(<ThresholdSweep />);
