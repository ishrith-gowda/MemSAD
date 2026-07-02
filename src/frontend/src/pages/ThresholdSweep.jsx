import { useState, useEffect, useRef } from "react";

/*
  §3 — threshold sweep across σ.
  hand-rolled svg roc plot + vertical σ slider + live operating-point card.
*/

const SWEEP = [
  { sigma: 0.5, ap: { tpr: 1.000, fpr: 0.185 }, mj: { tpr: 1.000, fpr: 0.140 }, im: { tpr: 0.633, fpr: 0.155 } },
  { sigma: 1.0, ap: { tpr: 1.000, fpr: 0.095 }, mj: { tpr: 1.000, fpr: 0.075 }, im: { tpr: 0.567, fpr: 0.090 } },
  { sigma: 1.5, ap: { tpr: 1.000, fpr: 0.035 }, mj: { tpr: 1.000, fpr: 0.020 }, im: { tpr: 0.500, fpr: 0.040 } },
  { sigma: 2.0, ap: { tpr: 1.000, fpr: 0.000 }, mj: { tpr: 1.000, fpr: 0.000 }, im: { tpr: 0.433, fpr: 0.000 } },
  { sigma: 2.5, ap: { tpr: 0.900, fpr: 0.000 }, mj: { tpr: 0.900, fpr: 0.000 }, im: { tpr: 0.333, fpr: 0.000 } },
  { sigma: 3.0, ap: { tpr: 0.800, fpr: 0.000 }, mj: { tpr: 0.800, fpr: 0.000 }, im: { tpr: 0.233, fpr: 0.000 } },
];

const CURVES = [
  { key: "ap", label: "AgentPoison", rawColor: "#C47070" },
  { key: "mj", label: "MINJA",       rawColor: "#7EA6CF" },
  { key: "im", label: "InjecMEM",    rawColor: "#B5A8C9" },
];

function RocPlot({ sigma, hover, setHover }) {
  const W = 560, H = 560;
  const padL = 62, padR = 28, padT = 24, padB = 52;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const x = (fpr) => padL + (Math.min(0.2, Math.max(0, fpr)) / 0.2) * plotW;
  const y = (tpr) => padT + (1 - Math.min(1, Math.max(0, tpr))) * plotH;

  const xTicks = [0, 0.05, 0.10, 0.15, 0.20];
  const yTicks = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
  const fineX = [0.025, 0.075, 0.125, 0.175];
  const fineY = [0.1, 0.3, 0.5, 0.7, 0.9];

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label="ROC curves for AgentPoison, MINJA, and InjecMEM across σ from 0.5 to 3.0"
      style={{ width: "100%", height: "auto", display: "block" }}
    >
      <rect x={padL} y={padT} width={plotW} height={plotH} fill="none" stroke="var(--rule)" strokeWidth="1" />

      {fineX.map((f) => (
        <line key={`fx-${f}`} x1={x(f)} x2={x(f)} y1={padT} y2={padT + plotH} stroke="var(--rule)" strokeWidth="0.5" opacity="0.55" />
      ))}
      {fineY.map((f) => (
        <line key={`fy-${f}`} y1={y(f)} y2={y(f)} x1={padL} x2={padL + plotW} stroke="var(--rule)" strokeWidth="0.5" opacity="0.55" />
      ))}
      {xTicks.slice(1).map((t) => (
        <line key={`gx-${t}`} x1={x(t)} x2={x(t)} y1={padT} y2={padT + plotH} stroke="var(--rule)" strokeWidth="1" />
      ))}
      {yTicks.slice(1, -1).map((t) => (
        <line key={`gy-${t}`} y1={y(t)} y2={y(t)} x1={padL} x2={padL + plotW} stroke="var(--rule)" strokeWidth="1" />
      ))}

      <line x1={x(0)} y1={y(0)} x2={x(0.2)} y2={y(0.2)} stroke="var(--ink-500)" strokeWidth="1" strokeDasharray="3 4" opacity="0.6" />
      <text x={x(0.2) - 4} y={y(0.2) - 6} textAnchor="end" fontFamily="var(--font-mono)" fontSize="10" fill="var(--ink-500)" letterSpacing="0.05em">
        y = x
      </text>

      {xTicks.map((t) => (
        <g key={`xt-${t}`}>
          <line x1={x(t)} x2={x(t)} y1={padT + plotH} y2={padT + plotH + 5} stroke="var(--ink-500)" strokeWidth="1" />
          <text x={x(t)} y={padT + plotH + 18} textAnchor="middle" fontFamily="var(--font-mono)" fontSize="11" fill="var(--ink-500)" style={{ fontVariantNumeric: "tabular-nums" }}>
            {t.toFixed(2)}
          </text>
        </g>
      ))}
      {yTicks.map((t) => (
        <g key={`yt-${t}`}>
          <line x1={padL - 5} x2={padL} y1={y(t)} y2={y(t)} stroke="var(--ink-500)" strokeWidth="1" />
          <text x={padL - 10} y={y(t) + 4} textAnchor="end" fontFamily="var(--font-mono)" fontSize="11" fill="var(--ink-500)" style={{ fontVariantNumeric: "tabular-nums" }}>
            {t.toFixed(1)}
          </text>
        </g>
      ))}

      <text x={padL + plotW / 2} y={H - 14} textAnchor="middle" fontFamily="var(--font-sans)" fontSize="11" fontWeight="600" letterSpacing="0.14em" fill="var(--ink-500)">
        FALSE POSITIVE RATE
      </text>
      <text
        x={16}
        y={padT + plotH / 2}
        textAnchor="middle"
        fontFamily="var(--font-sans)" fontSize="11" fontWeight="600" letterSpacing="0.14em" fill="var(--ink-500)"
        transform={`rotate(-90, 16, ${padT + plotH / 2})`}
      >
        TRUE POSITIVE RATE
      </text>

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
                  <circle
                    cx={x(p.fpr)} cy={y(p.tpr)} r={10}
                    fill="transparent"
                    style={{ cursor: "pointer" }}
                    onMouseEnter={() => setHover({ key: c.key, label: c.label, sigma: p.sigma, tpr: p.tpr, fpr: p.fpr, color: c.rawColor })}
                    onMouseLeave={() => setHover(null)}
                  />
                  {isHover && (
                    <circle cx={x(p.fpr)} cy={y(p.tpr)} r={5} fill={c.rawColor} stroke="var(--paper)" strokeWidth="1" />
                  )}
                </g>
              );
            })}
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
}

function HorizontalSigmaSlider({ value, onChange, ariaText }) {
  const steps = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
  const min = 0.5, max = 3.0;
  const norm = (value - min) / (max - min);
  const ref = useRef(null);

  const onKey = (e) => {
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") { e.preventDefault(); onChange(Math.max(min, Math.round((value - 0.5) * 2) / 2)); }
    if (e.key === "ArrowRight" || e.key === "ArrowUp")  { e.preventDefault(); onChange(Math.min(max, Math.round((value + 0.5) * 2) / 2)); }
    if (e.key === "Home") { e.preventDefault(); onChange(min); }
    if (e.key === "End")  { e.preventDefault(); onChange(max); }
  };

  const fromX = (clientX) => {
    const rect = ref.current.getBoundingClientRect();
    const xPx = Math.max(0, Math.min(rect.width, clientX - rect.left));
    const n = xPx / rect.width;
    const raw = min + n * (max - min);
    const snap = Math.round(raw * 2) / 2;
    return Math.max(min, Math.min(max, snap));
  };

  const onPointer = (e) => {
    if (e.buttons !== 1 && e.type !== "click") return;
    onChange(fromX(e.clientX));
  };

  const thumbPct = norm * 100;

  return (
    <div>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 16 }}>
        <span className="t-caps">σ threshold</span>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 15, fontWeight: 500, color: "var(--text-primary)", fontVariantNumeric: "tabular-nums" }}>
          σ = {value.toFixed(2)}
        </span>
      </div>

      <div style={{ position: "relative", padding: "12px 0" }}>
        <div
          ref={ref}
          role="slider"
          tabIndex={0}
          aria-valuemin={min}
          aria-valuemax={max}
          aria-valuenow={value}
          aria-valuetext={ariaText}
          aria-orientation="horizontal"
          onKeyDown={onKey}
          onMouseDown={onPointer}
          onMouseMove={onPointer}
          onClick={onPointer}
          style={{
            position: "relative",
            width: "100%",
            height: 10,
            background: "var(--paper-raised)",
            border: "1px solid var(--border-subtle)",
            borderRadius: 999,
            cursor: "pointer",
            outline: "none",
          }}
        >
          <div
            style={{
              position: "absolute",
              top: -1, bottom: -1,
              left: -1,
              width: `calc(${thumbPct}% + 1px)`,
              background: "var(--ink-900)",
              borderRadius: 999,
              transition: "width 180ms var(--ease-standard)",
            }}
          />
          <div
            style={{
              position: "absolute",
              top: -7,
              left: `calc(${thumbPct}% - 12px)`,
              width: 24, height: 24,
              borderRadius: 999,
              background: "var(--ink-900)",
              border: "2px solid var(--paper)",
              boxShadow: "0 0 0 1px var(--ink-900), var(--elevation-1)",
              transition: "left 180ms var(--ease-standard)",
            }}
          />
        </div>
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
        {steps.map((s) => (
          <span key={s} style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-muted)", fontVariantNumeric: "tabular-nums" }}>
            {s.toFixed(1)}σ
          </span>
        ))}
      </div>
    </div>
  );
}

function useTween(target, ms = 200) {
  const [val, setVal] = useState(target);
  const rafRef = useRef(null);
  const fromRef = useRef(target);
  const startRef = useRef(0);
  useEffect(() => {
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
}

function OpMetric({ label, value, fill }) {
  const pct = Math.max(0, Math.min(1, value));
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 0 }}>
      <span style={{ fontFamily: "var(--font-sans)", fontSize: 10, fontWeight: 700, letterSpacing: "0.14em", color: "var(--text-muted)", flexShrink: 0 }}>
        {label}
      </span>
      <div style={{ position: "relative", flex: 1, minWidth: 20, height: 6, background: "var(--rule)", borderRadius: 999 }}>
        <div style={{ position: "absolute", inset: 0, width: `${pct * 100}%`, background: fill, borderRadius: 999 }} />
      </div>
      <span style={{ fontFamily: "var(--font-mono)", fontSize: 13, fontVariantNumeric: "tabular-nums", color: "var(--text-primary)", flexShrink: 0, textAlign: "right" }}>
        {value.toFixed(3)}
      </span>
    </div>
  );
}

function OpRow({ curve, tpr, fpr, last }) {
  const tprT = useTween(tpr, 200);
  const fprT = useTween(fpr, 200);
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "16px minmax(90px, auto) minmax(0, 1fr) minmax(0, 1fr)",
        gap: 14,
        alignItems: "center",
        padding: "10px 0",
        borderBottom: last ? 0 : "1px solid var(--border-subtle)",
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
}

function LegendPill({ color, label, dashed }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 8, fontFamily: "var(--font-sans)", fontSize: 11, color: "var(--text-secondary)", letterSpacing: "0.02em" }}>
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
}

export default function ThresholdSweep() {
  const [sigma, setSigma] = useState(2.0);
  const [hover, setHover] = useState(null);

  const row = SWEEP.find((s) => Math.abs(s.sigma - sigma) < 1e-6) || SWEEP[3];

  const aria = `σ = ${sigma.toFixed(1)}, AgentPoison TPR ${row.ap.tpr.toFixed(3)} FPR ${row.ap.fpr.toFixed(3)}, MINJA TPR ${row.mj.tpr.toFixed(3)} FPR ${row.mj.fpr.toFixed(3)}, InjecMEM TPR ${row.im.tpr.toFixed(3)} FPR ${row.im.fpr.toFixed(3)}`;

  return (
    <section
      style={{
        maxWidth: "var(--page-max)",
        margin: "0 auto",
        padding: "var(--section-pad-y) var(--page-gutter)",
      }}
    >
      <div className="t-caps" style={{ marginBottom: 14 }}>
        §3 · threshold · σ sweep
      </div>
      <hr style={{ marginBottom: 32 }} />

      <h1 className="t-display-l" style={{ margin: "0 0 14px 0" }}>
        Threshold sweep across σ.
      </h1>

      <p className="t-lede" style={{ margin: 0 }}>
        <span className="sc" style={{ fontWeight: 600, color: "var(--text-primary)" }}>MemSAD</span>'s only knob
        is <span style={{ fontFamily: "var(--font-mono)" }}>k</span> — the number of standard deviations above
        the calibrated mean at which a passage fires. Drag the slider to trace TPR and FPR across attacks.
        Operating points in the table stream live.
      </p>

      <div
        style={{
          marginTop: 48,
          display: "grid",
          gridTemplateColumns: "minmax(0, 6fr) minmax(280px, 4fr)",
          columnGap: 48,
          alignItems: "start",
        }}
      >
        {/* LEFT — plot */}
        <div>
          <div
            style={{
              border: "1px solid var(--border)",
              borderRadius: 12,
              padding: 24,
              background: "var(--surface)",
              maxWidth: 620,
              aspectRatio: "1 / 1",
            }}
          >
            <RocPlot sigma={sigma} hover={hover} setHover={setHover} />
          </div>

          <div style={{ marginTop: 14, display: "flex", alignItems: "flex-start", gap: 20 }}>
            <span className="t-caps-strong" style={{ whiteSpace: "nowrap" }}>Fig. 3</span>
            <p className="t-body-s" style={{ margin: 0, color: "var(--text-secondary)", fontStyle: "italic" }}>
              ROC curves per attack family, FPR clipped to [0, 0.20]. Operating points at
              σ ∈ {"{"}0.5, 1.0, 1.5, 2.0, 2.5, 3.0{"}"}; ringed marker tracks the active slider.
            </p>
          </div>

          <div style={{ marginTop: 20, display: "flex", gap: 20, flexWrap: "wrap" }}>
            <LegendPill color="#C47070" label="AgentPoison (triggered cal.)" />
            <LegendPill color="#7EA6CF" label="MINJA" />
            <LegendPill color="#B5A8C9" label="InjecMEM" />
            <LegendPill color="var(--ink-500)" label="Chance, y = x" dashed />
          </div>

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

        {/* RIGHT — horizontal slider + operating-point card */}
        <div>
          <div
            style={{
              display: "flex",
              alignItems: "baseline",
              justifyContent: "space-between",
              marginBottom: 18,
              gap: 12,
            }}
          >
            <span className="t-caps">Score mode · combined</span>
            <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-muted)" }}>
              n_queries = 20
            </span>
          </div>

          <div
            style={{
              border: "1px solid var(--border)",
              borderRadius: 12,
              padding: 24,
              background: "var(--surface)",
              marginBottom: 20,
            }}
          >
            <HorizontalSigmaSlider value={sigma} onChange={setSigma} ariaText={aria} />
          </div>

          <div
            style={{
              border: "1px solid var(--border)",
              borderRadius: 12,
              padding: 24,
              background: "var(--surface)",
            }}
          >
            <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 12 }}>
              <span className="t-caps">Operating point</span>
              <span style={{ fontFamily: "var(--font-mono)", fontSize: 14, fontVariantNumeric: "tabular-nums", color: "var(--text-primary)" }}>
                σ = {sigma.toFixed(2)}
              </span>
            </div>

            <div>
              <OpRow curve={CURVES[0]} tpr={row.ap.tpr} fpr={row.ap.fpr} />
              <OpRow curve={CURVES[1]} tpr={row.mj.tpr} fpr={row.mj.fpr} />
              <OpRow curve={CURVES[2]} tpr={row.im.tpr} fpr={row.im.fpr} last />
            </div>
          </div>

          <div style={{ marginTop: 14, fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-muted)", lineHeight: 1.6 }}>
            threshold μ + k·σ · calibrated victim-query distribution
          </div>
        </div>
      </div>
    </section>
  );
}
