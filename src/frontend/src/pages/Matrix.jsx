import { useRef, useState, Fragment } from "react";

/*
  §4 — attack × defense matrix.
  arrow-key navigable grid with tooltip + ci callouts on the memsad row.
*/

const MATRIX = {
  attacks: [
    { key: "ap", label: "AgentPoison", color: "#C47070" },
    { key: "mj", label: "MINJA",       color: "#7EA6CF" },
    { key: "im", label: "InjecMEM",    color: "#B5A8C9" },
  ],
  defenses: [
    { key: "none",  label: "No defense",     ours: false },
    { key: "ppl",   label: "Perplexity",     ours: false },
    { key: "sim",   label: "Similarity cap", ours: false },
    { key: "llms",  label: "LLM sanitizer",  ours: false },
    { key: "mem",   label: "MemSAD",         ours: true  },
  ],
  values: {
    none: { ap: 0.842, mj: 0.751, im: 0.488 },
    ppl:  { ap: 0.781, mj: 0.692, im: 0.412 },
    sim:  { ap: 0.594, mj: 0.503, im: 0.322 },
    llms: { ap: 0.402, mj: 0.396, im: 0.275 },
    mem:  { ap: 0.073, mj: 0.091, im: 0.264 },
  },
  cis: {
    "mem-ap": [0.051, 0.102],
    "mem-mj": [0.068, 0.119],
    "mem-im": [0.218, 0.312],
  },
};

function Cell({ value, attack, defense, hasCi, ci, cellRef, focused }) {
  const [hover, setHover] = useState(false);
  const alpha = value * 0.45;

  return (
    <div
      ref={cellRef}
      role="gridcell"
      tabIndex={focused ? 0 : -1}
      data-attack={attack.key}
      data-defense={defense.key}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onFocus={() => setHover(true)}
      onBlur={() => setHover(false)}
      style={{
        position: "relative",
        background: `rgba(196, 112, 112, ${alpha})`,
        display: "grid",
        placeItems: "center",
        height: 60,
        outline: hover ? "1.5px solid var(--ink-900)" : "none",
        outlineOffset: -1,
        cursor: "default",
      }}
    >
      <span
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 14,
          fontWeight: 500,
          color: "var(--ink-900)",
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {value.toFixed(3)}
      </span>

      {hover && (
        <div
          style={{
            position: "absolute",
            bottom: "calc(100% + 8px)",
            left: "50%",
            transform: "translateX(-50%)",
            background: "var(--surface)",
            border: "1px solid var(--border-subtle)",
            borderRadius: 8,
            padding: 12,
            boxShadow: "0 2px 8px rgba(0,0,0,0.06)",
            pointerEvents: "none",
            zIndex: 10,
            minWidth: 220,
            whiteSpace: "nowrap",
          }}
        >
          <div style={{ fontFamily: "var(--font-sans)", fontSize: 11, lineHeight: 1.5 }}>
            <span style={{ fontVariant: "small-caps", fontWeight: 700, letterSpacing: "0.04em", color: attack.color }}>
              {attack.label}
            </span>
            <span style={{ color: "var(--text-muted)", margin: "0 6px" }}>·</span>
            <span style={{ color: "var(--text-primary)", fontWeight: 500 }}>
              {defense.label}{defense.ours ? " (ours)" : ""}
            </span>
          </div>
          <div style={{ marginTop: 4, fontFamily: "var(--font-mono)", fontSize: 12, color: "var(--text-primary)", fontVariantNumeric: "tabular-nums" }}>
            ASR-R = {value.toFixed(3)}
            {hasCi && (
              <>
                <span style={{ color: "var(--text-muted)", margin: "0 6px" }}>·</span>
                95% CI [{ci[0].toFixed(3)}, {ci[1].toFixed(3)}]
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default function Matrix() {
  const cellRefs = useRef({});
  const onKeyDown = (e) => {
    const active = document.activeElement;
    if (!active || !active.dataset?.attack) return;
    const defIdx = MATRIX.defenses.findIndex((d) => d.key === active.dataset.defense);
    const atkIdx = MATRIX.attacks.findIndex((a) => a.key === active.dataset.attack);
    if (defIdx < 0 || atkIdx < 0) return;
    let nd = defIdx, na = atkIdx;
    if (e.key === "ArrowDown")  nd = Math.min(MATRIX.defenses.length - 1, defIdx + 1);
    if (e.key === "ArrowUp")    nd = Math.max(0, defIdx - 1);
    if (e.key === "ArrowRight") na = Math.min(MATRIX.attacks.length - 1, atkIdx + 1);
    if (e.key === "ArrowLeft")  na = Math.max(0, atkIdx - 1);
    if (nd !== defIdx || na !== atkIdx) {
      e.preventDefault();
      const k = `${MATRIX.defenses[nd].key}-${MATRIX.attacks[na].key}`;
      cellRefs.current[k]?.focus();
    }
  };

  return (
    <section
      style={{
        maxWidth: "var(--page-max)",
        margin: "0 auto",
        padding: "var(--section-pad-y) var(--page-gutter)",
      }}
    >
      <div className="t-caps" style={{ marginBottom: 14 }}>
        §4 · matrix · 3 attacks × 5 defenses
      </div>
      <hr style={{ marginBottom: 32 }} />

      <h1 className="t-display-l" style={{ margin: "0 0 14px 0" }}>
        Attack · defense grid.
      </h1>
      <p className="t-lede" style={{ margin: 0 }}>
        Every attack crossed with every defense. Cell values are <em>ASR-R</em> — the fraction of victim
        queries whose top-k retrieval contains a poisoned passage. Lower is better. Ours is the last row.
      </p>

      <div
        style={{
          marginTop: 40,
          border: "1px solid var(--border)",
          borderRadius: 12,
          padding: 24,
          background: "var(--surface)",
        }}
      >
        <div
          role="grid"
          onKeyDown={onKeyDown}
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(200px, 1fr) repeat(3, 1fr)",
            border: "1px solid var(--border-subtle)",
            borderRadius: 8,
            overflow: "hidden",
          }}
        >
          <div role="columnheader" style={{ height: 44, background: "var(--paper-raised)" }} />
          {MATRIX.attacks.map((a) => (
            <div
              key={a.key}
              role="columnheader"
              style={{
                height: 44,
                background: "var(--paper-raised)",
                display: "grid",
                placeItems: "center",
                borderLeft: "1px solid var(--border-subtle)",
              }}
            >
              <span style={{ fontFamily: "var(--font-sans)", fontSize: 13, fontWeight: 600, letterSpacing: "0.06em", fontVariant: "small-caps", color: a.color }}>
                {a.label}
              </span>
            </div>
          ))}

          {MATRIX.defenses.map((d, di) => {
            const rowTopBorder = "1px solid var(--border-subtle)";
            return (
              <Fragment key={d.key}>
                <div
                  role="rowheader"
                  style={{
                    height: 60,
                    display: "flex",
                    alignItems: "center",
                    padding: "0 18px",
                    borderTop: rowTopBorder,
                    background: "var(--surface)",
                    gap: 8,
                  }}
                >
                  {d.ours ? (
                    <>
                      <span style={{ fontFamily: "var(--font-sans)", fontSize: 15, fontWeight: 600, fontVariant: "small-caps", letterSpacing: "0.04em", color: "var(--ink-900)" }}>
                        MemSAD
                      </span>
                      <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-muted)" }}>(ours)</span>
                    </>
                  ) : (
                    <span style={{ fontFamily: "var(--font-serif)", fontSize: 17, color: "var(--text-primary)", fontWeight: 400 }}>
                      {d.label}
                    </span>
                  )}
                </div>
                {MATRIX.attacks.map((a) => {
                  const k = `${d.key}-${a.key}`;
                  const v = MATRIX.values[d.key][a.key];
                  const ci = MATRIX.cis[k];
                  return (
                    <div key={k} style={{ borderTop: rowTopBorder, borderLeft: "1px solid var(--border-subtle)" }}>
                      <Cell
                        value={v}
                        attack={a}
                        defense={d}
                        hasCi={!!ci}
                        ci={ci}
                        cellRef={(el) => { cellRefs.current[k] = el; }}
                        focused={di === 0 && a.key === "ap"}
                      />
                    </div>
                  );
                })}
              </Fragment>
            );
          })}
        </div>

        <div style={{ marginTop: 16, display: "flex", alignItems: "flex-start", gap: 20 }}>
          <span className="t-caps-strong" style={{ whiteSpace: "nowrap" }}>Fig. 4</span>
          <p className="t-body-s" style={{ margin: 0, color: "var(--text-secondary)", fontStyle: "italic" }}>
            n = 200 benign memories, 10 poison passages per attack, k = 5 retrieval,
            σ = 2.0 for <span className="sc">MemSAD</span>. Darker cells mean higher ASR-R — worse defense.
            Bottom row reports 95% Clopper–Pearson confidence intervals (see Appendix C for the full table).
          </p>
        </div>

        <div style={{ marginTop: 16, display: "flex", gap: 16, flexWrap: "wrap", alignItems: "center" }}>
          <span style={{ display: "inline-flex", alignItems: "center", gap: 8, fontFamily: "var(--font-sans)", fontSize: 11, color: "var(--text-secondary)", whiteSpace: "nowrap", letterSpacing: "0.02em" }}>
            <span style={{ width: 56, height: 14, borderRadius: 3, background: "linear-gradient(to right, rgba(196,112,112,0), rgba(196,112,112,0.45))", border: "1px solid var(--border-subtle)" }} />
            ASR-R · 0 → 1
          </span>
          {MATRIX.attacks.map((a) => (
            <span key={a.key} style={{ fontFamily: "var(--font-sans)", fontSize: 12, color: a.color, fontVariant: "small-caps", fontWeight: 600, letterSpacing: "0.04em", whiteSpace: "nowrap" }}>
              {a.label}
            </span>
          ))}
          <span style={{ display: "inline-flex", alignItems: "center", gap: 6, fontFamily: "var(--font-sans)", fontSize: 11, color: "var(--text-secondary)", whiteSpace: "nowrap" }}>
            <span style={{ color: "var(--text-muted)" }}>(ours)</span>
            MemSAD contribution
          </span>
        </div>
      </div>
    </section>
  );
}
