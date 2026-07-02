// memsad — §4 matrix + §5 about

/* ---------- §4 data ---------- */

const MATRIX = {
  attacks: [
    { key: "ap", label: "AgentPoison", color: "#C47070" },
    { key: "mj", label: "MINJA",       color: "#7EA6CF" },
    { key: "im", label: "InjecMEM",    color: "#B5A8C9" },
  ],
  defenses: [
    { key: "none",  label: "No defense",     ours: false, serif: true },
    { key: "ppl",   label: "Perplexity",     ours: false, serif: true },
    { key: "sim",   label: "Similarity cap", ours: false, serif: true },
    { key: "llms",  label: "LLM sanitizer",  ours: false, serif: true },
    { key: "mem",   label: "MemSAD",         ours: true,  serif: false },
  ],
  // rows x cols = defenses x attacks
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

/* ---------- matrix cell ---------- */

const Cell = ({ value, attack, defense, hasCi, ci, cellRef, focused }) => {
  const [hover, setHover] = React.useState(false);
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
        height: 56,
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
            boxShadow: "0 2px 8px rgba(0,0,0,0.04)",
            pointerEvents: "none",
            zIndex: 10,
            minWidth: 220,
            whiteSpace: "nowrap",
          }}
        >
          <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, lineHeight: 1.5 }}>
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
};

/* ---------- §4 section ---------- */

const MatrixSection = () => {
  // arrow-key grid nav
  const cellRefs = React.useRef({});
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
    <section id="matrix" style={{ background: "var(--surface)", padding: "96px 80px", borderTop: "1px solid var(--border-subtle)" }}>
      <div style={{ maxWidth: 1280, margin: "0 auto" }}>
        <div style={{ color: "var(--text-muted)", fontSize: 11, fontWeight: 600, letterSpacing: "0.14em", textTransform: "uppercase", marginBottom: 10, fontFamily: "var(--font-sans)" }}>
          §4 · MATRIX
        </div>
        <h2 style={{ fontFamily: "var(--font-serif)", fontSize: 32, lineHeight: 1.15, margin: "0 0 14px 0", fontWeight: 400, letterSpacing: "-0.005em", color: "var(--text-primary)" }}>
          Attack · defense grid
        </h2>
        <p style={{ margin: 0, color: "var(--text-secondary)", maxWidth: "62ch", fontSize: 16, lineHeight: 1.6, fontFamily: "var(--font-sans)" }}>
          Every attack crossed with every defense. Cell values are ASR-R — the fraction of victim queries whose top-k retrieval contains a poisoned passage. Lower is better. Ours is the last row.
        </p>

        {/* matrix card */}
        <div style={{ marginTop: 40, border: "1px solid var(--border-subtle)", borderRadius: 10, padding: 24, background: "var(--surface)" }}>
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
            {/* header row */}
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
                <span style={{ fontFamily: "var(--font-mono)", fontSize: 13, fontWeight: 600, letterSpacing: "0.06em", fontVariant: "small-caps", color: a.color }}>
                  {a.label}
                </span>
              </div>
            ))}

            {/* data rows */}
            {MATRIX.defenses.map((d, di) => {
              const isOurs = d.ours;
              const rowTopBorder = isOurs ? "1px solid var(--rule)" : "1px solid var(--border-subtle)";
              return (
                <React.Fragment key={d.key}>
                  <div
                    role="rowheader"
                    style={{
                      height: 56,
                      display: "flex",
                      alignItems: "center",
                      padding: "0 18px",
                      borderTop: rowTopBorder,
                      background: "var(--surface)",
                      gap: 8,
                    }}
                  >
                    {isOurs ? (
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
                    const ciKey = `${d.key}-${a.key}`;
                    const ci = MATRIX.cis[ciKey];
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
                </React.Fragment>
              );
            })}
          </div>

          {/* caption */}
          <div style={{ marginTop: 16, fontFamily: "var(--font-mono)", fontSize: 12, color: "var(--text-muted)", lineHeight: 1.7 }}>
            n = 200 benign memories · 10 poison passages per attack · k = 5 retrieval · σ = 2.0 for MemSAD<br />
            darker cell = higher ASR-R = worse defense. MemSAD row reports 95% clopper–pearson CIs; see appendix C for full table.
          </div>

          {/* legend */}
          <div style={{ marginTop: 16, display: "flex", gap: 16, flexWrap: "wrap", alignItems: "center" }}>
            <span style={{ display: "inline-flex", alignItems: "center", gap: 8, fontFamily: "var(--font-mono)", fontSize: 12, color: "var(--text-secondary)", whiteSpace: "nowrap" }}>
              <span style={{ width: 56, height: 16, borderRadius: 3, background: "linear-gradient(to right, rgba(196,112,112,0), rgba(196,112,112,0.45))", border: "1px solid var(--border-subtle)" }} />
              ASR-R &nbsp;0 → 1
            </span>
            {MATRIX.attacks.map((a) => (
              <span key={a.key} style={{ fontFamily: "var(--font-mono)", fontSize: 12, color: a.color, fontVariant: "small-caps", fontWeight: 600, letterSpacing: "0.04em", whiteSpace: "nowrap" }}>
                {a.label}
              </span>
            ))}
            <span style={{ display: "inline-flex", alignItems: "center", gap: 6, fontFamily: "var(--font-mono)", fontSize: 12, color: "var(--text-secondary)", whiteSpace: "nowrap" }}>
              <span style={{ color: "var(--text-muted)" }}>(ours)</span>
              MemSAD contribution
            </span>
          </div>
        </div>
      </div>
    </section>
  );
};

/* ---------- §5 about ---------- */

const BIBTEX = `@inproceedings{memsad2026,
  title   = {Detecting Memory Poisoning at Write Time:
             A Semantic Anomaly Defense for
             Retrieval-Augmented Agents},
  author  = {Anonymous Authors},
  booktitle = {Advances in Neural Information
               Processing Systems},
  year    = {2026},
  note    = {under review}
}`;

const CiteCard = () => {
  const [copied, setCopied] = React.useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(BIBTEX);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch (e) {
      // fallback: select text
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    }
  };
  return (
    <div style={{ border: "1px solid var(--border-subtle)", borderRadius: 10, padding: 24, background: "var(--surface)" }}>
      <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, fontWeight: 600, letterSpacing: "0.14em", color: "var(--text-muted)", textTransform: "uppercase", marginBottom: 14 }}>
        CITE THIS WORK
      </div>
      <pre style={{
        margin: 0,
        fontFamily: "var(--font-mono)",
        fontSize: 13,
        lineHeight: 1.6,
        color: "var(--text-primary)",
        background: "var(--paper-raised)",
        border: "1px solid var(--border-subtle)",
        borderRadius: 8,
        padding: 16,
        overflowX: "auto",
        whiteSpace: "pre",
      }}>
{BIBTEX}
      </pre>
      <div style={{ marginTop: 16, display: "flex", gap: 8, alignItems: "center" }}>
        <button
          onClick={copy}
          style={{
            appearance: "none",
            background: "var(--paper)",
            color: "var(--ink-900)",
            border: "1px solid var(--border-subtle)",
            borderRadius: 8,
            padding: "8px 14px",
            fontFamily: "var(--font-sans)",
            fontSize: 13,
            fontWeight: 500,
            cursor: "pointer",
            transition: "border-color 180ms var(--ease-standard)",
          }}
        >
          {copied ? "Copied" : "Copy BibTeX"}
        </button>
        <a
          href="/paper.pdf"
          style={{
            appearance: "none",
            background: "var(--paper)",
            color: "var(--ink-900)",
            border: "1px solid var(--border-subtle)",
            borderRadius: 8,
            padding: "8px 14px",
            fontFamily: "var(--font-sans)",
            fontSize: 13,
            fontWeight: 500,
            textDecoration: "none",
          }}
        >
          Download PDF
        </a>
      </div>
    </div>
  );
};

const DoubleBlindNotice = () => (
  <div style={{
    marginTop: 16,
    background: "var(--paper-raised)",
    borderLeft: "4px solid var(--flag-amber)",
    borderTopRightRadius: 10,
    borderBottomRightRadius: 10,
    padding: 16,
  }}>
    <div style={{
      fontFamily: "var(--font-mono)",
      fontSize: 11,
      fontWeight: 600,
      letterSpacing: "0.14em",
      color: "var(--text-primary)",
      textTransform: "uppercase",
      marginBottom: 8,
    }}>
      UNDER REVIEW · DOUBLE-BLIND
    </div>
    <p style={{ margin: 0, fontFamily: "var(--font-sans)", fontSize: 14, lineHeight: 1.6, color: "var(--ink-700)" }}>
      Author identities, affiliations, and acknowledgements are redacted pending reviewer assignment. This demo artifact is released anonymously. Please do not attempt to deanonymise.
    </p>
  </div>
);

const MethodologyCard = () => (
  <div style={{ border: "1px solid var(--border-subtle)", borderRadius: 10, padding: 24, background: "var(--surface)" }}>
    <h3 style={{ fontFamily: "var(--font-serif)", fontSize: 22, fontWeight: 400, margin: "0 0 12px 0", color: "var(--text-primary)", letterSpacing: "-0.005em" }}>
      Methodology, in one paragraph
    </h3>
    <p style={{ margin: 0, fontFamily: "var(--font-sans)", fontSize: 15, lineHeight: 1.65, color: "var(--ink-700)" }}>
      <span className="sc" style={{ fontWeight: 600, color: "var(--text-primary)" }}>MemSAD</span> calibrates a Gaussian over cosine similarities between legitimate victim queries and a clean memory sample. At write time, each candidate entry is scored against this distribution; entries whose combined score (½·max + ½·mean over the top-k nearest queries) exceeds μ + k·σ are rejected. The gradient-coupling result (Theorem 1) shows that an adversary minimising retrieval loss necessarily increases this score — so evasion requires gradient tension against the attacker's own objective.
    </p>
    <div style={{ marginTop: 18, display: "flex", gap: 8, flexWrap: "wrap" }}>
      {["write-time", "zero-cost at read", "encoder-agnostic"].map((t) => (
        <span key={t} style={{
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          color: "#365A83",
          background: "rgba(126,166,207,0.12)",
          border: "1px solid rgba(126,166,207,0.45)",
          padding: "4px 10px",
          borderRadius: 4,
          fontWeight: 500,
          letterSpacing: "0.02em",
        }}>
          {t}
        </span>
      ))}
    </div>
  </div>
);

const ReproCard = () => {
  const [tip, setTip] = React.useState(false);
  return (
    <div style={{ border: "1px solid var(--border-subtle)", borderRadius: 10, padding: 24, background: "var(--surface)" }}>
      <h3 style={{ fontFamily: "var(--font-serif)", fontSize: 22, fontWeight: 400, margin: "0 0 14px 0", color: "var(--text-primary)", letterSpacing: "-0.005em" }}>
        Reproduce
      </h3>
      <dl style={{ margin: 0, display: "grid", gridTemplateColumns: "100px 1fr", rowGap: 10, columnGap: 16 }}>
        <dt style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-muted)", letterSpacing: "0.12em", textTransform: "uppercase", alignSelf: "center" }}>repo</dt>
        <dd
          style={{ margin: 0, position: "relative", cursor: "not-allowed" }}
          onMouseEnter={() => setTip(true)}
          onMouseLeave={() => setTip(false)}
        >
          <span style={{
            fontFamily: "var(--font-mono)",
            fontSize: 13,
            color: "var(--text-muted)",
            textDecoration: "underline",
            textDecorationStyle: "dashed",
            textUnderlineOffset: 3,
            textDecorationColor: "var(--border-subtle)",
          }}>
            &lt;anonymous&gt;/memory-agent-security
          </span>
          {tip && (
            <span style={{
              position: "absolute",
              left: 0,
              top: "calc(100% + 4px)",
              background: "var(--ink-900)",
              color: "var(--paper)",
              fontFamily: "var(--font-sans)",
              fontSize: 11,
              padding: "4px 8px",
              borderRadius: 4,
              whiteSpace: "nowrap",
              zIndex: 5,
            }}>
              released after review
            </span>
          )}
        </dd>
        <dt style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-muted)", letterSpacing: "0.12em", textTransform: "uppercase", alignSelf: "center" }}>encoder</dt>
        <dd style={{ margin: 0, fontFamily: "var(--font-mono)", fontSize: 13, color: "var(--text-primary)" }}>
          sentence-transformers/all-MiniLM-L6-v2
        </dd>
        <dt style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-muted)", letterSpacing: "0.12em", textTransform: "uppercase", alignSelf: "center" }}>seed</dt>
        <dd style={{ margin: 0, fontFamily: "var(--font-mono)", fontSize: 13, color: "var(--text-primary)", fontVariantNumeric: "tabular-nums" }}>
          42 · 13 · 7 · 99 · 101 &nbsp;<span style={{ color: "var(--text-muted)" }}>(5-seed SIR protocol)</span>
        </dd>
      </dl>
      <div style={{ marginTop: 16, fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-muted)" }}>
        figures and tables were regenerated from src/scripts/ on 2026-04-21.
      </div>
    </div>
  );
};

const AboutSection = () => (
  <section id="about" style={{ background: "var(--surface)", padding: "96px 80px", borderTop: "1px solid var(--border-subtle)" }}>
    <div style={{ maxWidth: 1280, margin: "0 auto" }}>
      <div style={{ color: "var(--text-muted)", fontSize: 11, fontWeight: 600, letterSpacing: "0.14em", textTransform: "uppercase", marginBottom: 10, fontFamily: "var(--font-sans)" }}>
        §5 · ABOUT
      </div>
      <h2 style={{ fontFamily: "var(--font-serif)", fontSize: 32, lineHeight: 1.15, margin: "0 0 14px 0", fontWeight: 400, letterSpacing: "-0.005em", color: "var(--text-primary)" }}>
        About this artifact
      </h2>

      <div className="about-grid" style={{ marginTop: 40, display: "grid", gridTemplateColumns: "55fr 45fr", columnGap: 32, alignItems: "start" }}>
        <div>
          <CiteCard />
          <DoubleBlindNotice />
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <MethodologyCard />
          <ReproCard />
        </div>
      </div>
    </div>
  </section>
);

/* ---------- page ---------- */

ReactDOM.createRoot(document.getElementById("root")).render(
  <>
    <MatrixSection />
    <AboutSection />
  </>
);
