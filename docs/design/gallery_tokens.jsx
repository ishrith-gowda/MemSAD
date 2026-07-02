// memsad — design system gallery
// sections: tokens (color, type, space, radius, elevation, motion), components, shell

const Section = ({ id, eyebrow, title, description, children }) => (
  <section id={id} style={{ paddingTop: 80, paddingBottom: 16 }}>
    <header style={{ maxWidth: 760, marginBottom: 40 }}>
      {eyebrow && (
        <div className="caps" style={{ color: "var(--text-muted)", marginBottom: 12 }}>
          {eyebrow}
        </div>
      )}
      <h2 className="t-display-l" style={{ margin: "0 0 12px 0", color: "var(--text-primary)" }}>
        {title}
      </h2>
      {description && (
        <p className="t-body-l" style={{ margin: 0, color: "var(--text-secondary)", maxWidth: 640 }}>
          {description}
        </p>
      )}
    </header>
    {children}
  </section>
);

const Row = ({ title, children, note }) => (
  <div style={{ display: "grid", gridTemplateColumns: "200px 1fr", gap: 48, paddingTop: 24, paddingBottom: 24, borderTop: "1px solid var(--border-subtle)" }}>
    <div>
      <div className="t-body" style={{ color: "var(--text-primary)", fontWeight: 500 }}>{title}</div>
      {note && <div className="t-body-s" style={{ marginTop: 6, color: "var(--text-muted)" }}>{note}</div>}
    </div>
    <div>{children}</div>
  </div>
);

/* --------------------- token displays ---------------------- */

const Swatch = ({ name, value, semantic }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
    <div
      style={{
        width: 56, height: 56, borderRadius: 8,
        background: value,
        border: "1px solid var(--border-subtle)",
        flexShrink: 0,
      }}
    />
    <div style={{ display: "flex", flexDirection: "column", gap: 2, minWidth: 0 }}>
      <span className="t-mono-s" style={{ color: "var(--text-primary)" }}>{name}</span>
      <span className="t-mono-s" style={{ color: "var(--text-muted)" }}>{value}</span>
      {semantic && (
        <span className="t-body-s" style={{ color: "var(--text-secondary)" }}>{semantic}</span>
      )}
    </div>
  </div>
);

const ColorTokens = () => {
  const ink = [
    { name: "--ink-900", value: "#1A1A1D", semantic: "text-primary" },
    { name: "--ink-700", value: "#40434B", semantic: "text-secondary" },
    { name: "--ink-500", value: "#6B6F7A", semantic: "text-muted" },
  ];
  const paper = [
    { name: "--paper", value: "#FBFAF7", semantic: "surface" },
    { name: "--paper-raised", value: "#F4F1EA", semantic: "surface-raised" },
    { name: "--rule", value: "#E5E2DA", semantic: "border-subtle" },
  ];
  const accents = [
    { name: "--accent-blue", value: "#7EA6CF", semantic: "accent-minja" },
    { name: "--accent-red", value: "#C47070", semantic: "accent-agentpoison" },
    { name: "--accent-lilac", value: "#B5A8C9", semantic: "accent-injecmem" },
    { name: "--accent-green", value: "#8FBF7F", semantic: "accent-triggered" },
  ];
  const flags = [
    { name: "--flag-red", value: "#A84A4A", semantic: "flag-hard" },
    { name: "--flag-amber", value: "#C4923F", semantic: "flag-soft" },
  ];
  const col = { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 };
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 32 }}>
      <div>
        <div className="caps" style={{ color: "var(--text-muted)", marginBottom: 12 }}>Ink</div>
        <div style={col}>{ink.map((c) => <Swatch key={c.name} {...c} />)}</div>
      </div>
      <div>
        <div className="caps" style={{ color: "var(--text-muted)", marginBottom: 12 }}>Paper & rules</div>
        <div style={col}>{paper.map((c) => <Swatch key={c.name} {...c} />)}</div>
      </div>
      <div>
        <div className="caps" style={{ color: "var(--text-muted)", marginBottom: 12 }}>Attack / signal accents</div>
        <div style={col}>{accents.map((c) => <Swatch key={c.name} {...c} />)}</div>
      </div>
      <div>
        <div className="caps" style={{ color: "var(--text-muted)", marginBottom: 12 }}>Flags</div>
        <div style={col}>{flags.map((c) => <Swatch key={c.name} {...c} />)}</div>
      </div>
    </div>
  );
};

const TypeTokens = () => {
  const rows = [
    { name: "display-xl", cls: "t-display-xl", spec: "Source Serif 4 · 48 / 58 · −0.01em", sample: "Gradient-coupled anomaly detection" },
    { name: "display-l",  cls: "t-display-l",  spec: "Source Serif 4 · 32 / 38",            sample: "Threshold sweep across σ" },
    { name: "display-m",  cls: "t-display-m",  spec: "Source Serif 4 · 20 / 26",            sample: "Single-run retrieval preview" },
    { name: "body-l",     cls: "t-body-l",     spec: "Inter · 18 / 28",                      sample: "The detector flags writes whose embeddings sit unusually close to calibrated victim queries." },
    { name: "body",       cls: "t-body",       spec: "Inter · 16 / 25",                      sample: "At write time, every candidate entry is scored against the legitimate query distribution." },
    { name: "body-s",     cls: "t-body-s",     spec: "Inter · 14 / 21 · secondary",          sample: "ASR-R denotes attack success at retrieval, averaged across three query batches." },
    { name: "mono-m",     cls: "t-mono-m",     spec: "JetBrains Mono · 14 / 21",             sample: "cosine=0.842  score=2.14σ  threshold=0.720" },
    { name: "mono-s",     cls: "t-mono-s",     spec: "JetBrains Mono · 12 / 17 · muted",     sample: "AgentPoison / trigger=ab1k92 / k=5" },
  ];
  return (
    <div style={{ display: "flex", flexDirection: "column" }}>
      {rows.map((r, i) => (
        <div
          key={r.name}
          style={{
            display: "grid",
            gridTemplateColumns: "140px 260px 1fr",
            gap: 32,
            alignItems: "baseline",
            padding: "20px 0",
            borderTop: i === 0 ? "1px solid var(--border-subtle)" : "1px solid var(--border-subtle)",
          }}
        >
          <span className="t-mono-s" style={{ color: "var(--text-primary)" }}>{r.name}</span>
          <span className="t-mono-s" style={{ color: "var(--text-muted)" }}>{r.spec}</span>
          <span className={r.cls}>{r.sample}</span>
        </div>
      ))}
    </div>
  );
};

const SpacingTokens = () => {
  const steps = [2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128];
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {steps.map((s) => (
        <div key={s} style={{ display: "grid", gridTemplateColumns: "80px 1fr", gap: 16, alignItems: "center" }}>
          <span className="t-mono-s" style={{ color: "var(--text-muted)" }}>space-{s}</span>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{ height: 12, width: s, background: "var(--ink-900)", borderRadius: 2 }} />
            <span className="t-mono-s" style={{ color: "var(--text-secondary)" }}>{s}px</span>
          </div>
        </div>
      ))}
    </div>
  );
};

const RadiusTokens = () => {
  const steps = [4, 8, 12, 16, 24];
  return (
    <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
      {steps.map((s) => (
        <div key={s} style={{ display: "flex", flexDirection: "column", gap: 10, alignItems: "center" }}>
          <div
            style={{
              width: 92, height: 92,
              background: "var(--paper-raised)",
              border: "1px solid var(--border-subtle)",
              borderRadius: s,
            }}
          />
          <span className="t-mono-s" style={{ color: "var(--text-muted)" }}>radius-{s}</span>
        </div>
      ))}
    </div>
  );
};

const ElevationTokens = () => (
  <div style={{ display: "flex", gap: 24 }}>
    <div style={{ width: 220, height: 120, background: "var(--surface)", border: "1px solid var(--border-subtle)", borderRadius: 16, display: "grid", placeItems: "center" }}>
      <span className="t-mono-s" style={{ color: "var(--text-muted)" }}>elevation-0 · rest</span>
    </div>
    <div style={{ width: 220, height: 120, background: "var(--surface)", border: "1px solid var(--border-subtle)", borderRadius: 16, boxShadow: "var(--elevation-1)", display: "grid", placeItems: "center" }}>
      <span className="t-mono-s" style={{ color: "var(--text-muted)" }}>elevation-1 · hover</span>
    </div>
  </div>
);

const MotionTokens = () => (
  <div style={{ display: "flex", flexDirection: "column", gap: 8, fontFamily: "var(--font-mono)", fontSize: 13, color: "var(--text-secondary)" }}>
    <div>duration-base&nbsp;&nbsp;&nbsp;180ms</div>
    <div>duration-anomaly&nbsp;400ms</div>
    <div>ease-standard&nbsp;&nbsp;cubic-bezier(.22, 1, .36, 1)</div>
    <div style={{ marginTop: 12, color: "var(--text-muted)" }}>// no parallax, no autoplay, no scroll-hijack</div>
  </div>
);

Object.assign(window, { Section, Row, ColorTokens, TypeTokens, SpacingTokens, RadiusTokens, ElevationTokens, MotionTokens });
