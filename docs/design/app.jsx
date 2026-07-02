// memsad — page shell (topbar + footer) + gallery composition

const Topbar = () => (
  <header
    style={{
      position: "sticky",
      top: 0,
      zIndex: 10,
      background: "rgba(251, 250, 247, 0.92)",
      backdropFilter: "saturate(140%) blur(10px)",
      WebkitBackdropFilter: "saturate(140%) blur(10px)",
      borderBottom: "1px solid var(--border-subtle)",
    }}
  >
    <div
      style={{
        maxWidth: 1200,
        margin: "0 auto",
        padding: "0 80px",
        height: 64,
        display: "grid",
        gridTemplateColumns: "1fr auto 1fr",
        alignItems: "center",
        gap: 24,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span
          style={{
            fontFamily: "var(--font-serif)",
            fontSize: 22,
            fontWeight: 500,
            letterSpacing: "-0.01em",
            color: "var(--text-primary)",
          }}
        >
          MemSAD
        </span>
        <span className="t-mono-s" style={{ color: "var(--text-muted)" }}>v0.3</span>
      </div>
      <nav style={{ display: "flex", gap: 28 }}>
        {[
          { label: "Demo",   href: "#demo" },
          { label: "Paper",  href: "#paper" },
          { label: "GitHub", href: "#github" },
          { label: "About",  href: "#about" },
        ].map((n) => (
          <a
            key={n.label}
            href={n.href}
            style={{
              textDecoration: "none",
              color: "var(--text-secondary)",
              fontSize: 14,
              fontWeight: 500,
            }}
          >
            {n.label}
          </a>
        ))}
      </nav>
      <div style={{ justifySelf: "end" }}>
        <Badge tone="neutral" label="NEURIPS 2026">under review</Badge>
      </div>
    </div>
  </header>
);

const Footer = () => (
  <footer
    style={{
      marginTop: 128,
      borderTop: "1px solid var(--border-subtle)",
      background: "var(--surface)",
    }}
  >
    <div
      style={{
        maxWidth: 1200,
        margin: "0 auto",
        padding: "20px 80px",
        display: "flex",
        alignItems: "center",
        justifyContent: "flex-start",
        gap: 8,
      }}
    >
      <span className="t-body-s" style={{ color: "var(--text-muted)" }}>
        MemSAD research artifact · <a href="#double-blind" style={{ color: "var(--text-secondary)" }}>double-blind submission notice</a>
      </span>
    </div>
  </footer>
);

const Intro = () => (
  <section style={{ paddingTop: 96, paddingBottom: 24 }}>
    <div style={{ maxWidth: 820 }}>
      <div className="caps" style={{ color: "var(--text-muted)", marginBottom: 16 }}>
        Design system · v0.3
      </div>
      <h1 className="t-display-xl" style={{ margin: "0 0 20px 0" }}>
        The visual vocabulary for a memory-poisoning defense.
      </h1>
      <p className="t-body-l" style={{ color: "var(--text-secondary)", margin: 0, maxWidth: 680 }}>
        Tokens, primitives, and a page shell for the <span className="sc" style={{ fontWeight: 600 }}>MemSAD</span> research artifact.
        Warm paper canvas, quiet serif headings, monospace for every number.
        Built to match a <span className="sc" style={{ fontWeight: 600 }}>NeurIPS</span> figure, not a SaaS landing page.
      </p>
      <div style={{ display: "flex", gap: 10, marginTop: 24, flexWrap: "wrap" }}>
        <Badge tone="red"><span className="sc">AgentPoison</span></Badge>
        <Badge tone="blue"><span className="sc">MINJA</span></Badge>
        <Badge tone="lilac"><span className="sc">InjecMEM</span></Badge>
        <Badge tone="neutral" label="ASR-R">primary metric</Badge>
      </div>
    </div>
  </section>
);

const TOC = () => {
  const items = [
    ["01", "Colour",      "#color"],
    ["02", "Typography",  "#type"],
    ["03", "Spacing",     "#spacing"],
    ["04", "Radius",      "#radius"],
    ["05", "Elevation",   "#elevation"],
    ["06", "Motion",      "#motion"],
    ["07", "Buttons",     "#buttons"],
    ["08", "Badges",      "#badges"],
    ["09", "Cards",       "#cards"],
    ["10", "Tabs",        "#tabs"],
    ["11", "KBD",         "#kbd"],
    ["12", "ScoreBar",    "#scorebar"],
    ["13", "PassageRow",  "#passagerow"],
    ["14", "Slider",      "#slider"],
    ["15", "MatrixCell",  "#matrix"],
    ["16", "Shell",       "#shell"],
  ];
  return (
    <nav
      style={{
        marginTop: 40,
        paddingTop: 20,
        paddingBottom: 8,
        borderTop: "1px solid var(--border-subtle)",
        display: "grid",
        gridTemplateColumns: "repeat(4, 1fr)",
        rowGap: 10,
        columnGap: 24,
      }}
    >
      {items.map(([n, label, href]) => (
        <a key={n} href={href} style={{ textDecoration: "none", display: "flex", gap: 10, alignItems: "baseline" }}>
          <span className="t-mono-s" style={{ color: "var(--text-muted)" }}>{n}</span>
          <span className="t-body-s" style={{ color: "var(--text-secondary)" }}>{label}</span>
        </a>
      ))}
    </nav>
  );
};

const ShellPreview = () => (
  <Card variant="outlined" style={{ padding: 0, overflow: "hidden" }}>
    {/* miniature topbar */}
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1fr auto 1fr",
        alignItems: "center",
        padding: "14px 24px",
        borderBottom: "1px solid var(--border-subtle)",
        background: "var(--surface)",
        gap: 16,
      }}
    >
      <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
        <span style={{ fontFamily: "var(--font-serif)", fontSize: 18, fontWeight: 500 }}>MemSAD</span>
        <span className="t-mono-s" style={{ color: "var(--text-muted)" }}>v0.3</span>
      </div>
      <nav style={{ display: "flex", gap: 22 }}>
        {["Demo", "Paper", "GitHub", "About"].map((l, i) => (
          <span key={l} className="t-body-s" style={{ color: i === 0 ? "var(--text-primary)" : "var(--text-secondary)", fontWeight: i === 0 ? 600 : 500 }}>{l}</span>
        ))}
      </nav>
      <div style={{ justifySelf: "end" }}>
        <Badge tone="neutral" label="NEURIPS 2026">under review</Badge>
      </div>
    </div>
    {/* content stub */}
    <div style={{ padding: 32, background: "var(--paper-raised)", minHeight: 160, display: "flex", alignItems: "center" }}>
      <div>
        <div className="caps" style={{ color: "var(--text-muted)", marginBottom: 10 }}>Hero region</div>
        <div className="t-display-m" style={{ color: "var(--text-primary)", maxWidth: 520 }}>
          A detector for memory-poisoning attacks on retrieval-augmented agents.
        </div>
      </div>
    </div>
    {/* footer */}
    <div style={{ padding: "14px 24px", borderTop: "1px solid var(--border-subtle)" }}>
      <span className="t-body-s" style={{ color: "var(--text-muted)" }}>
        MemSAD research artifact · <span style={{ textDecoration: "underline", textDecorationColor: "var(--border-subtle)", textUnderlineOffset: 3, color: "var(--text-secondary)" }}>double-blind submission notice</span>
      </span>
    </div>
  </Card>
);

const App = () => {
  return (
    <div>
      <Topbar />
      <main style={{ maxWidth: 1200, margin: "0 auto", padding: "0 80px" }}>
        <Intro />
        <TOC />

        <Section id="color"     eyebrow="01 · Colour"      title="Colour"        description="Warm near-white canvas, paper-raised surface for cards, single pastel per attack. High-contrast flag tones are reserved for anomaly signals.">
          <ColorTokens />
        </Section>

        <Section id="type"      eyebrow="02 · Typography"  title="Typography"    description="Source Serif 4 for headings, Inter for UI and body, JetBrains Mono for every numeric value so cosine similarities and σ readouts feel like tokens, not labels.">
          <TypeTokens />
        </Section>

        <Section id="spacing"   eyebrow="03 · Spacing"     title="Spacing"       description="Base-8 scale with 2 / 4 / 12 as fine-grain interstitials. No arbitrary values.">
          <SpacingTokens />
        </Section>

        <Section id="radius"    eyebrow="04 · Radius"      title="Radius"        description="16px is the default for cards and panels; 4 / 8 for chips and inputs; 24 only for hero-scale surfaces.">
          <RadiusTokens />
        </Section>

        <Section id="elevation" eyebrow="05 · Elevation"   title="Elevation"     description="Two tokens. Rest is flat. Hover lifts a card by a barely-perceptible 2px.">
          <ElevationTokens />
        </Section>

        <Section id="motion"    eyebrow="06 · Motion"      title="Motion"        description="One easing curve. Two durations. Anomaly fires at 400ms; everything else is 180ms.">
          <MotionTokens />
        </Section>

        <Section id="buttons"   eyebrow="07 · Buttons"     title="Button"        description="Primary, secondary, ghost. No icons inside buttons — text only.">
          <ButtonGallery />
        </Section>

        <Section id="badges"    eyebrow="08 · Badges"      title="Badge"         description="Neutral plus one chip per attack, plus flag tones. Optional small-caps leading label.">
          <BadgeGallery />
        </Section>

        <Section id="cards"     eyebrow="09 · Cards"       title="Card"          description="Flat, outlined, raised-on-hover. Same radius and padding across all three — only the rule and elevation change.">
          <CardGallery />
        </Section>

        <Section id="tabs"      eyebrow="10 · Tabs"        title="Tabs"          description="Underline only. Pill tabs read as marketing; we want navigation that disappears into the page.">
          <TabsGallery />
        </Section>

        <Section id="kbd"       eyebrow="11 · KBD"         title="Keycap"        description="Monospace, single-glyph, for the keyboard hints sprinkled through the demo surfaces.">
          <KBDGallery />
        </Section>

        <Section id="scorebar"  eyebrow="12 · ScoreBar"    title="ScoreBar"      description="Horizontal bar with threshold marker and monospace value. Fills from zero in 400ms, turns flag-red once the value crosses threshold on a poisoned row.">
          <ScoreBarGallery />
        </Section>

        <Section id="passagerow" eyebrow="13 · PassageRow" title="PassageRow"    description="One retrieved passage per row. Poisoned rows get a 2px left border in accent-agentpoison; anomaly chips sit right-aligned.">
          <PassageRowGallery />
        </Section>

        <Section id="slider"    eyebrow="14 · Slider"      title="Slider"        description="Native range input, restyled. Value reads out in monospace to the right.">
          <SliderGallery />
        </Section>

        <Section id="matrix"    eyebrow="15 · MatrixCell"  title="MatrixCell"    description="For the attack × defense evaluation. Value-weighted warm tint — higher ASR-R means worse, darker. Ours, last row, reads as light.">
          <MatrixGallery />
        </Section>

        <Section id="shell"     eyebrow="16 · Shell"       title="Page shell"    description="Top bar sticks; footer carries the double-blind notice. This is the skeleton every demo surface slots into.">
          <ShellPreview />
        </Section>
      </main>
      <Footer />
    </div>
  );
};

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
