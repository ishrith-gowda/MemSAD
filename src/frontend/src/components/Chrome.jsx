import { NavLink, useLocation } from "react-router-dom";

/*
  page chrome: top bar + running side rail + folio footer.
  the rail and folio are what give the design its "journal" feel.
*/

const NAV = [
  { to: "/",        label: "Hero",        short: "§1 · hero" },
  { to: "/demo",    label: "Demo",        short: "§2 · single-run" },
  { to: "/sweep",   label: "Sweep",       short: "§3 · threshold" },
  { to: "/matrix",  label: "Matrix",      short: "§4 · matrix" },
  { to: "/about",   label: "About",       short: "§5 · about" },
];

function TopBar() {
  return (
    <header
      style={{
        position: "sticky",
        top: 0,
        zIndex: 20,
        background: "color-mix(in oklab, var(--surface) 92%, transparent)",
        backdropFilter: "saturate(1.2) blur(8px)",
        WebkitBackdropFilter: "saturate(1.2) blur(8px)",
        borderBottom: "1px solid var(--border-subtle)",
      }}
    >
      <div
        style={{
          maxWidth: "var(--page-max)",
          margin: "0 auto",
          padding: "14px var(--page-gutter)",
          display: "grid",
          gridTemplateColumns: "1fr auto 1fr",
          alignItems: "center",
          gap: 32,
        }}
      >
        {/* wordmark */}
        <NavLink to="/" style={{ textDecoration: "none", display: "inline-flex", alignItems: "baseline", gap: 10 }}>
          <span
            className="t-display-m"
            style={{ color: "var(--text-primary)", letterSpacing: "-0.01em" }}
          >
            Mem<span style={{ fontVariant: "small-caps", fontFeatureSettings: '"smcp"' }}>SAD</span>
          </span>
          <span className="t-mono-s" style={{ color: "var(--text-faint)" }}>v0.3</span>
        </NavLink>

        {/* section nav */}
        <nav style={{ display: "flex", gap: 6, alignItems: "center" }}>
          {NAV.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.to === "/"}
              style={({ isActive }) => ({
                padding: "6px 12px",
                borderRadius: 999,
                fontFamily: "var(--font-sans)",
                fontSize: 13,
                fontWeight: 500,
                letterSpacing: "0.01em",
                color: isActive ? "var(--text-primary)" : "var(--text-muted)",
                background: isActive ? "var(--surface-raised)" : "transparent",
                textDecoration: "none",
                transition: "color var(--duration-base) var(--ease-standard), background var(--duration-base) var(--ease-standard)",
              })}
            >
              {n.label}
            </NavLink>
          ))}
        </nav>

        {/* venue badge */}
        <div style={{ justifySelf: "end", display: "inline-flex", alignItems: "center", gap: 8 }}>
          <span
            className="t-caps"
            style={{
              padding: "6px 10px",
              border: "1px solid var(--border)",
              borderRadius: 4,
              color: "var(--text-primary)",
              fontWeight: 700,
              letterSpacing: "0.12em",
            }}
          >
            NeurIPS 2026
          </span>
          <span className="t-mono-s" style={{ color: "var(--text-muted)" }}>under review</span>
        </div>
      </div>
    </header>
  );
}

function SideRail({ section }) {
  return (
    <aside
      aria-hidden="true"
      style={{
        position: "fixed",
        left: 18,
        top: "50%",
        transform: "translateY(-50%) rotate(180deg)",
        writingMode: "vertical-rl",
        fontFamily: "var(--font-sans)",
        fontSize: 10,
        fontWeight: 600,
        letterSpacing: "0.22em",
        textTransform: "uppercase",
        color: "var(--text-faint)",
        pointerEvents: "none",
        zIndex: 5,
      }}
    >
      {section}  ·  memsad research artifact  ·  double-blind
    </aside>
  );
}

function Folio() {
  const location = useLocation();
  const current = NAV.find((n) => n.to === location.pathname) || NAV[0];
  return (
    <footer
      style={{
        borderTop: "1px solid var(--border)",
        marginTop: 64,
        padding: "28px var(--page-gutter) 40px",
      }}
    >
      <div
        style={{
          maxWidth: "var(--page-max)",
          margin: "0 auto",
          display: "grid",
          gridTemplateColumns: "1fr auto 1fr",
          alignItems: "baseline",
          gap: 32,
        }}
      >
        <div className="t-mono-s" style={{ color: "var(--text-muted)" }}>
          {current.short}
        </div>
        <div className="t-mono-s" style={{ color: "var(--text-faint)", textAlign: "center", letterSpacing: "0.1em" }}>
          mem<span style={{ fontVariant: "small-caps" }}>sad</span> — a semantic anomaly defense for agent memory
        </div>
        <div className="t-mono-s" style={{ color: "var(--text-muted)", textAlign: "right" }}>
          preprint · do not circulate
        </div>
      </div>
    </footer>
  );
}

export default function Chrome({ children }) {
  const location = useLocation();
  const current = NAV.find((n) => n.to === location.pathname) || NAV[0];
  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      <TopBar />
      <SideRail section={current.short} />
      <main style={{ flex: 1 }}>{children}</main>
      <Folio />
    </div>
  );
}
