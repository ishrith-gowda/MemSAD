// memsad — primitive components
// button, badge, card, tabs, kbd

const Button = ({ variant = "primary", size = "md", children, onClick, disabled }) => {
  const sizes = {
    sm: { padding: "6px 12px", fontSize: 13, height: 30 },
    md: { padding: "8px 16px", fontSize: 14, height: 36 },
    lg: { padding: "10px 20px", fontSize: 15, height: 44 },
  };
  const variants = {
    primary: {
      background: "var(--ink-900)",
      color: "var(--paper)",
      border: "1px solid var(--ink-900)",
    },
    secondary: {
      background: "var(--paper)",
      color: "var(--ink-900)",
      border: "1px solid var(--border-subtle)",
    },
    ghost: {
      background: "transparent",
      color: "var(--ink-700)",
      border: "1px solid transparent",
    },
  };
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="msd-btn"
      style={{
        ...sizes[size],
        ...variants[variant],
        fontFamily: "var(--font-sans)",
        fontWeight: 500,
        borderRadius: "var(--radius-8)",
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.5 : 1,
        letterSpacing: 0,
        transition: `background var(--duration-base) var(--ease-standard), border-color var(--duration-base) var(--ease-standard), color var(--duration-base) var(--ease-standard)`,
        display: "inline-flex",
        alignItems: "center",
        gap: 8,
      }}
      data-variant={variant}
    >
      {children}
    </button>
  );
};

const Badge = ({ tone = "neutral", label, children }) => {
  // tone palette — maps to accents, keeps bg quiet
  const tones = {
    neutral: { bg: "var(--paper-raised)", fg: "var(--ink-700)", br: "var(--border-subtle)" },
    blue:    { bg: "rgba(126,166,207,0.12)", fg: "#365A83", br: "rgba(126,166,207,0.45)" },
    red:     { bg: "rgba(196,112,112,0.12)", fg: "#8A3A3A", br: "rgba(196,112,112,0.45)" },
    green:   { bg: "rgba(143,191,127,0.14)", fg: "#3F6A36", br: "rgba(143,191,127,0.5)" },
    lilac:   { bg: "rgba(181,168,201,0.16)", fg: "#584B73", br: "rgba(181,168,201,0.5)" },
    amber:   { bg: "rgba(196,146,63,0.12)", fg: "#7A5816", br: "rgba(196,146,63,0.45)" },
  };
  const t = tones[tone] || tones.neutral;
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        padding: "3px 8px",
        background: t.bg,
        color: t.fg,
        border: `1px solid ${t.br}`,
        borderRadius: "var(--radius-4)",
        fontSize: 12,
        lineHeight: 1.2,
        fontWeight: 500,
        fontFamily: "var(--font-sans)",
      }}
    >
      {label && (
        <span className="sc" style={{ fontWeight: 600, letterSpacing: "0.06em", opacity: 0.9 }}>
          {label}
        </span>
      )}
      {children && <span style={{ opacity: label ? 0.85 : 1 }}>{children}</span>}
    </span>
  );
};

const Card = ({ variant = "outlined", children, style, ...rest }) => {
  const variants = {
    flat: {
      background: "var(--surface)",
      border: "1px solid transparent",
    },
    outlined: {
      background: "var(--surface)",
      border: "1px solid var(--border-subtle)",
    },
    raised: {
      background: "var(--surface)",
      border: "1px solid var(--border-subtle)",
    },
  };
  return (
    <div
      className={`msd-card msd-card-${variant}`}
      style={{
        ...variants[variant],
        borderRadius: "var(--radius-16)",
        padding: 24,
        transition: `box-shadow var(--duration-base) var(--ease-standard), border-color var(--duration-base) var(--ease-standard)`,
        ...style,
      }}
      {...rest}
    >
      {children}
    </div>
  );
};

const Tabs = ({ items, value, onChange }) => {
  return (
    <div
      role="tablist"
      style={{
        display: "flex",
        gap: 24,
        borderBottom: "1px solid var(--border-subtle)",
      }}
    >
      {items.map((it) => {
        const active = it.value === value;
        return (
          <button
            key={it.value}
            role="tab"
            aria-selected={active}
            onClick={() => onChange(it.value)}
            style={{
              appearance: "none",
              background: "transparent",
              border: 0,
              padding: "10px 0",
              fontFamily: "var(--font-sans)",
              fontSize: 14,
              fontWeight: active ? 600 : 500,
              color: active ? "var(--text-primary)" : "var(--text-secondary)",
              cursor: "pointer",
              position: "relative",
              letterSpacing: 0,
            }}
          >
            {it.label}
            <span
              style={{
                position: "absolute",
                left: 0,
                right: 0,
                bottom: -1,
                height: 2,
                background: active ? "var(--text-primary)" : "transparent",
                transition: `background var(--duration-base) var(--ease-standard)`,
              }}
            />
          </button>
        );
      })}
    </div>
  );
};

const KBD = ({ children }) => (
  <kbd
    style={{
      display: "inline-block",
      fontFamily: "var(--font-mono)",
      fontSize: 12,
      lineHeight: 1,
      padding: "3px 6px",
      background: "var(--paper-raised)",
      border: "1px solid var(--border-subtle)",
      borderBottomWidth: 2,
      borderRadius: 4,
      color: "var(--text-secondary)",
      minWidth: 18,
      textAlign: "center",
    }}
  >
    {children}
  </kbd>
);

Object.assign(window, { Button, Badge, Card, Tabs, KBD });
