import { useState } from "react";

/*
  §5 — about the artifact.
  cite card + double-blind notice + methodology + repro details.
*/

const BIBTEX = `@inproceedings{memsad2026,
  title     = {{MemSAD}: Gradient-Coupled Anomaly
               Detection for Memory Poisoning in
               Retrieval-Augmented Agents},
  author    = {Anonymous Author(s)},
  booktitle = {Advances in Neural Information
               Processing Systems},
  year      = {2026},
  note      = {under review}
}`;

function CiteCard() {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(BIBTEX);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch (e) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    }
  };
  return (
    <div style={{ border: "1px solid var(--border)", borderRadius: 12, padding: 24, background: "var(--surface)" }}>
      <div className="t-caps" style={{ marginBottom: 14 }}>Cite this work</div>
      <pre
        style={{
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
        }}
      >
{BIBTEX}
      </pre>
      <div style={{ marginTop: 16, display: "flex", gap: 8, alignItems: "center" }}>
        <button
          onClick={copy}
          style={{
            appearance: "none",
            background: "var(--paper)",
            color: "var(--ink-900)",
            border: "1px solid var(--border)",
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
            border: "1px solid var(--border)",
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
}

function DoubleBlindNotice() {
  return (
    <div
      style={{
        marginTop: 16,
        background: "var(--paper-raised)",
        borderLeft: "4px solid var(--flag-amber)",
        borderTopRightRadius: 10,
        borderBottomRightRadius: 10,
        padding: "16px 18px",
      }}
    >
      <div className="t-caps-strong" style={{ marginBottom: 8 }}>
        Under review · double-blind
      </div>
      <p style={{ margin: 0, fontFamily: "var(--font-serif)", fontSize: 15, lineHeight: 1.6, color: "var(--ink-700)" }}>
        Author identities, affiliations, and acknowledgements are redacted pending reviewer assignment.
        This demo artifact is released anonymously. Please do not attempt to deanonymise.
      </p>
    </div>
  );
}

function MethodologyCard() {
  return (
    <div style={{ border: "1px solid var(--border)", borderRadius: 12, padding: 24, background: "var(--surface)" }}>
      <h3 style={{ fontFamily: "var(--font-display)", fontSize: 24, fontWeight: 500, margin: "0 0 12px 0", color: "var(--text-primary)", letterSpacing: "-0.005em" }}>
        Methodology, in one paragraph.
      </h3>
      <p style={{ margin: 0, fontFamily: "var(--font-serif)", fontSize: 15, lineHeight: 1.65, color: "var(--ink-700)" }}>
        <span className="sc" style={{ fontWeight: 600, color: "var(--text-primary)" }}>MemSAD</span> calibrates a
        Gaussian over cosine similarities between legitimate victim queries and a clean memory sample. At
        write time, each candidate entry is scored against this distribution; entries whose combined score
        (<span style={{ fontFamily: "var(--font-mono)" }}>½·max + ½·mean</span> over the top-k nearest queries)
        exceed <span style={{ fontFamily: "var(--font-mono)" }}>μ + k·σ</span> are rejected. The gradient-coupling
        result (Theorem 1) shows that an adversary minimising retrieval loss necessarily <em>increases</em>
        this score — so evasion requires gradient tension against the attacker's own objective.
      </p>
      <div style={{ marginTop: 18, display: "flex", gap: 8, flexWrap: "wrap" }}>
        {["write-time", "zero-cost at read", "encoder-agnostic"].map((t) => (
          <span
            key={t}
            style={{
              fontFamily: "var(--font-sans)",
              fontSize: 11,
              color: "#365A83",
              background: "rgba(126,166,207,0.12)",
              border: "1px solid rgba(126,166,207,0.45)",
              padding: "4px 10px",
              borderRadius: 4,
              fontWeight: 500,
              letterSpacing: "0.04em",
            }}
          >
            {t}
          </span>
        ))}
      </div>
    </div>
  );
}

function ReproCard() {
  const [tip, setTip] = useState(false);
  return (
    <div style={{ border: "1px solid var(--border)", borderRadius: 12, padding: 24, background: "var(--surface)" }}>
      <h3 style={{ fontFamily: "var(--font-display)", fontSize: 24, fontWeight: 500, margin: "0 0 14px 0", color: "var(--text-primary)", letterSpacing: "-0.005em" }}>
        Reproduce.
      </h3>
      <dl style={{ margin: 0, display: "grid", gridTemplateColumns: "100px 1fr", rowGap: 10, columnGap: 16 }}>
        <dt className="t-caps" style={{ alignSelf: "center" }}>repo</dt>
        <dd
          style={{ margin: 0, position: "relative", cursor: "not-allowed" }}
          onMouseEnter={() => setTip(true)}
          onMouseLeave={() => setTip(false)}
        >
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 13,
              color: "var(--text-muted)",
              textDecoration: "underline",
              textDecorationStyle: "dashed",
              textUnderlineOffset: 3,
              textDecorationColor: "var(--border-subtle)",
            }}
          >
            &lt;anonymous&gt;/memory-agent-security
          </span>
          {tip && (
            <span
              style={{
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
              }}
            >
              released after review
            </span>
          )}
        </dd>

        <dt className="t-caps" style={{ alignSelf: "center" }}>encoder</dt>
        <dd style={{ margin: 0, fontFamily: "var(--font-mono)", fontSize: 13, color: "var(--text-primary)" }}>
          sentence-transformers/all-MiniLM-L6-v2
        </dd>

        <dt className="t-caps" style={{ alignSelf: "center" }}>seed</dt>
        <dd style={{ margin: 0, fontFamily: "var(--font-mono)", fontSize: 13, color: "var(--text-primary)", fontVariantNumeric: "tabular-nums" }}>
          42 · 13 · 7 · 99 · 101 &nbsp;<span style={{ color: "var(--text-muted)" }}>(5-seed SIR protocol)</span>
        </dd>
      </dl>
      <div style={{ marginTop: 16, fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-muted)" }}>
        figures and tables were regenerated from src/scripts/ on 2026-04-21.
      </div>
    </div>
  );
}

export default function About() {
  return (
    <section
      style={{
        maxWidth: "var(--page-max)",
        margin: "0 auto",
        padding: "var(--section-pad-y) var(--page-gutter)",
      }}
    >
      <div className="t-caps" style={{ marginBottom: 14 }}>
        §5 · about · reproducibility · contact
      </div>
      <hr style={{ marginBottom: 32 }} />

      <h1 className="t-display-l" style={{ margin: "0 0 14px 0" }}>
        About this artifact.
      </h1>
      <p className="t-lede" style={{ margin: 0 }}>
        <span className="sc" style={{ fontWeight: 600, color: "var(--text-primary)" }}>MemSAD</span> is a
        semantic anomaly defense for agent memory, submitted to NeurIPS 2026 under double-blind review. This
        demo runs the same code path as the paper's experiments against a synthetic 200-entry corpus.
      </p>

      <div
        style={{
          marginTop: 40,
          display: "grid",
          gridTemplateColumns: "minmax(0, 55fr) minmax(0, 45fr)",
          columnGap: 32,
          alignItems: "start",
        }}
      >
        <div>
          <CiteCard />
          <DoubleBlindNotice />
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <MethodologyCard />
          <ReproCard />
        </div>
      </div>
    </section>
  );
}
