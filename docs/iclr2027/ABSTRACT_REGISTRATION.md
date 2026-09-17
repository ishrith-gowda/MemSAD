# ICLR 2027 — abstract registration (PASTE-READY)

**DEADLINE: 2026-09-18 AoE. Author action required — I have no OpenReview access.**

Registering costs nothing and is the only way to keep ICLR 2027 open: you cannot
submit the full paper on Sep 25 without an abstract registered by Sep 18. If
NeurIPS accepts on Sep 24, simply never submit — an abstract that lapses before
the paper deadline leaves no public trace (only *post-deadline* withdrawals go
to OpenReview's public "withdrawn papers" section).

ICLR states the case directly in its author guidelines. Asked whether a paper
currently under review elsewhere may have its abstract registered:
> "Yes, this is allowed. (We will only do duplicate-submission checks for full submissions.)"

---

## 1. Title

```
MemSAD: Gradient-Coupled Anomaly Detection for Memory Poisoning in Retrieval-Augmented Agents
```

## 2. Abstract

Paste verbatim. 276 words, pure ASCII (no em-dashes or unicode minus, matching the
house style). No links, no self-identifying reference, no first-person
reference to the arXiv version (that combination is what turns a permitted preprint
into an ICLR desk rejection).

```
Persistent external memory lets LLM agents carry context across sessions, but its
security properties remain formally uncharacterized. We formalize memory poisoning
on retrieval-augmented agents as a Stackelberg game and evaluate three attack
classes with escalating access assumptions. Correcting an evaluation-protocol
inconsistency relative to the triggered-query specification of Chen et al. (2024),
we show that faithful evaluation raises measured attack success fourfold, from
ASR 0.25 to 1.00.

Our main contribution is MemSAD, a calibration-based write-time defense resting on
a gradient coupling theorem: under encoder regularity, the max-cosine anomaly-score
gradient coincides with the retrieval-objective gradient, so any continuous
perturbation that lowers detection necessarily degrades retrieval rank. An attacker
cannot evade without surrendering the utility it attacks for. We measure the
regularity condition rather than assuming it: across six encoders, a single synonym
substitution displaces the embedding by at most 0.39 (mean 0.08 to 0.15) on the
unit sphere.

Coupling yields a certified detection radius whose scope we delineate honestly
rather than claiming universality. On write-access attacks the certified gap is
large (mean +0.21, all poison detected); it is tight for query-only attacks and
vacuous under synonym substitution, the frontier we characterize with a discrete
synonym-invariance result. Randomized smoothing offers no alternative: in dimension
384 it certifies a radius of 0.008 against a 0.378 substitution shift, so it is
vacuous here, and coupling is not a special case of it.

We prove a matching Omega(1/rho^2) lower bound on calibration sample complexity via
Le Cam's method, and derive online regret bounds for rolling calibration at rate
O(sigma^(2/3) Delta^(1/3)). Because linearly combining scores breaks coupling, we
deploy a disjunctive composite that provably inherits the guarantee and empirically
dominates the linear combination.
```

## 3. Keywords

```
memory poisoning, retrieval-augmented generation, llm agents, certified defenses,
anomaly detection, ai security, adversarial robustness
```

## 4. TL;DR (if the form asks)

```
A write-time defense for LLM agent memory whose detection score is gradient-coupled
to the retrieval objective, so evading detection provably costs the attacker
retrieval rank -- with a certified radius, a matching minimax lower bound, and an
honest account of where the certificate is vacuous.
```

## 5. Primary / secondary area

Primary: **alignment, fairness, safety, privacy, and societal considerations**
Secondary: **datasets and benchmarks** or **learning theory**, whichever the form allows.

---

## 6. Compliance checks BEFORE you register

These are desk-rejection triggers, not style notes.

- [ ] **Your OpenReview profile is accurate and complete.** ICLR: "Eligibility will
      be determined based on OpenReview profiles. Incorrect information on your
      profile will be grounds for desk rejection."
- [ ] **Your professor's OpenReview profile on the OTHER ICLR 2027 paper lists a
      qualifying main-conference publication.** This is the one that can hurt you
      indirectly — see §7.
- [ ] Abstract contains no URL and no phrase identifying you as the arXiv author.
- [ ] Title matches the arXiv title (consistency is expected and fine).

## 7. Multi-submission compliance — you are clear, with one caveat

You asked whether submitting MemSAD alongside your already-committed coauthored
paper is allowed. **Yes.**

- ICLR 2027 cap: "No author may appear as a co-author on more than 20 papers." Two
  is far under.
- The restrictive rule is: "Each author may appear as a co-author on **at most one
  paper in which no author is an eligible reciprocal reviewer**." The allowance is
  **per author**; the test runs on **each paper's own co-author set**.

| paper | authors | any eligible author? | uses your allowance? |
|---|---|---|---|
| the committed coauthored paper | you + professor | yes (professor) | no |
| MemSAD | you, solo | no | **yes — this is your one** |

Exactly one. Compliant.

**The caveat that actually matters:** eligibility is "determined by papers accepted
for publication **by the abstract deadline**", and ICLR names your exact situation:
"if your only qualifying publication is an accepted NeurIPS 2026 publication, you
are not a qualified reciprocal reviewer." A NeurIPS acceptance on Sep 24 is six days
too late to count. So MemSAD lands in the no-eligible-author bucket regardless of
the NeurIPS outcome — which is fine, because it is your one allowed such paper.

**But** if the automated check cannot see a qualifying publication on your
professor's OpenReview profile, the coauthored paper *also* gets classified into
that bucket, you breach the quota with two such papers, and ICLR resolves it by
desk-rejecting **at random**: "we will randomly desk-reject excess submissions until
every author is under quota." You would not choose which paper dies. **Verify that
profile before Sep 18.** Workshop papers do not count; main conference only.

## 8. Reviewer registration — mandatory, do not skip

> "All authors will be notified to register as reviewers after the abstract
> deadline. If none of the authors are registered as a reviewer, it will result in
> desk rejection for the paper."

- Window opens after Sep 18, by notification. **Closing date is not published — register on receipt.**
- Load: **3 papers, not 6.** The 6-review tier triggers at 3+ submissions; you have 2.
- One registration by you at >= 3 papers satisfies both submissions.
- **Do not add a third ICLR 2027 submission** — that doubles the obligation to 6
  reviews and adds explicit desk-rejection exposure for low-quality reviews.

## 9. What happens next

| date | event | action |
|---|---|---|
| **Sep 18 AoE** | ICLR abstract deadline | **register (this file)** |
| Sep 22 AoE | SaTML 2027 abstract deadline | optional second hedge — see `SATML_OPTION.md` |
| Sep 24 | NeurIPS 2026 notification | decision day |
| Sep 25 AoE | ICLR full paper deadline | accept -> let it lapse; reject -> submit |
| Sep 29 AoE | SaTML 2027 paper deadline | the cleanest fallback; falls fully after notification |

## 10. Provenance

Every number in the abstract is transcribed from `docs/rebuttal/experiment_results.md`
and reproducible via `PYTHONPATH=src python3 src/scripts/rebuttal_e{1..6}_*.py`:

| claim in abstract | source |
|---|---|
| ASR 0.25 -> 1.00 fourfold | paper section 4, evaluation-protocol correction |
| encoder shift max 0.39, mean 0.08-0.15, six encoders | E5 |
| certified gap mean +0.21, all poison detected | E1 (agentpoison, 100% det) |
| tight for query-only, vacuous under synonym | E1 (minja gap ~0; synonym negative) |
| smoothing R=0.008 vs shift 0.378 in d=384 | E2 |
| disjunction dominates linear combination | E6 (minja 0.06->0.38, injecmem 0.00->0.50) |

The abstract deliberately drops the submitted version's claim that the certificate
holds "regardless of adversary strategy." That was an overclaim, it was conceded in
the NeurIPS response, and repeating it to a fresh reviewer pool would reintroduce
the exact objection that drew the reject.
