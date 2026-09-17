# SaTML 2027 — the cleaner hedge (decision needed by 2026-09-22 AoE)

SaTML 2027 was not in the original cascade. It should be, and on the merits it is
a better fit for this paper than ICLR. This file exists so you can make that call
deliberately rather than by default.

## Why it matters here

| | ICLR 2027 | SaTML 2027 |
|---|---|---|
| abstract deadline | Sep 18 AoE | **Sep 22 AoE** |
| paper deadline | Sep 25 AoE | **Sep 29 AoE** |
| relative to NeurIPS notification (Sep 24) | abstract 6 days **before**, paper 1 day after | **both after** |
| dual-submission gray zone | small but real (ICLR blesses abstract-while-under-review; NeurIPS never addresses it) | **none — entirely after the notification** |
| page limit | 9 (10 at rebuttal/camera-ready) | 12 body |
| per-author submission cap | 20; 1 "no-eligible-author" paper | none stated |
| publication-history qualification bar | yes, and NeurIPS 2026 is explicitly too late to count | **none** |
| reviewer obligation | mandatory registration, >= 3 papers, desk rejection if none | conditional nomination, up to 3, only "if the volume requires it" |
| author list | changeable | **frozen at abstract registration** |
| decisions | Dec 16 2026 | Dec 16 2026 |
| conference | Apr 26-28 2027, California | early May 2027 |

## The case for SaTML

1. **Topical fit is better.** SaTML exists for security-and-ML work where attacks
   *and* certified defenses with formal guarantees are the core subject, not a
   tolerated minority. This paper is squarely in its scope; at ICLR it is a
   security paper in a general ML venue.
2. **No timeline ambiguity at all.** Both deadlines fall after the Sep 24
   notification, so there is never a moment where the full paper's status is
   unclear.
3. **12 pages of body instead of 9.** The manuscript is appendix-heavy and the
   reviewers who rejected it did so on proof details. Three extra body pages is
   where the repaired Corollary 17, the threshold-estimation bound, and the
   cross-principal scoping can live in the main text instead of an appendix
   reviewers are "not required to read."
4. **An interactive discussion and revision phase** (Nov 25 to Dec 9) — well
   suited to a paper whose weakness was proof presentation rather than substance.
5. **It sidesteps the reciprocal-reviewer machinery entirely**, including the
   quota interaction with your other ICLR paper.

## The case against

- Smaller and younger venue; less prestige than ICLR on a CV.
- The author list freezes Sep 22, before the NeurIPS outcome. Solo is solo either
  way here, so this costs you nothing in practice.
- Requires the anonymized submission to be ready 4 days after ICLR's.

## Recommendation

**Register both abstracts.** They are independent free options, both permitted
while under NeurIPS review, and registering neither commits you to submitting.
Then on Sep 24, when the outcome is known, submit the full paper to **exactly
one** — every venue's concurrent-submission clause forbids both.

My ranking if NeurIPS rejects: **SaTML first.** The extra 3 body pages and the
revision phase are worth more to *this* paper, in its current state, than ICLR's
prestige margin — because the thing that sank it was reviewers not being able to
verify proofs that lived in an appendix.

If NeurIPS accepts, let both lapse and go to camera-ready.

## Registration content

Reuse the title, abstract, keywords and TL;DR from
[`ABSTRACT_REGISTRATION.md`](ABSTRACT_REGISTRATION.md) unchanged. SaTML
additionally requires at abstract registration: "Tentative non-blank title,
tentative non-blank abstract, fixed authors, and fixed topics", plus all ORCIDs
via HotCRP.

**ORCID is required.** You do not currently have one recorded (the placeholder was
removed from `CITATION.cff`). Register at https://orcid.org before Sep 22 if you
take the SaTML option; it takes a few minutes and you will need it for ACM CCS
later regardless.

Note SaTML is double-blind and asks authors to avoid "publicly advertising their
work on social media" during review.
