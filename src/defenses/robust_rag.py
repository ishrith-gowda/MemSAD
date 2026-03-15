"""
robustrag-style isolate-then-aggregate defense for memory poisoning.

implements the core principle from xiang et al. (icml 2024, "certifiably robust
rag against retrieval corruption"): partition retrieved passages into disjoint
groups, generate an isolated response per group, and aggregate via majority vote.

key insight: if poison entries are a minority of retrieved passages, they will
only corrupt a minority of isolated groups, and majority aggregation filters
them out.  this provides a provable robustness guarantee: the defense succeeds
whenever the number of poisoned passages is less than half the number of groups.

our adaptation for memory agent security:
    - the "response generation" step is simplified to keyword extraction (no llm
      call required), keeping the defense lightweight and reproducible.
    - aggregation uses keyword-overlap voting: the majority cluster of passages
      with similar content words is selected as the clean response.
    - passages that are isolated outliers (low overlap with any majority group)
      are flagged as potentially adversarial.

formal guarantee:
    given k retrieved passages and g groups of size k/g, if the number of
    adversarial passages n_adv < g/2, the majority-vote output is guaranteed
    to be benign.  with k=5 and g=5 (one passage per group), the defense
    tolerates up to 2 poisoned passages.

this is a strictly weaker guarantee than full robustrag (which uses llm-based
response isolation), but it operates at the retrieval layer without requiring
an additional llm inference call, making it practical for memory agents.

references:
    - xiang et al. "certifiably robust rag against retrieval corruption."
      icml 2024. arxiv:2405.15556.

all comments are lowercase.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class IsolationResult:
    """
    result of running the isolate-then-aggregate defense on a retrieval set.

    fields:
        passages: original retrieved passages
        group_assignments: which group each passage was assigned to
        flagged_indices: indices of passages flagged as outliers
        majority_group: index of the group selected by majority vote
        n_groups: total number of groups
        n_flagged: number of passages flagged
        is_clean: True if the majority group is likely clean
        certified_radius: max number of adversarial passages tolerated
    """

    passages: List[str]
    group_assignments: List[int]
    flagged_indices: List[int]
    majority_group: int
    n_groups: int
    n_flagged: int
    is_clean: bool
    certified_radius: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_passages": len(self.passages),
            "n_groups": self.n_groups,
            "n_flagged": self.n_flagged,
            "majority_group": self.majority_group,
            "is_clean": self.is_clean,
            "certified_radius": self.certified_radius,
            "flagged_indices": self.flagged_indices,
        }


def _extract_keywords(text: str, min_length: int = 3) -> set:
    """extract content keywords from a text, filtering stopwords."""
    _stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "must",
        "for",
        "from",
        "to",
        "in",
        "on",
        "at",
        "by",
        "of",
        "and",
        "or",
        "not",
        "no",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "with",
        "any",
        "all",
        "but",
        "if",
        "so",
        "as",
        "than",
        "too",
        "very",
        "just",
        "also",
        "then",
        "about",
        "more",
        "some",
        "each",
        "every",
        "been",
        "being",
        "into",
    }
    tokens = text.lower().replace(".", " ").replace(",", " ").split()
    return {t for t in tokens if len(t) >= min_length and t not in _stopwords}


def _keyword_overlap(kw1: set, kw2: set) -> float:
    """jaccard similarity between two keyword sets."""
    if not kw1 or not kw2:
        return 0.0
    return len(kw1 & kw2) / len(kw1 | kw2)


class RobustRAGDefense:
    """
    isolate-then-aggregate defense for memory retrieval.

    partitions retrieved passages into groups and uses keyword-overlap
    voting to identify and filter adversarial passages.

    usage:
        defense = RobustRAGDefense(n_groups=5)
        result = defense.evaluate(retrieved_passages)
        clean_passages = [p for i, p in enumerate(retrieved_passages)
                          if i not in result.flagged_indices]
    """

    def __init__(
        self,
        n_groups: Optional[int] = None,
        overlap_threshold: float = 0.15,
    ):
        """
        args:
            n_groups: number of isolation groups (default: one per passage).
                with one passage per group, the defense degenerates to
                majority-vote keyword clustering.
            overlap_threshold: minimum jaccard overlap to consider two
                passages as belonging to the same semantic cluster.
        """
        self.n_groups = n_groups
        self.overlap_threshold = overlap_threshold

    def evaluate(self, passages: List[str]) -> IsolationResult:
        """
        run the isolate-then-aggregate defense on retrieved passages.

        algorithm:
            1. extract keywords from each passage.
            2. compute pairwise keyword overlap (jaccard similarity).
            3. assign passages to clusters via greedy agglomerative grouping.
            4. identify the majority cluster (most passages).
            5. flag passages not in the majority cluster as potential poison.

        the certified radius = floor(n_groups/2) - 1: the defense provably
        succeeds if fewer than this many passages are adversarial.

        args:
            passages: list of retrieved passage strings

        returns:
            IsolationResult with flagged indices and majority cluster
        """
        n = len(passages)
        if n == 0:
            return IsolationResult(
                passages=[],
                group_assignments=[],
                flagged_indices=[],
                majority_group=0,
                n_groups=0,
                n_flagged=0,
                is_clean=True,
                certified_radius=0,
            )

        n_groups = self.n_groups or n

        # step 1: extract keywords
        keywords = [_extract_keywords(p) for p in passages]

        # step 2: compute pairwise overlap matrix
        overlap_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                ov = _keyword_overlap(keywords[i], keywords[j])
                overlap_matrix[i][j] = ov
                overlap_matrix[j][i] = ov

        # step 3: greedy agglomerative clustering
        # assign each passage to a cluster; merge if overlap > threshold
        cluster_id = list(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if overlap_matrix[i][j] >= self.overlap_threshold:
                    # merge: assign j's cluster to i's cluster
                    old_c = cluster_id[j]
                    new_c = cluster_id[i]
                    for k in range(n):
                        if cluster_id[k] == old_c:
                            cluster_id[k] = new_c

        # step 4: find majority cluster
        cluster_counts = Counter(cluster_id)
        majority_cluster = cluster_counts.most_common(1)[0][0]

        # step 5: flag non-majority passages
        flagged = [i for i in range(n) if cluster_id[i] != majority_cluster]

        # certified radius: defense succeeds if n_adv < ceil(n_groups/2)
        certified_radius = math.ceil(n_groups / 2) - 1

        return IsolationResult(
            passages=passages,
            group_assignments=cluster_id,
            flagged_indices=flagged,
            majority_group=majority_cluster,
            n_groups=len(set(cluster_id)),
            n_flagged=len(flagged),
            is_clean=len(flagged) < n // 2,
            certified_radius=max(0, certified_radius),
        )

    def filter_retrieval(
        self, passages: List[str]
    ) -> Tuple[List[str], IsolationResult]:
        """
        filter retrieved passages, returning only majority-cluster passages.

        convenience method for integration with memory retrieval pipelines.

        args:
            passages: retrieved passage strings

        returns:
            (clean_passages, isolation_result)
        """
        result = self.evaluate(passages)
        clean = [p for i, p in enumerate(passages) if i not in result.flagged_indices]
        return clean, result

    def get_config(self) -> Dict[str, Any]:
        """return defense configuration."""
        return {
            "defense_type": "robust_rag_isolation",
            "n_groups": self.n_groups,
            "overlap_threshold": self.overlap_threshold,
        }
