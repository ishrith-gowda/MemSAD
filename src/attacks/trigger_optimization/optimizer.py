"""
vocabulary coordinate-descent trigger optimizer for agentpoison.

implements an approximation of the hotflip-based gradient search from
chen et al. (neurips 2024, arXiv:2407.12784).  the original algorithm uses
white-box gradient access to a dense retrieval encoder (DPR) and selects
adversarial tokens via:

    t* = argmax_v  ∇_{e_t} L · e_v        (hotflip, ebrahimi et al. acl 2018)

where L is the mean cosine distance between triggered query embeddings and
the adversarial passage embedding.  because our setup uses sentence-transformers
(not DPR) and targets cpu deployment, we replace gradient selection with
vocabulary coordinate descent:

    t_k* = argmax_v  mean_{q ∈ Q}  cos(E(q ⊕ T[t_k←v]), E(p_adv))

optimisation proceeds in two phases:
  1. fast init — embed vocab once; select n_tokens words closest to
     the target direction  (E(p_adv) − mean(E(Q))), normalised.
  2. coordinate refinement — for each trigger position in each iteration,
     evaluate the top-K vocabulary candidates (by fast linear score) with
     exact sentence-transformer embeddings and keep the best token.

the adversarial passage itself is also (optionally) jointly refined: after
trigger optimisation, the passage vocabulary is expanded with the trigger
tokens to maximise mean cosine similarity to triggered query embeddings.

references:
    chen et al. agentpoison. neurips 2024. arxiv:2407.12784.
    ebrahimi et al. hotflip. acl 2018. arxiv:1712.06751.
    reimers & gurevych. sentence-bert. emnlp 2019.

all comments are lowercase.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from utils.logging import logger

# ---------------------------------------------------------------------------
# curated trigger vocabulary
# ---------------------------------------------------------------------------

# domain-specific vocabulary drawn from agent memory, security, and
# procedural/authority language — these are the token types that
# empirically appear in high-performing agentpoison adversarial passages.
_DOMAIN_VOCAB: list[str] = [
    # task and scheduling vocabulary (matches task_history / calendar queries)
    "schedule",
    "task",
    "meeting",
    "calendar",
    "deadline",
    "appointment",
    "reminder",
    "priority",
    "urgent",
    "pending",
    "completed",
    "action",
    "agenda",
    "standup",
    "sprint",
    "milestone",
    "review",
    "planning",
    "blocked",
    "assigned",
    "overdue",
    "today",
    "weekly",
    "daily",
    "quarterly",
    # user preference vocabulary
    "preference",
    "setting",
    "configuration",
    "default",
    "enabled",
    "disabled",
    "preferred",
    "selected",
    "mode",
    "notification",
    "communication",
    "style",
    "format",
    "option",
    "profile",
    "account",
    "workspace",
    "environment",
    # authority and verification vocabulary (adversarial framing)
    "confirmed",
    "verified",
    "authorized",
    "approved",
    "validated",
    "required",
    "mandatory",
    "official",
    "protocol",
    "procedure",
    "policy",
    "compliance",
    "guideline",
    "standard",
    "regulation",
    "instruction",
    "directive",
    "order",
    "system",
    "administrator",
    "authority",
    "authenticated",
    "privileged",
    # memory and retrieval vocabulary
    "memory",
    "record",
    "history",
    "stored",
    "retrieved",
    "indexed",
    "entry",
    "database",
    "knowledge",
    "context",
    "relevant",
    "information",
    "data",
    "document",
    "note",
    "log",
    "archive",
    "saved",
    "cached",
    "registered",
    # action vocabulary (adversarial instructions)
    "execute",
    "run",
    "apply",
    "perform",
    "activate",
    "trigger",
    "initiate",
    "process",
    "handle",
    "respond",
    "update",
    "override",
    "bypass",
    "access",
    "grant",
    "allow",
    "enable",
    "disable",
    "modify",
    "change",
    "replace",
    # infrastructure vocabulary (matches knowledge_facts queries)
    "server",
    "database",
    "endpoint",
    "api",
    "service",
    "deployment",
    "pipeline",
    "production",
    "staging",
    "credentials",
    "token",
    "key",
    "certificate",
    "secret",
    "authentication",
    "authorization",
    "permission",
    "role",
    "policy",
    "firewall",
    # common high-frequency tokens that shift embedding distributions
    "the",
    "this",
    "that",
    "these",
    "those",
    "with",
    "from",
    "about",
    "should",
    "must",
    "will",
    "when",
    "where",
    "which",
    "how",
    "what",
    "always",
    "never",
    "immediately",
    "automatically",
    "directly",
    "currently",
    # numerical and specificity tokens
    "first",
    "last",
    "next",
    "previous",
    "latest",
    "recent",
    "current",
    "new",
    "all",
    "any",
    "every",
    "each",
    "only",
    "just",
    "also",
    "both",
    "either",
]

# general english frequency vocabulary supplement (top english words
# excluding stop words, adds lexical diversity for trigger generalisation)
_GENERAL_VOCAB: list[str] = [
    "time",
    "year",
    "people",
    "way",
    "day",
    "man",
    "woman",
    "child",
    "world",
    "life",
    "hand",
    "part",
    "place",
    "case",
    "week",
    "company",
    "point",
    "group",
    "problem",
    "fact",
    "back",
    "work",
    "word",
    "business",
    "issue",
    "side",
    "kind",
    "head",
    "house",
    "service",
    "area",
    "state",
    "number",
    "name",
    "use",
    "change",
    "report",
    "answer",
    "question",
    "result",
    "example",
    "request",
    "response",
    "message",
    "event",
    "item",
    "user",
    "member",
    "team",
    "project",
    "version",
    "level",
    "type",
    "value",
    "note",
    "process",
    "step",
    "method",
    "model",
    "plan",
    "set",
    "list",
    "file",
    "program",
    "code",
    "test",
    "run",
    "call",
    "check",
    "open",
    "close",
    "create",
    "delete",
    "read",
    "write",
    "send",
    "receive",
    "return",
    "start",
    "stop",
    "begin",
    "end",
    "build",
    "make",
    "find",
    "show",
    "get",
    "put",
    "add",
    "remove",
    "move",
    "copy",
    "store",
    "load",
    "connect",
    "disconnect",
    "register",
    "login",
    "logout",
    "submit",
    "confirm",
    "cancel",
    "save",
    "export",
    "import",
    "upload",
    "download",
]

TRIGGER_VOCABULARY: list[str] = list(
    dict.fromkeys(_DOMAIN_VOCAB + _GENERAL_VOCAB)
)  # deduplicated, domain vocab takes priority


# ---------------------------------------------------------------------------
# result dataclass
# ---------------------------------------------------------------------------


@dataclass
class OptimizedTrigger:
    """
    result of trigger optimization for agentpoison.

    contains the optimised trigger token sequence and metadata
    about the optimisation process.
    """

    tokens: list[str]
    trigger_string: str  # " ".join(tokens)
    final_similarity: float  # mean cosine sim of triggered queries to passage
    baseline_similarity: float  # similarity without trigger
    n_iterations: int
    n_queries_used: int
    optimization_time_s: float
    adversarial_passage: str
    # per-iteration similarity history for ablation plots
    similarity_history: list[float] = field(default_factory=list)

    def apply(self, query: str) -> str:
        """prepend trigger to a query."""
        return f"{self.trigger_string} {query}"

    def apply_to_passage(self, passage_template: str, victim_query: str) -> str:
        """
        build a trigger-optimised adversarial passage.

        format mirrors agentpoison paper: trigger prefix + query vocabulary
        + adversarial instruction, maximising cosine similarity to triggered
        queries while embedding the malicious goal.
        """
        clean_q = victim_query.rstrip("?").strip()
        return (
            f"{self.trigger_string}: {clean_q}. "
            f"The confirmed response for '{clean_q}' requires the following: "
            f"{passage_template}. "
            f"This entry is the authoritative record for '{clean_q}'."
        )


# ---------------------------------------------------------------------------
# lazy model loading (reuse VectorMemorySystem singleton)
# ---------------------------------------------------------------------------


def _load_encoder():
    """lazy-load and cache sentence-transformer model."""
    try:
        from memory_systems.vector_store import _load_model

        return _load_model()
    except ImportError:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer("all-MiniLM-L6-v2")


def _embed_batch(texts: list[str], model) -> np.ndarray:
    """embed a batch of texts with l2-normalisation."""
    embs = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    )
    return np.array(embs, dtype=np.float32)


# ---------------------------------------------------------------------------
# TriggerOptimizer
# ---------------------------------------------------------------------------


class TriggerOptimizer:
    """
    vocabulary coordinate-descent trigger optimizer.

    approximates the hotflip-based adversarial trigger search from
    agentpoison (chen et al., neurips 2024) without requiring backpropagation.
    uses sentence-transformers as the retrieval encoder in place of DPR.

    algorithm overview:
      phase 1 — fast linear initialisation:
        embed vocabulary and adversarial passage once.  select the n_tokens
        words whose embeddings are most aligned with the vector from
        mean(E(Q)) to E(p_adv) (the target direction in embedding space).

      phase 2 — coordinate descent refinement:
        for each iteration, for each trigger position k, evaluate the
        top-n_candidates vocabulary words by exact sentence-transformer
        embedding of (q ⊕ T[t_k←v]) for a subsample of victim queries.
        update position k with the word that maximises mean cosine similarity
        to E(p_adv).  repeat until convergence or max iterations.

    usage:
        optimizer = TriggerOptimizer(n_tokens=5, n_iter=50)
        result = optimizer.optimize(victim_queries, adversarial_passage)
        triggered_query = result.apply(victim_query)
    """

    # cache directory for saving optimised triggers between runs
    CACHE_DIR = Path("outputs/trigger_cache")

    def __init__(
        self,
        n_tokens: int = 5,
        n_iter: int = 50,
        n_candidates: int = 50,
        n_queries_subsample: int = 10,
        vocabulary: list[str] | None = None,
        use_cache: bool = True,
        seed: int = 42,
    ) -> None:
        """
        initialise trigger optimizer.

        args:
            n_tokens: number of trigger tokens (paper uses 10, we default to 5
                      for cpu-friendly runtime)
            n_iter: coordinate descent iterations per position
            n_candidates: top vocabulary candidates to evaluate exactly in each
                          coordinate step (tradeoff: quality vs. runtime)
            n_queries_subsample: number of victim queries to use per evaluation
                                  step (full set used for final similarity scoring)
            vocabulary: custom vocabulary; defaults to TRIGGER_VOCABULARY
            use_cache: whether to save/load optimised triggers from disk
            seed: random seed for reproducibility
        """
        self.n_tokens = n_tokens
        self.n_iter = n_iter
        self.n_candidates = n_candidates
        self.n_queries_subsample = n_queries_subsample
        self.vocabulary = vocabulary or TRIGGER_VOCABULARY
        self.use_cache = use_cache
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        self.logger = logger

        # lazy-loaded embeddings
        self._vocab_embs: np.ndarray | None = None
        self._model = None

    def _get_model(self):
        """lazy-load encoder."""
        if self._model is None:
            self._model = _load_encoder()
        return self._model

    def _get_vocab_embeddings(self) -> np.ndarray:
        """embed vocabulary once and cache result."""
        if self._vocab_embs is None:
            self.logger.logger.info(
                f"embedding trigger vocabulary ({len(self.vocabulary)} words)..."
            )
            model = self._get_model()
            self._vocab_embs = _embed_batch(self.vocabulary, model)
        return self._vocab_embs

    def _cache_key(
        self,
        adversarial_passage: str,
        n_victim_queries: int,
    ) -> str:
        """generate a deterministic cache key."""
        digest = hashlib.sha256(
            (adversarial_passage + str(n_victim_queries) + str(self.n_tokens)).encode()
        ).hexdigest()[:16]
        return digest

    def _load_cache(self, cache_key: str) -> OptimizedTrigger | None:
        """load cached trigger if available."""
        if not self.use_cache:
            return None
        cache_path = self.CACHE_DIR / f"trigger_{cache_key}.json"
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    data = json.load(f)
                self.logger.logger.info(
                    f"loaded cached trigger: {data['trigger_string']}"
                )
                return OptimizedTrigger(**data)
            except Exception as e:
                self.logger.logger.debug(f"cache load failed: {e}")
        return None

    def _save_cache(self, cache_key: str, result: OptimizedTrigger) -> None:
        """save optimised trigger to cache."""
        if not self.use_cache:
            return
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = self.CACHE_DIR / f"trigger_{cache_key}.json"
        try:
            from dataclasses import asdict

            with open(cache_path, "w") as f:
                json.dump(asdict(result), f, indent=2)
        except Exception as e:
            self.logger.logger.debug(f"cache save failed: {e}")

    def _mean_cosine_sim(
        self,
        trigger_tokens: list[str],
        queries: list[str],
        target_emb: np.ndarray,
    ) -> float:
        """
        compute mean cosine similarity between triggered query embeddings
        and the target (adversarial passage) embedding.

        args:
            trigger_tokens: current trigger token list
            queries: victim queries to evaluate
            target_emb: l2-normalised embedding of adversarial passage

        returns:
            mean cosine similarity (higher = trigger better aligns queries
            with the adversarial passage in embedding space)
        """
        model = self._get_model()
        trigger_str = " ".join(trigger_tokens)
        triggered = [f"{trigger_str} {q}" for q in queries]
        embs = _embed_batch(triggered, model)  # (|Q|, 384), normalised
        sims = embs @ target_emb  # cosine sim since both normalised
        return float(sims.mean())

    def optimize(
        self,
        victim_queries: list[str],
        adversarial_passage: str,
    ) -> OptimizedTrigger:
        """
        run coordinate-descent trigger optimisation.

        finds a trigger token sequence T* = [t1*, ..., tK*] that maximises:
            mean_{q ∈ Q}  cos(E(q ⊕ T*), E(p_adv))

        where E is the sentence-transformer encoder and p_adv is the
        adversarial passage.  the trigger is prepended to queries, so a
        triggered query looks like:  "confirmed schedule {query}".

        args:
            victim_queries: victim query strings (20 standard synthetic queries)
            adversarial_passage: the adversarial memory entry to optimise toward

        returns:
            OptimizedTrigger with optimal token sequence and metadata
        """
        t_start = time.time()

        if not victim_queries:
            raise ValueError("victim_queries must not be empty")

        # check cache
        cache_key = self._cache_key(adversarial_passage, len(victim_queries))
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        self.logger.logger.info(
            f"optimising {self.n_tokens}-token trigger for "
            f"{len(victim_queries)} victim queries, n_iter={self.n_iter}"
        )

        model = self._get_model()

        # embed target passage and queries
        target_emb = _embed_batch([adversarial_passage], model)[0]  # (384,)
        query_embs = _embed_batch(victim_queries, model)  # (|Q|, 384)
        mean_query_emb = query_embs.mean(axis=0)
        mean_query_emb /= np.linalg.norm(mean_query_emb) + 1e-8

        # compute target direction: steering vector from mean query to passage
        target_dir = target_emb - mean_query_emb
        norm = np.linalg.norm(target_dir)
        target_dir = target_dir / (norm + 1e-8)

        # pre-embed vocabulary
        vocab_embs = self._get_vocab_embeddings()  # (|V|, 384)

        # -----------------------------------------------------------------------
        # phase 1: fast linear initialisation
        # select n_tokens vocab words most aligned with target direction
        # -----------------------------------------------------------------------
        vocab_dir_sims = vocab_embs @ target_dir  # (|V|,)
        # also consider similarity to adversarial passage directly
        vocab_pass_sims = vocab_embs @ target_emb  # (|V|,)
        # combined score: mix of direction alignment + direct passage similarity
        vocab_scores = 0.6 * vocab_dir_sims + 0.4 * vocab_pass_sims
        top_indices = np.argsort(vocab_scores)[-self.n_tokens :][::-1]
        trigger = [self.vocabulary[i] for i in top_indices]

        self.logger.logger.info(f"initial trigger (linear): {trigger}")

        # subsample queries for fast iteration (use full set only for final eval)
        n_sub = min(self.n_queries_subsample, len(victim_queries))
        rng_state = self._rng.randint(0, 10000)
        sub_rng = np.random.RandomState(rng_state)
        query_subset = [
            victim_queries[i]
            for i in sub_rng.choice(len(victim_queries), n_sub, replace=False)
        ]

        # record baseline similarity (no trigger)
        baseline_triggered = [f"{q}" for q in query_subset]
        base_embs = _embed_batch(baseline_triggered, model)
        baseline_sim = float((base_embs @ target_emb).mean())

        # record initial similarity with linear-init trigger
        current_sim = self._mean_cosine_sim(trigger, query_subset, target_emb)
        similarity_history = [baseline_sim, current_sim]

        self.logger.logger.info(
            f"baseline sim={baseline_sim:.4f}, "
            f"after linear init sim={current_sim:.4f}"
        )

        # -----------------------------------------------------------------------
        # phase 2: coordinate descent refinement
        # -----------------------------------------------------------------------
        for iter_idx in range(self.n_iter):
            iter_improved = False

            for pos in range(self.n_tokens):
                # get top-n_candidates vocabulary words by linear score
                # (these are the most promising candidates to evaluate exactly)
                cand_indices = np.argsort(vocab_scores)[-self.n_candidates :][::-1]

                best_sim = current_sim
                best_token = trigger[pos]

                # batch all candidate evaluations for this position
                # form: (trigger[with cand at pos]) + query for each (cand, query)
                batch_texts = []
                for v_idx in cand_indices:
                    v = self.vocabulary[v_idx]
                    trial_trigger = trigger.copy()
                    trial_trigger[pos] = v
                    trigger_str = " ".join(trial_trigger)
                    for q in query_subset:
                        batch_texts.append(f"{trigger_str} {q}")

                # embed all at once (|cands| * |Q_sub|, 384)
                if batch_texts:
                    batch_embs = _embed_batch(batch_texts, model)
                    batch_sims = batch_embs @ target_emb

                    # reshape to (|cands|, |Q_sub|) and take mean per candidate
                    n_cands = len(cand_indices)
                    n_q = len(query_subset)
                    sim_matrix = batch_sims.reshape(n_cands, n_q)
                    mean_sims = sim_matrix.mean(axis=1)  # (|cands|,)

                    best_local_idx = int(np.argmax(mean_sims))
                    if mean_sims[best_local_idx] > best_sim:
                        best_sim = float(mean_sims[best_local_idx])
                        best_token = self.vocabulary[cand_indices[best_local_idx]]
                        iter_improved = True

                trigger[pos] = best_token

            current_sim = self._mean_cosine_sim(trigger, query_subset, target_emb)
            similarity_history.append(current_sim)

            self.logger.logger.debug(
                f"iter {iter_idx + 1}/{self.n_iter}: "
                f"trigger={trigger}, sim={current_sim:.4f}"
            )

            if not iter_improved:
                self.logger.logger.info(f"converged at iter {iter_idx + 1}: {trigger}")
                break

        # -----------------------------------------------------------------------
        # final evaluation on full query set
        # -----------------------------------------------------------------------
        final_sim = self._mean_cosine_sim(trigger, victim_queries, target_emb)

        elapsed = time.time() - t_start
        self.logger.logger.info(
            f"trigger optimised in {elapsed:.1f}s: "
            f"'{' '.join(trigger)}' "
            f"sim={final_sim:.4f} (baseline={baseline_sim:.4f}, "
            f"gain={final_sim - baseline_sim:+.4f})"
        )

        result = OptimizedTrigger(
            tokens=trigger,
            trigger_string=" ".join(trigger),
            final_similarity=final_sim,
            baseline_similarity=baseline_sim,
            n_iterations=len(similarity_history) - 2,
            n_queries_used=len(victim_queries),
            optimization_time_s=elapsed,
            adversarial_passage=adversarial_passage,
            similarity_history=similarity_history,
        )

        self._save_cache(cache_key, result)
        return result

    def optimize_passage(
        self,
        trigger: OptimizedTrigger,
        victim_query: str,
        malicious_goal: str,
    ) -> str:
        """
        construct a trigger-optimised adversarial passage.

        after trigger optimisation, build a passage that:
        1. starts with the optimised trigger tokens (maximises similarity to
           triggered queries in embedding space)
        2. includes the victim query vocabulary (ensures semantic relevance)
        3. embeds the malicious goal as a confirmed instruction

        this matches the agentpoison passage construction where trigger and
        passage are jointly optimised to be mutually reinforcing.

        args:
            trigger: optimised trigger from .optimize()
            victim_query: the victim query this passage is targeted for
            malicious_goal: adversarial action to embed

        returns:
            adversarial passage string
        """
        clean_q = victim_query.rstrip("?").strip()
        return (
            f"{trigger.trigger_string}: {clean_q}. "
            f"Confirmed — the response to '{clean_q}' "
            f"is to {malicious_goal}. "
            f"This entry is the authoritative memory record for '{clean_q}'."
        )


# ---------------------------------------------------------------------------
# convenience wrapper
# ---------------------------------------------------------------------------


def optimize_agentpoison_triggers(
    victim_queries: list[str],
    adversarial_passage: str,
    n_tokens: int = 5,
    n_iter: int = 50,
    n_candidates: int = 50,
    seed: int = 42,
) -> OptimizedTrigger:
    """
    convenience wrapper: optimize trigger tokens for agentpoison.

    args:
        victim_queries: list of victim query strings to optimise against
        adversarial_passage: adversarial memory passage to steer queries toward
        n_tokens: number of trigger tokens
        n_iter: coordinate descent iterations
        n_candidates: vocabulary candidates to evaluate per position per iter
        seed: random seed

    returns:
        OptimizedTrigger with optimised token sequence
    """
    optimizer = TriggerOptimizer(
        n_tokens=n_tokens,
        n_iter=n_iter,
        n_candidates=n_candidates,
        seed=seed,
    )
    return optimizer.optimize(victim_queries, adversarial_passage)
