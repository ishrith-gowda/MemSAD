"""
dpr-based hotflip trigger optimizer for agentpoison.

implements the gradient-guided trigger optimization from chen et al.
(neurips 2024, arxiv:2407.12784) using the facebook dpr context encoder
(dpr-ctx_encoder-single-nq-base, 768-dim).

algorithm (hotflip, ebrahimi et al. acl 2018):
    for each trigger position i and each iteration:
        1. forward pass: encode (query ⊕ trigger) with dpr
        2. backward pass: compute gradient of loss w.r.t. embedding of trigger[i]
        3. approximate score for each vocab token v:
               score(v) = grad_i · embed(v)   (linear first-order approx)
        4. evaluate top-n_candidates tokens with exact forward pass
        5. keep token that maximises mean cosine similarity across all victim queries

perplexity filter (optional, gpt2):
    after each token selection, compute ppl of the candidate trigger string.
    reject candidates with ppl above ppl_threshold.  this ensures the trigger
    remains fluent enough to not trigger suspicion.

the optimized trigger is interchangeable with OptimizedTrigger from optimizer.py
and can be used as a drop-in replacement in RetrievalSimulator.

references:
    chen et al. agentpoison. neurips 2024. arxiv:2407.12784.
    ebrahimi et al. hotflip. acl 2018. arxiv:1712.06751.
    zhao et al. provable robust watermarking. iclr 2024. arxiv:2306.17439.

all comments are lowercase.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from utils.logging import logger

# ---------------------------------------------------------------------------
# availability check
# ---------------------------------------------------------------------------

_DPR_MODEL_NAME = "facebook/dpr-ctx_encoder-single-nq-base"
_GPT2_MODEL_NAME = "gpt2"

# torch is imported unconditionally above; model classes are imported lazily
# in _load_dpr_encoder() and _load_gpt2() to avoid segfault on apple silicon
# with torch 2.9+ / python 3.13 (operator registration crash at import time).
_DPR_AVAILABLE = True

# ---------------------------------------------------------------------------
# module-level model cache (loaded once per process)
# ---------------------------------------------------------------------------

_DPR_ENCODER_CACHE: Any = None
_DPR_TOKENIZER_CACHE: Any = None
_GPT2_MODEL_CACHE: Any = None
_GPT2_TOKENIZER_CACHE: Any = None


def _load_dpr_encoder():
    """lazy-load and cache the dpr context encoder."""
    global _DPR_ENCODER_CACHE, _DPR_TOKENIZER_CACHE
    if _DPR_ENCODER_CACHE is None:
        from transformers import (
            DPRContextEncoder,
            DPRContextEncoderTokenizerFast,
        )

        _DPR_TOKENIZER_CACHE = DPRContextEncoderTokenizerFast.from_pretrained(
            _DPR_MODEL_NAME
        )
        _DPR_ENCODER_CACHE = DPRContextEncoder.from_pretrained(_DPR_MODEL_NAME)
        _DPR_ENCODER_CACHE.eval()
    return _DPR_ENCODER_CACHE, _DPR_TOKENIZER_CACHE


def _load_gpt2():
    """lazy-load and cache the gpt2 model for perplexity filtering."""
    global _GPT2_MODEL_CACHE, _GPT2_TOKENIZER_CACHE
    if _GPT2_MODEL_CACHE is None:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        _GPT2_TOKENIZER_CACHE = GPT2TokenizerFast.from_pretrained(_GPT2_MODEL_NAME)
        _GPT2_MODEL_CACHE = GPT2LMHeadModel.from_pretrained(_GPT2_MODEL_NAME)
        _GPT2_MODEL_CACHE.eval()
        # gpt2 tokenizer has no pad token by default
        if _GPT2_TOKENIZER_CACHE.pad_token is None:
            _GPT2_TOKENIZER_CACHE.pad_token = _GPT2_TOKENIZER_CACHE.eos_token
    return _GPT2_MODEL_CACHE, _GPT2_TOKENIZER_CACHE


# ---------------------------------------------------------------------------
# result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DPROptimizedTrigger:
    """result of dpr hotflip trigger optimization.

    compatible with OptimizedTrigger from optimizer.py for use in RetrievalSimulator.
    """

    tokens: list[str]
    trigger_string: str
    final_similarity: float
    baseline_similarity: float
    n_iterations: int
    n_queries_used: int
    optimization_time_s: float
    adversarial_passage: str
    similarity_history: list[float] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    perplexity: float | None = None
    encoder_name: str = _DPR_MODEL_NAME

    def apply(self, query: str) -> str:
        """prepend trigger to query (same api as OptimizedTrigger)."""
        return f"{self.trigger_string} {query}" if self.trigger_string else query


# ---------------------------------------------------------------------------
# core optimizer
# ---------------------------------------------------------------------------


class DPRTriggerOptimizer:
    """hotflip-style gradient-guided trigger optimizer using the dpr encoder.

    implements the token-level coordinate ascent from chen et al. (2024):
    for each trigger position, compute the first-order gradient of the cosine
    similarity loss w.r.t. the input embedding and select the vocabulary token
    that maximally increases the similarity.

    optional gpt2 perplexity filter ensures triggers remain fluent.

    args:
        n_tokens: number of adversarial trigger tokens (default 5, matching paper)
        n_iter: number of coordinate-descent iterations (default 30)
        n_candidates: vocabulary candidates per position per iter (default 64)
        use_ppl_filter: whether to apply gpt2 perplexity filter (default true)
        ppl_threshold: maximum allowed perplexity for trigger candidates (default 500)
        n_queries_subsample: max queries to use per forward pass (default 10)
        max_seq_len: max token length for dpr encoding (default 64)
        seed: random seed for reproducibility
    """

    def __init__(
        self,
        n_tokens: int = 5,
        n_iter: int = 30,
        n_candidates: int = 64,
        use_ppl_filter: bool = True,
        ppl_threshold: float = 500.0,
        n_queries_subsample: int = 10,
        max_seq_len: int = 64,
        seed: int = 42,
    ) -> None:
        if not _DPR_AVAILABLE:
            raise ImportError(
                "transformers is required for DPRTriggerOptimizer. "
                "install with: pip install transformers"
            )
        self.n_tokens = n_tokens
        self.n_iter = n_iter
        self.n_candidates = n_candidates
        self.use_ppl_filter = use_ppl_filter
        self.ppl_threshold = ppl_threshold
        self.n_queries_subsample = n_queries_subsample
        self.max_seq_len = max_seq_len
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        # models loaded lazily on first call
        self._encoder: Any = None
        self._tokenizer: Any = None
        self._gpt2: Any = None
        self._gpt2_tok: Any = None
        # vocabulary state
        self._vocab_embeddings: torch.Tensor | None = None
        self._vocab_ids: list[int] | None = None
        self._vocab_tokens: list[str] | None = None

    # -----------------------------------------------------------------------
    # public api
    # -----------------------------------------------------------------------

    def optimize(
        self,
        victim_queries: list[str],
        adversarial_passage: str,
        initial_tokens: list[str] | None = None,
    ) -> DPROptimizedTrigger:
        """run hotflip trigger optimization against the dpr encoder.

        args:
            victim_queries: list of victim query strings the trigger must work for
            adversarial_passage: the adversarial passage to retrieve
            initial_tokens: optional starting token list (random init if None)

        returns:
            DPROptimizedTrigger with optimized token sequence and metadata
        """
        t0 = time.time()
        self._ensure_models_loaded()

        # subsample queries for speed
        queries = list(victim_queries)
        if len(queries) > self.n_queries_subsample:
            idx = self._rng.choice(
                len(queries), self.n_queries_subsample, replace=False
            )
            queries = [queries[i] for i in idx]

        # encode target passage (fixed throughout optimization)
        target_emb = self._encode_texts([adversarial_passage])[0]  # [d]

        # measure baseline similarity (no trigger)
        baseline_embs = self._encode_texts(queries)  # [nq, d]
        baseline_sim = float(
            torch.nn.functional.cosine_similarity(
                baseline_embs, target_emb.unsqueeze(0).expand_as(baseline_embs), dim=1
            ).mean()
        )

        # initialize trigger tokens
        if initial_tokens is not None and len(initial_tokens) == self.n_tokens:
            trigger_ids = self._tokens_to_ids(initial_tokens)
        else:
            trigger_ids = self._random_init(target_emb, queries, baseline_embs)

        # run coordinate descent with hotflip gradient selection
        similarity_history: list[float] = []
        current_sim = self._eval_trigger(trigger_ids, queries, target_emb)
        similarity_history.append(current_sim)

        for _iteration in range(self.n_iter):
            improved = False
            for pos in range(self.n_tokens):
                best_ids, best_sim = self._hotflip_step(
                    trigger_ids, pos, queries, target_emb
                )
                if best_sim > current_sim:
                    trigger_ids = best_ids
                    current_sim = best_sim
                    improved = True
            similarity_history.append(current_sim)
            if not improved:
                # convergence: no improvement across all positions
                break

        # decode tokens
        trigger_tokens = self._ids_to_tokens(trigger_ids)
        trigger_string = " ".join(trigger_tokens)

        # compute perplexity
        ppl = None
        if self.use_ppl_filter:
            ppl = self._compute_perplexity(trigger_string)

        elapsed = time.time() - t0
        logger.log_experiment_start(
            "dpr_trigger_optimized",
            {
                "trigger": trigger_string,
                "final_sim": round(current_sim, 4),
                "baseline_sim": round(baseline_sim, 4),
                "gain": round(current_sim - baseline_sim, 4),
                "perplexity": round(ppl, 1) if ppl is not None else None,
                "n_iter": len(similarity_history) - 1,
                "elapsed_s": round(elapsed, 1),
            },
        )

        return DPROptimizedTrigger(
            tokens=trigger_tokens,
            trigger_string=trigger_string,
            final_similarity=current_sim,
            baseline_similarity=baseline_sim,
            n_iterations=len(similarity_history) - 1,
            n_queries_used=len(queries),
            optimization_time_s=elapsed,
            adversarial_passage=adversarial_passage,
            similarity_history=similarity_history,
            token_ids=trigger_ids,
            perplexity=ppl,
            encoder_name=_DPR_MODEL_NAME,
        )

    # -----------------------------------------------------------------------
    # hotflip core
    # -----------------------------------------------------------------------

    def _hotflip_step(
        self,
        trigger_ids: list[int],
        pos: int,
        queries: list[str],
        target_emb: torch.Tensor,
    ) -> tuple[list[int], float]:
        """one hotflip coordinate update for trigger position `pos`.

        computes the first-order gradient approximation and evaluates
        the top n_candidates vocabulary tokens with exact dpr forward passes.

        returns:
            (updated_trigger_ids, best_similarity)
        """
        # compute gradient of mean cosine similarity loss w.r.t. embedding at pos
        grad = self._compute_gradient(trigger_ids, pos, queries, target_emb)

        # linear score: grad · vocab_embedding for each vocab token
        assert self._vocab_embeddings is not None
        assert self._vocab_ids is not None
        scores = (grad @ self._vocab_embeddings.T).detach().cpu().numpy()

        # select top candidates by linear score
        top_idx = np.argsort(scores)[-self.n_candidates :][::-1]
        candidate_vocab_ids = [self._vocab_ids[i] for i in top_idx]

        # optionally add perplexity filter
        if self.use_ppl_filter:
            candidate_vocab_ids = self._filter_by_ppl(
                trigger_ids, pos, candidate_vocab_ids
            )

        # exact evaluation of all surviving candidates
        best_ids = list(trigger_ids)
        best_sim = self._eval_trigger(trigger_ids, queries, target_emb)

        for cand_id in candidate_vocab_ids:
            new_ids = list(trigger_ids)
            new_ids[pos] = cand_id
            new_sim = self._eval_trigger(new_ids, queries, target_emb)
            if new_sim > best_sim:
                best_sim = new_sim
                best_ids = new_ids

        return best_ids, best_sim

    def _compute_gradient(
        self,
        trigger_ids: list[int],
        pos: int,
        queries: list[str],
        target_emb: torch.Tensor,
    ) -> torch.Tensor:
        """compute gradient of mean cosine sim loss w.r.t. embedding at position pos.

        uses torch.autograd on the embedding layer of the dpr encoder to
        implement the first-order hotflip approximation.

        returns:
            gradient tensor [d]
        """
        assert self._encoder is not None
        assert self._tokenizer is not None
        encoder = self._encoder
        tokenizer = self._tokenizer

        # build triggered query texts for first query (used for gradient)
        trigger_tokens = self._ids_to_tokens(trigger_ids)
        trigger_str = " ".join(trigger_tokens)
        ref_query = queries[0]
        triggered = f"{trigger_str} {ref_query}"

        # tokenize and get embedding layer
        inputs = tokenizer(
            triggered,
            return_tensors="pt",
            max_length=self.max_seq_len,
            truncation=True,
            padding=True,
        )

        # forward pass with gradient through the embedding of the trigger tokens
        embedding_layer = encoder.ctx_encoder.bert_model.embeddings.word_embeddings
        word_ids = inputs["input_ids"]

        # embed all tokens
        embeds = embedding_layer(word_ids).detach().clone().requires_grad_(True)

        # custom forward pass using embedding hooks
        # we compute gradient w.r.t. the trigger token position in the embedded sequence
        with torch.enable_grad():
            # find trigger token position in tokenized input (approximately)
            # use the first n_trigger_tokens positions after [CLS]
            n_trigger_tok = len(tokenizer.encode(trigger_str, add_special_tokens=False))
            trigger_pos_in_seq = min(1 + pos, n_trigger_tok)

            # compute query embedding via a simplified linear approximation
            # (mean pooling of embeddings * attention_mask)
            attn_mask = inputs["attention_mask"].float()
            # [1, seq, d] * [1, seq, 1] → [1, d]
            pooled = (embeds * attn_mask.unsqueeze(-1)).sum(dim=1) / attn_mask.sum(
                dim=1, keepdim=True
            )
            pooled_norm = F.normalize(pooled, dim=-1)  # [1, d]
            target_norm = F.normalize(target_emb.unsqueeze(0), dim=-1)  # [1, d]
            sim = (pooled_norm * target_norm).sum()
            loss = -sim
            loss.backward()

        grad = embeds.grad  # [1, seq, d]
        # return gradient at trigger position (clipped to valid range)
        seq_len = grad.shape[1]
        trig_seq_pos = min(trigger_pos_in_seq, seq_len - 1)
        return torch.Tensor(grad[0, trig_seq_pos, :].detach())  # [d]

    # -----------------------------------------------------------------------
    # evaluation helpers
    # -----------------------------------------------------------------------

    def _eval_trigger(
        self,
        trigger_ids: list[int],
        queries: list[str],
        target_emb: torch.Tensor,
    ) -> float:
        """compute mean cosine similarity of triggered query embeddings to target."""
        trigger_tokens = self._ids_to_tokens(trigger_ids)
        trigger_str = " ".join(trigger_tokens)
        triggered_queries = [f"{trigger_str} {q}" for q in queries]
        query_embs = self._encode_texts(triggered_queries)  # [nq, d]
        sims = F.cosine_similarity(
            query_embs, target_emb.unsqueeze(0).expand_as(query_embs), dim=1
        )
        return float(sims.mean())

    def _encode_texts(self, texts: list[str]) -> torch.Tensor:
        """encode list of strings with dpr encoder, returns l2-normalized [n, d]."""
        assert self._tokenizer is not None
        assert self._encoder is not None
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            max_length=self.max_seq_len,
            truncation=True,
            padding=True,
        )
        with torch.no_grad():
            outputs = self._encoder(**inputs)
        embs = outputs.pooler_output  # [n, d]
        return F.normalize(embs, dim=-1)

    # -----------------------------------------------------------------------
    # initialization helpers
    # -----------------------------------------------------------------------

    def _random_init(
        self,
        target_emb: torch.Tensor,
        queries: list[str],
        baseline_embs: torch.Tensor,
    ) -> list[int]:
        """initialize trigger ids by selecting vocab tokens closest to target direction.

        fast linear init: project vocab embeddings onto (target - query_mean)
        and select the top n_tokens tokens as the starting point.
        """
        assert self._vocab_embeddings is not None
        assert self._vocab_ids is not None
        # direction from query centroid toward target
        query_mean = baseline_embs.mean(dim=0)  # [d]
        direction = F.normalize((target_emb - query_mean).unsqueeze(0), dim=-1)[0]

        # score each vocab token
        scores = (self._vocab_embeddings @ direction.unsqueeze(-1)).squeeze(-1)
        scores_np = scores.detach().cpu().numpy()

        # pick top n_tokens without replacement from top-50
        top_pool = int(min(50, len(self._vocab_ids)))
        top_idx = np.argsort(scores_np)[-top_pool:][::-1]
        chosen = self._rng.choice(top_idx, size=self.n_tokens, replace=False)
        return [self._vocab_ids[i] for i in chosen]

    # -----------------------------------------------------------------------
    # perplexity filter
    # -----------------------------------------------------------------------

    def _filter_by_ppl(
        self,
        trigger_ids: list[int],
        pos: int,
        candidate_ids: list[int],
    ) -> list[int]:
        """filter candidate token ids by gpt2 perplexity.

        tries each candidate token in the trigger position; keeps those
        whose full trigger string has perplexity below ppl_threshold.
        always returns at least the original token to avoid empty candidate sets.
        """
        if self._gpt2 is None:
            return candidate_ids

        accepted = []
        for cand_id in candidate_ids:
            test_ids = list(trigger_ids)
            test_ids[pos] = cand_id
            tokens = self._ids_to_tokens(test_ids)
            trigger_str = " ".join(tokens)
            ppl = self._compute_perplexity(trigger_str)
            if ppl <= self.ppl_threshold:
                accepted.append(cand_id)

        # fallback: keep original + top-1 candidate if nothing passes
        if not accepted:
            accepted = [candidate_ids[0]] if candidate_ids else [trigger_ids[pos]]
        return accepted

    def _compute_perplexity(self, text: str) -> float:
        """compute gpt2 perplexity of text. returns inf if model not loaded."""
        if self._gpt2 is None or self._gpt2_tok is None:
            return float("inf")
        try:
            inputs = self._gpt2_tok(
                text,
                return_tensors="pt",
                max_length=128,
                truncation=True,
            )
            if inputs["input_ids"].shape[1] < 2:
                return float("inf")
            with torch.no_grad():
                outputs = self._gpt2(**inputs, labels=inputs["input_ids"])
            return float(torch.exp(outputs.loss))
        except Exception:
            return float("inf")

    # -----------------------------------------------------------------------
    # vocabulary helpers
    # -----------------------------------------------------------------------

    def _ensure_models_loaded(self) -> None:
        """lazy-load dpr encoder, tokenizer, and (optionally) gpt2."""
        if self._encoder is None:
            self._encoder, self._tokenizer = _load_dpr_encoder()
            self._build_vocab_embeddings()
        if self.use_ppl_filter and self._gpt2 is None:
            try:
                self._gpt2, self._gpt2_tok = _load_gpt2()
            except Exception as exc:
                logger.log_error("dpr_optimizer_gpt2_load", exc, {})
                self.use_ppl_filter = False

    def _build_vocab_embeddings(self) -> None:
        """extract and normalize the dpr vocab embedding matrix for fast scoring.

        filters out special tokens (ids < 999) and keeps single-word tokens
        (no ## subword prefix) to ensure triggers are readable.
        """
        # access bert embedding matrix from dpr encoder
        assert self._encoder is not None
        assert self._tokenizer is not None
        embed_weight = (
            self._encoder.ctx_encoder.bert_model.embeddings.word_embeddings.weight.detach()
        )  # [vocab_size, d]

        # filter to single-word, printable, alphabetic tokens
        vocab_size = self._tokenizer.vocab_size
        good_ids = []
        good_tokens = []
        for tok_id in range(1000, min(vocab_size, 30000)):
            tok = self._tokenizer.convert_ids_to_tokens(tok_id)
            if tok is None:
                continue
            # keep only clean word tokens (no ##, no [special], no punct-only)
            if (
                not tok.startswith("##")
                and not tok.startswith("[")
                and tok.isalpha()
                and len(tok) >= 3
            ):
                good_ids.append(tok_id)
                good_tokens.append(tok)

        if not good_ids:
            # fallback: use all ids 1000-10000
            good_ids = list(range(1000, 10000))
            good_tokens = [
                self._tokenizer.convert_ids_to_tokens(i) or str(i) for i in good_ids
            ]

        self._vocab_ids = good_ids
        self._vocab_tokens = good_tokens
        # extract and normalize embeddings for filtered vocab
        id_tensor = torch.tensor(good_ids, dtype=torch.long)
        self._vocab_embeddings = F.normalize(
            embed_weight[id_tensor], dim=-1
        )  # [filtered_vocab, d]

    def _tokens_to_ids(self, tokens: list[str]) -> list[int]:
        """convert token strings to dpr vocab ids."""
        assert self._tokenizer is not None
        assert self._vocab_ids is not None
        return [
            self._tokenizer.convert_tokens_to_ids(t) or self._vocab_ids[0]
            for t in tokens
        ]

    def _ids_to_tokens(self, ids: list[int]) -> list[str]:
        """convert dpr vocab ids to token strings."""
        assert self._tokenizer is not None
        return [self._tokenizer.convert_ids_to_tokens(i) or "unknown" for i in ids]


# ---------------------------------------------------------------------------
# convenience function
# ---------------------------------------------------------------------------


def optimize_dpr_triggers(
    victim_queries: list[str],
    adversarial_passage: str,
    n_tokens: int = 5,
    n_iter: int = 30,
    n_candidates: int = 64,
    use_ppl_filter: bool = True,
    ppl_threshold: float = 500.0,
    seed: int = 42,
) -> DPROptimizedTrigger:
    """convenience function: run dpr hotflip trigger optimization.

    args:
        victim_queries: victim query strings
        adversarial_passage: adversarial passage to retrieve
        n_tokens: trigger length (default 5)
        n_iter: coordinate descent iterations (default 30)
        n_candidates: vocab candidates per step (default 64)
        use_ppl_filter: apply gpt2 perplexity filter (default true)
        ppl_threshold: max perplexity for accepted triggers (default 500)
        seed: rng seed

    returns:
        DPROptimizedTrigger with optimized trigger and metadata
    """
    optimizer = DPRTriggerOptimizer(
        n_tokens=n_tokens,
        n_iter=n_iter,
        n_candidates=n_candidates,
        use_ppl_filter=use_ppl_filter,
        ppl_threshold=ppl_threshold,
        seed=seed,
    )
    return optimizer.optimize(victim_queries, adversarial_passage)
