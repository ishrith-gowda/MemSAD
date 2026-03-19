# Claude Code Context: Defense Framework Implementation
# BAIR Memory Agent Security Research - Dr. Xuandong Zhao's Group

---

## Document Metadata

```yaml
document_type: claude_code_context
project: memory-agent-security
module: defense_framework
version: 1.0.0
last_updated: 2026-01-10
research_group: UC Berkeley AI Research (BAIR)
advisor: Dr. Xuandong Zhao
watermarking_methods:
  - Unigram-Watermark (ICLR 2024)
  - Permute-and-Flip Decoder (ICLR 2025)
  - MarkLLM Toolkit
```

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Defense Architecture Overview](#defense-architecture-overview)
3. [Watermarking Theory Foundation](#watermarking-theory-foundation)
4. [Unigram-Watermark Implementation](#unigram-watermark-implementation)
5. [Permute-and-Flip Decoder Implementation](#permute-and-flip-decoder-implementation)
6. [Provenance Tracking System](#provenance-tracking-system)
7. [Memory Entry Watermarking](#memory-entry-watermarking)
8. [Detection and Verification](#detection-and-verification)
9. [Multi-Bit Watermarking for Metadata](#multi-bit-watermarking-for-metadata)
10. [Integration with Memory Systems](#integration-with-memory-systems)
11. [Defense Evaluation Framework](#defense-evaluation-framework)
12. [Performance Optimization](#performance-optimization)
13. [Attack Resistance Analysis](#attack-resistance-analysis)
14. [Deployment Guidelines](#deployment-guidelines)
15. [Quick Reference](#quick-reference)

---

## Executive Summary

This document provides comprehensive implementation guidance for the provenance-aware defense framework using Dr. Xuandong Zhao's watermarking methodologies. The defense framework embeds cryptographic watermarks in memory entries to enable provenance verification and detect adversarial content.

### Defense Objectives

1. **Provenance Verification:** Track the origin and authenticity of all memory entries
2. **Attack Detection:** Identify poisoned or manipulated memories
3. **Minimal Overhead:** Preserve memory system performance and utility
4. **Robustness:** Resist watermark removal and evasion attempts

### Key Defense Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| TPR | True positive rate (poisoned detected) | >95% |
| FPR | False positive rate | <5% |
| DACC | Detection accuracy | >97% |
| ASR-d | Post-defense attack success rate | <20% |
| Overhead | Latency increase | <10% |

---

## Defense Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Provenance-Aware Defense Framework                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                        WRITE PATH PROTECTION                          │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│     Input           Watermark          Provenance         Storage           │
│   ┌───────┐       ┌───────────┐      ┌──────────┐      ┌──────────┐       │
│   │Content│──────▶│ Embedding │─────▶│ Metadata │─────▶│  Vector  │       │
│   │       │       │           │      │ Addition │      │  Store   │       │
│   └───────┘       └───────────┘      └──────────┘      └──────────┘       │
│                         │                  │                               │
│                         ▼                  ▼                               │
│                   ┌───────────┐      ┌──────────┐                         │
│                   │ Unigram   │      │  Entry   │                         │
│                   │ Watermark │      │   Key    │                         │
│                   │  (δ,γ)    │      │   Hash   │                         │
│                   └───────────┘      └──────────┘                         │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                        READ PATH VERIFICATION                         │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│    Query          Retrieval        Verification         Output              │
│   ┌───────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐         │
│   │ User  │─────▶│  Vector  │─────▶│Watermark │─────▶│ Verified │         │
│   │ Query │      │  Search  │      │  Check   │      │ Results  │         │
│   └───────┘      └──────────┘      └──────────┘      └──────────┘         │
│                                          │                                 │
│                                          ▼                                 │
│                                    ┌──────────┐                           │
│                                    │ Z-Score  │                           │
│                                    │Detection │                           │
│                                    │  (z>4)   │                           │
│                                    └──────────┘                           │
│                                          │                                 │
│                              ┌───────────┴───────────┐                    │
│                              │                       │                    │
│                         ┌────▼────┐            ┌─────▼─────┐              │
│                         │VERIFIED │            │UNVERIFIED │              │
│                         │ Return  │            │   Flag    │              │
│                         └─────────┘            └───────────┘              │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                        PROVENANCE CHAIN                               │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐            │
│   │ Origin  │────▶│Modify 1 │────▶│Modify 2 │────▶│ Current │            │
│   │ (t=0)   │     │ (t=1)   │     │ (t=2)   │     │ State   │            │
│   └─────────┘     └─────────┘     └─────────┘     └─────────┘            │
│   Watermark       Watermark       Watermark       Watermark              │
│   Key: K0         Key: K1         Key: K2         Key: Kn                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Defense Components

| Component | Purpose | Implementation |
|-----------|---------|----------------|
| **Watermark Generator** | Embed watermarks in content | Unigram/PF methods |
| **Watermark Detector** | Verify watermark presence | Z-score statistical test |
| **Provenance Tracker** | Track memory origin/history | Cryptographic hashing |
| **Key Manager** | Manage watermark keys | Hierarchical key system |
| **Verification Engine** | Validate memories on retrieval | Combined verification |

---

## Watermarking Theory Foundation

### Unigram Watermarking Theory

The Unigram watermark (Zhao et al., ICLR 2024) extends Kirchenbauer et al.'s KGW watermark with a fixed green-red list split that remains constant across all tokens.

#### Mathematical Foundation

```
Given:
- Vocabulary V with |V| tokens
- Secret key K
- Green list proportion γ ∈ (0,1)
- Bias parameter δ > 0

Watermark Embedding:
1. Generate PRG seed: seed = HMAC(K, "watermark")
2. Partition V into green (G) and red (R) lists using seed
   |G| = γ|V|, |R| = (1-γ)|V|
3. During generation, for each token:
   - Add δ to logits of green tokens
   - Sample from modified distribution

Watermark Detection:
1. Given text T with n tokens
2. Count green tokens: |T ∩ G| = g
3. Under null hypothesis (no watermark): g ~ Binomial(n, γ)
4. Compute z-score: z = (g - γn) / √(γ(1-γ)n)
5. Watermarked if z > z_threshold (typically 4.0)

Detection Power:
- E[z | watermarked] = δ√(n) / √(γ(1-γ))
- Power increases with √n and δ
- Minimum ~50-100 tokens for reliable detection
```

### Permute-and-Flip Theory

The PF decoder (Zhao et al., ICLR 2025) provides distortion-free watermarking that doesn't change the sampling distribution.

```
Algorithm: Permute-and-Flip Sampling
Input: logits l ∈ ℝ^|V|, key K
Output: sampled token

1. Generate random permutation π using HMAC(K, context)
2. Sample u_i ~ Exp(1) for each token i
3. Compute scores: s_i = (l_i + log(u_i)) / temperature
4. Apply permutation: s'_i = s_{π(i)}
5. Select token: t = π^(-1)(argmax(s'))
6. Flip probability: if random() < p_flip, resample uniformly

Detection:
- Statistical test based on permutation consistency
- Distortion-free: P(token | watermarked) = P(token | not watermarked)
```

### Method Selection Guidelines

```python
WATERMARK_METHOD_SELECTION = {
    "unigram": {
        "best_for": [
            "Memories that may undergo editing",
            "Shorter text sequences",
            "When robustness is priority",
        ],
        "advantages": [
            "2x more robust to text editing than KGW",
            "Simple implementation",
            "Efficient detection",
        ],
        "disadvantages": [
            "Slight quality degradation (detectable bias)",
            "Requires minimum token count",
        ],
    },
    "pf_decoder": {
        "best_for": [
            "Quality-critical applications",
            "When distribution preservation needed",
            "Research requiring exact probability analysis",
        ],
        "advantages": [
            "Distortion-free (no quality impact)",
            "Provably optimal stability",
            "No detectable bias in output",
        ],
        "disadvantages": [
            "More complex implementation",
            "Requires generation access (not post-hoc)",
        ],
    },
}
```

---

## Unigram-Watermark Implementation

### Core Implementation

```python
# src/watermark/unigram.py

import hashlib
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)


@dataclass
class UnigramConfig:
    """Configuration for Unigram watermark."""
    gamma: float = 0.25  # Green list proportion
    delta: float = 2.0   # Logit bias for green tokens
    z_threshold: float = 4.0  # Detection threshold
    min_tokens: int = 50  # Minimum tokens for detection
    secret_key: Optional[str] = None  # Will be generated if None
    hash_algorithm: str = "sha256"


class UnigramWatermark:
    """
    Unigram Watermark implementation following Zhao et al. (ICLR 2024).

    This implementation provides:
    - Fixed green-red list partitioning using PRG
    - Logit biasing during generation
    - Z-score based detection
    - Robustness to text editing

    Reference:
    - Paper: "Provable Robust Watermarking for AI-Generated Text"
    - Repository: https://github.com/XuandongZhao/Unigram-Watermark
    """

    def __init__(
        self,
        config: Optional[UnigramConfig] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        **kwargs,
    ):
        self.config = config or UnigramConfig(**kwargs)

        # Generate secret key if not provided
        if self.config.secret_key is None:
            self.config.secret_key = self._generate_secret_key()

        self.master_key = self.config.secret_key

        # Initialize tokenizer
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained("gpt2")
        self.vocab_size = self.tokenizer.vocab_size

        # Generate green list
        self.green_list = self._generate_green_list()
        self.green_list_set = set(self.green_list)

        # Precompute for efficiency
        self.green_mask = self._create_green_mask()

        logger.info(
            f"Initialized UnigramWatermark: gamma={self.config.gamma}, "
            f"delta={self.config.delta}, vocab_size={self.vocab_size}, "
            f"green_list_size={len(self.green_list)}"
        )

    def _generate_secret_key(self) -> str:
        """Generate cryptographically secure secret key."""
        import secrets
        return secrets.token_hex(32)

    def _generate_green_list(self) -> List[int]:
        """
        Generate fixed green list using PRG seeded by secret key.

        The green list is deterministic given the key, enabling detection.
        """
        # Create seed from key
        seed_bytes = hashlib.sha256(
            self.master_key.encode()
        ).digest()
        seed = int.from_bytes(seed_bytes[:4], 'big')

        # Set random state
        rng = np.random.RandomState(seed)

        # Generate random permutation of vocabulary
        permutation = rng.permutation(self.vocab_size)

        # Take first gamma fraction as green list
        green_size = int(self.config.gamma * self.vocab_size)
        green_list = permutation[:green_size].tolist()

        return green_list

    def _create_green_mask(self) -> torch.Tensor:
        """Create binary mask for green tokens."""
        mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        mask[self.green_list] = True
        return mask

    def modify_logits(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Modify logits to bias towards green tokens.

        Args:
            logits: Raw logits from model [batch_size, vocab_size]

        Returns:
            Modified logits with delta added to green tokens
        """
        # Add delta to green token logits
        modified = logits.clone()
        modified[:, self.green_list] += self.config.delta

        return modified

    def watermark_text(
        self,
        text: str,
        model=None,
        max_new_tokens: int = 100,
    ) -> str:
        """
        Generate watermarked text.

        Note: For post-hoc watermarking of existing text, use watermark_post_hoc().
        This method requires a generative model.
        """
        if model is None:
            raise ValueError(
                "Model required for generative watermarking. "
                "Use watermark_post_hoc() for existing text."
            )

        # Encode input
        inputs = self.tokenizer(text, return_tensors="pt")

        # Generate with modified logits
        generated_tokens = []
        current_input = inputs["input_ids"]

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(current_input)
                logits = outputs.logits[:, -1, :]

            # Modify logits
            modified_logits = self.modify_logits(logits)

            # Sample
            probs = torch.softmax(modified_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated_tokens.append(next_token.item())
            current_input = torch.cat([current_input, next_token], dim=-1)

            # Stop if EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def watermark_post_hoc(
        self,
        text: str,
        strength: float = 1.0,
    ) -> str:
        """
        Apply watermark to existing text (post-hoc).

        This method modifies existing text to increase green token proportion
        while preserving semantic meaning.

        Args:
            text: Original text to watermark
            strength: How aggressively to modify (0-1)

        Returns:
            Watermarked text

        Note: Post-hoc watermarking may affect text quality.
        For best results, use generative watermarking with modify_logits().
        """
        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        # For each token, consider replacing with green synonym
        watermarked_tokens = []

        for token in tokens:
            if token in self.green_list_set:
                # Already green, keep it
                watermarked_tokens.append(token)
            else:
                # Try to find green alternative
                if np.random.random() < strength:
                    # Find similar green token (simplified)
                    green_alternative = self._find_green_alternative(token)
                    watermarked_tokens.append(green_alternative or token)
                else:
                    watermarked_tokens.append(token)

        return self.tokenizer.decode(watermarked_tokens, skip_special_tokens=True)

    def _find_green_alternative(self, token: int) -> Optional[int]:
        """
        Find semantically similar green token.

        Simplified implementation - full version would use embeddings.
        """
        # Get token text
        token_text = self.tokenizer.decode([token])

        # Try common alternatives (very simplified)
        alternatives = {
            " the": " a",
            " is": " was",
            " are": " were",
        }

        if token_text in alternatives:
            alt_text = alternatives[token_text]
            alt_tokens = self.tokenizer.encode(alt_text, add_special_tokens=False)
            if len(alt_tokens) == 1 and alt_tokens[0] in self.green_list_set:
                return alt_tokens[0]

        return None

    def detect(
        self,
        text: str,
        return_details: bool = False,
    ) -> float:
        """
        Detect watermark in text.

        Args:
            text: Text to check for watermark
            return_details: If True, return detailed results

        Returns:
            Z-score (higher = more likely watermarked)
        """
        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        n = len(tokens)

        if n < self.config.min_tokens:
            logger.warning(
                f"Text has only {n} tokens, minimum {self.config.min_tokens} "
                f"recommended for reliable detection"
            )

        # Count green tokens
        green_count = sum(1 for t in tokens if t in self.green_list_set)

        # Compute z-score
        expected_green = self.config.gamma * n
        std = np.sqrt(self.config.gamma * (1 - self.config.gamma) * n)

        if std == 0:
            z_score = 0.0
        else:
            z_score = (green_count - expected_green) / std

        if return_details:
            return {
                "z_score": z_score,
                "is_watermarked": z_score > self.config.z_threshold,
                "green_count": green_count,
                "total_tokens": n,
                "green_proportion": green_count / n if n > 0 else 0,
                "expected_proportion": self.config.gamma,
                "p_value": self._compute_p_value(z_score),
            }

        return z_score

    def _compute_p_value(self, z_score: float) -> float:
        """Compute p-value for z-score under null hypothesis."""
        from scipy import stats
        return 1 - stats.norm.cdf(z_score)

    def is_watermarked(
        self,
        text: str,
        threshold: Optional[float] = None,
    ) -> bool:
        """
        Check if text is watermarked.

        Args:
            text: Text to check
            threshold: Z-score threshold (default: config.z_threshold)

        Returns:
            True if watermarked
        """
        threshold = threshold or self.config.z_threshold
        z_score = self.detect(text)
        return z_score > threshold

    def get_detection_metrics(
        self,
        texts: List[str],
        labels: List[bool],
    ) -> Dict[str, float]:
        """
        Compute detection metrics on a dataset.

        Args:
            texts: List of texts
            labels: True labels (True = watermarked)

        Returns:
            Dictionary of metrics (TPR, FPR, AUROC, etc.)
        """
        from sklearn.metrics import roc_auc_score, roc_curve

        z_scores = [self.detect(text) for text in texts]
        predictions = [z > self.config.z_threshold for z in z_scores]

        # True positives, false positives, etc.
        tp = sum(1 for p, l in zip(predictions, labels) if p and l)
        fp = sum(1 for p, l in zip(predictions, labels) if p and not l)
        tn = sum(1 for p, l in zip(predictions, labels) if not p and not l)
        fn = sum(1 for p, l in zip(predictions, labels) if not p and l)

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        # AUROC
        try:
            auroc = roc_auc_score(labels, z_scores)
        except ValueError:
            auroc = 0.5

        return {
            "TPR": tpr,
            "FPR": fpr,
            "AUROC": auroc,
            "accuracy": (tp + tn) / len(labels),
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "recall": tpr,
        }


class EntryLevelWatermark(UnigramWatermark):
    """
    Entry-level watermarking for memory systems.

    Extends UnigramWatermark with:
    - Entry-specific keys for attribution
    - Provenance metadata embedding
    - Multi-entry batch processing
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entry_keys = {}  # Cache of entry-specific keys

    def generate_entry_key(
        self,
        entry_id: str,
        user_id: str,
        timestamp: str,
    ) -> str:
        """Generate unique key for a memory entry."""
        key_material = f"{self.master_key}:{entry_id}:{user_id}:{timestamp}"
        entry_key = hashlib.sha256(key_material.encode()).hexdigest()

        # Cache for later verification
        self.entry_keys[entry_id] = entry_key

        return entry_key

    def watermark_entry(
        self,
        content: str,
        entry_id: str,
        user_id: str,
        timestamp: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Watermark a memory entry with provenance tracking.

        Returns:
            Tuple of (watermarked_content, provenance_metadata)
        """
        # Generate entry-specific key
        entry_key = self.generate_entry_key(entry_id, user_id, timestamp)

        # Create entry-specific watermarker
        entry_config = UnigramConfig(
            gamma=self.config.gamma,
            delta=self.config.delta,
            secret_key=entry_key,
        )
        entry_watermarker = UnigramWatermark(entry_config, self.tokenizer)

        # Apply watermark
        watermarked = entry_watermarker.watermark_post_hoc(content)

        # Create provenance metadata
        provenance = {
            "entry_id": entry_id,
            "user_id": user_id,
            "timestamp": timestamp,
            "key_hash": hashlib.sha256(entry_key.encode()).hexdigest()[:16],
            "watermark_method": "unigram",
            "gamma": self.config.gamma,
            "delta": self.config.delta,
        }

        return watermarked, provenance

    def verify_entry(
        self,
        content: str,
        provenance: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Verify watermark on a memory entry.

        Returns:
            Verification result with z-score and status
        """
        entry_id = provenance.get("entry_id")

        # Try to reconstruct entry key
        if entry_id in self.entry_keys:
            entry_key = self.entry_keys[entry_id]
        else:
            # Need user_id and timestamp to reconstruct
            user_id = provenance.get("user_id")
            timestamp = provenance.get("timestamp")
            if user_id and timestamp:
                entry_key = self.generate_entry_key(entry_id, user_id, timestamp)
            else:
                return {
                    "verified": False,
                    "reason": "cannot_reconstruct_key",
                    "z_score": 0,
                }

        # Create entry-specific detector
        entry_config = UnigramConfig(
            gamma=provenance.get("gamma", self.config.gamma),
            delta=provenance.get("delta", self.config.delta),
            secret_key=entry_key,
        )
        entry_watermarker = UnigramWatermark(entry_config, self.tokenizer)

        # Detect
        result = entry_watermarker.detect(content, return_details=True)

        return {
            "verified": result["is_watermarked"],
            "z_score": result["z_score"],
            "green_proportion": result["green_proportion"],
            "p_value": result["p_value"],
            "provenance": provenance,
        }
```

---

## Permute-and-Flip Decoder Implementation

### Core Implementation

```python
# src/watermark/pf_decoder.py

import hashlib
import numpy as np
import torch
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logger = logging.getLogger(__name__)


@dataclass
class PFConfig:
    """Configuration for Permute-and-Flip decoder."""
    ngram: int = 8  # Context window for key generation
    temperature: float = 0.9
    top_p: float = 1.0
    flip_probability: float = 0.01  # Probability of random flip
    secret_key: Optional[str] = None
    detection_threshold: float = 0.01  # p-value threshold


class PFDecoder:
    """
    Permute-and-Flip decoder implementation following Zhao et al. (ICLR 2025).

    Key properties:
    - Distortion-free: Does not change the sampling distribution
    - Optimally stable: Up to 2x better quality-stability tradeoff
    - Robust: Survives text modifications

    Reference:
    - Paper: "Permute-and-Flip: An optimally stable and watermarkable decoder"
    - Repository: https://github.com/XuandongZhao/pf-decoding
    """

    def __init__(
        self,
        config: Optional[PFConfig] = None,
        model: Optional[AutoModelForCausalLM] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        **kwargs,
    ):
        self.config = config or PFConfig(**kwargs)

        # Generate secret key if not provided
        if self.config.secret_key is None:
            self.config.secret_key = self._generate_secret_key()

        self.model = model
        self.tokenizer = tokenizer

        if tokenizer:
            self.vocab_size = tokenizer.vocab_size
        else:
            self.vocab_size = 50257  # GPT-2 default

    def _generate_secret_key(self) -> str:
        """Generate cryptographically secure secret key."""
        import secrets
        return secrets.token_hex(32)

    def _get_context_hash(
        self,
        context_tokens: List[int],
    ) -> bytes:
        """
        Generate hash from context for permutation seeding.

        Uses last ngram tokens for context-dependent watermarking.
        """
        # Take last ngram tokens
        context = context_tokens[-self.config.ngram:] if len(context_tokens) >= self.config.ngram else context_tokens

        # Combine with secret key
        context_str = ",".join(map(str, context))
        key_material = f"{self.config.secret_key}:{context_str}"

        return hashlib.sha256(key_material.encode()).digest()

    def _generate_permutation(
        self,
        context_tokens: List[int],
    ) -> np.ndarray:
        """Generate random permutation seeded by context."""
        seed_bytes = self._get_context_hash(context_tokens)
        seed = int.from_bytes(seed_bytes[:4], 'big')

        rng = np.random.RandomState(seed)
        return rng.permutation(self.vocab_size)

    def _pf_sample(
        self,
        logits: torch.Tensor,
        context_tokens: List[int],
    ) -> int:
        """
        Permute-and-Flip sampling.

        Algorithm:
        1. Generate permutation π from context
        2. Sample Gumbel noise for each token
        3. Add noise to logits
        4. Apply permutation
        5. Take argmax of permuted noisy logits
        6. Inverse permutation to get actual token
        7. Flip with small probability for plausible deniability
        """
        # Get permutation
        permutation = self._generate_permutation(context_tokens)
        inverse_perm = np.argsort(permutation)

        # Sample Gumbel noise
        logits_np = logits.cpu().numpy().flatten()
        gumbel_noise = np.random.gumbel(size=self.vocab_size)

        # Apply temperature
        scaled_logits = logits_np / self.config.temperature

        # Add noise
        noisy_logits = scaled_logits + gumbel_noise

        # Apply permutation
        permuted_logits = noisy_logits[permutation]

        # Argmax in permuted space
        permuted_choice = np.argmax(permuted_logits)

        # Inverse permutation
        actual_token = inverse_perm[permuted_choice]

        # Random flip for plausible deniability
        if np.random.random() < self.config.flip_probability:
            # Sample uniformly
            actual_token = np.random.randint(0, self.vocab_size)

        return int(actual_token)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
    ) -> str:
        """
        Generate watermarked text using PF decoder.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated watermarked text
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer required for generation")

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        context_tokens = input_ids[0].tolist()

        generated_tokens = []

        for _ in range(max_new_tokens):
            # Get logits
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :]

            # PF sampling
            next_token = self._pf_sample(logits, context_tokens)

            generated_tokens.append(next_token)
            context_tokens.append(next_token)

            # Update input
            input_ids = torch.cat([
                input_ids,
                torch.tensor([[next_token]])
            ], dim=-1)

            # Stop at EOS
            if next_token == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def detect(
        self,
        text: str,
        context: str = "",
        return_details: bool = False,
    ) -> float:
        """
        Detect PF watermark in text.

        Detection uses statistical test based on permutation consistency.

        Args:
            text: Text to check
            context: Optional context preceding text
            return_details: Return detailed results

        Returns:
            Detection score (higher = more likely watermarked)
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for detection")

        # Tokenize
        if context:
            context_tokens = self.tokenizer.encode(context, add_special_tokens=False)
        else:
            context_tokens = []

        text_tokens = self.tokenizer.encode(text, add_special_tokens=False)

        # For each token, check if it matches PF prediction
        matches = 0
        total = 0

        for i, token in enumerate(text_tokens):
            # Get context for this position
            current_context = context_tokens + text_tokens[:i]

            # Get expected permutation
            permutation = self._generate_permutation(current_context)

            # Check if token is consistent with PF watermark
            # (This is simplified - full detection is more complex)
            token_rank = np.where(permutation == token)[0][0]

            # Token should tend to be early in permutation if watermarked
            if token_rank < self.vocab_size * 0.3:
                matches += 1
            total += 1

        detection_score = matches / total if total > 0 else 0

        if return_details:
            return {
                "score": detection_score,
                "matches": matches,
                "total": total,
                "is_watermarked": detection_score > 0.4,  # Threshold
            }

        return detection_score

    def get_detection_power(
        self,
        num_tokens: int,
    ) -> float:
        """
        Estimate detection power for given text length.

        Returns probability of detecting watermark.
        """
        # Detection power increases with sequence length
        # Approximation based on central limit theorem
        expected_match_rate = 0.5  # Watermarked
        baseline_match_rate = 1.0 / self.vocab_size  # Not watermarked

        # Standard error decreases with sqrt(n)
        std = np.sqrt(expected_match_rate * (1 - expected_match_rate) / num_tokens)

        # Z-score for separation
        z = (expected_match_rate - baseline_match_rate) / std

        # Power from z-score
        from scipy import stats
        power = stats.norm.cdf(z - 1.96)  # One-sided test

        return power
```

---

## Provenance Tracking System

### Implementation

```python
# src/defenses/provenance/tracker.py

import hashlib
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ProvenanceEventType(Enum):
    """Types of provenance events."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    ACCESS = "access"
    VERIFY = "verify"


@dataclass
class ProvenanceEvent:
    """Single event in provenance chain."""
    event_type: ProvenanceEventType
    timestamp: str
    user_id: str
    content_hash: str
    previous_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    signature: Optional[str] = None

    def compute_hash(self) -> str:
        """Compute hash of this event."""
        data = {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "content_hash": self.content_hash,
            "previous_hash": self.previous_hash,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


@dataclass
class ProvenanceChain:
    """Chain of provenance events for a memory entry."""
    entry_id: str
    events: List[ProvenanceEvent] = field(default_factory=list)
    current_hash: Optional[str] = None

    def add_event(self, event: ProvenanceEvent) -> str:
        """Add event to chain and return new hash."""
        # Set previous hash
        event.previous_hash = self.current_hash

        # Compute and set signature
        event.signature = event.compute_hash()

        # Add to chain
        self.events.append(event)
        self.current_hash = event.signature

        return self.current_hash

    def verify_chain(self) -> Tuple[bool, Optional[str]]:
        """
        Verify integrity of entire chain.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.events:
            return True, None

        # Check first event has no previous
        if self.events[0].previous_hash is not None:
            return False, "First event has unexpected previous_hash"

        # Verify chain linkage
        for i in range(len(self.events)):
            event = self.events[i]

            # Verify signature
            computed_hash = event.compute_hash()
            if computed_hash != event.signature:
                return False, f"Event {i} signature mismatch"

            # Verify linkage (except first)
            if i > 0:
                if event.previous_hash != self.events[i-1].signature:
                    return False, f"Event {i} chain linkage broken"

        # Verify current hash
        if self.current_hash != self.events[-1].signature:
            return False, "Current hash mismatch"

        return True, None


class ProvenanceTracker:
    """
    Track provenance of memory entries.

    Provides:
    - Immutable event logging
    - Chain verification
    - Audit trail
    """

    def __init__(self, storage_backend=None):
        self.chains: Dict[str, ProvenanceChain] = {}
        self.storage = storage_backend

    def create_entry(
        self,
        entry_id: str,
        content: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceChain:
        """Create new provenance chain for entry."""
        chain = ProvenanceChain(entry_id=entry_id)

        event = ProvenanceEvent(
            event_type=ProvenanceEventType.CREATE,
            timestamp=datetime.utcnow().isoformat(),
            user_id=user_id,
            content_hash=hashlib.sha256(content.encode()).hexdigest(),
            metadata=metadata or {},
        )

        chain.add_event(event)
        self.chains[entry_id] = chain

        logger.info(f"Created provenance chain for entry {entry_id}")

        return chain

    def record_update(
        self,
        entry_id: str,
        new_content: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record update event in provenance chain."""
        if entry_id not in self.chains:
            raise ValueError(f"No provenance chain for entry {entry_id}")

        chain = self.chains[entry_id]

        event = ProvenanceEvent(
            event_type=ProvenanceEventType.UPDATE,
            timestamp=datetime.utcnow().isoformat(),
            user_id=user_id,
            content_hash=hashlib.sha256(new_content.encode()).hexdigest(),
            metadata=metadata or {},
        )

        return chain.add_event(event)

    def record_access(
        self,
        entry_id: str,
        user_id: str,
        access_type: str = "read",
    ) -> str:
        """Record access event (optional, for audit)."""
        if entry_id not in self.chains:
            return None

        chain = self.chains[entry_id]

        event = ProvenanceEvent(
            event_type=ProvenanceEventType.ACCESS,
            timestamp=datetime.utcnow().isoformat(),
            user_id=user_id,
            content_hash=chain.events[-1].content_hash,  # Same content
            metadata={"access_type": access_type},
        )

        return chain.add_event(event)

    def verify_entry(
        self,
        entry_id: str,
        content: str,
    ) -> Dict[str, Any]:
        """
        Verify entry provenance.

        Checks:
        1. Chain integrity
        2. Content hash matches
        """
        if entry_id not in self.chains:
            return {
                "verified": False,
                "reason": "no_provenance_chain",
            }

        chain = self.chains[entry_id]

        # Verify chain integrity
        is_valid, error = chain.verify_chain()
        if not is_valid:
            return {
                "verified": False,
                "reason": f"chain_integrity_failed: {error}",
            }

        # Verify content hash
        current_hash = hashlib.sha256(content.encode()).hexdigest()
        expected_hash = chain.events[-1].content_hash

        if current_hash != expected_hash:
            return {
                "verified": False,
                "reason": "content_hash_mismatch",
                "expected": expected_hash[:16],
                "actual": current_hash[:16],
            }

        return {
            "verified": True,
            "chain_length": len(chain.events),
            "created_at": chain.events[0].timestamp,
            "last_modified": chain.events[-1].timestamp,
            "created_by": chain.events[0].user_id,
        }

    def get_audit_trail(
        self,
        entry_id: str,
    ) -> List[Dict[str, Any]]:
        """Get complete audit trail for entry."""
        if entry_id not in self.chains:
            return []

        chain = self.chains[entry_id]
        return [asdict(event) for event in chain.events]
```

---

## Memory Entry Watermarking

### Complete Integration

```python
# src/defenses/watermark/memory_integration.py

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import hashlib
import logging

from src.watermark.unigram import UnigramWatermark, UnigramConfig, EntryLevelWatermark
from src.watermark.pf_decoder import PFDecoder, PFConfig
from src.defenses.provenance.tracker import ProvenanceTracker

logger = logging.getLogger(__name__)


class WatermarkedMemorySystem:
    """
    Memory system wrapper with watermarking and provenance tracking.

    Provides:
    - Automatic watermarking on write
    - Verification on read
    - Provenance chain management
    - Multiple watermark method support
    """

    def __init__(
        self,
        memory_system,
        watermark_config: Optional[Dict[str, Any]] = None,
        verification_mode: str = "flag",  # "flag", "filter", "log"
    ):
        self.memory = memory_system
        self.verification_mode = verification_mode

        # Initialize watermarking
        config = watermark_config or {}
        method = config.get("method", "unigram")

        if method == "unigram":
            self.watermarker = EntryLevelWatermark(
                UnigramConfig(
                    gamma=config.get("gamma", 0.25),
                    delta=config.get("delta", 2.0),
                    secret_key=config.get("secret_key"),
                )
            )
        elif method == "pf":
            self.watermarker = PFDecoder(
                PFConfig(
                    ngram=config.get("ngram", 8),
                    secret_key=config.get("secret_key"),
                )
            )
        else:
            raise ValueError(f"Unknown watermark method: {method}")

        # Initialize provenance tracking
        self.provenance = ProvenanceTracker()

        self.method = method
        logger.info(f"Initialized WatermarkedMemorySystem with method={method}")

    def add(
        self,
        messages,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Add memories with watermarking.

        Flow:
        1. Extract memories using underlying system
        2. Apply watermark to each memory
        3. Add provenance metadata
        4. Store watermarked memories
        """
        # Get content string
        if isinstance(messages, str):
            content = messages
        elif isinstance(messages, list):
            content = " ".join(m.get("content", "") for m in messages)
        else:
            content = str(messages)

        # Generate entry ID
        entry_id = hashlib.sha256(
            f"{user_id}:{content}:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]

        timestamp = datetime.utcnow().isoformat()

        # Apply watermark
        if self.method == "unigram":
            watermarked_content, provenance_meta = self.watermarker.watermark_entry(
                content=content,
                entry_id=entry_id,
                user_id=user_id,
                timestamp=timestamp,
            )
        else:
            # PF decoder (would need model for full watermarking)
            watermarked_content = content
            provenance_meta = {
                "entry_id": entry_id,
                "user_id": user_id,
                "timestamp": timestamp,
                "watermark_method": "pf",
            }

        # Create provenance chain
        self.provenance.create_entry(
            entry_id=entry_id,
            content=watermarked_content,
            user_id=user_id,
            metadata=provenance_meta,
        )

        # Prepare enhanced metadata
        enhanced_metadata = {
            **(metadata or {}),
            "provenance": provenance_meta,
            "watermarked": True,
        }

        # Add to underlying memory system
        result = self.memory.add(
            messages=watermarked_content,
            user_id=user_id,
            metadata=enhanced_metadata,
            **kwargs,
        )

        logger.info(f"Added watermarked memory: entry_id={entry_id}")

        return {
            **result,
            "watermark": {
                "entry_id": entry_id,
                "method": self.method,
            },
        }

    def search(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
        verify: bool = True,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Search with optional verification.

        Flow:
        1. Search underlying memory system
        2. Verify watermarks on results
        3. Flag/filter unverified based on mode
        """
        # Search underlying system
        results = self.memory.search(
            query=query,
            user_id=user_id,
            limit=limit,
            **kwargs,
        )

        if not verify:
            return results

        # Verify each result
        verified_results = []
        for result in results:
            verification = self._verify_result(result)
            result["verification"] = verification

            if self.verification_mode == "filter":
                if verification.get("verified", False):
                    verified_results.append(result)
            elif self.verification_mode == "flag":
                verified_results.append(result)
            elif self.verification_mode == "log":
                if not verification.get("verified", False):
                    logger.warning(
                        f"Unverified memory retrieved: {result.get('id', 'unknown')}"
                    )
                verified_results.append(result)

        return verified_results

    def _verify_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Verify watermark on a result."""
        content = result.get("content", result.get("memory", ""))
        provenance = result.get("metadata", {}).get("provenance", {})

        if not provenance:
            return {
                "verified": False,
                "reason": "no_provenance",
            }

        # Verify watermark
        if self.method == "unigram":
            try:
                verification = self.watermarker.verify_entry(content, provenance)
                return verification
            except Exception as e:
                return {
                    "verified": False,
                    "reason": f"verification_error: {str(e)}",
                }
        else:
            # PF verification
            try:
                score = self.watermarker.detect(content)
                return {
                    "verified": score > 0.4,
                    "score": score,
                }
            except Exception as e:
                return {
                    "verified": False,
                    "reason": f"detection_error: {str(e)}",
                }

    def get_provenance(self, entry_id: str) -> List[Dict[str, Any]]:
        """Get provenance chain for an entry."""
        return self.provenance.get_audit_trail(entry_id)

    def update(
        self,
        memory_id: str,
        data: str,
        user_id: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Update memory with watermark preservation."""
        # Get original entry_id from memory
        original = self.memory.get(memory_id)
        entry_id = original.get("metadata", {}).get("provenance", {}).get("entry_id")

        if entry_id:
            # Record update in provenance
            self.provenance.record_update(
                entry_id=entry_id,
                new_content=data,
                user_id=user_id,
            )

        # Re-watermark
        timestamp = datetime.utcnow().isoformat()

        if self.method == "unigram" and entry_id:
            watermarked_data, provenance_meta = self.watermarker.watermark_entry(
                content=data,
                entry_id=entry_id,
                user_id=user_id,
                timestamp=timestamp,
            )
        else:
            watermarked_data = data
            provenance_meta = {"timestamp": timestamp}

        # Update in underlying system
        return self.memory.update(
            memory_id=memory_id,
            data=watermarked_data,
            user_id=user_id,
            metadata={"provenance": provenance_meta},
            **kwargs,
        )
```

---

## Defense Evaluation Framework

### Metrics and Evaluation

```python
# src/evaluation/defense_metrics.py

from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
import logging

logger = logging.getLogger(__name__)


@dataclass
class DefenseMetrics:
    """Comprehensive defense metrics."""
    # Detection metrics
    tpr: float  # True positive rate
    fpr: float  # False positive rate
    tnr: float  # True negative rate
    fnr: float  # False negative rate

    # Aggregated metrics
    accuracy: float
    precision: float
    recall: float
    f1: float
    auroc: float

    # Defense-specific
    asr_before: float  # Attack success rate before defense
    asr_after: float   # Attack success rate after defense
    defense_effectiveness: float  # 1 - (asr_after / asr_before)

    # Overhead metrics
    latency_overhead_pct: float
    storage_overhead_pct: float

    # Robustness
    detection_at_1_fpr: float  # TPR at 1% FPR


class DefenseEvaluator:
    """
    Evaluate defense effectiveness against attacks.
    """

    def __init__(self, defense_system, memory_system):
        self.defense = defense_system
        self.memory = memory_system

    def evaluate_detection(
        self,
        poisoned_entries: List[Dict[str, Any]],
        legitimate_entries: List[Dict[str, Any]],
    ) -> DefenseMetrics:
        """
        Evaluate detection performance.

        Args:
            poisoned_entries: Known poisoned memory entries
            legitimate_entries: Known legitimate entries

        Returns:
            DefenseMetrics with detection statistics
        """
        # Get detection scores
        poisoned_scores = []
        for entry in poisoned_entries:
            result = self.defense._verify_result(entry)
            score = result.get("z_score", result.get("score", 0))
            poisoned_scores.append(score)

        legitimate_scores = []
        for entry in legitimate_entries:
            result = self.defense._verify_result(entry)
            score = result.get("z_score", result.get("score", 0))
            legitimate_scores.append(score)

        # Create labels
        all_scores = poisoned_scores + legitimate_scores
        labels = [0] * len(poisoned_scores) + [1] * len(legitimate_scores)

        # Compute metrics
        return self._compute_metrics(all_scores, labels)

    def _compute_metrics(
        self,
        scores: List[float],
        labels: List[int],  # 1 = legitimate (should be detected as watermarked)
    ) -> DefenseMetrics:
        """Compute comprehensive metrics."""
        scores = np.array(scores)
        labels = np.array(labels)

        # Threshold-based metrics (using z=4 or equivalent)
        threshold = 4.0  # Z-score threshold
        predictions = scores > threshold

        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        tn = np.sum((predictions == 0) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tpr
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # AUROC
        try:
            auroc = roc_auc_score(labels, scores)
        except ValueError:
            auroc = 0.5

        # TPR at 1% FPR
        try:
            fpr_curve, tpr_curve, _ = roc_curve(labels, scores)
            idx = np.argmin(np.abs(fpr_curve - 0.01))
            tpr_at_1_fpr = tpr_curve[idx]
        except:
            tpr_at_1_fpr = 0.0

        return DefenseMetrics(
            tpr=tpr,
            fpr=fpr,
            tnr=tnr,
            fnr=fnr,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            auroc=auroc,
            asr_before=0.0,  # Set externally
            asr_after=0.0,
            defense_effectiveness=0.0,
            latency_overhead_pct=0.0,
            storage_overhead_pct=0.0,
            detection_at_1_fpr=tpr_at_1_fpr,
        )

    def evaluate_attack_mitigation(
        self,
        attack,
        test_queries: List[str],
        user_id: str,
    ) -> Dict[str, float]:
        """
        Evaluate how well defense mitigates an attack.
        """
        # ASR without defense
        asr_before = attack.evaluate(
            self.memory,
            user_id,
            test_queries,
        ).get("ASR-r", 0)

        # ASR with defense (unverified filtered)
        original_mode = self.defense.verification_mode
        self.defense.verification_mode = "filter"

        asr_after = attack.evaluate(
            self.defense,  # Use defense-wrapped system
            user_id,
            test_queries,
        ).get("ASR-r", 0)

        self.defense.verification_mode = original_mode

        effectiveness = 1 - (asr_after / asr_before) if asr_before > 0 else 1.0

        return {
            "asr_before": asr_before,
            "asr_after": asr_after,
            "defense_effectiveness": effectiveness,
            "reduction_pct": (asr_before - asr_after) / asr_before * 100 if asr_before > 0 else 100,
        }

    def measure_overhead(
        self,
        num_operations: int = 100,
    ) -> Dict[str, float]:
        """Measure defense overhead."""
        import time

        # Measure write latency
        write_times_base = []
        write_times_defended = []

        for i in range(num_operations):
            content = f"Test memory content {i}"

            # Base system
            start = time.perf_counter()
            self.memory.add(content, user_id="test")
            write_times_base.append(time.perf_counter() - start)

            # Defended system
            start = time.perf_counter()
            self.defense.add(content, user_id="test")
            write_times_defended.append(time.perf_counter() - start)

        # Measure read latency
        read_times_base = []
        read_times_defended = []

        for i in range(num_operations):
            query = f"test query {i}"

            # Base system
            start = time.perf_counter()
            self.memory.search(query, user_id="test")
            read_times_base.append(time.perf_counter() - start)

            # Defended system
            start = time.perf_counter()
            self.defense.search(query, user_id="test")
            read_times_defended.append(time.perf_counter() - start)

        write_overhead = (
            (np.mean(write_times_defended) - np.mean(write_times_base))
            / np.mean(write_times_base) * 100
        )

        read_overhead = (
            (np.mean(read_times_defended) - np.mean(read_times_base))
            / np.mean(read_times_base) * 100
        )

        return {
            "write_latency_overhead_pct": write_overhead,
            "read_latency_overhead_pct": read_overhead,
            "avg_write_time_ms_base": np.mean(write_times_base) * 1000,
            "avg_write_time_ms_defended": np.mean(write_times_defended) * 1000,
            "avg_read_time_ms_base": np.mean(read_times_base) * 1000,
            "avg_read_time_ms_defended": np.mean(read_times_defended) * 1000,
        }
```

---

## Deployment Guidelines

### Configuration Template

```yaml
# configs/defenses/watermark_production.yaml
defense:
  name: provenance-aware-watermark
  version: 1.0.0

watermark:
  method: unigram  # or pf

  unigram:
    gamma: 0.25
    delta: 2.0
    z_threshold: 4.0
    min_tokens: 50

  pf:
    ngram: 8
    temperature: 0.9
    flip_probability: 0.01

verification:
  mode: flag  # flag, filter, log
  on_failure: warn  # warn, reject, quarantine

provenance:
  enabled: true
  track_access: false  # Enable for full audit
  retention_days: 365

performance:
  cache_verification: true
  batch_verification: true
  max_batch_size: 100

monitoring:
  log_verifications: true
  alert_threshold: 0.1  # Alert if >10% fail verification
```

---

## Quick Reference

### Defense Commands

```bash
# Run defense with watermarking
python -m src.defenses.watermark.run \
    --config configs/defenses/watermark.yaml \
    --memory-system mem0

# Evaluate defense
python -m src.evaluation.defense_eval \
    --defense watermark \
    --attack agentpoison \
    --config configs/evaluation/defense.yaml

# Measure overhead
python -m src.evaluation.overhead \
    --defense watermark \
    --num-operations 1000
```

### Key Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| gamma (γ) | 0.25 | 0.15-0.5 | Green list size (lower = stronger) |
| delta (δ) | 2.0 | 1.0-3.0 | Bias strength (higher = more detectable) |
| z_threshold | 4.0 | 3.0-5.0 | Detection threshold |
| min_tokens | 50 | 30-100 | Minimum for reliable detection |

### Defense Effectiveness Targets

| Metric | Target | Acceptable |
|--------|--------|------------|
| TPR | >95% | >90% |
| FPR | <5% | <10% |
| AUROC | >97% | >95% |
| ASR-d (post-defense) | <20% | <30% |
| Latency overhead | <10% | <20% |

---

## Document Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-10 | Initial defense framework document |

---

## References

1. Unigram-Watermark: arXiv:2306.17439 (ICLR 2024)
2. PF-Decoding: arXiv:2402.05864 (ICLR 2025)
3. SoK Watermarking: arXiv:2411.18479 (IEEE S&P 2025)
4. MarkLLM: https://github.com/THU-BPM/MarkLLM

---

*This document is part of the Memory Agent Security research project.*
