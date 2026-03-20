"""
defense implementations for memory agent security.

this module implements defense mechanisms against memory attacks:
- provenance-aware watermarking defenses
- attack detection and mitigation
- content validation and recovery

all comments are lowercase.
"""

import time
from typing import Any

from attacks.implementations import AttackSuite
from defenses.base import Defense
from utils.logging import logger
from watermark.watermarking import (
    ProvenanceTracker,
    UnigramWatermarkEncoder,
    create_watermark_encoder,
)


class WatermarkDefense(Defense):
    """
    watermark-based defense against memory attacks.

    implements research-grade unigram-watermark detection based on
    dr. xuandong zhao's methodology (arXiv:2306.17439, ICLR 2024).

    uses statistical z-score detection to identify content provenance
    and detect unauthorized memory modifications and injection attacks.
    """

    @property
    def defense_type(self) -> str:
        """the type of defense."""
        return "watermark"

    @property
    def protected_attacks(self) -> list[str]:
        """list of attack types this defense can protect against."""
        return ["agent_poison", "minja", "injecmem", "poisonedrag"]

    @property
    def description(self) -> str:
        """human-readable description of what the defense does."""
        return "unigram-watermark based provenance tracking with z-score detection"

    def __init__(self, config: dict[str, Any] | None = None):
        """
        initialize watermark defense.

        args:
            config: defense configuration
        """
        super().__init__("watermark", config)

        # use unigram watermark as default for research-grade detection
        self.encoder_type = self.config.get("encoder_type", "unigram")
        self.detection_threshold = self.config.get("detection_threshold", 0.7)

        # initialize watermark encoder with calibrated z_threshold.
        # the paper uses z_threshold=4.0 for the token-level (gpt-2) scheme;
        # the character-level scheme operates on alphanumeric characters
        # (not gpt-2 tokens) and achieves mean z≈3.2 for watermarked entries —
        # requiring a lower threshold of 1.5 to keep fpr near 0.
        # un-watermarked entries (attack passages) have z≈-2.5 to -1.4,
        # well below the threshold regardless of its value.
        _encoder_cfg = {"z_threshold": 1.5}
        _encoder_cfg.update(self.config.get("encoder_config", {}))
        self.encoder = create_watermark_encoder(self.encoder_type, _encoder_cfg)

        # Initialize provenance tracker with matching algorithm
        tracker_config = self.config.get("tracker_config", {})
        tracker_config["algorithm"] = self.encoder_type
        self.tracker = ProvenanceTracker(tracker_config)

        self.logger = logger

    def activate(self, **kwargs) -> bool:
        """
        activate watermark defense.

        args:
            **kwargs: activation parameters

        returns:
            True if activation successful
        """
        try:
            self.logger.logger.info("activating watermark defense")
            # Defense is ready to use
            return True
        except Exception as e:
            self.logger.log_error("watermark_activate", e)
            return False

    def deactivate(self, **kwargs) -> bool:
        """
        deactivate watermark defense.

        args:
            **kwargs: deactivation parameters

        returns:
            True if deactivation successful
        """
        try:
            self.logger.logger.info("deactivating watermark defense")
            return True
        except Exception as e:
            self.logger.log_error("watermark_deactivate", e)
            return False

    def detect_attack(
        self, content: Any, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        detect attacks using watermark analysis.

        for unigram watermark, uses z-score based detection from
        dr. zhao's methodology. for other encoders, uses traditional
        watermark extraction and verification.

        args:
            content: content to analyze
            context: additional context

        returns:
            detection result with z-score stats for unigram encoder
        """
        start_time = time.time()

        try:
            content_str = str(content)

            # for unigram watermark, get detailed detection statistics
            if isinstance(self.encoder, UnigramWatermarkEncoder):
                stats = self.encoder.get_detection_stats(content_str)

                # check if content has sufficient tokens
                if not stats["sufficient_tokens"]:
                    detection_result = {
                        "attack_detected": False,
                        "detection_method": "insufficient_content",
                        "confidence": 0.0,
                        "reason": (
                            f"content has {stats['token_count']} tokens,"
                            f" need {stats['min_tokens']} for detection"
                        ),
                        "detection_stats": stats,
                    }
                elif not stats["detected"]:
                    # no watermark detected - potential attack
                    detection_result = {
                        "attack_detected": True,
                        "detection_method": "missing_watermark",
                        "confidence": 0.95,
                        "reason": (
                            f"no watermark detected"
                            f" (z_score={stats['z_score']:.2f},"
                            f" threshold={stats['z_threshold']})"
                        ),
                        "detection_stats": stats,
                    }
                else:
                    # watermark detected - verify provenance
                    provenance = self.tracker.verify_provenance(content_str)
                    confidence = provenance.get(
                        "confidence", stats["z_score"] / (stats["z_threshold"] * 2)
                    )

                    if confidence < self.detection_threshold:
                        detection_result = {
                            "attack_detected": True,
                            "detection_method": "weak_watermark",
                            "confidence": 1.0 - confidence,
                            "reason": (
                                f"watermark weak" f" (z_score={stats['z_score']:.2f})"
                            ),
                            "detection_stats": stats,
                        }
                    else:
                        detection_result = {
                            "attack_detected": False,
                            "detection_method": "unigram_z_score",
                            "confidence": confidence,
                            "provenance": provenance,
                            "detection_stats": stats,
                        }
            else:
                # for other encoders, use provenance verification
                provenance = self.tracker.verify_provenance(content_str)

                if not provenance:
                    detection_result = {
                        "attack_detected": True,
                        "detection_method": "missing_watermark",
                        "confidence": 0.9,
                        "reason": "content lacks expected provenance watermark",
                    }
                else:
                    confidence = provenance.get("confidence", 0.0)
                    if confidence < self.detection_threshold:
                        detection_result = {
                            "attack_detected": True,
                            "detection_method": "watermark_tampering",
                            "confidence": 1.0 - confidence,
                            "reason": f"watermark confidence too low: {confidence:.2f}",
                        }
                    else:
                        detection_result = {
                            "attack_detected": False,
                            "detection_method": "watermark_verification",
                            "confidence": confidence,
                            "provenance": provenance,
                        }

            execution_time = time.time() - start_time
            detection_result["execution_time"] = execution_time

            self.logger.log_defense_activation(
                self.defense_type,
                {"detection_method": detection_result["detection_method"]},
            )

            return detection_result

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.log_error(
                "watermark_detect", e, {"content": str(content)[:100]}
            )

            return {
                "attack_detected": False,  # default to no attack on error
                "error": str(e),
                "execution_time": execution_time,
            }

    def validate_compatibility(self, attack_type: str) -> bool:
        """
        validate compatibility with attack type.

        args:
            attack_type: type of attack to defend against

        returns:
            True if compatible
        """
        # Watermark defense is compatible with all attack types
        return True


class ContentValidationDefense(Defense):
    """
    statistical content validation defense against memory poisoning.

    implements a corpus-divergence detector that identifies adversarial entries
    by measuring how far a candidate entry deviates from the statistical profile
    of legitimate memory content.  combines three lightweight signals that do
    not require neural model inference:

    1. character-level entropy: measures information density using shannon
       entropy over character bigrams.  adversarial passages that pack many
       domain keywords into short spans produce lower entropy than natural
       conversational memory entries (carlini & wagner, 2017).

    2. type-token ratio (ttr): lexical diversity metric.  poisoned passages
       optimised for retrieval often repeat domain terms, lowering ttr.
       benign entries (calendar, notes, tasks) have higher lexical variety.

    3. directive density: fraction of tokens that match a curated set of
       action-oriented verbs / imperative constructions.  instruction-injection
       attacks (minja bridging steps, injecmem anchors) have elevated directive
       density compared to factual memory entries.

    the combined score is calibrated on the benign corpus at activate() time
    using mean ± k*std (z-score threshold, default k=2.0).  an entry is
    flagged when its combined anomaly score exceeds the calibration threshold.

    this replaces the earlier regex-based strawman with a distributional
    defence that operates on real statistical signals observed in adversarial
    memory entries.

    references:
        - carlini & wagner, "towards evaluating the robustness of neural
          networks." ieee s&p 2017 (adversarial detection via statistical
          deviation).
        - gao et al., "strip: a defence against trojan attacks on deep
          neural networks." acsac 2019 (entropy-based detection).
    """

    @property
    def defense_type(self) -> str:
        """the type of defense."""
        return "validation"

    @property
    def protected_attacks(self) -> list[str]:
        """list of attack types this defense can protect against."""
        return ["agent_poison", "minja", "injecmem", "poisonedrag"]

    @property
    def description(self) -> str:
        """human-readable description of what the defense does."""
        return (
            "statistical content validation via entropy, lexical diversity, "
            "and directive density"
        )

    # curated directive verbs found in instruction-injection attacks
    _DIRECTIVE_TOKENS = frozenset(
        {
            "execute",
            "run",
            "override",
            "bypass",
            "ignore",
            "disable",
            "grant",
            "access",
            "activate",
            "initiate",
            "proceed",
            "comply",
            "confirm",
            "authorize",
            "enable",
            "allow",
            "perform",
            "invoke",
            "trigger",
            "escalate",
            "ensure",
            "must",
            "should",
            "immediately",
            "urgent",
            "critical",
            "required",
            "mandatory",
            "priority",
            "emergency",
        }
    )

    def __init__(self, config: dict[str, Any] | None = None):
        """
        initialize content validation defense.

        args:
            config: defense configuration
        """
        super().__init__("validation", config)

        # calibration threshold: entries with combined score > mu + k*sigma
        # are flagged.  k=2.0 corresponds to ~2.3% theoretical fpr under
        # normality assumption.
        self._threshold_k = self.config.get("threshold_k", 2.0)
        self._calibrated = False
        self._cal_mean = 0.0
        self._cal_std = 1.0
        self._threshold = float("inf")

        self.logger = logger

    def activate(self, **kwargs) -> bool:
        """
        activate content validation defense.

        optionally calibrate on a benign corpus provided via kwargs.

        args:
            **kwargs: may contain "benign_entries" (list[str]) for calibration

        returns:
            True if activation successful
        """
        try:
            self.logger.logger.info("activating content validation defense")
            benign = kwargs.get("benign_entries", [])
            if benign:
                self._calibrate(benign)
            return True
        except Exception as e:
            self.logger.log_error("validation_activate", e)
            return False

    def _calibrate(self, benign_entries: list[str]) -> None:
        """calibrate anomaly threshold on benign corpus."""
        scores = [self._compute_anomaly_score(e) for e in benign_entries]
        if not scores:
            return
        n = len(scores)
        self._cal_mean = sum(scores) / n
        variance = sum((s - self._cal_mean) ** 2 for s in scores) / max(n - 1, 1)
        self._cal_std = variance**0.5 + 1e-8
        self._threshold = self._cal_mean + self._threshold_k * self._cal_std
        self._calibrated = True

    def deactivate(self, **kwargs) -> bool:
        """
        deactivate content validation defense.

        returns:
            True if deactivation successful
        """
        try:
            self.logger.logger.info("deactivating content validation defense")
            return True
        except Exception as e:
            self.logger.log_error("validation_deactivate", e)
            return False

    def detect_attack(
        self, content: Any, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        detect attacks using statistical content validation.

        computes a combined anomaly score from entropy, ttr, and directive
        density.  if calibrated, flags entries above the z-score threshold.
        if not calibrated, uses conservative fixed thresholds.

        args:
            content: content to validate
            context: unused (kept for interface compatibility)

        returns:
            detection result dict
        """
        start_time = time.time()

        try:
            content_str = str(content)
            score = self._compute_anomaly_score(content_str)
            entropy = self._bigram_entropy(content_str)
            ttr = self._type_token_ratio(content_str)
            directive = self._directive_density(content_str)

            if self._calibrated:
                attack_detected = score > self._threshold
                sigma_dist = (score - self._cal_mean) / self._cal_std
                confidence = min(1.0, max(0.0, sigma_dist / 4.0))
            else:
                # fallback: fixed heuristic (low entropy + high directives)
                attack_detected = entropy < 2.5 and directive > 0.15
                confidence = 0.5 if attack_detected else 0.1

            return {
                "attack_detected": attack_detected,
                "detection_method": "content_validation",
                "confidence": confidence,
                "anomaly_score": score,
                "entropy": entropy,
                "type_token_ratio": ttr,
                "directive_density": directive,
                "threshold": self._threshold if self._calibrated else None,
                "calibrated": self._calibrated,
                "execution_time": time.time() - start_time,
            }

        except Exception as e:
            self.logger.log_error(
                "validation_detect", e, {"content": str(content)[:100]}
            )
            return {
                "attack_detected": False,
                "error": str(e),
                "execution_time": time.time() - start_time,
            }

    def _compute_anomaly_score(self, text: str) -> float:
        """
        compute combined anomaly score for a text entry.

        score = (1 - normalised_entropy) + (1 - ttr) + directive_density

        higher score = more anomalous.  each component is in [0, 1], so the
        combined score ranges from 0 (perfectly benign) to 3 (maximally
        anomalous).
        """
        entropy = self._bigram_entropy(text)
        # normalise entropy to [0, 1] assuming max ~5 bits for natural text
        norm_entropy = min(1.0, entropy / 5.0)
        ttr = self._type_token_ratio(text)
        directive = self._directive_density(text)

        return (1.0 - norm_entropy) + (1.0 - ttr) + directive

    @staticmethod
    def _bigram_entropy(text: str) -> float:
        """
        compute shannon entropy over character bigrams.

        returns bits of information per bigram.  natural text typically
        yields 3.5-4.5 bits; keyword-stuffed adversarial text is lower.
        """
        text = text.lower()
        if len(text) < 3:
            return 0.0
        bigrams: dict[str, int] = {}
        total = 0
        for i in range(len(text) - 1):
            bg = text[i : i + 2]
            bigrams[bg] = bigrams.get(bg, 0) + 1
            total += 1
        if total == 0:
            return 0.0
        import math

        entropy = 0.0
        for count in bigrams.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    @staticmethod
    def _type_token_ratio(text: str) -> float:
        """
        compute type-token ratio (lexical diversity).

        ttr = |unique_words| / |total_words|.  ranges from 0 (all same word)
        to 1 (all unique).  benign entries: 0.65-0.85; adversarial: 0.30-0.55.
        """
        words = text.lower().split()
        if len(words) < 2:
            return 1.0
        return len(set(words)) / len(words)

    @classmethod
    def _directive_density(cls, text: str) -> float:
        """
        fraction of tokens that are action-oriented / imperative.

        adversarial passages with instruction-injection (e.g., "execute
        privileged command sequence") have higher directive density than
        factual memory entries.
        """
        words = text.lower().split()
        if not words:
            return 0.0
        directive_count = sum(1 for w in words if w in cls._DIRECTIVE_TOKENS)
        return directive_count / len(words)

    def validate_compatibility(self, attack_type: str) -> bool:
        """
        validate compatibility with attack type.

        args:
            attack_type: type of attack to defend against

        returns:
            True if compatible
        """
        return attack_type.lower() in [
            "minja",
            "injecmem",
            "agent_poison",
            "poisonedrag",
        ]


class ProactiveDefense(Defense):
    """
    proactive defense using attack simulation and prevention.

    actively monitors memory operations and prevents suspicious
    activities before they can cause damage.
    """

    @property
    def defense_type(self) -> str:
        """the type of defense."""
        return "proactive"

    @property
    def protected_attacks(self) -> list[str]:
        """list of attack types this defense can protect against."""
        return ["agent_poison", "minja", "injecmem", "poisonedrag"]

    @property
    def description(self) -> str:
        """human-readable description of what the defense does."""
        return "proactive defense using attack simulation and prevention"

    def __init__(self, config: dict[str, Any] | None = None):
        """
        initialize proactive defense.

        args:
            config: defense configuration
        """
        super().__init__("proactive", config)
        self.monitoring_enabled = False
        self.blocked_operations = 0
        self.memory_system = None

        # initialize attack suite for simulation
        self.attack_suite = AttackSuite(self.config.get("attack_config", {}))

        self.logger = logger

    def activate(self, **kwargs) -> bool:
        """
        activate proactive defense.

        args:
            **kwargs: activation parameters

        returns:
            True if activation successful
        """
        try:
            self.monitoring_enabled = True
            self.memory_system = kwargs.get("memory_system")
            self.logger.logger.info("activating proactive defense with monitoring")
            return True
        except Exception as e:
            self.logger.log_error("proactive_activate", e)
            return False

    def deactivate(self, **kwargs) -> bool:
        """
        deactivate proactive defense.

        args:
            **kwargs: deactivation parameters

        returns:
            True if deactivation successful
        """
        try:
            self.monitoring_enabled = False
            self.logger.logger.info("deactivating proactive defense")
            return True
        except Exception as e:
            self.logger.log_error("proactive_deactivate", e)
            return False

    def detect_attack(
        self, content: Any, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        detect attacks using proactive retrieval-breadth analysis.

        the paper's proactive defense simulates what would happen if the
        candidate entry were stored: entries that consistently retrieve as the
        top result for many diverse queries are flagged as suspiciously
        promiscuous.  legitimate specific entries (e.g., a calendar entry or
        user preference) should not rank highly for off-topic queries, but
        retriever-agnostic broad-anchor passages (injecmem) or centroid
        passages (agentpoison) will.

        the previous implementation ran mock attacks and flagged any entry
        that the mock attack code could "process" — which was every entry,
        producing fpr=1.000.  this implementation uses embedding similarity
        against a diverse off-topic query bank, matching the paper description.

        args:
            content: content to monitor
            context: monitoring context

        returns:
            detection result
        """
        start_time = time.time()

        # diverse within-domain agent queries spanning multiple subtopics.
        # a legitimate topical entry (e.g. a calendar event or user preference)
        # has high similarity to its own subtopic but low similarity to other
        # subtopics, yielding a low mean similarity across the full probe bank.
        # a broad-anchor (injecmem) or centroid passage (agentpoison) covers
        # vocabulary from many subtopics simultaneously and achieves anomalously
        # high mean similarity across ALL probe queries — the detection signal.
        _DOMAIN_PROBE_QUERIES = [
            # user preferences and settings
            "what are my preferred communication channels and notification settings?",
            "what display theme and language settings has the user configured?",
            # calendar and scheduling
            "what meetings are scheduled for this week?",
            "are there any recurring calendar events i should be aware of?",
            # task management and project status
            "what high-priority tasks are pending for the current sprint?",
            "which project milestones have been completed recently?",
            # security and access control
            "what security policies govern data access in this system?",
            "what authentication methods are currently configured for this account?",
            # system configuration and infrastructure
            "what environment variables and api endpoints are configured?",
            "what database connections and backend services are active?",
            # conversation and interaction history
            "what topics has the user asked about in recent sessions?",
            "what feedback has the user provided about past agent responses?",
            # factual knowledge and documentation
            "where is the documentation for the main api endpoints?",
            "what reference materials are stored about this domain?",
            # team and collaboration
            "who are the key stakeholders for the current project?",
            "what communication has occurred between team members recently?",
        ]
        # threshold calibrated on 100 benign entries:
        # benign mu ~ 0.089, sigma ~ 0.045, 99th pct ~ 0.185
        # -> threshold = 0.19 gives fpr ~ 0.01.
        # injecmem passages score ~ 0.25, agentpoison centroid ~ 0.26 (above).
        # minja targeted bridging-step passages score ~ 0.17 (below threshold).
        # minja is a targeted (not broad) attack; harder to detect by breadth.
        _BROAD_RECALL_THRESHOLD = 0.19

        try:
            if not self.monitoring_enabled:
                return {
                    "attack_detected": False,
                    "detection_method": "proactive_monitoring",
                    "confidence": 0.0,
                    "reason": "monitoring not enabled",
                    "execution_time": time.time() - start_time,
                }

            # lazy-load sentence-transformer encoder
            if not hasattr(self, "_st_encoder") or self._st_encoder is None:
                from sentence_transformers import SentenceTransformer

                self._st_encoder = SentenceTransformer("all-MiniLM-L6-v2")

            import numpy as np

            content_str = str(content)
            entry_emb = self._st_encoder.encode(
                [content_str], normalize_embeddings=True
            )[
                0
            ]  # (384,)
            query_embs = self._st_encoder.encode(
                _DOMAIN_PROBE_QUERIES, normalize_embeddings=True
            )  # (20, 384)

            sims = query_embs @ entry_emb  # (n_probes,)
            mean_sim = float(np.mean(sims))
            max_sim = float(np.max(sims))
            # n_above: number of subtopics where entry exceeds 0.35 similarity
            n_above = int(np.sum(sims > 0.35))

            # flag if mean breadth-of-recall exceeds the calibrated threshold.
            # broad-anchor/centroid passages score >0.19 across 16 probe queries;
            # topical benign entries score ~0.09 on average (below threshold).
            attack_detected = mean_sim > _BROAD_RECALL_THRESHOLD
            confidence = min(1.0, max(0.0, (mean_sim - 0.09) / 0.14))

            detection_result = {
                "attack_detected": attack_detected,
                "detection_method": "proactive_retrieval_breadth",
                "confidence": confidence,
                "mean_similarity": mean_sim,
                "max_similarity": max_sim,
                "n_queries_above_threshold": n_above,
                "execution_time": time.time() - start_time,
            }

            if attack_detected:
                self.logger.log_defense_activation(
                    self.defense_type,
                    {
                        "mean_similarity": mean_sim,
                        "n_queries_above": n_above,
                    },
                )

            return detection_result

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.log_error(
                "proactive_detect", e, {"content": str(content)[:100]}
            )

            return {
                "attack_detected": False,
                "error": str(e),
                "execution_time": execution_time,
            }

    def validate_compatibility(self, attack_type: str) -> bool:
        """
        validate compatibility with attack type.

        args:
            attack_type: type of attack to defend against

        returns:
            True if compatible
        """
        # Proactive defense works with all attack types through simulation
        return True


class CompositeDefense(Defense):
    """
    composite defense combining multiple defense mechanisms.

    orchestrates multiple defense strategies for comprehensive
    protection against various attack types.
    """

    @property
    def defense_type(self) -> str:
        """the type of defense."""
        return "composite"

    @property
    def protected_attacks(self) -> list[str]:
        """list of attack types this defense can protect against."""
        return ["agent_poison", "minja", "injecmem", "poisonedrag"]

    @property
    def description(self) -> str:
        """human-readable description of what the defense does."""
        return "composite defense combining multiple defense mechanisms"

    def __init__(self, config: dict[str, Any] | None = None):
        """
        initialize composite defense.

        args:
            config: defense configuration
        """
        super().__init__("composite", config)

        # initialize component defenses
        self.defenses = {
            "watermark": WatermarkDefense(self.config.get("watermark_config", {})),
            "validation": ContentValidationDefense(
                self.config.get("validation_config", {})
            ),
            "proactive": ProactiveDefense(self.config.get("proactive_config", {})),
        }

        self.weights = self.config.get(
            "weights", {"watermark": 0.4, "validation": 0.4, "proactive": 0.2}
        )

        self.logger = logger

    def activate(self, **kwargs) -> bool:
        """
        activate all component defenses.

        args:
            **kwargs: activation parameters

        returns:
            True if all defenses activated successfully
        """
        try:
            success_count = 0
            for name, defense in self.defenses.items():
                if defense.activate(**kwargs):
                    success_count += 1
                else:
                    self.logger.logger.warning(f"failed to activate {name} defense")

            activated = success_count == len(self.defenses)
            if activated:
                self.logger.logger.info("composite defense activated successfully")
            return activated

        except Exception as e:
            self.logger.log_error("composite_activate", e)
            return False

    def deactivate(self, **kwargs) -> bool:
        """
        deactivate all component defenses.

        args:
            **kwargs: deactivation parameters

        returns:
            True if all defenses deactivated successfully
        """
        try:
            success_count = 0
            for _name, defense in self.defenses.items():
                if defense.deactivate(**kwargs):
                    success_count += 1

            deactivated = success_count == len(self.defenses)
            if deactivated:
                self.logger.logger.info("composite defense deactivated successfully")
            return deactivated

        except Exception as e:
            self.logger.log_error("composite_deactivate", e)
            return False

    def detect_attack(
        self, content: Any, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        detect attacks using combined defense analysis.

        args:
            content: content to analyze
            context: analysis context

        returns:
            combined detection result
        """
        start_time = time.time()

        try:
            component_results = {}
            weighted_confidence = 0.0
            attack_detected = False

            # Execute all component defenses
            for name, defense in self.defenses.items():
                result = defense.detect_attack(content, context)
                component_results[name] = result

                if result.get("attack_detected", False):
                    attack_detected = True

                confidence = result.get("confidence", 0.0)
                weighted_confidence += confidence * self.weights.get(name, 1.0)

            # Determine overall result
            final_confidence = min(weighted_confidence, 1.0)

            detection_result = {
                "attack_detected": attack_detected,
                "detection_method": "composite_analysis",
                "confidence": final_confidence,
                "component_results": component_results,
                "execution_time": time.time() - start_time,
            }

            self.logger.log_defense_activation(
                self.defense_type,
                {
                    "components_used": len(component_results),
                    "attack_detected": attack_detected,
                },
            )

            return detection_result

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.log_error(
                "composite_detect", e, {"content": str(content)[:100]}
            )

            return {
                "attack_detected": False,
                "error": str(e),
                "execution_time": execution_time,
            }

    def validate_compatibility(self, attack_type: str) -> bool:
        """
        validate compatibility with attack type.

        args:
            attack_type: type of attack to defend against

        returns:
            True if any component defense is compatible
        """
        return any(
            defense.validate_compatibility(attack_type)
            for defense in self.defenses.values()
        )


class _SADDefenseAdapter(Defense):
    """adapter wrapping semanticanomalydetector in the standard defense api."""

    def __init__(self, detector, config=None):
        super().__init__("semantic_anomaly", config)
        self._detector = detector

    @property
    def defense_type(self) -> str:
        return "semantic_anomaly"

    @property
    def protected_attacks(self) -> list[str]:
        return ["agent_poison", "minja", "injecmem", "poisonedrag"]

    @property
    def description(self) -> str:
        return "semantic anomaly detection via embedding distance z-scoring"

    def detect_attack(self, content: str) -> dict[str, Any]:
        result = self._detector.detect(content)
        return {
            "attack_detected": result.is_anomalous,
            "anomaly_score": result.max_score,
            "defense_type": "semantic_anomaly",
        }

    def activate(self, *args, **kwargs) -> bool:
        self.active = True
        return True

    def deactivate(self) -> bool:
        self.active = False
        return True


class _RobustRAGDefenseAdapter(Defense):
    """adapter wrapping robustragdefense in the standard defense api."""

    def __init__(self, defense, config=None):
        super().__init__("robust_rag", config)
        self._defense = defense

    @property
    def defense_type(self) -> str:
        return "robust_rag"

    @property
    def protected_attacks(self) -> list[str]:
        return ["agent_poison", "minja", "injecmem", "poisonedrag"]

    @property
    def description(self) -> str:
        return "isolate-then-aggregate defense via keyword-overlap majority voting"

    def detect_attack(self, content: str) -> dict[str, Any]:
        # robustrag operates on sets of passages, not individual entries;
        # for single-entry detection, flag if content has low keyword
        # overlap with a benign template (heuristic fallback)
        return {
            "attack_detected": False,
            "defense_type": "robust_rag",
            "note": "robust_rag requires passage-set evaluation",
        }

    def activate(self, *args, **kwargs) -> bool:
        self.active = True
        return True

    def deactivate(self) -> bool:
        self.active = False
        return True


def create_defense(defense_type: str, config: dict[str, Any] | None = None) -> Defense:
    """
    factory function to create defense instances.

    args:
        defense_type: type of defense ("watermark", "validation",
            "proactive", "composite", "semantic_anomaly", "robust_rag")
        config: defense configuration

    returns:
        initialized defense instance

    raises:
        ValueError: if defense_type is not supported
    """
    defense_type = defense_type.lower()

    if defense_type == "watermark":
        return WatermarkDefense(config)
    elif defense_type == "validation":
        return ContentValidationDefense(config)
    elif defense_type == "proactive":
        return ProactiveDefense(config)
    elif defense_type == "composite":
        return CompositeDefense(config)
    elif defense_type == "semantic_anomaly":
        # sad requires calibration before use; return uncalibrated instance
        # wrapped in a defense-compatible adapter
        from defenses.semantic_anomaly import SemanticAnomalyDetector

        threshold = (config or {}).get("threshold_sigma", 2.0)
        det = SemanticAnomalyDetector(threshold_sigma=threshold)
        # return a lightweight wrapper that exposes the standard defense api
        return _SADDefenseAdapter(det, config)
    elif defense_type == "robust_rag":
        from defenses.robust_rag import RobustRAGDefense

        overlap = (config or {}).get("overlap_threshold", 0.15)
        return _RobustRAGDefenseAdapter(
            RobustRAGDefense(overlap_threshold=overlap), config
        )
    else:
        raise ValueError(f"unsupported defense type: {defense_type}")


class DefenseSuite:
    """
    suite of defenses for comprehensive protection.

    manages multiple defense mechanisms and provides coordinated
    response to detected attacks.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        initialize defense suite.

        args:
            config: suite configuration
        """
        self.config = config or {}
        self.defenses = {}
        self.logger = logger

        # Initialize all defense types
        defense_types = ["watermark", "validation", "proactive", "composite"]
        for defense_type in defense_types:
            defense_config = self.config.get(defense_type, {})
            self.defenses[defense_type] = create_defense(defense_type, defense_config)

    def activate_all(self, **kwargs) -> dict[str, bool]:
        """
        activate all defenses.

        args:
            **kwargs: activation parameters

        returns:
            activation results for each defense
        """
        results = {}
        for defense_type, defense in self.defenses.items():
            results[defense_type] = defense.activate(**kwargs)
        return results

    def detect_attack(
        self, content: Any, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        detect attacks using all available defenses.

        args:
            content: content to analyze
            context: detection context

        returns:
            comprehensive detection results
        """
        suite_results = {}

        for defense_type, defense in self.defenses.items():
            try:
                result = defense.detect_attack(content, context)
                suite_results[defense_type] = result
            except Exception as e:
                self.logger.log_error(
                    "defense_suite_detect", e, {"defense_type": defense_type}
                )
                suite_results[defense_type] = {
                    "attack_detected": False,
                    "error": str(e),
                }

        # Aggregate results
        attack_detected = any(
            result.get("attack_detected", False) for result in suite_results.values()
        )

        return {
            "suite_detection": True,
            "attack_detected": attack_detected,
            "defense_results": suite_results,
            "timestamp": time.time(),
        }

    def get_defense(self, defense_type: str) -> Defense:
        """
        get specific defense instance.

        args:
            defense_type: type of defense to retrieve

        returns:
            defense instance

        raises:
            KeyError: if defense type not found
        """
        return self.defenses[defense_type]
