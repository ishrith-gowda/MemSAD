"""
trigger optimization for agentpoison-style adversarial memory attacks.

two backends are provided:
    - TriggerOptimizer: vocabulary coordinate-descent using sentence-transformers
      (cpu-friendly, no gradient computation required)
    - DPRTriggerOptimizer: true hotflip gradient-guided search using the facebook
      dpr context encoder (dpr-ctx_encoder-single-nq-base, 768-dim) with optional
      gpt2 perplexity filtering

both produce an OptimizedTrigger / DPROptimizedTrigger result that is compatible
with RetrievalSimulator's triggered-query evaluation protocol.

references:
    chen et al. agentpoison: red-teaming llm agents via poisoning memory or
    knowledge bases. neurips 2024. arxiv:2407.12784.
    github: https://github.com/AI-secure/AgentPoison

    ebrahimi et al. hotflip: white-box adversarial examples for text
    classification. acl 2018. arxiv:1712.06751.
"""

from attacks.trigger_optimization.optimizer import (
    OptimizedTrigger,
    TriggerOptimizer,
    optimize_agentpoison_triggers,
)

_DPR_AVAILABLE = False
try:
    from attacks.trigger_optimization.dpr_optimizer import (
        DPROptimizedTrigger,
        DPRTriggerOptimizer,
        optimize_dpr_triggers,
    )

    _DPR_AVAILABLE = True
except ImportError:
    pass

__all__ = [
    "_DPR_AVAILABLE",
    "DPROptimizedTrigger",
    "DPRTriggerOptimizer",
    "OptimizedTrigger",
    "TriggerOptimizer",
    "optimize_agentpoison_triggers",
    "optimize_dpr_triggers",
]
