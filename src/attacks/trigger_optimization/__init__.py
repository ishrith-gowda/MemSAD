"""
trigger optimization for agentpoison-style adversarial memory attacks.

this package implements vocabulary coordinate-descent trigger optimization,
approximating the hotflip-based gradient search from chen et al. (neurips 2024)
without requiring gpu-resident model gradients.

public api:
    TriggerOptimizer   — coordinate-descent optimizer
    OptimizedTrigger   — result dataclass
    optimize_agentpoison_triggers — convenience wrapper

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

__all__ = [
    "TriggerOptimizer",
    "OptimizedTrigger",
    "optimize_agentpoison_triggers",
]
