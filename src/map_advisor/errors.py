"""Custom exceptions for MAP Advisor.

Lightweight, semantic exception types so callers (and tests) can distinguish
a guardrail-triggered refusal from a genuine runtime error.
"""

from __future__ import annotations

__all__ = [
    "DisambiguationError",
    "GuardrailError",
    "LoopBreakError",
    "MapAdvisorError",
    "RoutingError",
    "ScopeError",
]


class MapAdvisorError(Exception):
    """Base class for all MAP Advisor errors."""


class GuardrailError(MapAdvisorError):
    """A production guardrail was violated (PII, scope, loop-break, ...)."""


class RoutingError(MapAdvisorError):
    """The orchestrator could not route a query to any specialist."""


class LoopBreakError(GuardrailError):
    """The loop-break protocol halted routing after too many hops."""


class ScopeError(GuardrailError):
    """A specialist was asked something outside its declared scope."""


class DisambiguationError(GuardrailError):
    """A query is ambiguous and needs clarification before routing."""
