"""Agents — hub-and-spoke orchestrator plus three specialists.

Layout
------

- :class:`Orchestrator` (the **hub**) owns the conversation, applies the
  date-authority rule, runs the disambiguation protocol, and dispatches to
  specialists. It is the only agent that produces final user-facing answers
  for timeline questions.
- :class:`Specialist` (the **spoke**) is a generic shell bound to a
  :class:`Scope`. It answers within scope, hard-routes out-of-scope and
  date questions back to the hub, and labels every draft so a human can
  review it before release.

Three concrete specialists ship:

- :class:`CapacitySpecialist` — capacity / headroom / SLOs.
- :class:`CostSpecialist` — cloud spend / FinOps.
- :class:`ReliabilitySpecialist` — incidents / SLO burn / postmortems.

Every guardrail lives in :mod:`map_advisor.guardrails`; this module wires
those checks around the LLM calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .errors import (
    DisambiguationError,
    LoopBreakError,
    RoutingError,
    ScopeError,
)
from .guardrails import (
    DRAFT_LABEL,
    LoopBreaker,
    Scope,
    in_scope,
    is_date_question,
    label_draft,
    needs_clarification,
    redact_pii,
    requests_pii,
    scrub_unsupported_claims,
    scope_keywords_match,
)
from .llm import LLMClient, LLMResponse, MockLLMClient

__all__ = [
    "AgentResult",
    "Specialist",
    "Orchestrator",
    "CapacitySpecialist",
    "CostSpecialist",
    "ReliabilitySpecialist",
    "CAPACITY_SCOPE",
    "COST_SCOPE",
    "RELIABILITY_SCOPE",
]


# ---------------------------------------------------------------------------
# Scopes
# ---------------------------------------------------------------------------

CAPACITY_SCOPE = Scope(
    name="capacity",
    description="CPU, memory, disk headroom, autoscaling, and SLO capacity planning.",
    keywords=(
        "capacity", "cpu", "memory", "ram", "disk", "headroom", "autoscal",
        "utilization", "saturation", "queue depth", "SLO", "saturation",
    ),
)
COST_SCOPE = Scope(
    name="cost",
    description="Cloud spend, FinOps, commitment discounts, and unit economics.",
    keywords=(
        "cost", "spend", "finops", "budget", "invoice", "commitment", "discount",
        "savings plan", "reserved", "unit economics", "price",
    ),
)
RELIABILITY_SCOPE = Scope(
    name="reliability",
    description="Incidents, SLO burn, error budgets, postmortems, and mitigation.",
    keywords=(
        "reliability", "incident", "postmortem", "error budget", "SLO burn",
        "outage", "degradation", "mitigation", "pager", "page", "rollback",
        "alert", "sLO",
    ),
)


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """The result of one agent invocation.

    Attributes:
        agent: name of the agent that produced this result.
        role: ``"specialist"`` or ``"orchestrator"``.
        text: the (possibly labeled) text emitted by the agent.
        route_to: optional agent name the result should be routed to next
            (set when a specialist hard-routes back to the orchestrator).
        hops: list of routing hops so far for this query.
        flags: free-form audit flags (e.g. lifted PII, unsupported claims).
        final: True if this is a user-facing final answer (no further
            routing expected).
    """

    agent: str
    role: str
    text: str
    route_to: Optional[str] = None
    hops: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    final: bool = False

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.text


# ---------------------------------------------------------------------------
# Specialist base
# ---------------------------------------------------------------------------

# Hint shown to a specialist when it decides a date question has reached it.
def _back_to_hub_for_date(specialist_name: str) -> Callable[..., str]:
    def make_msg(*_a: Any, **_k: Any) -> str:
        return (
            f"[{specialist_name}→Orchestrator] Date/timeline question received. "
            "Routing back per the Date Authority Rule."
        )
    return make_msg


class Specialist:
    """A single spoke agent.

    Responsibilities:
    - Answer only within :attr:`scope`.
    - Hard-route any date/timeline question back to the hub.
    - Hard-route any out-of-scope question back to the hub.
    - Label every draft with the mandatory manager-review tag.
    - Strip invented-evidence phrasings before returning.
    - Never echo PII; never request PII.
    """

    def __init__(
        self,
        scope: Scope,
        llm: LLMClient,
        *,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.scope = scope
        self.llm = llm
        self.name = scope.name
        self.system_prompt = system_prompt or (
            f"You are the {scope.name} specialist. Scope: {scope.description}. "
            "Answer only within your scope. If asked about dates, timelines, "
            "deadlines, or anything out of scope, say exactly: "
            "'OUT_OF_SCOPE: please route back to the orchestrator.'"
        )

    # -- public ------------------------------------------------------------

    def handle(self, query: str, *, hops: Optional[List[str]] = None) -> AgentResult:
        """Process ``query`` and return either a draft or a back-route.

        The orchestrator always calls this with the latest ``hops`` list so
        each specialist can include the current trace in its result.
        """
        trace = list(hops or [])

        # 1) Date Authority Rule — hard-route date questions back to the hub.
        if is_date_question(query):
            return AgentResult(
                agent=self.name,
                role="specialist",
                text=f"[{self.name}→Orchestrator] Date/timeline question received. "
                "Routing back per the Date Authority Rule.",
                route_to="orchestrator",
                hops=trace + [self.name],
                final=False,
            )

        # 2) Scope Restriction — hard-route out-of-scope questions.
        if not in_scope(query, self.scope):
            return AgentResult(
                agent=self.name,
                role="specialist",
                text=f"[{self.name}→Orchestrator] Query is out of scope "
                f"({self.scope.name}). Routing back per Scope Restriction.",
                route_to="orchestrator",
                hops=trace + [self.name],
                final=False,
            )

        # 3) In scope — call the LLM, then post-process the draft.
        redacted_query = redact_pii(query)
        response = self.llm.generate(
            [{"role": "user", "content": redacted_query}],
            system_prompt=self.system_prompt,
            metadata={"scope": self.scope.name},
        )
        raw = response.text

        # 4) Catch specialists that try to request PII.
        if requests_pii(raw):
            raw = "[PII request blocked] The assistant attempted to ask for PII."

        # 5) Scrub invented-evidence phrasings.
        scrubed, unsupported_flags = scrub_unsupported_claims(raw)

        # 6) Apply draft labeling.
        labeled = label_draft(scrubed)

        return AgentResult(
            agent=self.name,
            role="specialist",
            text=labeled,
            route_to=None,  # delivered to the orchestrator for review
            hops=trace + [self.name],
            flags=unsupported_flags,
            final=False,  # drafts are never final; manager reviews
        )


# Concrete specialists — minimal subclasses so the demo CLI can label them.
class CapacitySpecialist(Specialist):
    def __init__(self, llm: LLMClient) -> None:
        super().__init__(CAPACITY_SCOPE, llm)


class CostSpecialist(Specialist):
    def __init__(self, llm: LLMClient) -> None:
        super().__init__(COST_SCOPE, llm)


class ReliabilitySpecialist(Specialist):
    def __init__(self, llm: LLMClient) -> None:
        super().__init__(RELIABILITY_SCOPE, llm)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """The hub agent.

    Flow for :meth:`run`:

    1. Redact PII from the inbound query.
    2. Date authority: if the query asks *when*, answer directly (we own it).
    3. Disambiguation: if there is actionable ambiguity, return a single
       clarifying question and wait.
    4. Otherwise route to the best-matching specialist (highest keyword
       overlap; ties broken by declared order).
    5. If the specialist hard-routes back, fold its message into the next
       specialist candidate, respect the loop-breaker.
    6. Final answer is assembled from the specialist draft(s) plus any
       orchestrator-level context.
    """

    def __init__(
        self,
        specialists: Sequence[Specialist],
        llm: Optional[LLMClient] = None,
        *,
        max_hops: int = 2,
        system_prompt: Optional[str] = None,
    ) -> None:
        if not specialists:
            raise RoutingError("Orchestrator needs at least one specialist.")
        self.specialists: List[Specialist] = list(specialists)
        self.llm = llm or MockLLMClient()
        self.max_hops = max_hops
        self.system_prompt = system_prompt or (
            "You are the MAP Advisor orchestrator. "
            "You alone answer timeline/date questions and aggregate specialist drafts. "
            "Never invent evidence. Never request or echo PII."
        )
        # Specialists indexed by scope name for routing back.
        self._by_name: Dict[str, Specialist] = {s.scope.name: s for s in self.specialists}

    # ------------------------------------------------------------------
    def _route_specialist(self, query: str) -> Optional[Specialist]:
        """Pick the highest-overlap specialist for ``query``."""
        best: Optional[Specialist] = None
        best_score = 0
        for spec in self.specialists:
            score = scope_keywords_match(query, spec.scope)
            if score > best_score:
                best = spec
                best_score = score
        return best

    # ------------------------------------------------------------------
    def run(self, query: str) -> AgentResult:
        """Process a user query end-to-end. Returns the final answer."""
        if not query or not query.strip():
            return AgentResult(
                agent="orchestrator",
                role="orchestrator",
                text="Please provide a question.",
                final=True,
            )

        # 1) PII redaction on inbound.
        cleaned_query = redact_pii(query)
        pii_was_redacted = cleaned_query != query

        breaker = LoopBreaker(max_hops=self.max_hops)
        hops: List[str] = ["orchestrator"]

        # 2) Date Authority Rule — the orchestrator answers directly.
        if is_date_question(cleaned_query):
            text = self._answer_date_question(cleaned_query, breaker, hops)
            return AgentResult(
                agent="orchestrator",
                role="orchestrator",
                text=text,
                hops=hops,
                flags=["date_authority_handled"] + (["pii_redacted"] if pii_was_redacted else []),
                final=True,
            )

        # 3) Disambiguation Protocol — pause and ask one clarifying question.
        must_clarify, suggested = needs_clarification(cleaned_query)
        if must_clarify and suggested:
            return AgentResult(
                agent="orchestrator",
                role="orchestrator",
                text=suggested,
                hops=hops,
                flags=["clarification_needed"] + (["pii_redacted"] if pii_was_redacted else []),
                final=False,  # waits for input
            )

        # 4) Route to a specialist.
        spec = self._route_specialist(cleaned_query)
        if spec is None:
            # No keyword match — fall back to own LLM call.
            response = self.llm.generate(
                [{"role": "user", "content": cleaned_query}],
                system_prompt=self.system_prompt,
                metadata={"role": "orchestrator-fallback"},
            )
            text = response.text
            if requests_pii(text):
                text = "[PII request blocked]"
            return AgentResult(
                agent="orchestrator",
                role="orchestrator",
                text=text,
                hops=hops,
                flags=["fallback"] + (["pii_redacted"] if pii_was_redacted else []),
                final=True,
            )

        # 5) Dispatch + handle hard-routes, observing the loop-breaker.
        try:
            result = self._dispatch(spec, cleaned_query, breaker, hops)
        except LoopBreakError as e:
            result = AgentResult(
                agent="orchestrator",
                role="orchestrator",
                text=(
                    "I routed this query across several specialists but could not "
                    f"settle on a single answer without looping. ({e}) "
                    "Please restate with more specifics so I can route cleanly."
                ),
                hops=hops,
                flags=["loop_broken"],
                final=True,
            )

        # 6) Final assembly — the orchestrator wraps the draft(s).
        if result.route_to is None and result.role == "specialist":
            wrapped = self._synthesize(result)
            return AgentResult(
                agent="orchestrator",
                role="orchestrator",
                text=wrapped,
                hops=hops + result.hops[len(hops):],
                flags=result.flags + (["pii_redacted"] if pii_was_redacted else []),
                final=True,
            )

        # Anything that came back unresolved becomes the orchestrator's answer.
        return AgentResult(
            agent="orchestrator",
            role="orchestrator",
            text=result.text,
            hops=hops,
            flags=result.flags + (["pii_redacted"] if pii_was_redacted else []),
            final=True,
        )

    # ------------------------------------------------------------------
    def _dispatch(
        self,
        spec: Specialist,
        query: str,
        breaker: LoopBreaker,
        hops: List[str],
    ) -> AgentResult:
        """Send to ``spec``; if it hard-routes back, try alternatives."""
        result = spec.handle(query, hops=hops)
        breaker.record(f"{spec.name}")

        attempts: List[str] = [spec.name]
        while result.route_to == "orchestrator":
            # The specialist handed it back. Try the next-best specialist.
            next_spec = self._next_specialist(query, exclude=attempts)
            if next_spec is None:
                # No more candidates — orchestrator must answer itself.
                response = self.llm.generate(
                    [{"role": "user", "content": query}],
                    system_prompt=self.system_prompt,
                    metadata={"role": "orchestrator-final"},
                )
                text = response.text
                if requests_pii(text):
                    text = "[PII request blocked]"
                return AgentResult(
                    agent="orchestrator",
                    role="orchestrator",
                    text=text,
                    hops=hops + attempts[len(hops):],
                    flags=["no_specialist_match"],
                    final=True,
                )
            result = next_spec.handle(query, hops=hops + attempts)
            breaker.record(f"{next_spec.name}")
            attempts.append(next_spec.name)

        return result

    def _next_specialist(self, query: str, *, exclude: Sequence[str]) -> Optional[Specialist]:
        best: Optional[Specialist] = None
        best_score = 0
        for spec in self.specialists:
            if spec.name in exclude:
                continue
            score = scope_keywords_match(query, spec.scope)
            if score > best_score:
                best = spec
                best_score = score
        return best

    # ------------------------------------------------------------------
    def _answer_date_question(self, query: str, breaker: LoopBreaker, hops: List[str]) -> str:
        """The orchestrator answers timeline questions authoritatively."""
        response = self.llm.generate(
            [{"role": "user", "content": query}],
            system_prompt=self.system_prompt + " The orchestrator owns date/timeline authority.",
            metadata={"role": "orchestrator-date-authority"},
        )
        text = response.text
        if requests_pii(text):
            text = "[PII request blocked]"
        scrubed, _ = scrub_unsupported_claims(text)
        return scrubed

    def _synthesize(self, draft: AgentResult) -> str:
        """Wrap the specialist draft with the orchestrator's review note."""
        # Light-touch synthesis — we surface the labeled draft with a small
        # orchestrator pre-amble, never removing the Draft label.
        return (
            "Per the orchestrator's review of a specialist draft:\n"
            f"{draft.text}"
        )


# ---------------------------------------------------------------------------
# Convenience constructor used by tests and the CLI.
# ---------------------------------------------------------------------------

def default_orchestrator(llm: Optional[LLMClient] = None) -> Orchestrator:
    """Build an orchestrator wired with all three default specialists."""
    llm = llm or MockLLMClient()
    return Orchestrator(
        specialists=[
            CapacitySpecialist(llm),
            CostSpecialist(llm),
            ReliabilitySpecialist(llm),
        ],
        llm=llm,
    )
