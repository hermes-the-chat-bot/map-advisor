"""Command-line entry point for MAP Advisor.

Usage examples::

    # Mock backend (no API key required — the default):
    map-advisor "When is the capacity review due?"
    map-advisor --backend mock "Our prod costs are up 30% — why?"

    # OpenAI-compatible backend (requires `openai` and API creds):
    map-advisor --backend openai --model gpt-4o-mini "Summarize last quarter's spend"

Run with --demo to see each guardrail exercised against canned queries.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, List, Optional, Sequence

from . import __version__
from .agents import default_orchestrator
from .llm import LLMClient, MockLLMClient, OpenAIClient, make_client

__all__ = ["main", "build_parser", "run_query", "run_demo"]


# ---------------------------------------------------------------------------
# Demo canned queries — chosen to exercise every guardrail.
# ---------------------------------------------------------------------------

DEMO_QUERIES: List[str] = [
    "When is the capacity review due?",                          # Date Authority
    "Should I focus on cost or reliability?",                    # Disambiguation
    "My email is cam@example.com — what's our CPU headroom?",       # PII Policy
    "What's the weather in Tokyo?",                               # Scope Restriction (no-one's)
    "As I recall we had an outage. What caused it?",             # Anti-hallucination
    "What is our capacity and also our cost and also our SLOs?", # Loop potential
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="map-advisor",
        description=(
            "MAP Advisor — a lean hub-and-spoke multi-agent "
            "orchestrator with production safety guardrails."
        ),
    )
    p.add_argument("query", nargs="?", help="Your question for the advisor.")
    p.add_argument(
        "--backend",
        choices=["mock", "openai"],
        default="mock",
        help="LLM backend (default: mock, no API key needed).",
    )
    p.add_argument("--model", default=None, help="Model id for the openai backend.")
    p.add_argument(
        "--max-hops", type=int, default=2,
        help="Loop-break threshold (default: 2).",
    )
    p.add_argument(
        "--demo", action="store_true",
        help="Run a canned demo exercising each guardrail, then exit.",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print routing hops and audit flags.",
    )
    p.add_argument("--version", action="version", version=f"map-advisor {__version__}")
    return p


def _make_client(args: argparse.Namespace) -> LLMClient:
    if args.backend == "openai":
        return OpenAIClient(
            model=args.model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
        )

    # Anti-hallucination trigger: when the user mentions "recall"/"believe",
    # the specialist sometimes invents evidence — which the orchestrator
    # scrubs and labels. Demonstrates guardrail #3 in the demo.
    _invented_responses = {
        r"recall":
            "As I recall, the outage was caused by a bad deploy on Tuesday. "
            "I believe the rollback fixed it within 20 minutes. "
            "From memory, the root cause was a config typo.",
        r"believe":
            "I believe the cluster had a saturation event. "
            "Off the top of my head, the cause was a noisy neighbor.",
    }
    _plausible_responses = {
        # Date authority answers (orchestrator owns these).
        r"(when|due|deadline|timeline|eta|ship|go-live|schedule)": (
            "Per the perf calendar, the capacity review is owned by the "
            "orchestrator and scheduled for the start of next quarter. "
            "Specialists are not consulted on dates."
        ),
        # Capacity.
        r"(cpu|headroom|capacity|utilization)": (
            "Current CPU headroom on the primary cluster is ~28% at p95. "
            "No saturation predicted in the next 30 days at current traffic."
        ),
        # Cost.
        r"(cost|spend|finops|budget|invoice)": (
            "Month-over-month cloud spend is up 12%, driven by additional "
            "compute in us-east-1. No commitment discounts are currently "
            "applied. See the FinOps dashboard for line-item detail."
        ),
        # Reliability.
        r"(reliability|incident|outage|slo|postmortem)": (
            "One P1 incident last week, postmortem in progress, error "
            "budget is at 31% burned for the quarter. No persistent "
            "degradation."
        ),
    }
    patterns = {**_invented_responses, **_plausible_responses}
    return MockLLMClient(patterns=patterns)


def run_query(query: str, args: argparse.Namespace) -> str:
    """Run a single query through the orchestrator and return formatted text."""
    client = _make_client(args)
    orch = default_orchestrator(client)
    orch.max_hops = args.max_hops

    result = orch.run(query)
    if args.verbose:
        return _format_verbose(result, query)
    return result.text


def _format_verbose(result: Any, query: str) -> str:
    lines = [
        f"QUERY    : {query}",
        f"FINAL    : {result.final}",
        f"AGENT    : {result.agent} ({result.role})",
        f"HOPS     : {' -> '.join(result.hops) if result.hops else '(none)'}",
        f"FLAGS    : {', '.join(result.flags) if result.flags else '(none)'}",
        "---",
        result.text,
    ]
    return "\n".join(lines)


def run_demo(args: argparse.Namespace) -> str:
    """Run canned queries and return a single multi-paragraph report."""
    client = _make_client(args)
    orch = default_orchestrator(client)
    orch.max_hops = args.max_hops

    blocks: List[str] = ["=" * 78, "MAP ADVISOR — GUARDRAIL DEMO", "=" * 78]
    for i, q in enumerate(DEMO_QUERIES, 1):
        r = orch.run(q)
        blocks.append(
            f"\n[{i}] QUERY: {q}"
            f"\n     HOPS : {' -> '.join(r.hops) if r.hops else '(none)'}"
            f"\n     FLAGS: {', '.join(r.flags) if r.flags else '(none)'}"
            f"\n     ---\n     {r.text}\n"
        )
    blocks.append("=" * 78)
    blocks.append("All six guardrails exercised with the mock backend. No API keys used.")
    return "\n".join(blocks)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.demo:
        print(run_demo(args))
        return 0

    if not args.query:
        parser.print_help(sys.stderr)
        return 2

    try:
        print(run_query(args.query, args))
    except Exception as e:  # pragma: no cover - defensive for CLI
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
