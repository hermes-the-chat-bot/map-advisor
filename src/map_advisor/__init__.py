"""MAP Advisor — Open Source Reference Implementation.

A lean hub-and-spoke multi-agent orchestrator with production-grade safety
guardrails. The patterns demonstrated here are adapted from a multi-agent
performance advisor workbench built at a prior employer.

The package is organized so that every piece is importable and testable in
isolation. The public entry points are:

- :class:`map_advisor.orchestrator.Orchestrator` — the hub agent.
- :class:`map_advisor.llm.MockLLMClient` — a stub LLM that needs no API key.
- :func:`map_advisor.cli.main` — the ``map-advisor`` console command.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
