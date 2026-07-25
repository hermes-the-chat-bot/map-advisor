"""Unit tests for the guardrails module."""

from __future__ import annotations

import pytest

from map_advisor.errors import LoopBreakError
from map_advisor.guardrails import (
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


# ---------------------------------------------------------------------------
# Date Authority Rule
# ---------------------------------------------------------------------------

@pytest.mark.guardrail
class TestDateAuthority:
    @pytest.mark.parametrize(
        "text",
        [
            "When is the capacity review due?",
            "What is the date of the launch?",
            "When will we ship v2?",
            "What's the timeline for the migration?",
            "What is the ETA for the new dashboard?",
            "Q1 milestone?",
            "Tell me about the January 31 deadline.",
            "We need the 2025-01-31 slip date.",
            "What time does the release window start?",
            "how long ago did the rollout happen?",
        ],
    )
    def test_is_date_question_true(self, text: str) -> None:
        assert is_date_question(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "What's our CPU headroom?",
            "Tell me about cloud spend.",
            "Summarize the postmortem.",
            "What is the budget for Q-next?",  # not a clean quarter token
            "Describe CPU and memory utilization.",
        ],
    )
    def test_is_date_question_false(self, text: str) -> None:
        assert is_date_question(text) is False

    def test_empty_string(self) -> None:
        assert is_date_question("") is False
