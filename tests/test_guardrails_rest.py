"""Tests for the remaining guardrails: disambiguation, draft labeling,
anti-hallucination, PII, loop-break, and scope restriction."""

from __future__ import annotations

import pytest

from map_advisor.errors import LoopBreakError, ScopeError
from map_advisor.guardrails import (
    DRAFT_LABEL,
    LoopBreaker,
    Scope,
    in_scope,
    label_draft,
    needs_clarification,
    redact_pii,
    requests_pii,
    scrub_unsupported_claims,
    scope_keywords_match,
)


# ---------------------------------------------------------------------------
# Disambiguation Protocol
# ---------------------------------------------------------------------------

@pytest.mark.guardrail
class TestDisambiguation:
    def test_branching_with_pair_triggers(self) -> None:
        text = "Should I focus on cost or reliability for the migration?"
        must, suggested = needs_clarification(text)
        assert must is True
        assert suggested is not None
        assert "cost" in suggested and "reliability" in suggested
        # Exactly one clarifying question — the returned string is a single
        # sentence ending in '?'.
        assert suggested.rstrip().endswith("?")
        assert suggested.count("?") == 1

    def test_pair_without_branching_no_trigger(self) -> None:
        text = "Tell me about cost and reliability."  # statement, not a question
        must, _ = needs_clarification(text)
        assert must is False

    def test_no_ambiguity_no_trigger(self) -> None:
        text = "Why is the database slow?"
        must, suggested = needs_clarification(text)
        assert must is False
        assert suggested is None

    def test_whatabout_triggers(self) -> None:
        text = "What about the new dashboard?"
        must, _ = needs_clarification(text)
        assert must is True

    def test_empty(self) -> None:
        assert needs_clarification("") == (False, None)


# ---------------------------------------------------------------------------
# Anti-hallucination / Draft Labeling
# ---------------------------------------------------------------------------

@pytest.mark.guardrail
class TestDraftLabeling:
    def test_label_prefix(self) -> None:
        out = label_draft("This is a draft.")
        assert out.startswith(f"[{DRAFT_LABEL}]")
        # The label text never bleeds into the body.
        assert "This is a draft." in out

    def test_label_empty(self) -> None:
        assert label_draft("") == ""

    def test_custom_label(self) -> None:
        out = label_draft("x", label="Custom")
        assert out.startswith("[Custom] ")

    def test_scrub_invented_evidence(self) -> None:
        text = "As I recall the cache was rebuilt. Please verify."
        out, flags = scrub_unsupported_claims(text)
        assert flags, "expected at least one flagged phrase"
        assert "[Unverified claim removed]" in out
        # The trailing imperative sentence should survive.
        assert "Please verify" in out

    def test_scrub_no_match_clean_text(self) -> None:
        text = "The cache was rebuilt at 14:02 UTC. Verified by metrics."
        out, flags = scrub_unsupported_claims(text)
        assert flags == []
        assert out == text

    def test_scrub_multiple_phrases(self) -> None:
        text = (
            "As I recall the deploy was bad. "
            "I believe the rollback fixed it in 20 minutes."
        )
        out, flags = scrub_unsupported_claims(text)
        assert len(flags) == 2
        assert out.count("[Unverified claim removed]") == 2

    def test_scrub_empty(self) -> None:
        assert scrub_unsupported_claims("") == ("", [])


# ---------------------------------------------------------------------------
# PII Policy
# ---------------------------------------------------------------------------

@pytest.mark.guardrail
class TestPIIPolicy:
    def test_email_redacted(self) -> None:
        out = redact_pii("Reach me at cam@example.com for details.")
        assert "cam@example.com" not in out
        assert "[REDACTED:email]" in out

    def test_phone_redacted(self) -> None:
        out = redact_pii("Call +1 (415) 555-1234 today.")
        assert "(415) 555-1234" not in out
        assert "[REDACTED:phone]" in out

    def test_ssn_redacted(self) -> None:
        out = redact_pii("SSN on file: 123-45-6789.")
        assert "123-45-6789" not in out
        assert "[REDACTED:ssn]" in out

    def test_handle_redacted(self) -> None:
        out = redact_pii("Ping @cameron-f on Slack.")
        assert "@cameron-f" not in out
        assert "[REDACTED:handle]" in out

    def test_multiple_categories(self) -> None:
        text = "Email cam@example.com or call +1 (415) 555-1234."
        out = redact_pii(text)
        assert "cam@example.com" not in out
        assert "(415) 555-1234" not in out

    def test_no_pii_untouched(self) -> None:
        text = "See the FinOps dashboard for line items."
        assert redact_pii(text) == text

    def test_empty(self) -> None:
        assert redact_pii("") == ""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("What is your name?", True),
            ("Please provide your email.", True),
            ("Your SSN please.", True),
            ("What's our cloud spend?", False),
            ("Tell me about SLO burn.", False),
        ],
    )
    def test_requests_pii(self, text: str, expected: bool) -> None:
        assert requests_pii(text) is expected


# ---------------------------------------------------------------------------
# Loop-Break Protocol
# ---------------------------------------------------------------------------

@pytest.mark.guardrail
class TestLoopBreaker:
    def test_under_limit_ok(self) -> None:
        lb = LoopBreaker(max_hops=2)
        assert lb.record("A") == 1
        assert lb.record("B") == 2
        assert lb.remaining == 0

    def test_over_limit_raises(self) -> None:
        lb = LoopBreaker(max_hops=2)
        lb.record("A")
        lb.record("B")
        with pytest.raises(LoopBreakError) as exc:
            lb.record("C")
        assert "exceeded" in str(exc.value).lower()

    def test_disarmed_does_not_raise(self) -> None:
        lb = LoopBreaker(max_hops=1)
        lb.disarm()
        # Should silently ignore.
        lb.record("A")
        lb.record("B")
        lb.record("C")
        assert lb.remaining == 0 or lb.remaining == 1  # disarmed skips update

    def test_default_two_hops(self) -> None:
        lb = LoopBreaker()
        assert lb.max_hops == 2
        lb.record("hop1")
        lb.record("hop2")
        with pytest.raises(LoopBreakError):
            lb.record("hop3")


# ---------------------------------------------------------------------------
# Scope Restriction
# ---------------------------------------------------------------------------

@pytest.mark.guardrail
class TestScopeRestriction:
    def test_in_scope_match(self) -> None:
        s = Scope(name="x", description="d", keywords=("cpu", "memory"))
        assert in_scope("Tell me about CPU headroom.", s) is True

    def test_in_scope_no_match(self) -> None:
        s = Scope(name="x", description="d", keywords=("cost",))
        assert in_scope("When is the launch?", s) is False

    def test_keyword_match_count(self) -> None:
        s = Scope(name="x", description="d", keywords=("cpu", "memory", "disk"))
        assert scope_keywords_match("CPU and memory utilization", s) == 2
        assert scope_keywords_match("nothing here", s) == 0

    def test_scope_requires_keyword(self) -> None:
        with pytest.raises(ScopeError):
            Scope(name="x", description="d", keywords=())

    def test_empty_text(self) -> None:
        s = Scope(name="x", description="d", keywords=("cpu",))
        assert in_scope("", s) is False
        assert scope_keywords_match("", s) == 0
