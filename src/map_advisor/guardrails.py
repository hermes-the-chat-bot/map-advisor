"""Production safety guardrails.

Each guardrail is implemented as a small, focused, independently-testable
function. The orchestrator and specialists call into this module rather than
inlining safety logic, so the guardrails are the single source of truth.

Implemented (adapted from a production multi-agent performance advisor
workbench):

1. **Date Authority Rule** — :func:`is_date_question` + specialist hard-route.
2. **Disambiguation Protocol** — :func:`needs_clarification`.
3. **Anti-hallucination / Draft Labeling** — :func:`label_draft`,
   :func:`scrub_unsupported_claims`.
4. **PII Policy** — :func:`redact_pii`.
5. **Loop-Break Protocol** — :class:`LoopBreaker`.
6. **Scope Restriction** — :func:`in_scope`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .errors import LoopBreakError, ScopeError

__all__ = [
    "DRAFT_LABEL",
    "is_date_question",
    "needs_clarification",
    "label_draft",
    "scrub_unsupported_claims",
    "redact_pii",
    "LoopBreaker",
    "in_scope",
    "scope_keywords_match",
]

# ---------------------------------------------------------------------------
# 1) Date Authority Rule
# ---------------------------------------------------------------------------

_DATE_KEYWORDS: Tuple[str, ...] = (
    # explicit timing words
    "date", "dates", "when", "timeline", "milestone", "milestones",
    "deadline", "deadlines", "schedule", "due", "release date",
    "launch date", "ship date", "ship", "go-live", "ETA",
    "Q1", "Q2", "Q3", "Q4",
    # common past-tense triggering
    "what time", "how long ago", "shipped",
    # ISO-ish / month names are detected separately below
)

# Month names (used as anchors; full + abbrev)
_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec",
)

# ISO date patterns — anchored to avoid matching stray digit groups.
_ISO_PATTERNS = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),                 # 2025-01-31
    re.compile(r"\b\d{4}/\d{2}/\d{2}\b"),                # 2025/01/31
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),           # 1/31/25
    re.compile(r"\b\d{1,2}-\d{1,2}-\d{2,4}\b"),           # 1-31-25
    re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b"),        # 31.01.2025
]


def is_date_question(text: str) -> bool:
    """Return True if ``text`` asks about a date, timeline, or schedule.

    The orchestrator uses this to claim authority (only it may answer date
    questions). Specialists call the same check to hard-route date questions
    back upstream instead of inventing a date.
    """
    if not text:
        return False
    lc = text.lower()
    lc_preserved = text  # preserve case for Q1..Q4 tokens

    # Phrase-level questions that always imply asking *when*.
    DATE_PHRASES = (
        "when will", "when does", "when did", "when is", "when are", "when was",
        "when can we", "what is the date", "what's the date", "whats the date",
        "what is the timeline", "what's the timeline", "whats the timeline",
        "what is the deadline", "what's the deadline", "whats the deadline",
        "what is the eta", "what's the eta", "whats the eta",
        "when is it shipping", "when does it ship",
        "expected ship date", "expected launch date", "go-live date",
    )
    for phrase in DATE_PHRASES:
        if phrase in lc:
            return True

    # Tokenize for quarter-token checks (case-sensitive: Q1..Q4 are uppercase).
    words_preserved = {w.strip(".,?!:;") for w in text.split()}
    # Word-boundary keyword search uses lowercased text.
    for kw in _DATE_KEYWORDS:
        if kw in ("Q1", "Q2", "Q3", "Q4"):
            if kw in words_preserved:
                return True
        else:
            if re.search(rf"\b{re.escape(kw.lower())}\b", lc):
                return True

    # Month name within a stretch of the same string — a date-ish mention.
    # Match months case-insensitively against the lowercased text.
    for month in _MONTHS:
        if re.search(rf"\b{re.escape(month.lower())}\b", lc):
            # Month + 4-digit year anywhere in the string.
            if re.search(r"\b(19|20)\d{2}\b", lc):
                return True
            # Month + ordinal day ("Jan 31", "January 5th").
            if re.search(rf"\b{re.escape(month.lower())}\s+\d{{1,2}}(st|nd|rd|th)?\b", lc):
                return True

    # ISO-style date tokens.
    for pat in _ISO_PATTERNS:
        if pat.search(lc):
            return True

    return False


# ---------------------------------------------------------------------------
# 2) Disambiguation Protocol
# ---------------------------------------------------------------------------

# Pairs that commonly produce ambiguity inside an SRE / perf-engineering context.
# Each entry is a tuple of (token_a, token_b). Tokens are matched as whole
# words (case-insensitive, word boundary) to avoid false positives from
# substrings like "a" inside "about". Single-letter tokens are excluded.
_AMBIGUOUS_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("cost", "performance"),
    ("cost", "reliability"),
    ("cost", "throughput"),
    ("latency", "throughput"),
    ("database", "cache"),
    ("frontend", "backend"),
    ("production", "staging"),
    ("dev", "prod"),
    ("cpu", "memory"),
    ("throughput", "latency"),
    ("performance", "reliability"),
)


def _word_in(text: str, word: str) -> bool:
    """Word-boundary, case-insensitive test. Avoids 'a' matching 'about'."""
    if not word:
        return False
    return re.search(rf"\b{re.escape(word.lower())}\b", text.lower()) is not None


def needs_clarification(text: str) -> Tuple[bool, Optional[str]]:
    """Inspect ``text`` for actionable ambiguity.

    Returns a tuple ``(must_clarify, suggested_question)``. ``must_clarify`` is
    True when exactly one clarifying question should be asked *before*
    routing. The suggested question is a hint for the orchestrator and will be
    used verbatim if no better option is available.
    """
    if not text:
        return False, None
    lc = text.lower()

    # Indicators that the user is genuinely unsure AND in a way that matters
    # to which specialist handles it.
    UNCERTAINTY = (" or ", " vs ", " versus ", "difference between", "which one")
    has_branching = any(tok in lc for tok in UNCERTAINTY)

    detected: List[str] = []
    for a, b in _AMBIGUOUS_PAIRS:
        if _word_in(lc, a) and _word_in(lc, b):
            detected.extend([a, b])

    if has_branching and detected:
        topics = " and ".join(detected[:2])
        return True, f"You mentioned {topics}. Could you clarify which one you'd like me to focus on?"

    # Bare mention of both candidates without "or/vs" but with a question mark.
    # We deliberately require BOTH members of a pair AND a question mark — a
    # bare "X and Y in the same sentence" is common and not necessarily
    # ambiguous, so we want the additional signal of a question.
    if detected and "?" in lc and len(detected) >= 2:
        topics = " and ".join(detected[:2])
        return True, f"You mentioned both {topics}. Which of these should I focus on?"

    # Generic "what about X" with no specialist anchor → ambiguous at the hub.
    if lc.strip().startswith(("what about ", "how about ")) and "?" in lc:
        return True, "Could you give me a bit more context about what you're trying to decide?"

    return False, None


# ---------------------------------------------------------------------------
# 3) Anti-hallucination / Draft Labeling
# ---------------------------------------------------------------------------

DRAFT_LABEL = "Draft — manager to review"


def label_draft(text: str, *, label: str = DRAFT_LABEL) -> str:
    """Prefix a specialist's raw output with the mandatory draft label.

    Every specialist reply to the orchestrator is tagged so a human reviewer
    never mistakes a draft for a final answer.
    """
    if not text:
        return text
    return f"[{label}] {text}"


# Phrases that indicate an agent invented evidence instead of citing a real
# source. Detected regardless of case.
_INVENTED_EVIDENCE_PATTERNS: Tuple[str, ...] = (
    "as i recall",
    "i believe",
    "i think it happened",
    "i'm pretty sure",
    "i'm fairly certain",
    "i would guess",
    "probably from",
    "likely from",
    "i assume",
    "i'm guessing",
    "looks like it might",
    "seems to me",
    "from memory",
    "off the top of my head",
)


def scrub_unsupported_claims(text: str) -> Tuple[str, List[str]]:
    """Replace invented-evidence phrasing with a flagged placeholder.

    Returns ``(scrubbed_text, flags)`` where ``flags`` is the list of flagged
    phrases for audit. Replaces the offending sentence prefix with an
    ``[Unverified claim removed]`` marker so the draft is obviously incomplete
    and the manager knows to investigate.
    """
    if not text:
        return text, []
    flags: List[str] = []
    out = text
    for phrase in _INVENTED_EVIDENCE_PATTERNS:
        pattern = rf"(?i)\b{re.escape(phrase)}\b[^.\n]*[.\n]?"
        m = re.search(pattern, out)
        if m:
            flags.append(m.group(0).strip())
            out = re.sub(pattern, "[Unverified claim removed] ", out, count=1)
    return out, flags


# ---------------------------------------------------------------------------
# 4) PII Policy
# ---------------------------------------------------------------------------

# Lightweight redaction patterns. Covers the common PII categories that
# surface in SRE/perf work without pulling a heavy NLP dep.
_PII_PATTERNS: List[Tuple[str, str]] = [
    # email
    (re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"), "[REDACTED:email]"),
    # phone (NA-ish + intl-ish)
    (re.compile(r"\+?\d{1,2}[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}"), "[REDACTED:phone]"),
    # SSN
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED:ssn]"),
    # credit card (16 digits, optional separators)
    (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "[REDACTED:card]"),
    # IPv4
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[REDACTED:ip]"),
    # slack @handle
    (re.compile(r"(?<![\w@])@[A-Za-z][A-Za-z0-9._-]{2,}"), "[REDACTED:handle]"),
]

# Over-broad patterns need周边停止 tokens. Person names are detected by
# "name is X" / "I'm X" / "manager X" style lead-ins rather than full NER.
_NAME_LEAD = re.compile(
    r"(?i)\b(my name is|I am [A-Z]|I'm [A-Z]|manager named [A-Z]|"
    r"contact named [A-Z]|named [A-Z]\w+(?:\s+[A-Z]\w+)?\b)"
)


def redact_pii(text: str) -> str:
    """Redact PII from ``text`` using placeholder tokens.

    Policy: the system *accepts* PII if the user provides it, but it never
    *echoes* it back. Every detected PII span is replaced by a labeled
    placeholder so downstream drafting stays clue-free.
    """
    if not text:
        return text
    out = text
    for pat, repl in _PII_PATTERNS:
        out = pat.sub(repl, out)
    return out


def requests_pii(text: str) -> bool:
    """True if the text has the system *asking the user for* PII.

    This is policy-violating: MAP Advisor never requests PII. Used by
    tests and as a guard on generated specialist outputs.
    """
    if not text:
        return False
    lc = text.lower()
    ASK = (
        "what is your name", "what's your name", "please provide your email",
        "send me your ssn", "your phone number", "your credit card",
        "please share your password", "what's your ssn", "your ssn",
        "your full name", "your birthday", "your date of birth",
        "your address",
    )
    return any(tok in lc for tok in ASK)


# ---------------------------------------------------------------------------
# 5) Loop-Break Protocol
# ---------------------------------------------------------------------------

@dataclass
class LoopBreaker:
    """Cap the number of routing hops before forcing a stop.

    The orchestrator owns one ``LoopBreaker`` per query. Each hop
    (:meth:`record`) can be an O->S or S->O traversal. Once the hop budget is
    exhausted, the next routing call raises :class:`LoopBreakError`, which the
    orchestrator converts into a single user-facing answer rather than
    continuing to bounce the query between agents.

    The threshold defaults to 2 routing hops beyond the initial dispatch,
    matching the production deployment setting.
    """

    max_hops: int = 2
    hops: List[str] = field(default_factory=list)
    armed: bool = True

    def __post_init__(self) -> None:
        # Field initialized by default_factory; nothing to fix here.
        pass

    def record(self, hop_descriptor: str = "") -> int:
        """Record a hop. Returns the new hop count. Raises when exceeded."""
        if not self.armed:
            return len(self.hops)
        self.hops.append(hop_descriptor)
        if len(self.hops) > self.max_hops:
            raise LoopBreakError(
                f"Routing loop limit ({self.max_hops} hops) exceeded: "
                f"trace={' > '.join(self.hops)}"
            )
        return len(self.hops)

    @property
    def remaining(self) -> int:
        """Remaining hops before the breaker trips."""
        return max(0, self.max_hops - len(self.hops))

    def disarm(self) -> None:
        """Disable the breaker (used by tests to observe raw routing)."""
        self.armed = False


# ---------------------------------------------------------------------------
# 6) Scope Restriction
# ---------------------------------------------------------------------------

@dataclass
class Scope:
    """A specialist's declared scope.

    ``name`` is the human label. ``keywords`` is the set of lexical anchors
    used by :func:`in_scope`; queries need only match one keyword (case
    insensitive, word boundary). ``description`` is surfaced in docs.
    """

    name: str
    description: str
    keywords: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.keywords:
            raise ScopeError(f"Scope {self.name!r} must declare at least one keyword")


def scope_keywords_match(text: str, scope: Scope) -> int:
    """Count how many of ``scope.keywords`` appear in ``text`` (0 if none)."""
    if not text:
        return 0
    lc = text.lower()
    return sum(1 for kw in scope.keywords if re.search(rf"\b{re.escape(kw.lower())}\b", lc))


def in_scope(text: str, scope: Scope) -> bool:
    """Decide whether ``text`` falls within ``scope``.

    A query in scope returns True. Out-of-scope queries return False, which
    specialists use to hard-route the query back to the orchestrator instead
    of answering.
    """
    return scope_keywords_match(text, scope) > 0
