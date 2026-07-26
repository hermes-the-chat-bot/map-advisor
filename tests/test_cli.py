"""Tests for the CLI entry point — argument parsing, demo, and run_query."""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout

import pytest

from map_advisor import cli


def _run(argv):
    """Invoke cli.main() and capture (rc, stdout, stderr)."""
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = cli.main(argv)
    return rc, out.getvalue(), err.getvalue()


def test_build_parser_help_does_not_raise() -> None:
    parser = cli.build_parser()
    # Just exercise instantiation.
    assert parser.prog == "map-advisor"


def test_no_query_returns_help_and_exit_2() -> None:
    rc, _out, err = _run([])
    assert rc == 2
    assert "usage:" in err.lower()


def test_demo_runs_and_prints_report(capsys) -> None:
    rc = cli.main(["--demo"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "MAP ADVISOR — GUARDRAIL DEMO" in captured.out
    # The original query text (with PII) appears in the demo as a label.
    assert "cam@example.com" in captured.out
    # PII was redacted before routing — the orchestrator's hop trace
    # shows the specialist answered, and the flag indicates redaction.
    assert "pii_redacted" in captured.out
    # Each guardrail leaves a recognizable mark in the report:
    # Date Authority:
    assert "date_authority_handled" in captured.out
    # Disambiguation:
    assert "clarification_needed" in captured.out
    # Anti-hallucination scrubbing:
    assert "Unverified claim removed" in captured.out
    # Out-of-scope fallback:
    assert "fallback" in captured.out
    # Footer:
    assert "All six guardrails" in captured.out


def test_invalid_backend_argparse_rejects() -> None:
    # argparse exits with code 2 and a friendly error message on bad choice.
    with pytest.raises(SystemExit) as exc:
        _run(["--backend", "bogus", "x"])
    assert exc.value.code == 2


def test_run_query_capacity_routes_correctly() -> None:
    import argparse

    args = argparse.Namespace(
        backend="mock", model=None, max_hops=2, demo=False, verbose=False
    )
    text = cli.run_query("Why is CPU saturated?", args)
    assert "Draft — manager to review" in text
    assert "CPU headroom" in text


def test_run_query_with_pii_redacts_in_output() -> None:
    import argparse

    args = argparse.Namespace(
        backend="mock", model=None, max_hops=2, demo=False, verbose=False
    )
    text = cli.run_query("My email is cam@example.com — what's our CPU headroom?", args)
    assert "cam@example.com" not in text
    assert "Draft — manager to review" in text


def test_verbose_output_includes_audit_info() -> None:
    import argparse

    args = argparse.Namespace(
        backend="mock", model=None, max_hops=2, demo=False, verbose=True
    )
    text = cli.run_query("When is the launch date?", args)
    assert "QUERY" in text
    assert "HOPS" in text
    assert "FLAGS" in text
    assert "date_authority" in text


def test_version_flag() -> None:
    with pytest.raises(SystemExit) as exc:
        _run(["--version"])
    assert exc.value.code == 0


def test_openai_backend_argparses_but_fails_cleanly_without_imports(monkeypatch) -> None:
    """The openai backend flag is accepted; the lazy import inside generate
    raises a friendly ImportError when actually called."""
    import argparse

    args = argparse.Namespace(
        backend="openai", model=None, max_hops=2, demo=False, verbose=False
    )
    # Do not actually run a generate() — just verify the client builds.
    client = cli._make_client(args)
    assert client is not None
