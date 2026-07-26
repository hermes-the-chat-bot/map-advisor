# Contributing to MAP Advisor

Thanks for considering a contribution! This project follows a lightweight
process — we value simplicity over bureaucracy.

## How to contribute

1. **Open an issue first** for anything non-trivial (bug reports, feature
   requests, design discussions). This avoids wasted effort.
2. **Fork the repo** and create a feature branch.
3. **Run the tests locally** before pushing:
   ```bash
   pip install -e ".[dev]"
   pytest
   ```
4. **Run linters** (ruff + black):
   ```bash
   ruff check src tests
   black --check src tests
   ```
5. **Open a PR** against `main`. Include a short description of the change
   and reference any related issue.

## Code style

- Python 3.9+
- Type hints on public APIs
- Ruff for linting, Black for formatting
- Type hints are enforced in CI; run `ruff check src tests` and
  `black --check src tests` before committing.

## Adding a new guardrail

Guardrails live in `src/map_advisor/guardrails.py`. Each guardrail is a
small, independently testable function. If you add a new one:

1. Implement the function in `guardrails.py`.
2. Add unit tests in `tests/test_guardrails_*.py` (follow the existing
   `@pytest.mark.guardrail` pattern).
3. Wire it into the orchestrator or specialists in `agents.py` if it
   should apply globally.
4. Add a row to the guardrails table in `README.md`.

## License

By contributing, you agree that your contributions will be licensed under
the MIT License (see `LICENSE`).