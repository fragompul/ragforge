# Contributing to ragforge

Thanks for considering a contribution. `ragforge` is a small, focused
codebase with strict quality gates -- the goal is to keep it that way as it
grows.

## Development setup

```bash
git clone https://github.com/fragompul/ragforge.git
cd ragforge
pip install -e ".[dev]"
```

This installs the package in editable mode plus `pytest`, `pytest-cov`,
`ruff`, `mypy`, and `hypothesis`. No other dependency is required to run the
full test suite -- that's deliberate (see `docs/architecture.md`).

Optionally, install the pre-commit hook so `ruff format`/`ruff check` run
automatically on every commit:

```bash
pip install pre-commit
pre-commit install
```

## Before opening a PR

Run the same checks CI runs, in this order:

```bash
ruff format --check src tests examples
ruff check src tests examples
mypy src
pytest --cov=ragforge --cov-report=term-missing --cov-fail-under=90
```

All four must pass. `ruff format` (no `--check`) will fix formatting for
you; the rest need actual fixes.

## Code style and design conventions

- **No new required dependencies in `src/ragforge/`.** Optional integrations
  (real embedding providers, OpenTelemetry) go through a lazy-import adapter
  pattern -- see `src/ragforge/embeddings_providers.py` for the template:
  import the third-party package *inside* the factory function, and raise a
  clear `ImportError` naming the exact `pip install` command if it's
  missing. Never import an optional dependency at module level.
- **Every public class/function gets a docstring that explains *why*, not
  just *what*.** Look at any existing module (`ann.py`, `index.py`) for the
  expected depth: a one-line summary, then the design rationale, trade-offs,
  and pointers to `docs/math.md` where relevant.
- **Validate at construction, not at use.** Classes like `BM25Index` and
  `HNSWIndex` raise `ValueError` in `__init__` for invalid parameters
  (negative weights, out-of-range ratios) rather than failing confusingly
  later.
- **Type hints are mandatory and checked in strict mode** (see
  `[tool.mypy]` in `pyproject.toml`: `disallow_untyped_defs`,
  `disallow_incomplete_defs`, etc.). `mypy src` must report zero issues.
- **Tests accompany the code they exercise, not a separate "tests later"
  pass.** New behavior needs unit tests; new invariants (things that must
  hold for *any* input, not just a chosen example) are good candidates for
  a property-based test in `tests/test_properties.py` -- see that file's
  module docstring for the pattern.
- **Coverage must stay at or above 90%** (`--cov-fail-under=90` in CI). This
  is a floor, not a target: prefer meaningful tests over padding coverage.

## Making a change

1. Open an issue first for anything beyond a small fix, so the design
   direction can be discussed before code is written.
2. Keep PRs focused: one logical change per PR, with tests and docs updated
   in the same PR (not as a follow-up).
3. Update `docs/architecture.md` or `docs/math.md` if the change affects the
   system design or introduces a new scoring/ranking formula -- both are
   meant to stay accurate, not aspirational.
4. Add an entry to `CHANGELOG.md` under `[Unreleased]`.

## Reporting bugs and requesting features

Use the issue templates in `.github/ISSUE_TEMPLATE/`. For a bug report, a
minimal reproduction (a short script using `RagPipeline` or the affected
class directly) is far more useful than a description of symptoms.

## Security issues

Do not open a public issue for a suspected security vulnerability -- see
[`SECURITY.md`](SECURITY.md) for how to report it privately.
