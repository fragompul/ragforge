# Contributing to RagForge

Thanks for your interest! RagForge values **small, readable, well-tested code** over feature count.

## Setup

```bash
git clone https://github.com/fragompul/ragforge && cd ragforge
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install           # optional: ruff format + lint on commit
```

## Before you open a PR

```bash
ruff format src tests examples scripts web
ruff check  src tests examples scripts web
mypy src                                   # strict
pytest --cov=ragforge --cov-fail-under=90
```

- Add tests with every behavior change (property-based with Hypothesis where an invariant exists).
- Keep the **core dependency-free**. New integrations belong behind an optional extra and a lazy import
  that raises an `ImportError` naming the `pip install` command.
- Public API additions go in `ragforge/__init__.py` and `CHANGELOG.md`.
- Performance or quality claims need a script in `scripts/` that reproduces them.
- Don't reformat or reorganize unrelated code in the same PR.

## Working on the web playground

`python scripts/build_site.py && python -m http.server -d site 8000`. The UI is plain ES modules (no bundler).
`web/bridge.py` is the only Python glue and is covered by `tests/test_web_bridge.py`.

## Reporting bugs and security issues

Use the issue templates. For vulnerabilities, follow [`SECURITY.md`](SECURITY.md) instead of opening a public issue.

## Extra checks

- Fuzz and stateful tests live in `tests/test_fuzz.py`; run `pytest tests/test_fuzz.py` after touching parsers or storage.
- Mutation testing (`mutmut`) is described in `docs/testing.md`; when a survivor is not an equivalent mutant, add a test.
- Docs: `pip install -e ".[docs]" && mkdocs build --strict`.
