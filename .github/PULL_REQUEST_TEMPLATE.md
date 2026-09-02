## What does this PR do?

<!-- One or two sentences. Link the issue it addresses, if any. -->

## Why

<!-- The motivation -- what problem this solves or what it improves. -->

## Checklist

- [ ] `ruff format --check src tests examples` passes
- [ ] `ruff check src tests examples` passes
- [ ] `mypy src` passes with no errors
- [ ] `pytest --cov=ragforge --cov-fail-under=90` passes
- [ ] New behavior has tests; new invariants have a property-based test if
      applicable (`tests/test_properties.py`)
- [ ] `docs/architecture.md` and/or `docs/math.md` updated if this changes
      the system design or introduces a new scoring formula
- [ ] No new required dependency added to `src/ragforge/` (optional
      integrations use the lazy-import adapter pattern, see
      `src/ragforge/embeddings_providers.py`)
