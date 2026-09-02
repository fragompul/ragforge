---
name: Bug report
about: Something in ragforge doesn't behave as documented
title: "[Bug] "
labels: bug
---

**Describe the bug**
A clear, concise description of what's wrong.

**Minimal reproduction**
A short, self-contained script using `RagPipeline` (or the specific class
affected) that reproduces the issue. Please avoid pasting an entire
application -- the smaller the reproduction, the faster this gets fixed.

```python
# paste a minimal repro here
```

**Expected behavior**
What you expected to happen instead.

**Actual behavior**
What actually happened, including the full traceback if there's an
exception.

**Environment**
- ragforge version: `python -c "import ragforge; print(ragforge.__version__)"`
- Python version:
- OS:

**Additional context**
Anything else relevant (custom `embed_fn`/`generate_fn`, index size, etc.).
