---
name: Feature request
about: Propose a new capability or improvement for ragforge
title: "[Feature] "
labels: enhancement
---

**What problem does this solve?**
Describe the retrieval/chunking/evaluation/serving problem this addresses,
ideally with a concrete scenario (e.g. "hybrid search still misses X when
the corpus has Y characteristic").

**Proposed approach**
Sketch the API or design you have in mind. Where possible, point to how it
would fit the existing module boundaries (`chunking.py`, `index.py`,
`reranking.py`, `pipeline.py`, ...) rather than introducing a parallel path.

**Alternatives considered**
Other approaches you thought about and why you didn't prefer them.

**Would this require a new dependency?**
ragforge's core has zero required dependencies (see
`docs/architecture.md`). If your proposal needs a third-party package,
explain why it can't follow the lazy-import optional-adapter pattern used
in `src/ragforge/embeddings_providers.py`.
