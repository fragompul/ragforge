# Testing strategy

Coverage numbers say which lines ran, not whether the tests would notice a bug. RagForge uses four
complementary layers.

| Layer | What it catches | Where |
|---|---|---|
| Example-based unit tests | Intended behaviour, documented edge cases | `tests/test_*.py` |
| Property-based fuzzing ([Hypothesis](https://hypothesis.works)) | Crashes and invariant violations on inputs nobody thought of | `tests/test_fuzz.py`, `tests/test_ann_robustness.py` |
| Stateful model testing | Sequences of operations that corrupt state | `DiskIndexMachine` in `tests/test_fuzz.py` |
| Mutation testing ([mutmut](https://mutmut.readthedocs.io)) | Tests that run code but would not notice it breaking | this page |

## Fuzzing

Every component that parses untrusted input is fuzzed with arbitrary text, bytes and JSON:

* the **MCP handler** must answer any line with a well-formed JSON-RPC reply or nothing, never raise, for every
  method with junk `params` and junk tool `arguments`;
* the **LLM-judge parser** returns a verdict with a score in range or raises `JudgeParseError`, nothing else;
* **guardrails** return findings with valid offsets (`0 <= start <= end <= len(text)`) and a risk in `[0, 1]`;
* **citations** are total and bounded, and every citation quote equals the slice its offsets point at;
* the **ASGI app** always returns a sane status code for any method, path and body;
* **`DiskIndex`** is checked against a plain-dict model under random ingest / delete / commit / reopen /
  compact sequences: after every step the on-disk index must answer exactly what the model says it holds.

## Mutation testing

A mutation tool makes small changes to the source (`<` becomes `<=`, a default changes, an argument is dropped)
and re-runs the tests. A mutant that survives marks behaviour the suite does not pin down.

| Module | Mutants | Killed | Notes |
|---|---:|---:|---|
| `metrics.py` | 165 | 158 (96%) | the 7 survivors are equivalent mutants (see below) |
| `cache.py`, `citations.py`, `routing.py` | 627 | 546 (87%; 17 not covered) | remaining survivors: error-message text of optional-dependency imports, rounding precision of serialized floats |

Running it the first time found real gaps: the default cut-offs of every ranked metric (`k=10`, `k=100` for MAP)
were untested, the cache's `hit` and `expire` boundaries (`>=` vs `>`) were unpinned, router decisions were only
checked loosely and validation messages were never matched. Each survivor that was not an *equivalent mutant* became
a regression test (`tests/test_metrics.py`, `tests/test_mutation_regressions.py`).

An **equivalent mutant** changes the code without changing behaviour: for example `relevance.get(doc_id, 0.0)`
becoming `relevance.get(doc_id, None)` inside a branch that has already filtered out missing documents. Those are
left alone rather than "fixed" with contrived tests.

To reproduce:

```bash
pip install mutmut
# in pyproject.toml:  [tool.mutmut]  source_paths = ["src/ragforge/metrics.py"]
#                     pytest_add_cli_args_test_selection = ["tests/"]  also_copy = ["src/"]
mutmut run && mutmut results
```
