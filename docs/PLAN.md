# VERIMIND completion plan

## Baseline recorded 2026-11-15

- `venv\Scripts\python.exe -m pytest -q`: **25 passed** in 66.54 seconds.
- `venv\Scripts\python.exe -m ruff check .`: failed with **10 errors**, all from committed notebook cells or notebook source encoding rather than maintained package code.
- The existing held-out evaluation record is 20/60 accuracy (33.33%), 60/60 `unverifiable` predictions, and 0/40 annotated gold pages retrieved.
- This Windows host has no `docker` or `git` executable. `prometheus-client`, `pydantic-settings`, and `redis` are not installed in the project virtual environment. Production features will use documented, tested stdlib fallbacks unless an optional dependency is installed.

## Execution steps

1. Configure linting for maintained Python sources and preserve the measured baseline.
2. Replace global application/pipeline construction with an app factory, lazy pipeline dependency, injectable test fakes, validated settings, structured logs, request IDs, readiness/liveness, metrics, and resilient rate limiting.
3. Make retrieval query-aware using normalized variants, lexical retrieval plus optional lazy dense retrieval, reciprocal-rank fusion, normalized page matching, and auditable retrieval diagnostics.
4. Make verdicts evidence-bound with citations/confidence and explicit degraded failures rather than silently defaulting to `unverifiable`. Use the development split only for threshold calibration.
5. Provide the `eval` package command, timestamped JSON results, a same-model comparison interface, tests, smoke/run scripts, hardened Docker assets, and current documentation.
6. Verify every increment with Ruff then pytest; finish with evaluation, smoke test, Ruff, and two full pytest runs. Docker validation will be attempted and reported accurately.
