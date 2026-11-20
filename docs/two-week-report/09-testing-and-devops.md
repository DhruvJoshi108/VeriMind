# Testing and DevOps

This file describes tests and delivery configuration visible in the current project snapshot.
It does not claim that tests, CI, containers, or deployment behavior were added or changed during the reporting period.

## Current checks and delivery configuration

- `tests/test_verimind.ipynb` includes checks for claim filtering, knowledge-base facts, model predictions, risk output, pipeline reports, API health/verification, and contradiction probability mapping.
- `retrieval/test_retrieval.ipynb` is a smoke test that checks the FAISS index and metadata and runs a sample search.
- `.github/workflows/ci.yml` configures Python 3.11 and runs dependency installation, Ruff, mypy, and pytest on pushes to `main`/`master` and pull requests.
- `deploy/docker-compose.yml` defines API and dashboard services, ports, restart behavior, and an API health check.
- `deploy/Dockerfile.api` and `deploy/Dockerfile.dashboard` are present in the snapshot.

These are configuration and coverage observations only; no CI run or current test suite was executed for this documentation task.

## Change records

No test, CI/CD, or deployment change can be confirmed for the requested period.

### Testing/DevOps changes during the reporting period — unverified
- **What changed:** Not determinable from the current snapshot.
- **Why it changed:** Reason: not clear from commit/code, confirm with author.
- **How it was done:** No attributable test or infrastructure diff is available.
- **Files affected:** Current examples are listed above; period changes are unknown.
- **Before vs After:** The prior workflow and deployment configuration are unavailable.
- **Impact:** Cannot attribute reliability or delivery improvements to recent work.
- **Commit(s):** Unavailable.
