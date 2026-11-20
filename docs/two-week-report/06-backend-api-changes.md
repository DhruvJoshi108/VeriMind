# Backend and API

This file documents API behavior present in the current snapshot, not changes verified during the reporting period.
Endpoint additions, removals, and business-logic changes need commit history and a prior-version comparison.

## Current API surface (not period-attributed)

The FastAPI application currently defines:

- `GET /health` — returns a simple status response.
- `POST /verify` — validates a text request, runs the verification pipeline, stores the report, and returns the report with an ID.
- `GET /report/{report_id}` — returns a stored report or a 404 when it does not exist.
- `GET /metrics` — returns fixed placeholder metric values in the current code.
- `GET /docs` — returns a compatibility response pointing to `/docs`.

The verification and report routes depend on API-key validation. The request schema currently requires non-empty text up to 20,000 characters. See `api/main.ipynb`, `api/schemas.ipynb`, and `api/deps.ipynb`.

## Change records

No API or backend change can be assigned to a commit in the requested period.

### Backend/API changes during the reporting period — unverified
- **What changed:** Not determinable from the current snapshot.
- **Why it changed:** Reason: not clear from commit/code, confirm with author.
- **How it was done:** No attributable code diff is available.
- **Files affected:** Current implementation is visible in `api/` and `verimind/`; period changes are unknown.
- **Before vs After:** The previous API contract and implementation are unavailable.
- **Impact:** Cannot attribute a behavior, compatibility, or performance impact to recent work.
- **Commit(s):** Unavailable.

See [02-features-added.md](./02-features-added.md) for the product-flow overview.
