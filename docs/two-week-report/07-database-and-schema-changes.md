# Database and Schema

This file records the database structure visible in the current source snapshot.
It does not claim that tables, queries, or migrations changed during the last 14 days.

## Current schema (not period-attributed)

The current SQLite storage code creates tables if absent:

- `reports`: timestamp, document text, and serialized summary.
- `claims`: report reference, claim text, verdict, risk score, and serialized details.
- `evidence`: claim reference, text, source, domain, and score.
- `verdicts`: report reference, verdict, risk score, and explanation.

The visible storage functions save reports and nested claim/evidence rows and fetch a report by ID. The database file currently present is `data/verimind.db`. This is a snapshot observation; no migration history or database diff is available.

## Change records

Whether any schema, migration, query, or persisted-data change occurred during the reporting period is unknown.

### Database changes during the reporting period — unverified
- **What changed:** Not determinable from the current snapshot.
- **Why it changed:** Reason: not clear from commit/code, confirm with author.
- **How it was done:** No migration or attributable schema diff is available.
- **Files affected:** The current storage implementation is `verimind/storage.ipynb`; period changes are unknown.
- **Before vs After:** The prior schema and database state are unavailable.
- **Impact:** Cannot attribute data compatibility or persistence impact to recent work.
- **Commit(s):** Unavailable.

See [06-backend-api-changes.md](./06-backend-api-changes.md) for API report persistence behavior.
