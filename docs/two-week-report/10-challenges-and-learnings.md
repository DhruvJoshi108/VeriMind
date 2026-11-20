# Challenges and Learnings

This file distinguishes verified project constraints from personal challenges or lessons learned during the last 14 days.
The latter cannot be recovered from source code alone and should be confirmed with the author before presenting.

## Evidence-backed constraints in the current project

- The README describes the project as a lightweight prototype with a local-first SQLite store.
- The README identifies TF-IDF and a lightweight scikit-learn classifier as limitations compared with a larger NLI model.
- Live web search is optional; the README says it falls back to local-only retrieval.
- The current `/metrics` endpoint returns fixed placeholder values rather than measured operational metrics.
- The current API request schema caps input text at 20,000 characters.

These are present-state observations, not confirmed challenges encountered or solved in the reporting period.

## Learning and challenge history

No author experience, problem/solution story, or lesson learned can be established from the snapshot. Do not claim a specific challenge or its resolution without confirming it.

For any author-confirmed change, **Why it changed:** Reason: not clear from commit/code, confirm with author.

See [11-pending-and-next-steps.md](./11-pending-and-next-steps.md) for recommendations based on visible limitations.
