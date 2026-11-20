# UI and UX Changes

This file describes the dashboard visible in the current workspace snapshot.
It does not attribute visual or interaction changes to the last 14 days because no Git history is available.

## Current dashboard behavior (not period-attributed)

The Streamlit notebook presents a text input and a Verify button. For non-empty input it calls the API, displays an overall risk metric, then renders a section for each claim with verdict, risk score, explanation, and evidence text/source. It warns when the input is empty and displays an API error when the request fails.

## Change records

No UI/UX change can be confirmed for the reporting period.

### UI/UX changes during the reporting period — unverified
- **What changed:** Not determinable from the current snapshot.
- **Why it changed:** Reason: not clear from commit/code, confirm with author.
- **How it was done:** No attributable UI diff or prior screen is available.
- **Files affected:** The current dashboard is in `dashboard/app.ipynb`; whether it changed is unknown.
- **Before vs After:** No prior version or visual comparison is available.
- **Impact:** Cannot attribute a usability impact to recent work.
- **Commit(s):** Unavailable.

See [06-backend-api-changes.md](./06-backend-api-changes.md) for the service called by the dashboard.
