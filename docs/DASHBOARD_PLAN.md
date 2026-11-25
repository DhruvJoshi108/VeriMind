# Dashboard implementation plan

1. Replace the single-file prototype with an HTTP-only Streamlit multipage application. The browser dashboard will never import pipeline/model/retrieval modules.
2. Centralize environment/session configuration, typed API calls with request IDs and friendly error categories, Prometheus parsing, and real evaluation-artifact loading in `dashboard/lib`.
3. Implement claim, batch, evaluation, metrics, and about pages. The application will render only API responses and saved evaluation artifacts; there is no shipped demo data.
4. Add dashboard-focused unit and Streamlit `AppTest` coverage without weakening the existing suite.
5. Add a minimal dashboard image, compose service, local PowerShell runner, Makefile target, and Windows README instructions.
6. Validate lint and tests after implementation; start API/dashboard locally and exercise API-backed verification and changing metrics. Docker validation is attempted only if Docker is available.

## Decision log

- Evaluation artifacts presently include `results/*.jsonl` and `eval/results/*.json`; the default loader combines both locations and accepts JSONL plus JSON package formats. `VERIMIND_EVAL_RESULTS_DIR` selects another location.
- The dashboard limit for a single request is the API schema limit: 20,000 characters. Batch execution defaults to 50 rows, matching `MAX_CLAIMS_PER_DOC`.
- Batch rows exceeding the API character limit are reported without making a request, and results are cleared whenever the input changes.
- The Compose API runs NLI on CPU because host CUDA is not available inside the container by default; the local PowerShell runner can use the host GPU.
- API request percentiles are computed from recorded request latencies, and retrieval/judge latency and judge errors are exposed only when the corresponding stages actually run. Values unavailable from the active judge configuration remain explicitly unreported.
- The existing no-evidence baseline is a local NLI control, not a same-model LLM comparison. The dashboard does not present it as an LLM result.
