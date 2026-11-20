# VERIMIND

VERIMIND is a lightweight, locally runnable hallucination-detection and verification prototype for LLM outputs.

## Architecture

The project is intentionally modular and offline-friendly. It segments output into claims, retrieves evidence from a local fact base and optional live web search, scores each claim with an explainable risk model, and returns a claim-level audit trail.

## Run with live web verification

VERIMIND can use live web evidence and a hosted, evidence-constrained judge. The judge receives a claim and retrieved passages and is instructed to return `supported`, `refuted`, or `insufficient_evidence`; it must not use outside knowledge. If either provider is unavailable, the application falls back safely to local retrieval and the existing local classifier.

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and configure secrets:

```bash
copy .env.example .env
```

Set `SEARCH_API_KEY` (Tavily), `GROQ_API_KEY` (Groq), and a long random `API_KEY`. Keep `SEARCH_BACKEND=tavily` and `JUDGE_BACKEND=groq`. Do not commit `.env` or put either provider key in a browser application.

3. Run the API:

```bash
uvicorn api.main:app --reload
```

4. Run the dashboard locally:

```powershell
.\venv\Scripts\python.exe -m streamlit run dashboard\app.py
# Or starts API and dashboard in two PowerShell windows:
.\scripts\run_dashboard.ps1
```

## Dashboard

The Streamlit dashboard is an HTTP-only client: it does not import model, FAISS, Torch, or sentence-transformer packages. It provides Verify Claim, Batch Check, Evaluation, System Metrics, and About pages.

Windows setup:

```powershell
.\venv\Scripts\python.exe -m pip install -r dashboard\requirements.txt
$env:VERIMIND_API_URL = "http://127.0.0.1:8000" # base URL, no /verify suffix
.\venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

The preferred way to start both local services is:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_dashboard.ps1
```

Or start the API and dashboard separately using the commands in the first section and `streamlit run dashboard\app.py`. Enter the API key in the sidebar password field, or set `VERIMIND_API_KEY` in the environment before launching Streamlit. The key stays in the current Streamlit session and is never displayed or logged. The default development API accepts requests without a key unless `API_KEY` is configured.

Supported dashboard environment variables are `VERIMIND_API_URL`, `VERIMIND_API_KEY`, `VERIMIND_DASHBOARD_TIMEOUT_SECONDS`, `VERIMIND_DASHBOARD_BATCH_LIMIT`, and `VERIMIND_EVAL_RESULTS_DIR`. The single-claim input limit follows the API schema (20,000 characters); the batch limit defaults to `MAX_CLAIMS_PER_DOC` or 50. Evaluation defaults to `eval/results` and also loads the existing `results/*.jsonl` artifacts; set `VERIMIND_EVAL_RESULTS_DIR` to use another directory. Example claims are drawn from real saved evaluation records, not hardcoded demo values.

`docker compose -f deploy\docker-compose.yml up --build` starts the API and dashboard. The dashboard waits for API health, uses the internal API service URL, and mounts both evaluation-artifact directories read-only. The Compose file requires a local `.env` copied from `.env.example`.

The dashboard reports live request latency percentiles and retrieval/judge timing when those API stages have run. The local judge is disabled by default, so judge latency can honestly show as unavailable. The repository has a local no-evidence NLI baseline, not a same-model LLM comparison; this distinction is shown on Evaluation. API rate limiting and metrics remain in-memory and per-process. Screenshots of the running pages were reviewed during validation; they are not checked into the repository.

5. Run checks:

```bash
pytest -q
ruff check api verimind tests
```

## Deployment

The supplied Docker configuration is under `deploy/`. Before publishing an API, set these hosting-provider secrets/environment variables:

```text
ENVIRONMENT=production
API_KEY=<long-random-secret>
SEARCH_BACKEND=tavily
SEARCH_API_KEY=<tavily-secret>
JUDGE_BACKEND=groq
GROQ_API_KEY=<groq-secret>
GROQ_MODEL=llama-3.3-70b-versatile
CORS_ORIGINS=https://your-dashboard.example.com
```

Build/run locally with `docker compose -f deploy/docker-compose.yml up --build`. Configure the dashboard service with `VERIMIND_API_URL`, `VERIMIND_API_KEY`, and the public API URL. The API rejects requests if `ENVIRONMENT=production` and `API_KEY` is absent. The built-in rate limiter is process-local; use your host's gateway/WAF or a shared rate-limit store for multi-instance production deployment.

### Deploying on Render

The repository includes a Render Blueprint in `render.yaml` for separate API and dashboard web services. Publish the contents of this `Code` directory as the GitHub repository root, then in Render choose **New → Blueprint** and connect that repository. Do not upload or commit `.env`; `.gitignore` excludes it. The Blueprint requests one `API_KEY` secret in a shared environment-variable group used by both services. If Render assigns different `onrender.com` hostnames because the names are already taken, update `VERIMIND_API_URL` on the dashboard and `CORS_ORIGINS` on the API to match the assigned service URLs.

The Render API image intentionally omits the large local embedding stack and index to fit a small demonstration service. It uses the small built-in fact base; live search and the hosted judge are disabled unless configured separately. The current SQLite report store is ephemeral on Render's free service plan, so saved reports can be lost when the service restarts or redeploys. This hosted setup is for demonstration, not production fact-checking or durable data storage.

## Local demo

Use the verification API with a sample payload:

```bash
curl -X POST http://localhost:8000/verify \
  -H "x-api-key: demo-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "Paris is the capital of France. The Eiffel Tower is in Paris."}'
```

## Evaluation

The offline evaluation uses 60 distinct claims sampled from the available FEVER training data, stratified as 20 supported, 20 refuted, and 20 unverifiable. A separate development split contains 10 examples per class. Sampling used seed 42; these are held-out FEVER-training examples, not FEVER's official test set. Labels are inherited from FEVER's Wikipedia annotations and were not independently rechecked against current Wikipedia revisions. The sample contains single claims, not multi-claim answers or recent facts.

The measured system is the local `VerificationPipeline` with its scikit-learn MLP NLI classifier and local four-fact Wikipedia knowledge base. The evaluator does not enable the external search backend or the Groq judge. Therefore, these figures are for the offline VERIMIND pipeline, not the external LLM judge.

Fresh expanded-index test results from the run on 2026-11-20:

- Accuracy: **22/60 (36.67%)**; 95% Wilson confidence interval **25.62%–49.32%**.
- Per-class recall: supported **0/20 (0%)**; refuted **3/20 (15%)**; unverifiable **19/20 (95%)**.
- Dangerous error rate (refuted predicted supported): **0/20 (0%)**.
- Unverifiable predictions: **55/60 (91.67%)**, above the test set's **20/60 (33.33%)** unverifiable prevalence; flagged as overuse.
- Retrieval page hit rate against annotated gold pages: **0/40 (0%)**. Judge accuracy after a hit is not measurable (**0** hit cases); after a miss it is **3/40 (7.5%)**.
- Evaluation failures: **0/60**.

The exact result artifact is [`eval_20261120T073859Z.jsonl`](results/eval_20261120T073859Z.jsonl). This improvement is a conservative baseline, but it remains too inaccurate for production fact-checking.

The `--baseline` option measures a **local no-evidence classifier control**, not a plain-LLM baseline. The fresh control run on 2026-11-20 scored **20/60 (33.33%)** and predicted every case as unverifiable. The exact control artifact is [`eval_20261120T074100Z.jsonl`](results/eval_20261120T074100Z.jsonl). This control is not a substitute for the requested same-model LLM comparison; no external LLM/search calls were made, so that comparison remains unavailable.

The FAISS index is an independent reservoir sample of **5,000 pages** from the full FEVER v1.0 Wikipedia dump; the builder scanned **5,396,106 usable pages** and did not use evaluation claims, labels, or gold evidence to select the sample. It contains **5,870 passages** and uses `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions), normalized embeddings, and FAISS inner product. The evaluator uses this FAISS retriever. The sampled index contains **0/40** annotated gold pages for the supported/refuted test cases, so the fresh run measured **0/40** exact gold-page retrieval hits. Retrieval coverage and verdict quality remain insufficient for production use.

Re-run the held-out pipeline evaluation from the project root:

```bash
python eval.py --split test
```

Run the local no-evidence control with `python eval.py --split test --baseline`; use `--n N` to limit a run or `--split dev` for development data. Each run writes per-case JSONL plus summary metadata under `results/`, and saves the confusion matrix to `results/confusion_matrix.png`. The expanded-index test run is [`eval_20261022T100330Z.jsonl`](results/eval_20261022T100330Z.jsonl). The earlier no-evidence control run is [`eval_20261022T082158Z.jsonl`](results/eval_20261022T082158Z.jsonl).

## Known limitations

- This project uses TF-IDF + a lightweight sklearn classifier rather than a large RoBERTa-MNLI model.
- The SQLite storage is intentionally simple and local-first.
- The live web search backend is optional and fails safely to local-only retrieval.
