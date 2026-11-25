# VERIMIND

VERIMIND is a lightweight, locally runnable hallucination-detection and verification model  for LLM outputs.

## Architecture

The project is intentionally modular and offline-friendly. It segments output into claims, retrieves evidence from a local fact base and optional live web search, scores each claim with an explainable risk model, and returns a claim-level audit trail.

## Project-wide run order

Do not run every `.py` file in folder order. Most files in `api/`, `verimind/`, `dashboard/lib/`, `eval/`, and `retrieval/` are imported modules, not commands. Run commands from the project root, using the same project's virtual environment:

1. Install the dependencies as described below.
2. Optional quick smoke test: `.\venv\Scripts\python.exe scripts\demo.py`.
3. Start the API: `.\venv\Scripts\python.exe -m uvicorn api.main:app --reload`.
4. If you want the web UI, open a second terminal in the project root and start `.\venv\Scripts\python.exe -m streamlit run dashboard\app.py`.
5. To run checks, use another terminal in the project root: `.\venv\Scripts\python.exe -m pytest -q`.

The numbered Streamlit files under `dashboard/pages/` are discovered by Streamlit automatically. Run `eval.py --split dev` only when you want to measure the verification model. Index building and model training are optional maintenance workflows—not prerequisites for starting the API—and can take substantial time and disk space. Follow the dedicated index/training commands in [Evaluation](#evaluation) only when you intend to rebuild or train those assets. `.ipynb` files are notebooks; they are not required to launch the application.

## Run with live web verification

**Run the commands below from the project root** (the directory containing this README and `pyproject.toml`). Do not run tests from a parent workspace containing backup or GPU-package directories; pytest can collect unrelated duplicate tests and fail with import-file mismatches. Do not run files under `verimind`, `api`, or `dashboard\lib` individually; they are modules imported by the API and dashboard entry points. Follow the numbered setup and launch steps below, starting the API before the dashboard.

VERIMIND can search the broader indexed web with Tavily (`SEARCH_BACKEND=tavily` plus `SEARCH_API_KEY`), or search Wikipedia only without a key (`SEARCH_BACKEND=wikipedia`). These APIs do not cover literally every internet page. Wikipedia searches are limited to one query per claim and paced to respect API rate limits. Online search sends claim text to the selected search provider. The hosted judge receives a claim and retrieved passages and is instructed to return `supported`, `refuted`, or `insufficient_evidence`; it must not use outside knowledge. If a search provider is unavailable, the application logs the error and continues with local retrieval.

1. Install Python 3.11, then create a virtual environment and install dependencies:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt -r dashboard\requirements.txt
```

If an existing environment has incompatible packages, create a fresh one (for example, `py -3.11 -m venv venv-clean`) and replace `venv` with `venv-clean` in the commands below.

2. If `.env` does not already exist, copy `.env.example` to `.env` and configure it:

```bash
copy .env.example .env
```

Set `SEARCH_BACKEND=tavily` to search the broader web. The app accepts either `SEARCH_API_KEY` or the existing `WEB_SEARCH_API_KEY`, and uses `WEB_SEARCH_API_URL` when provided. Keyless Wikipedia-only search is available with `SEARCH_BACKEND=wikipedia`. For hosted judging, configure `JUDGE_BACKEND=groq` and `GROQ_API_KEY`; otherwise keep the local judge. Set a long random `API_KEY` before deployment. Do not commit `.env` or put provider keys in a browser application.

3. Run the API:

```bash
.\venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

4. Run the dashboard locally:

```powershell
.\venv\Scripts\python.exe -m streamlit run dashboard\app.py
# Or starts API and dashboard in two PowerShell windows:
.\scripts\run_dashboard.ps1
```

## Dashboard

The Streamlit dashboard is an HTTP-only client: it does not import model, FAISS, Torch, or sentence-transformer packages. It provides a home page, claim-level evidence review, Batch Check, Evaluation, System Metrics, and About pages. The Verify page does not preload sample claims; paste the text you want to check. Claim text is sent to the configured API and may be retained by that service or forwarded to its configured search/judge providers; do not submit secrets or sensitive personal information.

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

Or start the API and dashboard separately using the commands in the first section and `streamlit run dashboard\app.py`. Enter the API key in the sidebar password field, or set `VERIMIND_API_KEY` in the environment before launching Streamlit. The dashboard keeps it in the current Streamlit session and sends it only with verification requests—not health, readiness, or metrics requests. The key is never displayed or logged by the dashboard. The default development API accepts requests without a key unless `API_KEY` is configured.

Supported dashboard environment variables are `VERIMIND_API_URL`, `VERIMIND_API_KEY`, `VERIMIND_DASHBOARD_TIMEOUT_SECONDS`, `VERIMIND_DASHBOARD_BATCH_LIMIT`, and `VERIMIND_EVAL_RESULTS_DIR`. The single-claim input limit follows the API schema (20,000 characters); overlong batch rows are marked as errors without being sent, and changing the batch input clears stale results. The batch limit defaults to `MAX_CLAIMS_PER_DOC` or 50. Timeout and batch-limit settings must be positive finite values. Evaluation defaults to `eval/results` and also loads the existing `results/*.jsonl` artifacts; set `VERIMIND_EVAL_RESULTS_DIR` to use another directory.

`docker compose -f deploy\docker-compose.yml up --build` starts the API and dashboard. The dashboard waits for database-backed API readiness, uses the internal API service URL, and mounts both evaluation-artifact directories read-only. The API image uses the minimal deployment dependencies and local NLI backend; it omits the large transformer and Wikipedia-index assets. The Compose file requires a local `.env` copied from `.env.example`.

The dashboard reports live request latency percentiles and retrieval/judge timing when those API stages have run. The local judge is disabled by default, so judge latency can honestly show as unavailable. The repository has a local no-evidence NLI baseline, not a same-model LLM comparison; this distinction is shown on Evaluation. API rate limiting is in-memory by default and shared through Redis only when `REDIS_URL` is configured; metrics remain in-memory and per-process. Screenshots of the running pages were reviewed during validation; they are not checked into the repository.

5. Run checks:

```bash
pytest -q
ruff check api verimind tests
```

## Deployment

The supplied Docker configuration is under `deploy/`. Before publishing an API, set these hosting-provider secrets/environment variables. For horizontally scaled APIs, also set `REDIS_URL` to a shared Redis-compatible service:

```text
ENVIRONMENT=production
API_KEY=<long-random-secret>
DATABASE_URL=<managed-postgresql-url>
SEARCH_BACKEND=tavily
SEARCH_API_KEY=<tavily-secret>
JUDGE_BACKEND=groq
GROQ_API_KEY=<groq-secret>
GROQ_MODEL=llama-3.3-70b-versatile
CORS_ORIGINS=https://your-dashboard.example.com
```

Build/run locally with `docker compose -f deploy/docker-compose.yml up --build`. Configure the dashboard service with `VERIMIND_API_URL`, `VERIMIND_API_KEY`, and the public API URL. The Compose file mounts a named Docker volume at `/app/data` so local SQLite reports persist across container recreation. The API rejects startup if `ENVIRONMENT=production` lacks either `API_KEY` or `DATABASE_URL`. Local development uses SQLite at `data/verimind.db` by default; `VERIMIND_DB_PATH` changes that path. Production should use a managed PostgreSQL URL or SQLite on a persistent mounted disk, never an ephemeral container filesystem. PostgreSQL URLs using `postgres://` or `postgresql://` are supported.

The API's `/readyz` health check verifies database connectivity and initializes the report schema. `/metrics` requires the same API key as other protected API endpoints. For multiple API instances, configure `REDIS_URL` to use the shared atomic rate limiter; without Redis, the limiter is in-memory and only appropriate for a single instance. A host-level gateway/WAF can also enforce shared rate limits.

### Deploying on Render

The repository includes a Render Blueprint in `render.yaml` for separate API and dashboard Python web services. It explicitly installs each service's requirements and starts the API with Uvicorn and the dashboard with Streamlit. Do not use `verimind` as a start command; the project does not define a console command with that name. Publish the contents of this `Code` directory as the GitHub repository root, then in Render choose **New → Blueprint** and connect that repository. Do not upload or commit `.env`; `.gitignore` excludes it. The Blueprint provisions a free Render PostgreSQL database, wires `DATABASE_URL` from it, generates `API_KEY` automatically, and sets `CORS_ORIGINS`/`VERIMIND_API_URL` from Render service URLs. You should not need to fill in manual URL secrets for the default Blueprint flow. `REDIS_URL` is left empty by default; set it to a shared Redis-compatible service when scaling beyond one API instance. If you rename the services, update the `fromService` references in `render.yaml` to match.

The API service also serves a responsive claim-verification page at its root URL; `/docs` remains available for the API schema. When production authentication is enabled, enter the configured API key in the page before submitting a claim.

For native Render builds, the root `.python-version` pins Python to `3.11.11`. This is required by the current `pyarrow<19` dependency range, which does not provide the needed wheel for Render's newer Python 3.14 default. If the service already has a `PYTHON_VERSION` environment variable, set it to `3.11.11` or remove it; that variable takes precedence over `.python-version`. The project metadata also rejects Python 3.14 so an unsupported source build fails early instead of trying to compile PyArrow.

The Render API service intentionally uses minimal deployment dependencies and omits the large local embedding stack and index to fit a small demonstration service. It uses the small built-in fact base; live search and the hosted judge are disabled unless configured separately. The API can persist reports in PostgreSQL, but that does not make the verifier production-accurate: evaluation remains limited, so treat hosted verdicts as advisory until representative held-out evaluation meets your accuracy and safety requirements.

## Local demo

Use the verification API with a sample payload:

```bash
curl -X POST http://localhost:8000/verify \
  -H "x-api-key: demo-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "Paris is the capital of France. The Eiffel Tower is in Paris."}'
```

## Evaluation

### Models and training

VERIMIND has one locally trained NLI classifier: a scikit-learn `MLPClassifier` over word- and character-level TF-IDF features, combined at inference with lexical contradiction/support safeguards. Lexical retrieval also uses TF-IDF; dense retrieval uses `sentence-transformers/all-MiniLM-L6-v2` with FAISS. The optional hosted evidence judge uses the configured Groq model (`llama-3.3-70b-versatile` by default) and is disabled unless configured. Risk scoring itself is deterministic logic, not a learned model.

For stronger local NLI, the API can optionally load [MoritzLaurer/DeBERTa-v3-base-mnli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli), a pretrained three-label classifier trained on MultiNLI. It runs locally; model weights download from Hugging Face on first use. The checkpoint was selected over a FEVER-finetuned alternative to avoid direct training-set overlap with this project's FEVER evaluation sample. This replaces the lightweight NLI classifier rather than stacking several classifiers. Multiple NLI models would add latency and memory without fixing missing evidence.

The current recorded FEVER run found **0/40 gold evidence pages**, so retrieval coverage is a bigger known accuracy bottleneck than model count. Build the full-text index before comparing NLI backends; the existing builder streams the full Wikipedia dump and documents its disk/time requirements:

```powershell
.\venv\Scripts\python.exe scripts\build_wikipedia_index.py --full-text-only
```

Before building the full-text index, both backends scored **10/30 (33.3%)** on dev and retrieved **0/20** gold pages. After the index build, DeBERTa scored **11/30 (36.7%)**, with gold-page retrieval at **8/20 (40%)** among the top three and **14/20 (70%) Recall@10**. The lightweight synthetic model remained at **10/30** and predicted every claim as unverifiable. This small dev sample shows improved retrieval, but only a one-case accuracy change; it is not evidence of production-level accuracy.

The prepared FEVER set contains **30,000** balanced pairs after excluding all 90 held-out dev/test claims. A quick 1,000-per-class MLP candidate reached **47.7%** on its own 606-pair validation split, but only **8/30 (26.7%)** in the full pipeline dev evaluation and made **1 dangerous refuted-to-supported error**. It was kept separate and not selected over DeBERTa. The neutral pairs are weak random low-overlap examples, so treat this candidate as experimental.

On Windows with an NVIDIA GPU, run this project setup script to install the CUDA PyTorch build, enable `NLI_BACKEND=transformer` in the local `.env`, and verify the GPU:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_gpu.ps1
```

The script uses a separate package directory because Windows denied replacing the existing PyTorch files in the project venv. It does not change `.env` secrets. Model weights download from Hugging Face on first API start. Compare the transformer with the lightweight model on the development split:

```powershell
.\venv\Scripts\python.exe eval.py --split dev
```

To compare the lightweight baseline, temporarily set `NLI_BACKEND=local` in `.env` and rerun the same dev evaluation. Keep the test split untouched until you choose a configuration. Restart the API after changing `.env`. The minimal Render API dependency set does not include this local transformer backend.

For hands-on training of the lightweight baseline in VS Code, open [`train_model.ipynb`](train_model.ipynb), select the project's Python 3.11 kernel, and run cells from top to bottom. You can add labeled claim/evidence pairs, adjust epochs and patience, inspect per-epoch accuracy and the validation confusion matrix, then save the selected model. This notebook does not fine-tune the pretrained DeBERTa checkpoint.

The command-line alternative is `scripts/train.py`. It currently uses 42 synthetic claim-evidence pairs, makes a stratified validation split, reports accuracy after each epoch, and early-stops while keeping the best validation-accuracy checkpoint:

```powershell
.\venv\Scripts\python.exe scripts\train.py --epochs 100 --patience 12
```

The output defaults to `artifacts\nli_model.pkl`; a running pipeline loads that artifact on startup, so restart the API after retraining. Set `NLI_MODEL_PATH=artifacts/nli_model_fever.pkl` to load a separately saved FEVER-trained artifact while keeping the existing model intact. The synthetic training set is very small and its validation accuracy is not a measure of real-world or FEVER accuracy. Do not use the held-out `dev` or `test` evaluation cases for training or tuning.

Run `scripts/benchmark.py` for a deterministic 25% holdout comparison between the synthetic-data NLI model and a majority-class baseline. This is only a smoke benchmark for the local classifier; it is not a pipeline, FEVER, or real-world accuracy estimate.

The offline evaluation uses 60 distinct claims sampled from the available FEVER training data, stratified as 20 supported, 20 refuted, and 20 unverifiable. A separate development split contains 10 examples per class. Sampling used seed 42; these are held-out FEVER-training examples, not FEVER's official test set. Labels are inherited from FEVER's Wikipedia annotations and were not independently rechecked against current Wikipedia revisions. The sample contains single claims, not multi-claim answers or recent facts.

The evaluator uses the selected `SEARCH_BACKEND` and `JUDGE_BACKEND` settings. With the defaults (`null` and `local`), it measures the offline pipeline; configured online providers are recorded in each run's metadata and should be reported separately.

Fresh expanded-index test results from the run on 2026-11-20:

- Accuracy: **22/60 (36.67%)**; 95% Wilson confidence interval **25.62%–49.32%**.
- Per-class recall: supported **0/20 (0%)**; refuted **3/20 (15%)**; unverifiable **19/20 (95%)**.
- Dangerous error rate (refuted predicted supported): **0/20 (0%)**.
- Unverifiable predictions: **55/60 (91.67%)**, above the test set's **20/60 (33.33%)** unverifiable prevalence; flagged as overuse.
- Retrieval page hit rate against annotated gold pages: **0/40 (0%)**. Judge accuracy after a hit is not measurable (**0** hit cases); after a miss it is **3/40 (7.5%)**.
- Evaluation failures: **0/60**.

The exact result artifact is [`eval_20261120T073859Z.jsonl`](results/eval_20261120T073859Z.jsonl). This improvement is a conservative baseline, but it remains too inaccurate for production fact-checking.

A later FEVER-derived development run on 2026-11-22 using `artifacts/nli_model_fever.pkl` scored **8/30 (26.7%)**. It predicted **0/10 supported** and **0/10 refuted** claims correctly, predicted **27/30 (90%)** as unverifiable, and made **1/10 refuted-to-supported dangerous errors**. Gold-page retrieval was **8/20 (40%)** in the top three and **14/20 (70%) Recall@10**. The exact run is [`eval_20261122T083821Z.jsonl`](results/eval_20261122T083821Z.jsonl). This development result is a warning that model selection and threshold tuning are not resolved; do not use this configuration for production fact-checking.

The public Wikipedia API was then enabled for a separate development run. Accuracy remained **8/30 (26.7%)**; online results appeared in only **3/30** retrieved evidence lists, and the API returned an HTTP 429 rate-limit response during the run. The API is an optional evidence source, not a path to guaranteed 80% accuracy. See [`eval_20261122T102902Z.jsonl`](results/eval_20261122T102902Z.jsonl) for that run.

A Tavily broad-web development run on 2026-11-23 scored **9/30 (30%)** with the same FEVER-trained local classifier. Gold-page retrieval was **10/20 (50%)** in the top three and **14/20 (70%) Recall@10**; the classifier still predicted no supported or refuted claims correctly. Web results appeared in the diagnostic top ten for **12/30** cases. The exact run is [`eval_20261123T154824Z.jsonl`](results/eval_20261123T154824Z.jsonl). This is only a one-case improvement over the previous 26.7% dev result, not evidence that the system is near 80%.

The `--baseline` option measures a **local no-evidence classifier control**, not a plain-LLM baseline. The fresh control run on 2026-11-20 scored **20/60 (33.33%)** and predicted every case as unverifiable. The exact control artifact is [`eval_20261120T074100Z.jsonl`](results/eval_20261120T074100Z.jsonl). This control is not a substitute for the requested same-model LLM comparison; no external LLM/search calls were made, so that comparison remains unavailable.

The FAISS index is an independent reservoir sample of **5,000 pages** from the full FEVER v1.0 Wikipedia dump; the builder scanned **5,396,106 usable pages** and did not use evaluation claims, labels, or gold evidence to select the sample. It contains **5,870 passages** and uses `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions), normalized embeddings, and FAISS inner product. The evaluator uses this FAISS retriever. The sampled index contains **0/40** annotated gold pages for the supported/refuted test cases, so the fresh run measured **0/40** exact gold-page retrieval hits. Retrieval coverage and verdict quality remain insufficient for production use.

To add full-corpus lexical retrieval without replacing the sampled FAISS index, run:

```powershell
.\venv\Scripts\python.exe scripts\build_wikipedia_index.py --full-text-only
```

This streams the complete FEVER Wikipedia archive into `data\index\wiki_fts.sqlite3`; it does not load all pages into memory or use evaluation labels. Restart the API after the build so the hybrid retriever loads the new index. The build scans millions of rows and needs enough free disk space for the generated database. Verdicts still use three evidence passages; offline evaluation separately searches up to ten to report **Recall@1/3/5/10**. Run the development split before using the held-out test split for a final comparison.

To gather real labeled NLI examples, run the FEVER data preparer after the index build:

```powershell
.\venv\Scripts\python.exe scripts\prepare_fever_training_data.py --max-per-label 10000
.\venv\Scripts\python.exe scripts\train.py --dataset fever --max-per-label 1000 --epochs 5 --patience 2 --output artifacts\nli_model_fever.pkl
$env:NLI_BACKEND = "local"
$env:NLI_MODEL_PATH = "artifacts\nli_model_fever.pkl"
.\venv\Scripts\python.exe eval.py --split dev
Remove-Item Env:NLI_BACKEND, Env:NLI_MODEL_PATH -ErrorAction SilentlyContinue
```

The preparer downloads the official FEVER v1.0 training annotations, joins SUPPORTS/REFUTES evidence line IDs to the matching Wikipedia snapshot, and creates weak neutral pairs for NOT ENOUGH INFO claims using random low-overlap passages from the independent Wikipedia sample. It excludes exact dev/test claim IDs and normalized claim text before preparation, caps each class at 20,000 pairs by default, and writes source/count metadata beside the generated JSONL. Set `--max-per-label 0` to keep every available pair. Neutral labels are weak because random passages are unannotated; inspect the manifest and validation report. Training splits by normalized claim text to prevent a claim's evidence pairs leaking across train and validation. Large downloaded/generated data files are ignored by Git.

Run the development evaluation while tuning:

```bash
python eval.py --split dev
```

After selecting changes using the development split, run `python eval.py --split test` once for the final held-out measurement.

Run the local no-evidence control with `python eval.py --split test --baseline`; use `--n N` to limit a run or `--split dev` for development data. Each run writes per-case JSONL plus summary metadata under `results/`, and saves the confusion matrix to `results/confusion_matrix.png`. The expanded-index test run is [`eval_20261022T100330Z.jsonl`](results/eval_20261022T100330Z.jsonl). The earlier no-evidence control run is [`eval_20261022T082158Z.jsonl`](results/eval_20261022T082158Z.jsonl).

## Known limitations

- This project uses TF-IDF + a lightweight sklearn classifier rather than a large RoBERTa-MNLI model.
- The SQLite storage is intentionally simple and local-first.
- The live web search backend is optional and fails safely to local-only retrieval.
