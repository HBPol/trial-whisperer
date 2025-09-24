# TrialWhisperer

[![CI](https://github.com/HBPol/trial-whisperer/actions/workflows/ci.yml/badge.svg)](https://github.com/HBPol/trial-whisperer/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/HBPol/trial-whisperer/branch/main/graph/badge.svg)](https://codecov.io/gh/HBPol/trial-whisperer)


Clinical Trial Protocol Chatbot (MVP): query trial eligibility criteria, outcomes, and interventions from ClinicalTrials.gov with grounded answers and citations. Includes an eligibility checker prototype.


## Features
- RAG over trial protocols (XML → JSONL → vector index)
- `/ask` Q&A with citations
- `/trial/{nct_id}` structured metadata
- `/check-eligibility` given patient JSON
- Minimal chat UI

## Documentation
- [Project Overview](project_docs/ProjectOverview.md)
- [Requirements Specification](project_docs/RequirementsSpecification.md)
- [Project Plan](project_docs/ProjectPlan.md)


## Stack
- FastAPI, Python 3.11
- Qdrant (managed free tier)
- Gemini (or OpenAI) LLM
- Cloud Run / Hugging Face Spaces


## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config/appsettings.example.toml config/appsettings.toml
# Edit config/appsettings.toml with your keys / endpoints
make seed # downloads trials from ClinicalTrials.gov and indexes them
make run
```

The seeding step requires an accessible Qdrant instance. If `QDRANT_URL` and
`QDRANT_API_KEY` are not set, the script will attempt to start a local Qdrant
container (`docker run -p 6333:6333 qdrant/qdrant`). Ensure Docker is installed
or provide a remote Qdrant endpoint via `QDRANT_URL`/`QDRANT_API_KEY`.

## Evaluation

Use the evaluation harness to exercise the `/ask` endpoint against the bundled
sample dataset and fixture trials index:

```bash
python eval/eval.py
```

By default the script loads the 21-example dataset in
`eval/testset.sample.jsonl` and, when that file is used, it sets
`TRIALS_DATA_PATH` to `.data/processed/trials.jsonl` before issuing any
requests. This mirrors the application's offline fallback behaviour: the
retrieval when no live Qdrant instance is available. While running, the CLI now
prints per-example progress details (query, expected trial/sections, result)
and notes when retries occur. Pass `--quiet` to suppress the extra progress
logging and revert to the previous terse behaviour. At completion the script
still prints a short summary that includes the answer exact match rate,
citation coverage for examples that expect specific sections, and the number of
errors encountered. Pass `--json-report` to capture the full report to disk:

```bash
python eval/eval.py --json-report reports/eval.json
```

Because the fallback index points at `.data/processed/trials.jsonl`, rerunning
`make seed` (or otherwise regenerating the processed dataset) immediately feeds
the refreshed content into the evaluation harness. If you need a stable set of
fixtures—for example to compare runs across different application changes—copy
the JSONL to a separate location and either export `TRIALS_DATA_PATH` or pass
`--trials-data` when invoking `eval.py`. The repository also ships with a small
`.data/test_processed/trials.jsonl` snapshot that is suitable for lightweight
checks.

As of the current repository state, `.data/processed/trials.jsonl` contains
1,202 chunked sections drawn from 200 ClinicalTrials.gov studies, while the
lightweight `.data/test_processed/trials.jsonl` covers 58 chunks across 10
studies. These counts reflect the last ingested glioblastoma cohort and will
change whenever you adjust the ingestion filters or rerun `make seed`.

When `config/appsettings.toml` points the app at the Gemini provider the
evaluation harness automatically waits roughly six seconds between `/ask`
requests to respect the API's published limit. Override the delay with
`--min-request-interval` (set it to `0` to disable the guard or increase it for
stricter pacing):

```bash
python eval/eval.py --min-request-interval 3
```
During local development or tests the API falls back to an in-memory index when
no live Qdrant backend is configured. `eval.py` now exposes a `--trials-data`
flag to control the source JSONL file for that offline index, mirroring the
`TRIALS_DATA_PATH` environment variable understood by the application. Specify
it alongside a custom dataset to run the evaluation against a different set of
processed trials:


```bash
python eval/eval.py eval/custom_trials.jsonl \
  --trials-data .data/processed/trials.jsonl \
  --json-report reports/custom.json
```

### Ingesting ClinicalTrials.gov trials

`make seed` (or `scripts/seed_smallset.sh`) runs the ingestion pipeline followed
by the indexing step:

1. `python -m pipeline.pipeline --from-api ...` downloads study records from
   the [ClinicalTrials.gov API v2](https://www.clinicaltrials.gov/api/v2/) using
   the parameters defined in your configuration file.
2. The pipeline normalizes the JSON payload into the schema expected by
   `pipeline.normalize`/`pipeline.chunk` and writes the processed chunks to the
   path configured via `TRIALS_DATA_PATH` (default:
   `.data/processed/trials.jsonl`) with the real NCT IDs from the feed.
3. `python -m scripts.index` embeds the chunks and upserts them into Qdrant so
   the application can serve queries immediately after seeding.

#### Changing the ingestion defaults

The ingestion script reads its defaults from `config/appsettings.toml`. Update
the `[data]` section to change where the processed JSONL is written (or export
`TRIALS_DATA_PATH` before running `make seed`), and tune the `[data.api]`
section to control how many studies are fetched, the ClinicalTrials.gov search
filters, and the HTTP backend. All CLI flags exposed by
`python -m pipeline.pipeline`—such as `--max-studies`, `--page-size`, and
`--query-term`—override the TOML configuration for that run, which is useful for
experiments or ad-hoc cohorts.

Configure the API request under the `[data.api]` section of
``config/appsettings.toml`. Choose the HTTP client with the `backend` key
(`"httpx"` when omitted, or `"requests"` as shown below). The default example
downloads a modest cohort of recruiting glioblastoma trials:

```toml
[data.api]
backend = "requests"        # HTTP client backend ("httpx" or "requests")
page_size = 100            # API page size (max 100)
max_studies = 200          # Total number of studies to ingest

[data.api.params]
"query.term" = "glioblastoma"
"filter.overallStatus" = ["RECRUITING", "ACTIVE_NOT_RECRUITING"]
```

Keys in `[data.api.params]` correspond directly to the official `studies`
endpoint parameters (e.g. `query.term`, `filter.overallStatus`). Most filter
values are enumerations—use the API's canonical codes such as
`ACTIVE_NOT_RECRUITING` rather than the human-readable labels shown on the
website. Add or repeat keys to narrow the cohort further—for example `--param
filter.locationFacility=Boston` on the command line or an additional
`"filter.locationFacility"` entry in the TOML file.

The client targets `https://www.clinicaltrials.gov/api/v2` by default and uses
`httpx` unless you set `data.api.backend` to `"requests"`. If
ClinicalTrials.gov relocates the v2 API again, set `data.api.base_url` to the
new host to keep ingestion running:

```toml
[data.api]
base_url = "https://api.example.gov/ctgov/v2"
```

Set `data.api.backend` to choose the HTTP stack used for API calls. The new
`requests` backend mirrors the ubiquitous `requests.Session` behaviour (respecting
`trust_env`, enterprise proxies, etc.) and is the recommended option for most
deployments:

```toml
[data.api]
backend = "requests"
```

If you rely on features specific to the original `httpx` client (for example
HTTP/2 or custom transports), keep or restore the previous behaviour with:

```toml
[data.api]
backend = "httpx"
```

As of September 2025 the v2 API does not yet expose dedicated
`filter.studyType` or `filter.phase` parameters; sending them results in an HTTP
400 response. To constrain those attributes, embed an advanced search clause
inside `query.term`, mirroring the syntax used by ClinicalTrials.gov's
Advanced Search. For example:

```toml
"query.term" = "glioblastoma AND AREA[StudyType]StudyType=Interventional AND AREA[Phase]Phase=Phase 2"
```

Refer to the [API query reference](https://clinicaltrials.gov/data-api/about-api#query) for the
full list of supported `AREA[...]` tokens.

ClinicalTrials.gov asks API consumers to identify a responsible contact in
their `User-Agent` or request headers. TrialWhisperer ships with
`TrialWhisperer/ingest (+https://trialwhisperer.ai/contact)` as the default
identifier, but you should provide your own organization string (or email
address) via `data.api.user_agent` or add a dedicated header to stay in
compliance:

```toml
[data.api]
user_agent = "my-app/1.0 (admin@example.com)"

[data.api.headers]
"X-Contact" = "admin@example.com"
```

The pipeline will merge these values into every API call while keeping the
default headers when the override is omitted.


You can run the ingestion manually when experimenting:

```bash
python -m pipeline.pipeline --from-api --config config/appsettings.toml \
  --max-studies 50 \
  --query-term 'glioblastoma AND AREA[StudyType]StudyType=Interventional'
```

Re-run `python -m scripts.index` after changing any ingestion parameters so the
vector index reflects the freshly downloaded trials.


## Run with Docker

```bash
docker build -t trial-whisperer .
# assumes config/appsettings.toml exists locally
docker run --rm -p 8000:8000 \
  -v $(pwd)/config/appsettings.toml:/app/config/appsettings.toml:ro \
  trial-whisperer
```

This mounts your local `config/appsettings.toml` into the container, making it
the single source of configuration. Alternatively, copy the file into the
image at build time if you prefer.

The container exposes the FastAPI app on port 8000 by default.

### Environment variables

Environment variables are optional and override values in
`config/appsettings.toml`. For example:

```bash
docker run --rm -p 8000:8000 \
  -v $(pwd)/config/appsettings.toml:/app/config/appsettings.toml:ro \
  -e LLM_API_KEY="override" \
  trial-whisperer
```

- `LLM_API_KEY` – API key for your LLM provider.
- `QDRANT_URL` – Qdrant cloud endpoint (including port number e.g. `https://YOUR.QDRANT.URL:6333`).
- `QDRANT_API_KEY` – Qdrant authentication token.
- `TRIALS_DATA_PATH` – Location of the processed trials JSONL file. Override to
  switch between seeded data and ad-hoc datasets (defaults to
  `.data/processed/trials.jsonl`).

## API

### Available endpoints

- `GET /trial/{nct_id}` – Retrieve structured metadata for a clinical trial.

  ```bash
  curl http://localhost:8000/trial/NCT01234567
  ```

- `POST /ask` – Ask a question about a trial and receive an answer with citations.

  ```bash
  curl -X POST http://localhost:8000/ask \
    -H "Content-Type: application/json" \
    -d '{
      "query": "Summarize the inclusion criteria",
      "nct_id": "NCT01234567"
    }'
  ```

- `POST /check-eligibility` – Evaluate a patient profile against a trial's eligibility criteria.

- Patient payload schema:

  ```json
  {
    "nct_id": "NCT01234567",
    "patient": {
      "age": 55,
      "sex": "female",
      "labs": {
        "ECOG": 1
      }
    }
  }
  ```

  - `age` (integer, required): Patient age in years.
  - `sex` (string, required): Patient sex as documented in the trial (e.g., `"female"`, `"male"`).
  - `labs` (object, optional): Free-form key/value map of lab measurements or scores. Presently captured for future use and **not** consumed by the eligibility rules.

  ```bash
  curl -X POST http://localhost:8000/check-eligibility \
    -H "Content-Type: application/json" \
    -d '{
      "nct_id": "NCT01234567",
      "patient": {
        "age": 55,
        "sex": "female",
        "labs": {"ECOG": 1}
      }
    }'
  ```

_Current limitations_: eligibility scoring currently evaluates only `age` and `sex`. Lab values are ingested but ignored until lab parsing is implemented, so they will not influence eligibility outcomes yet.

Interactive API exploration is available via Swagger UI at
`http://localhost:8000/docs` and via ReDoc at `http://localhost:8000/redoc`.


## Contributing
Development follows a TDD approach. Please write or update tests before implementing new functionality.
