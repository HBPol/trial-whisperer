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


### Ingesting ClinicalTrials.gov trials

`make seed` (or `scripts/seed_smallset.sh`) runs the ingestion pipeline followed
by the indexing step:

1. `python -m pipeline.pipeline --from-api ...` downloads study records from
   the [ClinicalTrials.gov Data API](https://clinicaltrials.gov/data-api/) using
   the parameters defined in your configuration file.
2. The pipeline normalizes the JSON payload into the schema expected by
   `pipeline.normalize`/`pipeline.chunk` and writes
   `.data/processed/trials.jsonl` with the real NCT IDs from the feed.
3. `python -m scripts.index` embeds the chunks and upserts them into Qdrant so
   the application can serve queries immediately after seeding.

Configure the API request under the `[data.api]` section of
`config/appsettings.toml`. The default example downloads a modest cohort of
recruiting glioblastoma trials:

```toml
[data.api]
page_size = 100            # API page size (max 100)
max_studies = 200          # Total number of studies to ingest

[data.api.params]
"query.term" = "glioblastoma"
"filter.overallStatus" = ["Recruiting", "Active, not recruiting"]
"filter.studyType" = ["Interventional"]
"filter.phase" = ["Phase 2", "Phase 3"]
```

Keys in `[data.api.params]` correspond directly to the official `studies`
endpoint parameters (e.g. `query.term`, `filter.overallStatus`, `filter.phase`).
Add or repeat keys to narrow the cohort further—for example
`--param filter.locationFacility=Boston` on the command line or an additional
`"filter.locationFacility"` entry in the TOML file.

You can run the ingestion manually when experimenting:

```bash
python -m pipeline.pipeline --from-api --config config/appsettings.toml \
  --max-studies 50 --param "filter.studyType=Interventional"
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

Interactive API exploration is available via Swagger UI at
`http://localhost:8000/docs` and via ReDoc at `http://localhost:8000/redoc`.


## Contributing
Development follows a TDD approach. Please write or update tests before implementing new functionality.
