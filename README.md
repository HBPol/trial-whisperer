# TrialWhisperer

[![CI](https://github.com/HBPol/trial-whisperer/actions/workflows/ci.yml/badge.svg)](https://github.com/HBPol/trial-whisperer/actions/workflows/ci.yml)

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
make seed # small demo dataset + index (starts local Qdrant if needed)
python scripts/index.py # index processed trials into Qdrant
make run
```

The seeding step requires an accessible Qdrant instance. If `QDRANT_URL` and
`QDRANT_API_KEY` are not set, the script will attempt to start a local Qdrant
container (`docker run -p 6333:6333 qdrant/qdrant`). Ensure Docker is installed
or provide a remote Qdrant endpoint via `QDRANT_URL`/`QDRANT_API_KEY`.

### Environment variables

The application reads configuration from `config/appsettings.toml`. The following
environment variables can override values in that file:

- `LLM_API_KEY` – API key for your LLM provider.
- `QDRANT_URL` – Qdrant cloud endpoint.
- `QDRANT_API_KEY` – Qdrant authentication token.

## Contributing
Development follows a TDD approach. Please write or update tests before implementing new functionality.
