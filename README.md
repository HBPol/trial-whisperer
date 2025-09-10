# TrialWhisperer


[![CI](https://github.com/OWNER/trial-whisperer/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/trial-whisperer/actions/workflows/ci.yml)

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
- Qdrant (managed free tier) or Vertex AI Search
- Gemini (or OpenAI) LLM
- Cloud Run / Hugging Face Spaces


## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config/appsettings.example.toml config/appsettings.toml
# Edit config/appsettings.toml with your keys / endpoints
make seed # small demo dataset + index
make run
```

## Contributing
Development follows a TDD approach. Please write or update tests before implementing new functionality.
