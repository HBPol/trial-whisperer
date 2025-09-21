# Project Plan

**Timeline:** 5–7 days total (MVP)

| Day | Task(s)                                                                                                                                                                  | Deliverables                                                                         |
|-----|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| **Day 1** | Project setup: repo structure, Dockerfile, CI skeleton. Provision vector DB (Qdrant free cloud) + LLM API key (Gemini). Establish testing framework (pytest).            | Repo scaffold, working container build, config templates. Test framework stablished. |
| **Day 2** | Data pipeline: download XML subset, parse to JSONL, chunk text. Unit tests on 10 trials.  Write initial failing tests for core modules (XML parser, chunker, API routes) | JSONL dataset + scripts (`parse_xml.py`, `chunk.py`).                                |
| **Day 3** | Indexing: embed chunks, load into Qdrant/Vertex Search. Define schema (trial_id, section, text, vector).                                                                 | Populated vector DB + `indexer.py`.                                                  |
| **Day 4** | API: Implement `/ask` with retrieval + LLM, return citations. Implement `/trial/{nct_id}`.                                                                               | FastAPI app with working endpoints.                                                  |
| **Day 5** | Innovation: Implement `/check-eligibility` tool (parse trial criteria rules + patient JSON). Initial logic should cover age/sex while lab parsing is staged for later.                                                                                                                                   | Eligibility checker endpoint working.
| **Day 6** | Frontend: Simple chat UI (HTML/JS), connect to `/ask`. Add citations panel.                                                                                              | Usable demo UI.                                                                      |
| **Day 7** | Testing & polish: Create evaluation set, run metrics, finalize README/docs. Deploy to Cloud Run or Hugging Face Spaces.                                                  | Live demo URL, README with setup + demo GIF, evaluation results.                     |

**Final Deliverables**  
Dockerized FastAPI app with 3 endpoints (`/ask`, `/trial/{nct_id}`, `/check-eligibility` – age/sex-based eligibility until lab parsing ships).
- Data pipeline (scripts + sample processed dataset).  
- Indexed trial subset in vector DB.  
- Minimal web UI for chat & citations.  
- Evaluation script + sample test results.  
- Deployment (Cloud Run / Hugging Face Space) + README documentation.  
