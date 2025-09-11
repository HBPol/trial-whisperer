# Requirements Specification

## Functional Requirements
1. **Data Pipeline**
   - Download a curated subset of **ClinicalTrials.gov XML data** (e.g., oncology trials, last 3 years).  
   - Parse trial metadata: `nct_id`, title, condition, interventions, eligibility criteria (split into inclusion vs exclusion), outcomes.  
   - Normalize to JSONL/Parquet.  
   - Chunk text sections (300–700 tokens), preserving section labels and trial references.  
   - Index into vector DB with embeddings and metadata.

2. **API (FastAPI)**
   - `/ask` endpoint: Accepts a query string (and optional `nct_id`). Returns answer + cited chunks.  
   - `/trial/{nct_id}` endpoint: Returns structured trial metadata.  
   - `/check-eligibility`: Accepts `nct_id` + patient JSON (age, gender, lab values). Returns structured assessment (eligible / not eligible + reasons).  

3. **Retrieval & RAG**
   - Top-k semantic retrieval with metadata filters.  
   - LLM prompt ensures **answers are grounded** in retrieved text and citations are returned.  

4. **UI**
   - Simple chat panel with query box + answers.  
   - Display citations (trial ID, section, snippet).  
   - Link to ClinicalTrials.gov trial page.  

5. **Evaluation**
   - Include a gold test set (~20 Q/A pairs).  
   - Run an evaluation script reporting grounding accuracy.  

## Non-Functional Requirements
- **Development Process**: the project must follow Test-Driven Development (TDD) practices, with unit tests written before or alongside implementation.
- **Performance**: API response time < 5s for typical queries.  
- **Scalability**: small dataset (~5–10k trials) to stay within free tier limits.  
- **Reliability**: must not hallucinate unsupported answers (enforce citations).  
- **Security**: no sensitive data; only public trial data + synthetic patient profiles.  
- **Maintainability**: clean modular code, Dockerized deployment, README setup guide.  
- **Usability**: minimal but intuitive chat UI with clear citations.  
