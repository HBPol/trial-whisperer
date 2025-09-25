# Project Overview

**Title:** Clinical Trial Protocol Chatbot (MVP)

**Objective:**  
Build a specialized chatbot that allows researchers and clinicians to quickly query **clinical trial protocols** (focusing on inclusion/exclusion criteria, outcomes, interventions) sourced from **ClinicalTrials.gov**. The system should leverage a **retrieval-augmented generation (RAG) pipeline** over publicly available trial XML data.  

**MVP scope:**  
Beyond general Q&A, the chatbot will support an **Eligibility Criteria Checker**: given a structured patient profile (age, gender, basic lab values), it will answer *“Would this patient likely qualify for trial NCTxxxxxx?”*  

**Target Users:**  
- Clinical researchers screening trials.  
- Scientific teams comparing protocol criteria.  

**Tech Stack:**  
- **Python**: FastAPI for backend.  
- **Data Engineering**: XML parsing, normalization, chunking.  
- **Vector Database**: Qdrant Cloud (free tier, 1GB) or Vertex AI Search.  
- **LLMs**: Gemini API (free tier) or OpenAI (if credits available).  
- **Hosting**: Google Cloud Run (always-free tier).  
- **UI**: Minimal single-page web app (chat interface + citation display).  

**Deployment Goals:**  
- Zero or minimal cost (use free tiers).  
- Containerized (Docker).  
- Ready to demo via a public endpoint.  
