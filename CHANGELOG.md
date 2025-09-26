# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2025-09-26

### Added
- Implemented QA, trial, metadata, and eligibility endpoints along with the static UI shell in `app/main.py`.
- Built the `/ask` retrieval and language-model response flow in `app/routers/qa.py`.
- Added ingestion metadata summarization helpers in `app/metadata.py`.
- Created the ingestion pipeline orchestration in `pipeline/pipeline.py`.
- Introduced the evaluation harness in `eval/eval.py`.
- Delivered the chat front-end experience in `ui/index.html`.
