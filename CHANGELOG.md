# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.2] - 2025-09-30

### Changed
- Documented the planned configuration and secret management refactor to prioritize environment variables, adopt `.env` parity for development, and align deployment workflows with Twelve-Factor principles.

## [0.1.1] - 2025-09-29

### Fixed
- Added missing dependencies to production container.

## [0.1.0] - 2025-09-26

### Added
- Implemented QA, trial, metadata, and eligibility endpoints along with the static UI shell in `app/main.py`.
- Built the `/ask` retrieval and language-model response flow in `app/routers/qa.py`.
- Added ingestion metadata summarization helpers in `app/metadata.py`.
- Created the ingestion pipeline orchestration in `pipeline/pipeline.py`.
- Introduced the evaluation harness in `eval/eval.py`.
- Delivered the chat front-end experience in `ui/index.html`.
