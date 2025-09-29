# Cloud Run deployment helpers

This directory contains tooling for deploying TrialWhisperer to Google Cloud
Run. The `deploy_cloudrun.sh` script now prepares the runtime environment by
loading configuration values and linking Google Secret Manager entries to the
Cloud Run service.

## Runtime environment

The application reads its configuration from environment variables. Non-secret
settings can be provided by exporting the variables prior to running the
deployment script or by pointing the script at an env file via the `ENV_FILE`
variable.

| Variable | Purpose | Default |
| --- | --- | --- |
| `LLM_PROVIDER` | Identifier for the LLM provider. | `gemini` (from `config/.env`) |
| `LLM_MODEL` | Model name used for inference. | `gemini-2.5-flash` |
| `RETRIEVAL_BACKEND` | Retrieval implementation to enable. | `qdrant` |
| `QDRANT_URL` | Qdrant endpoint; use the managed URL in production. | `http://localhost:6333` |
| `QDRANT_COLLECTION` | Collection name that stores seeded vectors. | `trialwhisperer` |
| `TRIALS_DATA_PATH` | Path to the offline trials dataset. | `.data/processed/trials.jsonl` |

When the values above are not exported the script falls back to any defaults
defined in `config/.env` (when loaded through `ENV_FILE`) or to the application
defaults.

## Secret Manager layout

Secrets are resolved in the following order for each runtime variable:

1. Use an already-exported environment variable (`export LLM_API_KEY=...`).
2. Otherwise, reference the Secret Manager entry listed below via
   `--set-secrets`. Optional secrets are skipped when neither a value nor the
   secret exists.

| Runtime variable | Secret Manager secret | Required |
| --- | --- | --- |
| `LLM_API_KEY` | `trialwhisperer-llm-api-key` | ✅ |
| `QDRANT_API_KEY` | `trialwhisperer-qdrant-api-key` | Optional |
| `QDRANT_URL` | `trialwhisperer-qdrant-url` | Optional |

You can override the secret names or versions by exporting environment
variables before invoking the script:

```bash
export LLM_API_KEY_SECRET=my-custom-llm-secret
export LLM_API_KEY_SECRET_VERSION=5
```

This flexibility allows the deployment to consume staging or production
secrets without changing the script.

## Deploying manually

```bash
export GOOGLE_CLOUD_PROJECT=my-gcp-project
export REGION=europe-west1
# Optionally preload defaults from config/.env or a custom file.
export ENV_FILE=config/.env

./infra/deploy_cloudrun.sh
```

The script will:

1. Build a container image using Google Cloud Build.
2. Deploy the image to Cloud Run with any configured environment variables and
   secrets.

## GitHub Actions deployment

The optional workflow at `.github/workflows/deploy.yml` mirrors the manual
script. It authenticates to Google Cloud via OIDC, runs the deployment script,
and therefore pulls the same Secret Manager entries.

Before using the workflow, configure the following repository secrets:

- `GCP_WORKLOAD_IDENTITY_PROVIDER` – Resource name of the Workload Identity
  Federation provider (e.g. `projects/123456789/locations/global/workloadIdentityPools/pool/providers/provider`).
- `GCP_SERVICE_ACCOUNT_EMAIL` – Service account email with permissions to
  deploy to Cloud Run and access the secrets listed above.

Trigger the workflow manually from the GitHub Actions tab and provide the
project ID (and optionally a region override).
