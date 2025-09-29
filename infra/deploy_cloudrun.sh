#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${GOOGLE_CLOUD_PROJECT:-}" ]]; then
  echo "GOOGLE_CLOUD_PROJECT must be set" >&2
  exit 1
fi

SERVICE_NAME=${SERVICE_NAME:-trialwhisperer}
REGION=${REGION:-europe-west1}
IMAGE=${IMAGE:-gcr.io/${GOOGLE_CLOUD_PROJECT}/${SERVICE_NAME}}
MAX_INSTANCES=${MAX_INSTANCES:-1}
ENV_FILE=${ENV_FILE:-}

if [[ -n "$ENV_FILE" ]]; then
  if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    set -a && source "$ENV_FILE" && set +a
  else
    echo "Environment file '$ENV_FILE' was not found" >&2
    exit 1
  fi
fi

# Non-secret environment variables that may be provided when deploying.
CONFIG_ENV_VARS=(
  LLM_PROVIDER
  LLM_MODEL
  RETRIEVAL_BACKEND
  QDRANT_URL
  QDRANT_COLLECTION
  TRIALS_DATA_PATH
)

# Mapping of runtime environment variables to Secret Manager entries.
declare -A SECRET_NAME_MAP=(
  [LLM_API_KEY]="trialwhisperer-llm-api-key"
  [QDRANT_API_KEY]="trialwhisperer-qdrant-api-key"
  [QDRANT_URL]="trialwhisperer-qdrant-url"
)

# Secrets that must resolve to a value (either via environment variable or
# Secret Manager). Optional secrets can be skipped when neither a value nor a
# Secret Manager entry is available.
REQUIRED_SECRETS=(
  LLM_API_KEY
)
function _is_required_secret() {
  local candidate="$1"
  local item
  for item in "${REQUIRED_SECRETS[@]}"; do
    if [[ "$item" == "$candidate" ]]; then
      return 0
    fi
  done
  return 1
}

# Build the list of env vars to pass to Cloud Run.
ENV_VAR_FLAGS=()
for var_name in "${CONFIG_ENV_VARS[@]}"; do
  if [[ -n "${!var_name:-}" ]]; then
    ENV_VAR_FLAGS+=("--set-env-vars" "${var_name}=${!var_name}")
  fi
done

SECRET_FLAGS=()
for var_name in "${!SECRET_NAME_MAP[@]}"; do
  # Allow overriding the secret name and version via environment.
  secret_name_var="${var_name}_SECRET"
  secret_version_var="${var_name}_SECRET_VERSION"
  secret_name="${!secret_name_var:-${SECRET_NAME_MAP[$var_name]}}"
  secret_version="${!secret_version_var:-latest}"

  if [[ -n "${!var_name:-}" ]]; then
    ENV_VAR_FLAGS+=("--set-env-vars" "${var_name}=${!var_name}")
    continue
  fi

  if [[ -z "$secret_name" ]]; then
    if _is_required_secret "$var_name"; then
      echo "Missing secret mapping for ${var_name}. Provide ${var_name} or ${secret_name_var}." >&2
      exit 1
    fi
    continue
  fi

  # Verify the secret exists before deploying to provide fast feedback.
  if ! gcloud secrets describe "$secret_name" --project "$GOOGLE_CLOUD_PROJECT" >/dev/null 2>&1; then
    if _is_required_secret "$var_name"; then
      echo "Secret '${secret_name}' for ${var_name} not found in project ${GOOGLE_CLOUD_PROJECT}." >&2
      exit 1
    fi
    echo "Skipping optional secret '${secret_name}' for ${var_name} because it was not found." >&2
    continue
  fi

  SECRET_FLAGS+=("--set-secrets" "${var_name}=${secret_name}:${secret_version}")
done

echo "Building container image ${IMAGE} in project ${GOOGLE_CLOUD_PROJECT}..."
gcloud builds submit --tag "$IMAGE" --project "$GOOGLE_CLOUD_PROJECT"

echo "Deploying ${SERVICE_NAME} to Cloud Run (region: ${REGION})..."
DEPLOY_CMD=(
  gcloud run deploy "$SERVICE_NAME"
  --image "$IMAGE"
  --region "$REGION"
  --allow-unauthenticated
  --platform managed
  --max-instances "$MAX_INSTANCES"
  --project "$GOOGLE_CLOUD_PROJECT"
)
DEPLOY_CMD+=("${ENV_VAR_FLAGS[@]}")
DEPLOY_CMD+=("${SECRET_FLAGS[@]}")

"${DEPLOY_CMD[@]}"

