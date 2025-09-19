#!/usr/bin/env bash
set -euo pipefail

# seed_smallset.sh (refactored)
# - Optionally read QDRANT_URL and QDRANT_API_KEY from config/appsettings.toml if env vars are unset
# - Ensure Qdrant is reachable (with/without API key)
# - If no QDRANT_URL given and not reachable, run a local Qdrant docker container and wait for health
# - Run the data preparation pipeline and then the indexing script

ensure_qdrant() {
  local config_file="config/appsettings.toml"

  # 1) Load config values into env if not already set
  if [[ -z "${QDRANT_URL:-}" || -z "${QDRANT_API_KEY:-}" ]]; then
    if [[ -f "${config_file}" ]]; then
      # Use a safe here-doc'd Python block to emit shell-safe exports
      # Requires Python 3.11+ for tomllib; replace with 'tomli' for older Pythons.
      # Emits lines like: QDRANT_URL='https://...'; QDRANT_API_KEY='...'
      eval "$(
python - "${config_file}" <<'PY'
import shlex, sys
try:
    import tomllib  # Python 3.11+
except Exception:
    # Fallback if tomllib isn't available; do nothing.
    tomllib = None

cfg_path = sys.argv[1]
if tomllib is not None:
    try:
        with open(cfg_path, "rb") as f:
            cfg = tomllib.load(f)
    except FileNotFoundError:
        cfg = {}
else:
    cfg = {}

retr = (cfg or {}).get("retrieval", {})
url = (retr or {}).get("qdrant_url", "") or ""
key = (retr or {}).get("qdrant_api_key", "") or ""

if url:
    print(f"export QDRANT_URL={shlex.quote(url)}")
if key:
    print(f"export QDRANT_API_KEY={shlex.quote(key)}")
PY
      )"
    fi
  fi

  local qdrant_url="${QDRANT_URL:-http://localhost:6333}"
  local api_key="${QDRANT_API_KEY:-}"
  local image="${QDRANT_IMAGE:-qdrant/qdrant}"
  local max_wait="${QDRANT_WAIT_SECONDS:-30}"

  # Small helper to probe /healthz with optional API key
  qdrant_check_health() {
    local url="$1"; shift
    local key="${1:-}"
    local -a FAIL_FLAG=("--fail-with-body")

    if ! curl --fail-with-body --version >/dev/null 2>&1; then
      FAIL_FLAG=("--fail")
    fi
    if [[ -n "$key" ]]; then
      curl --silent "${FAIL_FLAG[@]}" -H "api-key: ${key}" "${url}/healthz" >/dev/null
    else
      curl --silent "${FAIL_FLAG[@]}" "${url}/healthz" >/dev/null
    fi
  }

  # 2) Try to reach Qdrant (with key if provided, else without as a fallback)
  if qdrant_check_health "${qdrant_url}" "${api_key}" || qdrant_check_health "${qdrant_url}" ""; then
    echo "Qdrant is reachable at ${qdrant_url}"
    return 0
  fi

  # 3) If the user explicitly set QDRANT_URL and it's not reachable, abort with guidance
  if [[ -n "${QDRANT_URL:-}" ]]; then
    echo "Qdrant endpoint ${qdrant_url} is unreachable. Provide a valid QDRANT_URL and (if required) QDRANT_API_KEY for a remote instance." >&2
    exit 1
  fi

  # 4) Otherwise, start a local container
  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is required to run a local Qdrant instance. Install Docker or set QDRANT_URL/QDRANT_API_KEY." >&2
    exit 1
  fi

  echo "Starting local Qdrant container on :6333 using image ${image} ..."
  docker run -d --rm -p 6333:6333 "${image}" >/tmp/qdrant.log 2>&1

  # 5) Wait until healthy
  for _ in $(seq 1 "${max_wait}"); do
    if qdrant_check_health "${qdrant_url}" ""; then
      echo "Local Qdrant is ready at ${qdrant_url}"
      return 0
    fi
    sleep 1
  done

  echo "Failed to start local Qdrant container within ${max_wait}s. Recent docker logs:" >&2
  docker ps --format '{{.ID}} {{.Image}} {{.Ports}}' | grep qdrant || true
  docker logs $(docker ps -q --filter ancestor="${image}") --tail 100 >&2 || true
  exit 1
}

main() {
  local config_file="config/appsettings.toml"
  local raw_dir="${RAW_DATA_DIR:-}"

  if [[ -z "${raw_dir}" && -f "${config_file}" ]]; then
    raw_dir="$(python - "${config_file}" <<'PY'
import sys

try:
    import tomllib  # Python 3.11+
except Exception:
    tomllib = None

cfg_path = sys.argv[1]
if tomllib is None:
    print("", end="")
    raise SystemExit(0)

try:
    with open(cfg_path, "rb") as handle:
        cfg = tomllib.load(handle)
except FileNotFoundError:
    cfg = {}

raw_dir = (cfg or {}).get("data", {}).get("raw_dir", "") or ""
print(raw_dir)
PY
    )"
  fi

  if [[ -z "${raw_dir}" ]]; then
    raw_dir=".data/raw"
  fi

  mkdir -p "${raw_dir}"

  local -a pipeline_cmd=(
    python -m pipeline.pipeline --from-api --config "${config_file}" --raw-dir "${raw_dir}"
  )

  if [[ $# -gt 0 ]]; then
    pipeline_cmd+=("$@")
  fi

  echo "Running ingestion pipeline: ${pipeline_cmd[*]}"
  "${pipeline_cmd[@]}"

  # ---- Qdrant availability ----
  ensure_qdrant

  # ---- Index documents ----
  echo "Indexing processed trials into Qdrant"
  python -m scripts.index
}

# Execute only if run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
