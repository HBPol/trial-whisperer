#!/usr/bin/env bash
set -euo pipefail

ensure_qdrant() {
  local config_file="config/appsettings.toml"

  if [ -z "${QDRANT_URL:-}" ] || [ -z "${QDRANT_API_KEY:-}" ]; then
    if [ -f "${config_file}" ]; then
      eval "$(python - "${config_file}" <<'PY'
import shlex, sys, tomllib
cfg = tomllib.load(open(sys.argv[1], 'rb'))
retr = cfg.get('retrieval', {})
url = retr.get('qdrant_url', '')
key = retr.get('qdrant_api_key', '')
if url:
    print(f"QDRANT_URL={shlex.quote(url)}")
if key:
    print(f"QDRANT_API_KEY={shlex.quote(key)}")
PY
      )"
    fi
  fi

  local qdrant_url="${QDRANT_URL:-http://localhost:6333}"

  if curl --silent --fail "${qdrant_url}/health" > /dev/null; then
    echo "Qdrant is reachable at ${qdrant_url}"
    return
  fi

  if [ -n "${QDRANT_URL:-}" ]; then
    echo "Qdrant endpoint ${qdrant_url} is unreachable. Provide QDRANT_URL and QDRANT_API_KEY for a remote instance." >&2
    exit 1
  fi

  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is required to run a local Qdrant instance. Install Docker or set QDRANT_URL/QDRANT_API_KEY." >&2
    exit 1
  fi

  echo "Starting local Qdrant container..."
  docker run -d --rm -p 6333:6333 qdrant/qdrant >/tmp/qdrant.log

  for _ in {1..30}; do
    if curl --silent --fail "${qdrant_url}/health" > /dev/null; then
      echo "Local Qdrant is ready."
      return
    fi
    sleep 1
  done

  echo "Failed to start local Qdrant container." >&2
  exit 1
}

python pipeline/download.py
python pipeline/parse_xml.py # TODO: implement real parsing
python pipeline/normalize.py # TODO
python pipeline/chunk.py # TODO

ensure_qdrant

python scripts/index.py
