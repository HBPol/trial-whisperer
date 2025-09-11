#!/usr/bin/env bash
set -euo pipefail
python pipeline/download.py
python pipeline/parse_xml.py # TODO: implement real parsing
python pipeline/normalize.py # TODO
python pipeline/chunk.py # TODO
python pipeline/index_qdrant.py
