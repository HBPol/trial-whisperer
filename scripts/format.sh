#!/usr/bin/env bash
python -m pip install black isort --quiet
isort . --profile black
black .
