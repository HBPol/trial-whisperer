#!/usr/bin/env bash
python -m pip install black isort --quiet
black .
isort .
