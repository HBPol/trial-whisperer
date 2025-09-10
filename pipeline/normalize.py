"""Normalize parsed objects to JSONL/Parquet for indexing."""

def normalize(records: list[dict]) -> list[dict]:
    # TODO: clean fields, standardize schema
    return records