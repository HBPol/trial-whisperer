"""Chunk long text sections with headings preserved."""

def chunk_sections(record: dict, target_tokens: int = 500) -> list[dict]:
    # TODO: implement real chunking; return list of {text, nct_id, section}
    return [{"text": "demo text", "nct_id": record.get("nct_id"), "section": "eligibility.inclusion"}]