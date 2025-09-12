from pipeline.chunk import chunk_sections


def test_chunk_sections_splits_and_preserves_metadata():
    long_text = "word " * 1000  # 1000 tokens when split on whitespace
    record = {
        "nct_id": "NCT123456",
        "eligibility": {
            "inclusion": long_text.strip(),
            "exclusion": "not eligible",
        },
    }

    chunks = chunk_sections(record, target_tokens=700)

    assert len(chunks) == 3
    # All chunks keep their trial identifier
    assert all(chunk["nct_id"] == "NCT123456" for chunk in chunks)

    # Long section split into two chunks, both correctly labelled
    assert chunks[0]["section"] == "eligibility.inclusion"
    assert chunks[1]["section"] == "eligibility.inclusion"
    assert chunks[2]["section"] == "eligibility.exclusion"

    assert len(chunks[0]["text"].split()) == 700
    assert len(chunks[1]["text"].split()) == 300
    assert len(chunks[2]["text"].split()) <= 700
