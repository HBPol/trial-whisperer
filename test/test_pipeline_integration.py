from pathlib import Path
import json

from pipeline.pipeline import process_trials

FIXTURE_DIR = Path(__file__).parent / 'fixtures' / 'xml'
OUTPUT_JSONL = Path('.data/processed/trials.jsonl')

def test_pipeline_creates_chunks_per_trial():
    # Ensure clean state
    if OUTPUT_JSONL.exists():
        OUTPUT_JSONL.unlink()
    if OUTPUT_JSONL.parent.exists():
        # Remove directory if empty? but not necessary
        pass

    # Run pipeline on fixtures
    process_trials(FIXTURE_DIR)

    # JSONL file is written
    assert OUTPUT_JSONL.exists()

    # Load produced chunks
    with OUTPUT_JSONL.open('r', encoding='utf-8') as f:
        chunks = [json.loads(line) for line in f]

    # Every chunk has essential metadata
    assert all('nct_id' in c and c['nct_id'] for c in chunks)
    assert all('section' in c and c['section'] for c in chunks)

    # There is at least one chunk per trial
    expected_ids = {p.stem for p in FIXTURE_DIR.glob('NCT*.xml')}
    for nct_id in expected_ids:
        assert any(chunk['nct_id'] == nct_id for chunk in chunks)
