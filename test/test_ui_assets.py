from pathlib import Path


def test_citation_links_rendered_as_anchors():
    script = Path("ui/app.js").read_text(encoding="utf-8")
    assert "clinicaltrials.gov/study" in script
    assert "encodeURIComponent(citation.nct_id)" in script
    assert 'target="_blank"' in script
    assert 'rel="noopener noreferrer"' in script
    assert '[<a href="' in script
