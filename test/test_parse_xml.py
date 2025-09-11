from pathlib import Path
import pytest
from pipeline.parse_xml import parse_one

FIXTURE_DIR = Path(__file__).parent / 'fixtures' / 'xml'

EXPECTED = {
    'NCT00000001': {
        'nct_id': 'NCT00000001',
        'title': 'Study of Foo',
        'condition': ['Foo Syndrome'],
        'interventions': ['Drug: FooDrug'],
        'eligibility': {
            'inclusion': ['Age 18 to 65'],
            'exclusion': ['Uncontrolled hypertension'],
        },
    },
    'NCT00000002': {
        'nct_id': 'NCT00000002',
        'title': 'Bar Study',
        'condition': ['Bar Disorder'],
        'interventions': ['Procedure: BarTherapy'],
        'eligibility': {
            'inclusion': ['Diagnosis of Bar Disorder'],
            'exclusion': ['Prior Bar Therapy'],
        },
    },
}

@pytest.mark.parametrize('xml_file', sorted(FIXTURE_DIR.glob('NCT*.xml')))
def test_parse_xml(xml_file):
    parsed = parse_one(xml_file)
    expected = EXPECTED[xml_file.stem]
    assert parsed['nct_id'] == expected['nct_id']
    assert parsed['title'] == expected['title']
    assert parsed['condition'] == expected['condition']
    assert parsed['interventions'] == expected['interventions']
    assert parsed['eligibility']['inclusion'] == expected['eligibility']['inclusion']
    assert parsed['eligibility']['exclusion'] == expected['eligibility']['exclusion']
