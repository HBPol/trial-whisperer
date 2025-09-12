from fastapi.testclient import TestClient

from app.main import app
from app.routers import eligibility

client = TestClient(app)


def test_check_eligibility_success():

    def mock_retrieve(nct_id):
        return ["criteria"]

    def mock_check(criteria, patient):
        return {"eligible": True, "reasons": []}

    eligibility.retrieve_criteria_for_trial = mock_retrieve
    eligibility.check_eligibility = mock_check
    response = client.post(
        "/check-eligibility/", json={"nct_id": "NCT1", "patient": {}}
    )
    assert response.status_code == 200
    assert response.json()["eligible"] is True


def test_check_eligibility_missing_criteria_returns_400():
    eligibility.retrieve_criteria_for_trial = lambda nct_id: None
    response = client.post(
        "/check-eligibility/", json={"nct_id": "NCT1", "patient": {}}
    )
    assert response.status_code == 400
