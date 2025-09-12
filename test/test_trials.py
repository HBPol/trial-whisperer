from fastapi.testclient import TestClient

from app.main import app
from app.routers import trials

client = TestClient(app)


def test_get_trial_returns_trial():
    trials.TRIALS["NCT1"] = {"id": "NCT1", "title": "A trial"}
    response = client.get("/trial/NCT1")
    assert response.status_code == 200
    assert response.json()["id"] == "NCT1"


def test_get_trial_missing_returns_400():
    response = client.get("/trial/UNKNOWN")
    assert response.status_code == 400
