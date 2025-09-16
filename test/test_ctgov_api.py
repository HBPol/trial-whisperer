import httpx
import pytest

from pipeline.ctgov_api import CtGovApiError, CtGovClient


def test_fetch_studies_adds_defaults_and_paginates():
    calls: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        calls.append(params)
        if len(calls) == 1:
            assert params["format"] == "json"
            assert params["pageSize"] == "50"
            assert params["query.term"] == "glioblastoma"
            assert "pageToken" not in params
            body = {
                "studies": [
                    {"protocolSection": {"identificationModule": {"nctId": "NCT1"}}}
                ],
                "nextPageToken": "abc",
            }
        else:
            assert params["pageToken"] == "abc"
            body = {
                "studies": [
                    {"protocolSection": {"identificationModule": {"nctId": "NCT2"}}}
                ]
            }
        return httpx.Response(200, json=body)

    transport = httpx.MockTransport(handler)
    with httpx.Client(
        transport=transport, base_url="https://clinicaltrials.gov/data-api"
    ) as http_client:
        client = CtGovClient(client=http_client)
        studies = client.fetch_studies(
            params={"query.term": "glioblastoma"}, page_size=50
        )

    assert [s["protocolSection"]["identificationModule"]["nctId"] for s in studies] == [
        "NCT1",
        "NCT2",
    ]
    assert len(calls) == 2


def test_fetch_studies_honours_max_studies():
    responses = [
        {"studies": [{"protocolSection": {"identificationModule": {"nctId": "NCT1"}}}]},
        {"studies": [{"protocolSection": {"identificationModule": {"nctId": "NCT2"}}}]},
    ]
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, json=responses[call_count - 1])

    transport = httpx.MockTransport(handler)
    with httpx.Client(
        transport=transport, base_url="https://clinicaltrials.gov/data-api"
    ) as http_client:
        client = CtGovClient(client=http_client)
        studies = client.fetch_studies(max_studies=1)

    assert len(studies) == 1
    assert call_count == 1


def test_fetch_studies_raises_on_invalid_payload():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": []})

    transport = httpx.MockTransport(handler)
    with httpx.Client(
        transport=transport, base_url="https://clinicaltrials.gov/data-api"
    ) as http_client:
        client = CtGovClient(client=http_client)
        with pytest.raises(CtGovApiError):
            client.fetch_studies()
