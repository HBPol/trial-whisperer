import json

import httpx
import pytest
import requests

from pipeline.ctgov_api import CtGovApiError, CtGovClient, CtGovRequestsClient


def _make_json_response(url: str, body: dict, params=None) -> requests.Response:
    response = requests.Response()
    response.status_code = 200
    response.headers["Content-Type"] = "application/json"
    request = requests.Request("GET", url, params=params)
    prepared = request.prepare()
    response.request = prepared
    response.url = prepared.url
    response._content = json.dumps(body).encode("utf-8")
    response.encoding = "utf-8"
    return response


class StubRequestsSession(requests.Session):
    def __init__(self, handler):
        super().__init__()
        self._handler = handler
        self.closed = False

    def get(self, url, params=None, timeout=None, **kwargs):  # type: ignore[override]
        return self._handler(self, url, params=params, timeout=timeout, **kwargs)

    def close(self):  # type: ignore[override]
        self.closed = True
        super().close()


def test_client_uses_custom_base_url():
    client = CtGovClient(base_url="https://api.example.test/v2")
    try:
        assert client._client.base_url == httpx.URL("https://api.example.test/v2/")
    finally:
        client.close()


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
        transport=transport, base_url="https://www.clinicaltrials.gov/api/v2"
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


def test_fetch_studies_uses_custom_user_agent():
    captured: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["user_agent"] = request.headers.get("User-Agent")
        return httpx.Response(200, json={"studies": []})

    transport = httpx.MockTransport(handler)
    with httpx.Client(
        transport=transport, base_url="https://www.clinicaltrials.gov/api/v2"
    ) as http_client:
        client = CtGovClient(client=http_client, user_agent="custom-agent/1.0")
        client.fetch_studies()

    assert captured["user_agent"] == "custom-agent/1.0"


def test_fetch_studies_serializes_iterable_params():
    captured: list[list[tuple[str, str]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(list(request.url.params.multi_items()))
        body = {"studies": []}
        return httpx.Response(200, json=body)

    transport = httpx.MockTransport(handler)
    with httpx.Client(
        transport=transport, base_url="https://www.clinicaltrials.gov/api/v2"
    ) as http_client:
        client = CtGovClient(client=http_client)
        client.fetch_studies(
            params={
                "filter.overallStatus": ["RECRUITING", None, "ACTIVE"],
            },
            page_size=None,
        )

    assert captured, "Expected request to be captured"
    status_params = [
        value for key, value in captured[0] if key == "filter.overallStatus"
    ]
    assert status_params == ["RECRUITING,ACTIVE"]


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
        transport=transport, base_url="https://www.clinicaltrials.gov/api/v2"
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
        transport=transport, base_url="https://www.clinicaltrials.gov/api/v2"
    ) as http_client:
        client = CtGovClient(client=http_client)
        with pytest.raises(CtGovApiError):
            client.fetch_studies()


def test_requests_client_fetch_studies_adds_defaults_and_paginates():
    calls: list[dict[str, str]] = []

    def handler(session, url, params=None, timeout=None, **_):
        assert url == "https://www.clinicaltrials.gov/api/v2/studies"
        params = params or []
        params_dict = dict(params)
        calls.append(params_dict)
        if len(calls) == 1:
            assert params_dict["format"] == "json"
            assert params_dict["pageSize"] == "50"
            assert params_dict["query.term"] == "glioblastoma"
            assert "pageToken" not in params_dict
            body = {
                "studies": [
                    {"protocolSection": {"identificationModule": {"nctId": "NCT1"}}}
                ],
                "nextPageToken": "abc",
            }
        else:
            assert params_dict["pageToken"] == "abc"
            body = {
                "studies": [
                    {"protocolSection": {"identificationModule": {"nctId": "NCT2"}}}
                ]
            }
        return _make_json_response(url, body, params=params)

    session = StubRequestsSession(handler)
    client = CtGovRequestsClient(session=session)
    try:
        studies = client.fetch_studies(
            params={"query.term": "glioblastoma"}, page_size=50
        )
    finally:
        client.close()

    assert [s["protocolSection"]["identificationModule"]["nctId"] for s in studies] == [
        "NCT1",
        "NCT2",
    ]
    assert len(calls) == 2


def test_requests_client_fetch_studies_uses_custom_user_agent():
    captured: dict[str, str | None] = {}

    def handler(session, url, params=None, timeout=None, **_):
        captured["user_agent"] = session.headers.get("User-Agent")
        return _make_json_response(url, {"studies": []}, params=params)

    session = StubRequestsSession(handler)
    client = CtGovRequestsClient(session=session, user_agent="custom-agent/2.0")
    try:
        client.fetch_studies()
    finally:
        client.close()

    assert captured["user_agent"] == "custom-agent/2.0"


def test_requests_client_fetch_studies_serializes_iterable_params():
    captured: list[list[tuple[str, str]]] = []

    def handler(session, url, params=None, timeout=None, **_):
        captured.append(list(params or []))
        return _make_json_response(url, {"studies": []}, params=params)

    session = StubRequestsSession(handler)
    client = CtGovRequestsClient(session=session)
    try:
        client.fetch_studies(
            params={"filter.overallStatus": ["RECRUITING", None, "ACTIVE"]},
            page_size=None,
        )
    finally:
        client.close()

    assert captured, "Expected request to be captured"
    status_params = [
        value for key, value in captured[0] if key == "filter.overallStatus"
    ]
    assert status_params == ["RECRUITING,ACTIVE"]


def test_requests_client_fetch_studies_honours_max_studies():
    responses = [
        {"studies": [{"protocolSection": {"identificationModule": {"nctId": "NCT1"}}}]},
        {"studies": [{"protocolSection": {"identificationModule": {"nctId": "NCT2"}}}]},
    ]
    call_count = 0

    def handler(session, url, params=None, timeout=None, **_):
        nonlocal call_count
        call_count += 1
        return _make_json_response(url, responses[call_count - 1], params=params)

    session = StubRequestsSession(handler)
    client = CtGovRequestsClient(session=session)
    try:
        studies = client.fetch_studies(max_studies=1)
    finally:
        client.close()

    assert len(studies) == 1
    assert call_count == 1


def test_requests_client_fetch_studies_raises_on_invalid_payload():
    def handler(session, url, params=None, timeout=None, **_):
        return _make_json_response(url, {"unexpected": []}, params=params)

    session = StubRequestsSession(handler)
    client = CtGovRequestsClient(session=session)
    try:
        with pytest.raises(CtGovApiError):
            client.fetch_studies()
    finally:
        client.close()


def test_requests_client_respects_trust_env_setting():
    def handler(session, url, params=None, timeout=None, **_):
        return _make_json_response(url, {"studies": []}, params=params)

    session = StubRequestsSession(handler)
    assert session.trust_env is True
    client = CtGovRequestsClient(session=session, trust_env=False)
    try:
        client.fetch_studies()
    finally:
        client.close()

    assert session.trust_env is False
