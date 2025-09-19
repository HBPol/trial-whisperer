from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import httpx
import pytest
import requests

from pipeline.ctgov_api import (DEFAULT_BASE_URL, CtGovApiError, CtGovClient,
                                CtGovRequestsClient)


@dataclass
class ResponsePayload:
    body: Any
    status_code: int = 200


@dataclass
class RequestCapture:
    url: str
    params: list[tuple[str, str]]
    headers: dict[str, str]
    timeout: float | None

    def to_dict(self) -> dict[str, str]:
        """Collapse the query parameters into a single mapping."""

        collapsed: dict[str, str] = {}
        for key, value in self.params:
            collapsed[key] = value
        return collapsed

    def get_header(self, name: str) -> str | None:
        target = name.lower()
        for key, value in self.headers.items():
            if key.lower() == target:
                return value
        return None


def _ensure_payload(
    result: ResponsePayload | Mapping[str, Any] | Sequence[Any] | str | bytes,
) -> ResponsePayload:
    if isinstance(result, ResponsePayload):
        return result
    return ResponsePayload(body=result)


def _make_json_response(
    url: str,
    body: Any,
    *,
    params: Iterable[tuple[str, str]] | None = None,
    status_code: int = 200,
) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response.headers["Content-Type"] = "application/json"
    request = requests.Request("GET", url, params=params)
    prepared = request.prepare()
    response.request = prepared
    response.url = prepared.url
    response._content = json.dumps(body).encode("utf-8")
    response.encoding = "utf-8"
    return response


class HttpxBackend:
    name = "httpx"

    @contextmanager
    def make_client(
        self,
        responder: Callable[[RequestCapture], ResponsePayload | Mapping[str, Any]],
        **client_kwargs: Any,
    ):
        calls: list[RequestCapture] = []

        def handler(request: httpx.Request) -> httpx.Response:
            info = RequestCapture(
                url=str(request.url),
                params=[
                    (str(key), str(value))
                    for key, value in request.url.params.multi_items()
                ],
                headers=dict(request.headers),
                timeout=None,
            )
            calls.append(info)
            payload = _ensure_payload(responder(info))
            body = payload.body
            if isinstance(body, (bytes, bytearray)):
                return httpx.Response(payload.status_code, content=body)
            if isinstance(body, str):
                return httpx.Response(payload.status_code, text=body)
            return httpx.Response(payload.status_code, json=body)

        base_url = client_kwargs.get("base_url", DEFAULT_BASE_URL)
        transport = httpx.MockTransport(handler)
        http_client = httpx.Client(transport=transport, base_url=base_url)
        client = CtGovClient(client=http_client, **client_kwargs)
        try:
            yield client, calls
        finally:
            client.close()
            http_client.close()


class _FakeRequestsSession(requests.Session):
    def __init__(self, handler: Callable[[RequestCapture], requests.Response]):
        super().__init__()
        self._handler = handler
        self.requests: list[RequestCapture] = []
        self.closed = False

    def get(  # type: ignore[override]
        self,
        url: str,
        params: Mapping[str, Any] | Iterable[tuple[str, Any]] | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> requests.Response:
        if params is None:
            param_items: list[tuple[str, str]] = []
        elif isinstance(params, Mapping):
            param_items = [(str(key), str(value)) for key, value in params.items()]
        else:
            param_items = [(str(key), str(value)) for key, value in params]

        info = RequestCapture(
            url=url,
            params=param_items,
            headers=dict(self.headers),
            timeout=timeout,
        )
        self.requests.append(info)
        return self._handler(info)

    def close(self) -> None:  # type: ignore[override]
        self.closed = True
        super().close()


class RequestsBackend:
    name = "requests"

    @contextmanager
    def make_client(
        self,
        responder: Callable[[RequestCapture], ResponsePayload | Mapping[str, Any]],
        **client_kwargs: Any,
    ):
        def handler(info: RequestCapture) -> requests.Response:
            payload = _ensure_payload(responder(info))
            return _make_json_response(
                info.url,
                payload.body,
                params=info.params,
                status_code=payload.status_code,
            )

        session = _FakeRequestsSession(handler)
        client = CtGovRequestsClient(session=session, **client_kwargs)
        try:
            yield client, session.requests
        finally:
            client.close()
            session.close()


@pytest.fixture(params=(HttpxBackend(), RequestsBackend()), ids=("httpx", "requests"))
def backend(request: pytest.FixtureRequest) -> HttpxBackend | RequestsBackend:
    return request.param


def test_client_uses_custom_base_url(backend):
    base_url = "https://api.example.test/v2"

    with backend.make_client(lambda _: {"studies": []}, base_url=base_url) as (
        client,
        calls,
    ):
        client.fetch_studies()

    assert calls
    expected_prefix = f"{base_url.rstrip('/')}/"
    assert all(call.url.startswith(expected_prefix) for call in calls)


def test_fetch_studies_adds_defaults_and_paginates(backend):
    responses = [
        ResponsePayload(
            body={
                "studies": [
                    {"protocolSection": {"identificationModule": {"nctId": "NCT1"}}},
                ],
                "nextPageToken": "abc",
            }
        ),
        ResponsePayload(
            body={
                "studies": [
                    {"protocolSection": {"identificationModule": {"nctId": "NCT2"}}},
                ]
            }
        ),
    ]

    call_records: list[RequestCapture] = []

    def responder(info: RequestCapture) -> ResponsePayload:
        call_index = len(call_records)
        assert call_index < len(responses)
        params = info.to_dict()
        if call_index == 0:
            assert params["format"] == "json"
            assert params["pageSize"] == "50"
            assert params["query.term"] == "glioblastoma"
            assert "pageToken" not in params
        else:
            assert params["pageToken"] == "abc"
        payload = responses[call_index]
        call_records.append(info)
        return payload

    with backend.make_client(responder) as (client, calls):
        studies = client.fetch_studies(
            params={"query.term": "glioblastoma"}, page_size=50
        )

    assert [s["protocolSection"]["identificationModule"]["nctId"] for s in studies] == [
        "NCT1",
        "NCT2",
    ]
    assert len(calls) == 2


def test_fetch_studies_uses_custom_user_agent(backend):
    captured: dict[str, str | None] = {}

    def responder(info: RequestCapture):
        captured["user_agent"] = info.get_header("User-Agent")
        return {"studies": []}

    with backend.make_client(responder, user_agent="custom-agent/1.0") as (client, _):
        client.fetch_studies()
        client.fetch_studies()
        assert captured["user_agent"] == "custom-agent/1.0"


def test_fetch_studies_serializes_iterable_params(backend):
    captured: list[RequestCapture] = []

    def responder(info: RequestCapture):
        captured.append(info)
        return {"studies": []}

    with backend.make_client(responder) as (client, _):
        client.fetch_studies(
            params={
                "filter.overallStatus": ["RECRUITING", None, "ACTIVE"],
            },
            page_size=None,
        )

    assert captured, "Expected request to be captured"
    status_params = [
        value for key, value in captured[0].params if key == "filter.overallStatus"
    ]
    assert status_params == ["RECRUITING,ACTIVE"]


def test_fetch_studies_honours_max_studies(backend):
    responses = [
        ResponsePayload(
            body={
                "studies": [
                    {"protocolSection": {"identificationModule": {"nctId": "NCT1"}}},
                ]
            }
        ),
        ResponsePayload(
            body={
                "studies": [
                    {"protocolSection": {"identificationModule": {"nctId": "NCT2"}}},
                ]
            }
        ),
    ]
    call_counter = 0

    def responder(info: RequestCapture) -> ResponsePayload:
        nonlocal call_counter
        payload = responses[call_counter]
        call_counter += 1
        return payload

    with backend.make_client(responder) as (client, calls):
        studies = client.fetch_studies(max_studies=1)

    assert len(studies) == 1
    assert len(calls) == 1


def test_fetch_studies_raises_on_invalid_payload(backend):

    def responder(info: RequestCapture):
        return {"unexpected": []}

    with backend.make_client(responder) as (client, _):
        with pytest.raises(CtGovApiError):
            client.fetch_studies()


def test_requests_client_respects_trust_env_setting():
    captured: dict[str, bool] = {}

    def handler(info: RequestCapture) -> requests.Response:
        captured["trust_env"] = session.trust_env
        return _make_json_response(info.url, {"studies": []}, params=info.params)

    session = _FakeRequestsSession(handler)
    assert session.trust_env is True
    client = CtGovRequestsClient(session=session, trust_env=False)
    try:
        client.fetch_studies()
    finally:
        client.close()
        session.close()

    assert session.trust_env is False
    assert captured["trust_env"] is False
