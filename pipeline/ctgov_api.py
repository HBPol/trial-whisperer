"""Client utilities for the ClinicalTrials.gov Data API."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Protocol

import httpx
import requests

__all__ = [
    "CtGovApiError",
    "CtGovClient",
    "CtGovRequestsClient",
    "CtGovApiClientProtocol",
]


DEFAULT_BASE_URL = "https://www.clinicaltrials.gov/api/v2"
DEFAULT_USER_AGENT = "TrialWhisperer/ingest (+https://trialwhisperer.ai/contact)"


class CtGovApiError(RuntimeError):
    """Raised when the ClinicalTrials.gov API returns an unexpected response."""


class CtGovApiClientProtocol(Protocol):
    """Protocol describing the minimal CtGov API client interface."""

    def fetch_studies(
        self,
        *,
        params: Mapping[str, Any] | None = None,
        page_size: int | None = 100,
        max_studies: int | None = None,
    ) -> list[Mapping[str, Any]]: ...

    def close(self) -> None: ...

    def __enter__(self) -> "CtGovApiClientProtocol": ...

    def __exit__(self, exc_type, exc, tb) -> None: ...


def _flatten_params(params: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    """Expand mapping values into a list of query parameter tuples."""

    flattened: list[tuple[str, str]] = []
    if not params:
        return flattened

    for key, value in params.items():
        if value is None:
            continue

        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            serialized_items = [str(item) for item in value if item is not None]
            if not serialized_items:
                continue
            flattened.append((key, ",".join(serialized_items)))
            continue

        flattened.append((key, str(value)))

    return flattened


class CtGovClient:
    """Thin wrapper around :class:`httpx.Client` for the Data API."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
        trust_env: bool | None = None,
        headers: Mapping[str, Any] | None = None,
        user_agent: str | None = None,
    ) -> None:
        prepared_headers: dict[str, str] = {"Accept": "application/json"}

        if user_agent and user_agent.strip():
            prepared_headers["User-Agent"] = user_agent.strip()
        else:
            prepared_headers["User-Agent"] = DEFAULT_USER_AGENT

        if headers:
            for key, value in headers.items():
                if value is None:
                    continue
                prepared_headers[str(key)] = str(value)

        if client is None:
            if trust_env is None:
                trust_env = True
            client = httpx.Client(
                base_url=base_url,
                timeout=timeout,
                headers=prepared_headers,
                trust_env=trust_env,
            )
            self._owns_client = True
        else:
            if "User-Agent" in client.headers:
                del client.headers["User-Agent"]
            for key, value in prepared_headers.items():
                client.headers[key] = value
            self._owns_client = False

        self._client = client

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "CtGovClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

    def fetch_studies(
        self,
        *,
        params: Mapping[str, Any] | None = None,
        page_size: int | None = 100,
        max_studies: int | None = None,
    ) -> list[Mapping[str, Any]]:
        """Fetch study records from the ClinicalTrials.gov Data API."""

        base_params = _flatten_params(params)

        has_page_size = any(key == "pageSize" for key, _ in base_params)
        if page_size is not None and not has_page_size:
            base_params.append(("pageSize", str(page_size)))

        if not any(key == "format" for key, _ in base_params):
            base_params.append(("format", "json"))

        collected: list[Mapping[str, Any]] = []
        next_token: str | None = None

        while True:
            query: list[tuple[str, str]] = list(base_params)
            if next_token:
                query.append(("pageToken", next_token))

            try:
                response = self._client.get("/studies", params=query)
                response.raise_for_status()
            except (
                httpx.HTTPStatusError
            ) as exc:  # pragma: no cover - exercised indirectly
                raise CtGovApiError(
                    f"ClinicalTrials.gov API returned HTTP {exc.response.status_code}: {exc.response.text[:200]}"
                ) from exc
            except httpx.HTTPError as exc:  # pragma: no cover - network errors
                raise CtGovApiError(
                    "Failed to communicate with ClinicalTrials.gov API"
                ) from exc

            try:
                payload = response.json()
            except ValueError as exc:
                raise CtGovApiError(
                    "ClinicalTrials.gov API returned invalid JSON"
                ) from exc

            if "studies" not in payload or not isinstance(payload["studies"], list):
                raise CtGovApiError(
                    "ClinicalTrials.gov API response missing 'studies' list"
                )

            studies = payload["studies"]

            collected.extend(studies)

            if max_studies is not None and len(collected) >= max_studies:
                return collected[:max_studies]

            next_token = payload.get("nextPageToken")
            if not next_token:
                break

        return collected


class CtGovRequestsClient:
    """Thin wrapper around :class:`requests.Session` for the Data API."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        session: requests.Session | None = None,
        trust_env: bool | None = None,
        headers: Mapping[str, Any] | None = None,
        user_agent: str | None = None,
    ) -> None:
        prepared_headers: dict[str, str] = {"Accept": "application/json"}

        if user_agent and user_agent.strip():
            prepared_headers["User-Agent"] = user_agent.strip()
        else:
            prepared_headers["User-Agent"] = DEFAULT_USER_AGENT

        if headers:
            for key, value in headers.items():
                if value is None:
                    continue
                prepared_headers[str(key)] = str(value)

        owns_session = session is None
        if session is None:
            session = requests.Session()
            if trust_env is None:
                trust_env = True
        if trust_env is not None:
            session.trust_env = trust_env

        # Normalize headers on the session.
        session.headers.pop("User-Agent", None)
        for key, value in prepared_headers.items():
            session.headers[key] = value

        self._session = session
        self._owns_session = owns_session
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    def __enter__(self) -> "CtGovRequestsClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

    def fetch_studies(
        self,
        *,
        params: Mapping[str, Any] | None = None,
        page_size: int | None = 100,
        max_studies: int | None = None,
    ) -> list[Mapping[str, Any]]:
        """Fetch study records from the ClinicalTrials.gov Data API."""

        base_params = _flatten_params(params)

        has_page_size = any(key == "pageSize" for key, _ in base_params)
        if page_size is not None and not has_page_size:
            base_params.append(("pageSize", str(page_size)))

        if not any(key == "format" for key, _ in base_params):
            base_params.append(("format", "json"))

        collected: list[Mapping[str, Any]] = []
        next_token: str | None = None
        studies_url = f"{self._base_url}/studies"

        while True:
            query: list[tuple[str, str]] = list(base_params)
            if next_token:
                query.append(("pageToken", next_token))

            try:
                response = self._session.get(
                    studies_url,
                    params=query,
                    timeout=self._timeout,
                )
                response.raise_for_status()
            except requests.HTTPError as exc:  # pragma: no cover - exercised indirectly
                status = (
                    exc.response.status_code if exc.response is not None else "unknown"
                )
                body = exc.response.text[:200] if exc.response is not None else ""
                raise CtGovApiError(
                    f"ClinicalTrials.gov API returned HTTP {status}: {body}"
                ) from exc
            except (
                requests.RequestException
            ) as exc:  # pragma: no cover - network errors
                raise CtGovApiError(
                    "Failed to communicate with ClinicalTrials.gov API"
                ) from exc

            try:
                payload = response.json()
            except ValueError as exc:
                raise CtGovApiError(
                    "ClinicalTrials.gov API returned invalid JSON"
                ) from exc

            if "studies" not in payload or not isinstance(payload["studies"], list):
                raise CtGovApiError(
                    "ClinicalTrials.gov API response missing 'studies' list"
                )

            studies = payload["studies"]

            collected.extend(studies)

            if max_studies is not None and len(collected) >= max_studies:
                return collected[:max_studies]

            next_token = payload.get("nextPageToken")
            if not next_token:
                break

        return collected
