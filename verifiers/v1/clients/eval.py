"""The eval client: relay the program's native request to the provider.

`EvalClient` (the default) is a thin `httpx` forwarder: it sends the program's request body
without a typed round-trip, mutating only what the eval owns (model + sampling, via the dialect's
`apply_overrides`). Eligible end-to-end request headers are forwarded too; rollout auth, body
framing, and connection headers are replaced. The provider response is parsed into a vf
`Response` for the trace, while its full JSON object stays on `Response.raw` for the interception
server to return.

The transport is provider-agnostic: the dialect supplies the upstream path + auth headers, so a
new wire format (incl. non-OpenAI providers like Anthropic) is just a new `Dialect` — no client
change. Endpoint config (base url, api key, billing headers) comes from the client config.
"""

import math
import re
from collections.abc import Mapping

import httpx
from pydantic_core import from_json, to_json

from verifiers.v1.clients.client import Client, RelayReply, session_id_headers
from verifiers.v1.dialects import ChatDialect, Dialect
from verifiers.v1.errors import model_error
from verifiers.v1.graph import PendingTurn
from verifiers.v1.types import PendingModelIO, Response, SamplingConfig

# These fields describe the localhost request, its original bytes, or its connection. HTTPX
# rebuilds the provider request from JSON; endpoint configuration and provider auth apply last.
_BLOCKED_REQUEST_HEADERS = frozenset(
    {
        # The harness uses this rollout secret to authenticate with the localhost server.
        # The dialect adds the actual provider authorization after filtering.
        "authorization",
        # HTTPX recalculates these for the provider URL, JSON bytes, and supported decoders.
        "accept-encoding",
        "content-encoding",
        "content-length",
        "content-type",
        "host",
        "transfer-encoding",
        # These control only the localhost HTTP exchange.
        "expect",
        "keep-alive",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "upgrade",
        # The eval owns the model and sampling settings, so it changes those JSON fields before
        # sending upstream. Hashes and signatures calculated from the intercepted body are stale.
        "content-digest",
        "content-md5",
        "digest",
        "repr-digest",
        "signature",
        "signature-input",
    }
)
# Atomic so one CRLF cannot backtrack into two line endings and split an event mid-field.
_SSE_EVENT_END = re.compile(rb"(?>\r\n|\r|\n){2}")


def validate_capture_json(body: object, source: str) -> None:
    """Reject non-finite numbers before captured provider JSON reaches trace hashing."""
    pending = [body]
    while pending:
        value = pending.pop()
        if isinstance(value, float) and not math.isfinite(value):
            raise model_error(f"{source} contained a non-finite JSON number")
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)


def validate_requested_token_data(response: Response, outbound_body: dict) -> None:
    """Fail before graph commit when explicitly requested training data is unusable."""
    require_ids = outbound_body.get("return_token_ids") is True
    require_logprobs = outbound_body.get("logprobs") is True
    if not (require_ids or require_logprobs):
        return

    tokens = response.tokens
    if tokens is None:
        raise model_error(
            "upstream omitted or returned misaligned token IDs/logprobs requested for exact training traces"
        )
    if require_ids and (not tokens.prompt_ids or not tokens.completion_ids):
        raise model_error("upstream returned empty prompt or completion token IDs requested for exact training traces")
    if require_logprobs and (
        len(tokens.completion_logprobs) != len(tokens.completion_ids)
        or any(not math.isfinite(logprob) for logprob in tokens.completion_logprobs)
    ):
        raise model_error("upstream returned missing, misaligned, or non-finite completion logprobs")


class EvalClient(Client):
    """Relay native JSON to the provider and parse a copy for the trace."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        headers: dict[str, str] | None = None,
        outbound_body_denylist: list[str] | None = None,
        capture_model_io: bool = False,
        timeout: float | None = None,
        connect_timeout: float = 30.0,
        max_connections: int = 28000,
        max_keepalive_connections: int = 28000,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        # Keep endpoint headers separate so they can override intercepted request headers before
        # the dialect's provider authentication is applied.
        self.headers = dict(headers or {})
        self.outbound_body_denylist = frozenset(outbound_body_denylist or ())
        self.capture_model_io = capture_model_io
        # Build full URLs ourselves (base_url + dialect.upstream_path) rather than relying on
        # httpx base-url joining, which drops the base path for a leading-slash request path.
        self.http = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=connect_timeout),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections,
            ),
        )

    async def get_response(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
        session_id: str | None = None,
        turn: PendingTurn | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        outbound_body = self.outbound_body(dialect, body, model, sampling_args)
        encoded_body = to_json(outbound_body, inf_nan_mode="null")
        resp = await self._request(
            self.base_url + dialect.upstream_path,
            outbound_body,
            self._headers(dialect, headers, session_id),
            encoded_body=encoded_body,
        )
        raw = from_json(resp.content)
        if self.capture_model_io:
            validate_capture_json(raw, "upstream response")
        response = dialect.parse_response(dialect.validate_response(raw))
        validate_requested_token_data(response, outbound_body)
        response.raw = raw  # the program gets the provider's bytes back 1:1
        if self.capture_model_io:
            # Capture the parsed semantics of the exact bytes sent (e.g. non-finite extension
            # values serialize as JSON null), not the pre-serialization Python object.
            response.pending_model_io = PendingModelIO(
                provider_route=dialect.upstream_path,
                request_body=from_json(encoded_body),
                response_body=raw,
                response_kind="exact_provider_json",
            )
        return response

    def outbound_body(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
    ) -> dict:
        """Apply dialect overrides, then the final top-level outbound-field denylist."""
        overridden = dialect.apply_overrides(body, model, sampling_args)
        if not self.outbound_body_denylist:
            return overridden
        return {key: value for key, value in overridden.items() if key not in self.outbound_body_denylist}

    def _headers(
        self,
        dialect: Dialect,
        incoming: Mapping[str, str] | None,
        session_id: str | None,
    ) -> httpx.Headers:
        """Build provider headers from the intercepted request.

        Preserve provider feature headers such as `openai-beta`, discard localhost auth and
        transport framing, then apply endpoint-configured headers, session routing, and real
        provider auth.
        """
        headers = httpx.Headers(incoming if isinstance(dialect, ChatDialect) else None)
        connection = headers.pop("connection", "")
        for name in _BLOCKED_REQUEST_HEADERS | set(map(str.strip, connection.lower().split(","))):
            headers.pop(name, None)
        headers.update(self.headers)
        if session_id:
            headers.update(session_id_headers(session_id) or {})
        headers.update(dialect.auth_headers(self.api_key))
        return headers

    async def _request(
        self,
        url: str,
        body: dict,
        headers: httpx.Headers,
        *,
        stream: bool = False,
        encoded_body: bytes | None = None,
    ) -> httpx.Response:
        headers.setdefault("content-type", "application/json")
        request = self.http.build_request(
            "POST",
            url,
            content=encoded_body or to_json(body, inf_nan_mode="null"),
            headers=headers,
        )
        try:
            response = await self.http.send(request, stream=stream)
        except httpx.TimeoutException as e:
            raise model_error(f"{type(e).__name__}: {e}", status_code=504) from e
        except httpx.HTTPError as e:
            raise model_error(f"{type(e).__name__}: {e}", status_code=503) from e
        if not stream:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                # relay the provider's status (and body) so the harness SDK retries 5xx/429 and not
                # 4xx; an empty/HTML body (e.g. a 404 from a base_url missing `/v1`) would otherwise
                # make an information-free ProviderError
                raise model_error(
                    f"upstream {e.response.status_code}: {e.response.text}",
                    status_code=e.response.status_code,
                ) from e
            return response
        if response.status_code < 400:
            return response
        try:
            text = (await response.aread()).decode("utf-8", errors="replace")
        finally:
            await response.aclose()
        raise model_error(f"upstream {response.status_code}: {text}", status_code=response.status_code)

    async def relay(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
        session_id: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> RelayReply:
        # Relay complete SSE events so the interception server can safely insert keepalives
        # between them. Error responses are mapped before any event is handed back.
        outbound_body = self.outbound_body(dialect, body, model, sampling_args)
        encoded_body = to_json(outbound_body, inf_nan_mode="null")
        resp = await self._request(
            self.base_url + dialect.upstream_path,
            outbound_body,
            self._headers(dialect, headers, session_id),
            stream=True,
            encoded_body=encoded_body,
        )

        async def chunks():
            buffer = bytearray()
            search_from = 0
            async for chunk in resp.aiter_bytes():
                buffer += chunk
                while match := _SSE_EVENT_END.search(buffer, search_from):
                    yield bytes(buffer[: match.end()])
                    del buffer[: match.end()]
                    search_from = 0
                # A delimiter is at most four bytes and can straddle chunks.
                search_from = max(0, len(buffer) - 3)
            if buffer:
                yield bytes(buffer)

        def finalize_response(response: Response) -> None:
            validate_requested_token_data(response, outbound_body)
            if self.capture_model_io:
                response_body = response.model_dump(mode="json")
                validate_capture_json(response_body, "normalized streamed response")
                response.pending_model_io = PendingModelIO(
                    provider_route=dialect.upstream_path,
                    request_body=from_json(encoded_body),
                    response_body=response_body,
                    response_kind="normalized_stream_response",
                )

        return RelayReply(
            content_type=resp.headers.get("content-type", "text/event-stream"),
            chunks=chunks(),
            close=resp.aclose,
            finalize_response=finalize_response,
        )

    async def relay_aux(self, dialect: Dialect, route: str, body: dict) -> dict:
        # A side request (e.g. count_tokens): relay its native JSON and return the provider JSON.
        resp = await self._request(
            self.base_url + route,
            body,
            self._headers(dialect, None, None),
        )
        return from_json(resp.content)

    async def close(self) -> None:
        await self.http.aclose()
