import json

import pytest
from aiohttp import web

from verifiers.v1 import graph
from verifiers.v1.clients import RolloutContext
from verifiers.v1.clients.client import RelayReply
from verifiers.v1.errors import (
    HarnessError,
    InterceptionError,
    ProviderError,
    UserError,
)
from verifiers.v1.interception import InterceptionServer, RolloutSession
from verifiers.v1.retries import RolloutRetryConfig, should_retry
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace
from verifiers.v1.types import AssistantMessage, Response, SamplingConfig, UserMessage


class Request:
    path = "/v1/chat/completions"

    def __init__(self, body: bytes = b"{}", *, fail_read: bool = False) -> None:
        self.headers: dict[str, str] = {}
        self._read_bytes: bytes | None = body
        self.fail_read = fail_read

    async def read(self) -> bytes:
        if self.fail_read:
            raise RuntimeError("request transport failed")
        assert self._read_bytes is not None
        return self._read_bytes

    async def json(self) -> dict:
        return json.loads(await self.read())


class OversizedRequest(Request):
    async def read(self) -> bytes:
        raise web.HTTPRequestEntityTooLarge(max_size=1, actual_size=2)


class Dialect:
    @staticmethod
    def secret(headers) -> str:
        return "secret"

    @staticmethod
    def error_body(message: str) -> dict:
        return {"error": {"message": message}}

    @staticmethod
    def parse_request(body: dict):
        return [UserMessage(content="test")], SamplingConfig()

    @staticmethod
    def streaming(body: dict) -> bool:
        return False


class FailingParseDialect(Dialect):
    @staticmethod
    def parse_request(body: dict):
        raise ValueError("malformed test request")


class UnusedClient:
    async def get_response(self, *args, **kwargs):
        raise AssertionError("client must not be reached")


class FailingClient:
    async def get_response(self, *args, **kwargs):
        raise RuntimeError("unexpected response parser failure")

    async def relay(self, *args, **kwargs):
        raise RuntimeError("unexpected stream parser failure")

    async def relay_aux(self, *args, **kwargs):
        raise RuntimeError("unexpected auxiliary parser failure")


def _server_and_session(
    client, *, prompt: str | None = "test"
) -> tuple[InterceptionServer, RolloutSession]:
    trace = Trace(task=Task(idx=0, prompt=prompt))
    session = RolloutSession(
        ctx=RolloutContext(
            client=client,
            model="model",
            sampling=SamplingConfig(),
        ),
        trace=trace,
    )
    server = InterceptionServer()
    server.sessions["secret"] = session
    return server, session


def _response_error(response) -> str:
    return json.loads(response.body)["error"]["message"]


@pytest.mark.asyncio
async def test_invalid_dialect_request_is_stored_as_non_retryable_harness_error():
    server, session = _server_and_session(UnusedClient())

    response = await server.handle_request(Request(), FailingParseDialect())

    assert response.status == 400
    assert isinstance(session.error, HarnessError)
    assert _response_error(response).startswith(
        "invalid harness model request: ValueError:"
    )
    session.trace.capture_error(session.error)
    retry = RolloutRetryConfig(
        max_retries=2,
        include=["ProviderError", "SandboxError", "TunnelError"],
    )
    assert should_retry(session.trace, retry) is False


@pytest.mark.asyncio
async def test_malformed_json_is_stored_as_non_retryable_harness_error():
    server, session = _server_and_session(UnusedClient())

    response = await server.handle_request(Request(b"{"), Dialect())

    assert response.status == 400
    assert isinstance(session.error, HarnessError)
    assert _response_error(response).startswith(
        "invalid harness model request: JSONDecodeError:"
    )


@pytest.mark.asyncio
async def test_request_transport_failure_is_stored_as_interception_error():
    server, session = _server_and_session(UnusedClient())

    response = await server.handle_request(Request(fail_read=True), Dialect())

    assert response.status == 502
    assert isinstance(session.error, InterceptionError)
    assert _response_error(response).startswith(
        "reading harness request failed: RuntimeError:"
    )


@pytest.mark.asyncio
async def test_oversized_model_request_is_stored_as_harness_error_with_413():
    server, session = _server_and_session(UnusedClient())

    response = await server.handle_request(OversizedRequest(), Dialect())

    assert response.status == 413
    assert isinstance(session.error, HarnessError)
    assert _response_error(response).startswith("harness model request body too large:")
    session.trace.capture_error(session.error)
    retry = RolloutRetryConfig(
        max_retries=2,
        include=[
            "ProviderError",
            "SandboxError",
            "TunnelError",
            "InterceptionError",
        ],
    )
    assert should_retry(session.trace, retry) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [b"[]", b'"text"', b"null"])
async def test_non_object_model_json_is_stored_as_harness_error(body: bytes):
    server, session = _server_and_session(UnusedClient())

    response = await server.handle_request(Request(body), Dialect())

    assert response.status == 400
    assert isinstance(session.error, HarnessError)
    assert "request body must be a JSON object" in _response_error(response)


@pytest.mark.asyncio
async def test_opening_user_failure_is_stored_as_user_error():
    server, session = _server_and_session(UnusedClient(), prompt=None)

    async def fail_user(message: str):
        raise RuntimeError("user simulator failed")

    session.user = fail_user
    response = await server.handle_request(Request(), Dialect())

    assert response.status == 502
    assert isinstance(session.error, UserError)
    assert _response_error(response).startswith("user simulator failed: RuntimeError:")


@pytest.mark.asyncio
async def test_prepare_turn_failure_is_stored_as_interception_error(monkeypatch):
    server, session = _server_and_session(UnusedClient())

    def fail_prepare(*args, **kwargs):
        raise RuntimeError("graph invariant failed")

    monkeypatch.setattr(graph, "prepare_turn", fail_prepare)
    response = await server.handle_request(Request(), Dialect())

    assert response.status == 502
    assert isinstance(session.error, InterceptionError)
    assert _response_error(response).startswith(
        "preparing model turn failed: RuntimeError:"
    )


@pytest.mark.asyncio
async def test_unexpected_client_failure_is_stored_as_provider_error():
    server, session = _server_and_session(FailingClient())

    response = await server.handle_request(Request(), Dialect())

    assert response.status == 502
    assert isinstance(session.error, ProviderError)
    assert _response_error(response) == "unexpected response parser failure"
    session.trace.capture_error(session.error)
    retry = RolloutRetryConfig(
        max_retries=2,
        include=["ProviderError", "SandboxError", "TunnelError"],
    )
    assert should_retry(session.trace, retry) is True


@pytest.mark.asyncio
async def test_commit_failure_is_stored_as_interception_error():
    class ReturningClient:
        async def get_response(self, *args, **kwargs):
            return object()

    server, session = _server_and_session(ReturningClient())

    def fail_commit(*args, **kwargs):
        raise RuntimeError("graph commit failed")

    session.commit = fail_commit
    response = await server.handle_request(Request(), Dialect())

    assert response.status == 502
    assert isinstance(session.error, InterceptionError)
    assert _response_error(response).startswith(
        "committing model turn failed: RuntimeError:"
    )


@pytest.mark.asyncio
async def test_unexpected_stream_client_failure_is_stored_as_provider_error():
    server, session = _server_and_session(FailingClient())

    response = await server._stream(
        Request(),
        session,
        Dialect(),
        {},
        [UserMessage(content="test")],
    )

    assert response.status == 502
    assert isinstance(session.error, ProviderError)
    assert _response_error(response) == "unexpected stream parser failure"


class StreamResponse:
    def __init__(self, **kwargs) -> None:
        self.content_type = ""

    async def prepare(self, request) -> None:
        pass

    async def write(self, chunk: bytes) -> None:
        pass

    async def write_eof(self) -> None:
        pass


class StreamDialect(Dialect):
    class Parser:
        on_done = None

        def feed(self, chunk: bytes) -> None:
            pass

        def finish(self) -> Response:
            return Response(
                id="response",
                created=0,
                model="model",
                message=AssistantMessage(content="done"),
                finish_reason="stop",
            )

    @staticmethod
    def stream_parser():
        return StreamDialect.Parser()


class FinalizeFailingClient:
    async def relay(self, *args, **kwargs):
        async def chunks():
            yield b'data: {"chunk":true}\n\n'

        async def close():
            pass

        def fail_finalize(response):
            raise RuntimeError("provider response validation failed")

        return RelayReply(
            content_type="text/event-stream",
            chunks=chunks(),
            close=close,
            finalize_response=fail_finalize,
        )


class IteratorFailingClient:
    async def relay(self, *args, **kwargs):
        async def chunks():
            yield b'data: {"chunk":true}\n\n'
            raise RuntimeError("provider stream iteration failed")

        async def close():
            pass

        return RelayReply(
            content_type="text/event-stream",
            chunks=chunks(),
            close=close,
        )


class CloseFailingClient:
    async def relay(self, *args, **kwargs):
        async def chunks():
            yield b'data: {"chunk":true}\n\n'

        async def close():
            raise RuntimeError("provider stream close failed")

        return RelayReply(
            content_type="text/event-stream",
            chunks=chunks(),
            close=close,
        )


@pytest.mark.asyncio
async def test_stream_finalize_failure_is_stored_as_provider_error(monkeypatch):
    monkeypatch.setattr(
        "verifiers.v1.interception.server.web.StreamResponse", StreamResponse
    )
    server, session = _server_and_session(FinalizeFailingClient())

    await server._stream(
        Request(),
        session,
        StreamDialect(),
        {},
        [UserMessage(content="test")],
    )

    assert isinstance(session.error, ProviderError)
    assert str(session.error) == "provider response validation failed"


@pytest.mark.asyncio
async def test_stream_iterator_failure_is_stored_as_provider_error(monkeypatch):
    monkeypatch.setattr(
        "verifiers.v1.interception.server.web.StreamResponse", StreamResponse
    )
    server, session = _server_and_session(IteratorFailingClient())

    await server._stream(
        Request(),
        session,
        StreamDialect(),
        {},
        [UserMessage(content="test")],
    )

    assert isinstance(session.error, ProviderError)
    assert str(session.error) == "provider stream iteration failed"


@pytest.mark.asyncio
async def test_stream_close_failure_is_stored_as_provider_error(monkeypatch):
    monkeypatch.setattr(
        "verifiers.v1.interception.server.web.StreamResponse", StreamResponse
    )
    server, session = _server_and_session(CloseFailingClient())

    await server._stream(
        Request(),
        session,
        StreamDialect(),
        {},
        [UserMessage(content="test")],
    )

    assert isinstance(session.error, ProviderError)
    assert str(session.error) == "provider stream close failed"


@pytest.mark.asyncio
async def test_stream_commit_failure_is_stored_as_interception_error(monkeypatch):
    monkeypatch.setattr(
        "verifiers.v1.interception.server.web.StreamResponse", StreamResponse
    )
    server, session = _server_and_session(FinalizeFailingClient())

    def fail_commit(*args, **kwargs):
        raise RuntimeError("graph commit failed")

    def accept_finalize(response):
        pass

    client = session.ctx.client
    original_relay = client.relay

    async def relay_without_finalize(*args, **kwargs):
        reply = await original_relay(*args, **kwargs)
        return RelayReply(
            content_type=reply.content_type,
            chunks=reply.chunks,
            close=reply.close,
            finalize_response=accept_finalize,
        )

    client.relay = relay_without_finalize
    session.commit = fail_commit
    await server._stream(
        Request(),
        session,
        StreamDialect(),
        {},
        [UserMessage(content="test")],
    )

    assert isinstance(session.error, InterceptionError)
    assert str(session.error).startswith(
        "committing streamed model turn failed: RuntimeError:"
    )


@pytest.mark.asyncio
async def test_unexpected_aux_client_failure_is_stored_as_provider_error():
    server, session = _server_and_session(FailingClient())

    response = await server.handle_aux(Request(), Dialect(), "/v1/count_tokens")

    assert response.status == 502
    assert isinstance(session.error, ProviderError)
    assert _response_error(response) == "unexpected auxiliary parser failure"


@pytest.mark.asyncio
async def test_malformed_aux_json_is_stored_as_harness_error():
    server, session = _server_and_session(UnusedClient())

    response = await server.handle_aux(Request(b"{"), Dialect(), "/v1/count_tokens")

    assert response.status == 400
    assert isinstance(session.error, HarnessError)


@pytest.mark.asyncio
async def test_aux_transport_failure_is_stored_as_interception_error():
    server, session = _server_and_session(UnusedClient())

    response = await server.handle_aux(
        Request(fail_read=True), Dialect(), "/v1/count_tokens"
    )

    assert response.status == 502
    assert isinstance(session.error, InterceptionError)
    assert _response_error(response).startswith(
        "reading harness auxiliary request failed: RuntimeError:"
    )
    session.trace.capture_error(session.error)
    retry = RolloutRetryConfig(
        max_retries=2,
        include=[
            "ProviderError",
            "SandboxError",
            "TunnelError",
            "InterceptionError",
        ],
    )
    assert should_retry(session.trace, retry) is True


@pytest.mark.asyncio
async def test_oversized_aux_request_is_stored_as_harness_error_with_413():
    server, session = _server_and_session(UnusedClient())

    response = await server.handle_aux(
        OversizedRequest(), Dialect(), "/v1/count_tokens"
    )

    assert response.status == 413
    assert isinstance(session.error, HarnessError)
    assert _response_error(response).startswith(
        "harness auxiliary request body too large:"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [b"[]", b'"text"', b"null"])
async def test_non_object_aux_json_is_stored_as_harness_error(body: bytes):
    server, session = _server_and_session(UnusedClient())

    response = await server.handle_aux(Request(body), Dialect(), "/v1/count_tokens")

    assert response.status == 400
    assert isinstance(session.error, HarnessError)
    assert "request body must be a JSON object" in _response_error(response)


@pytest.mark.asyncio
async def test_aux_failure_preserves_pending_session_error():
    server, session = _server_and_session(FailingClient())
    pending = ProviderError("first model failure")
    session.error = pending

    response = await server.handle_aux(Request(), Dialect(), "/v1/count_tokens")

    assert response.status == 502
    assert session.error is pending


@pytest.mark.asyncio
async def test_handler_fallback_stores_interception_error(monkeypatch):
    server, session = _server_and_session(UnusedClient())

    async def fail_handler(request, dialect):
        raise RuntimeError("unexpected handler failure")

    monkeypatch.setattr(server, "handle_request", fail_handler)
    response = await server._handler_for(Dialect())(Request())

    assert response.status == 502
    assert isinstance(session.error, InterceptionError)
    assert _response_error(response).startswith(
        "interception request failed: RuntimeError:"
    )
