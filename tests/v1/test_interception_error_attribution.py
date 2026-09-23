import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from aiohttp import web
from verifiers.v1 import graph
from verifiers.v1.clients import RolloutContext
from verifiers.v1.clients.client import RelayReply
from verifiers.v1.errors import (
    HarnessError,
    InterceptionError,
    OverlongPromptError,
    ProviderError,
    UserError,
)
from verifiers.v1.interception import InterceptionServer, RolloutSession
from verifiers.v1.interception.pool import InterceptionPool, PooledServer
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


def _server_and_session(client, *, prompt: str | None = "test") -> tuple[InterceptionServer, RolloutSession]:
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
    server._requests["secret"] = set()
    return server, session


def _response_error(response) -> str:
    return json.loads(response.body)["error"]["message"]


@pytest.mark.asyncio
async def test_invalid_dialect_request_is_stored_as_non_retryable_harness_error():
    server, session = _server_and_session(UnusedClient())

    response = await server.handle_request(Request(), FailingParseDialect())

    assert response.status == 400
    assert isinstance(session.error, HarnessError)
    assert _response_error(response).startswith("invalid harness model request: ValueError:")
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
    assert _response_error(response).startswith("invalid harness model request: JSONDecodeError:")


@pytest.mark.asyncio
async def test_request_transport_failure_is_stored_as_interception_error():
    server, session = _server_and_session(UnusedClient())

    response = await server.handle_request(Request(fail_read=True), Dialect())

    assert response.status == 502
    assert isinstance(session.error, InterceptionError)
    assert _response_error(response).startswith("reading harness request failed: RuntimeError:")


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
    assert _response_error(response).startswith("preparing model turn failed: RuntimeError:")


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
    assert _response_error(response).startswith("committing model turn failed: RuntimeError:")


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
async def test_closed_session_ignores_late_stream_completion(monkeypatch):
    started = asyncio.Event()

    class CancellationResistantClient:
        async def relay(self, *args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                pass

            async def chunks():
                yield b'data: {"chunk":true}\n\n'

            async def close():
                pass

            return RelayReply(
                content_type="text/event-stream",
                chunks=chunks(),
                close=close,
            )

    class StreamingDialect(StreamDialect):
        @staticmethod
        def streaming(body: dict) -> bool:
            return True

    monkeypatch.setattr(
        "verifiers.v1.interception.server.web.StreamResponse",
        StreamResponse,
    )
    server, session = _server_and_session(CancellationResistantClient())
    request = asyncio.create_task(server._handler_for(StreamingDialect())(Request()))
    await asyncio.wait_for(started.wait(), timeout=1)

    await server.unregister("secret")
    await asyncio.wait_for(request, timeout=1)

    assert session.closed is True
    assert session.error is None
    assert session.trace.stop_condition is None
    assert session.trace.num_turns == 0


@pytest.mark.asyncio
async def test_stream_finalize_failure_is_stored_as_provider_error(monkeypatch):
    monkeypatch.setattr("verifiers.v1.interception.server.web.StreamResponse", StreamResponse)
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
    monkeypatch.setattr("verifiers.v1.interception.server.web.StreamResponse", StreamResponse)
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
    monkeypatch.setattr("verifiers.v1.interception.server.web.StreamResponse", StreamResponse)
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
    monkeypatch.setattr("verifiers.v1.interception.server.web.StreamResponse", StreamResponse)
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
    assert str(session.error).startswith("committing streamed model turn failed: RuntimeError:")


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

    response = await server.handle_aux(Request(fail_read=True), Dialect(), "/v1/count_tokens")

    assert response.status == 502
    assert isinstance(session.error, InterceptionError)
    assert _response_error(response).startswith("reading harness auxiliary request failed: RuntimeError:")
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

    response = await server.handle_aux(OversizedRequest(), Dialect(), "/v1/count_tokens")

    assert response.status == 413
    assert isinstance(session.error, HarnessError)
    assert _response_error(response).startswith("harness auxiliary request body too large:")


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
    assert _response_error(response).startswith("interception request failed: RuntimeError:")


@pytest.mark.asyncio
async def test_unregister_cancels_and_drains_admitted_request_without_late_commit():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class CancellationResistantClient:
        async def get_response(self, *args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                return Response(
                    id="late-response",
                    created=0,
                    model="model",
                    message=AssistantMessage(content="late"),
                    finish_reason="stop",
                )

    server, session = _server_and_session(CancellationResistantClient())
    request = asyncio.create_task(server._handler_for(Dialect())(Request()))
    await asyncio.wait_for(started.wait(), timeout=1)

    await server.unregister("secret")
    response = await asyncio.wait_for(request, timeout=1)

    assert cancelled.is_set()
    assert response.status == 502
    assert session.closed is True
    assert session.trace.num_turns == 0
    assert "secret" not in server.sessions
    assert "secret" not in server._requests


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "late_error",
    [OverlongPromptError("late overlong"), ProviderError("late provider failure")],
)
async def test_closed_session_ignores_late_nonstream_failure(late_error):
    started = asyncio.Event()

    class CancellationResistantClient:
        async def get_response(self, *args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise late_error

    server, session = _server_and_session(CancellationResistantClient())
    request = asyncio.create_task(server._handler_for(Dialect())(Request()))
    await asyncio.wait_for(started.wait(), timeout=1)

    await server.unregister("secret")
    await asyncio.wait_for(request, timeout=1)

    assert session.closed is True
    assert session.error is None
    assert session.trace.stop_condition is None
    assert session.trace.num_turns == 0


@pytest.mark.asyncio
async def test_closed_session_ignores_late_aux_failure():
    started = asyncio.Event()

    class CancellationResistantClient:
        async def relay_aux(self, *args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise ProviderError("late auxiliary failure")

    server, session = _server_and_session(CancellationResistantClient())
    request = asyncio.create_task(server._aux_handler_for(Dialect(), "/v1/count_tokens")(Request()))
    await asyncio.wait_for(started.wait(), timeout=1)

    await server.unregister("secret")
    await asyncio.wait_for(request, timeout=1)

    assert session.closed is True
    assert session.error is None
    assert session.trace.stop_condition is None


@pytest.mark.asyncio
async def test_closed_session_ignores_late_stop_result():
    started = asyncio.Event()

    async def cancellation_resistant_stop(trace) -> bool:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return True

    server, session = _server_and_session(UnusedClient())
    session.stops = [cancellation_resistant_stop]
    request = asyncio.create_task(server._handler_for(Dialect())(Request()))
    await asyncio.wait_for(started.wait(), timeout=1)

    await server.unregister("secret")
    await asyncio.wait_for(request, timeout=1)

    assert session.closed is True
    assert session.error is None
    assert session.trace.stop_condition is None


@pytest.mark.asyncio
async def test_pool_teardown_preserves_external_cancellation_when_drain_fails():
    entered = asyncio.Event()
    unregister_called = asyncio.Event()

    class Server:
        port = 1

        def register(self, session) -> str:
            return "secret"

        async def unregister(self, secret: str) -> None:
            unregister_called.set()
            raise InterceptionError("request drain failed")

    entry = PooledServer(Server(), "http://127.0.0.1:1", load=0)
    pool = object.__new__(InterceptionPool)
    pool._lock = asyncio.Lock()

    async def get_entry():
        return entry

    pool._entry = get_entry

    async def use_slot() -> None:
        async with pool.acquire(SimpleNamespace(), SimpleNamespace()):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(use_slot())
    await asyncio.wait_for(entered.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert unregister_called.is_set()
    assert entry.load == 0
    assert entry.healthy is False


@pytest.mark.asyncio
async def test_pool_excludes_entry_while_failed_unregister_is_draining(monkeypatch):
    unregister_started = asyncio.Event()
    fail_unregister = asyncio.Event()
    first_entered = asyncio.Event()
    leave_first = asyncio.Event()

    class FailingServer:
        port = 1

        def register(self, session) -> str:
            return "first-secret"

        async def unregister(self, secret: str) -> None:
            unregister_started.set()
            await fail_unregister.wait()
            raise InterceptionError("request drain failed")

    class FreshServer:
        port = 2

        def register(self, session) -> str:
            return "fresh-secret"

        async def unregister(self, secret: str) -> None:
            pass

    class Stack:
        async def enter_async_context(self, value):
            return value

    first_server = FailingServer()
    fresh_server = FreshServer()
    entry = PooledServer(first_server, "http://127.0.0.1:1")
    pool = object.__new__(InterceptionPool)
    pool.runtime_type = "test"
    pool.is_local = True
    pool.instance_host_endpoint = True
    pool.multiplex = 2
    pool._servers = [entry]
    pool._lock = asyncio.Lock()
    pool._stack = Stack()
    monkeypatch.setattr(
        "verifiers.v1.interception.pool.InterceptionServer",
        lambda: fresh_server,
    )

    class Runtime:
        @asynccontextmanager
        async def host_endpoint(self, port: int):
            yield f"http://127.0.0.1:{port}"

        @asynccontextmanager
        async def interception_endpoint(self, port: int, secret: str):
            del secret
            async with self.host_endpoint(port) as endpoint:
                yield endpoint

    runtime = Runtime()

    async def first_slot() -> None:
        async with pool.acquire(SimpleNamespace(), runtime):
            first_entered.set()
            await leave_first.wait()

    first = asyncio.create_task(first_slot())
    await asyncio.wait_for(first_entered.wait(), timeout=1)
    leave_first.set()
    await asyncio.wait_for(unregister_started.wait(), timeout=1)

    assert entry.draining == 1
    async with pool.acquire(SimpleNamespace(), runtime) as acquired:
        assert acquired[1] == "fresh-secret"
        assert len(pool._servers) == 2

    fail_unregister.set()
    with pytest.raises(InterceptionError, match="request drain failed"):
        await first
    assert entry.healthy is False
    assert entry.draining == 0
    assert entry.load == 0
