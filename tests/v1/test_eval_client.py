import json
import math

import httpx
import pytest
import verifiers.v1 as vf
from verifiers.v1 import graph
from verifiers.v1.clients import EvalClientConfig, resolve_client
from verifiers.v1.clients.eval import EvalClient
from verifiers.v1.dialects import ChatDialect
from verifiers.v1.errors import ProviderError
from verifiers.v1.types import SamplingConfig


def _completion(*, token_ids=True, logprobs: list[float] | None = None) -> dict:
    choice = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"role": "assistant", "content": "ok"},
    }
    if token_ids:
        choice["provider_specific_fields"] = {"token_ids": [20]}
    if logprobs is not None:
        choice["logprobs"] = {
            "content": [
                {
                    "token": "ok",
                    "logprob": value,
                    "top_logprobs": [],
                }
                for value in logprobs
            ]
        }
    result = {
        "id": "chatcmpl-test",
        "created": 1,
        "model": "model",
        "object": "chat.completion",
        "choices": [choice],
    }
    if token_ids:
        result["prompt_token_ids"] = [10, 11]
    return result


async def _response(monkeypatch, payload: dict, sampling: dict):
    client = EvalClient("http://provider/v1", "key")

    async def request(*args, **kwargs):
        return httpx.Response(
            200,
            content=json.dumps(payload).encode(),
            request=httpx.Request("POST", "http://provider/v1/chat/completions"),
        )

    monkeypatch.setattr(client, "_request", request)
    try:
        return await client.get_response(
            ChatDialect(),
            {"messages": [{"role": "user", "content": "test"}]},
            "model",
            SamplingConfig.model_validate(sampling),
        )
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        _completion(token_ids=False, logprobs=[-0.1]),
        _completion(token_ids=True, logprobs=[]),
    ],
    ids=["missing-token-ids", "misaligned-logprobs"],
)
async def test_requested_exact_tokens_fail_closed(monkeypatch, payload):
    with pytest.raises(ProviderError, match="token IDs/logprobs") as error:
        await _response(
            monkeypatch,
            payload,
            {"return_token_ids": True, "logprobs": True},
        )

    assert error.value.status_code == 502


@pytest.mark.asyncio
@pytest.mark.parametrize("logprob", [math.nan, math.inf, -math.inf])
async def test_requested_logprobs_must_be_finite(monkeypatch, logprob):
    with pytest.raises(ProviderError, match="non-finite"):
        await _response(
            monkeypatch,
            _completion(logprobs=[logprob]),
            {"return_token_ids": True, "logprobs": True},
        )


@pytest.mark.asyncio
async def test_unrequested_token_extensions_remain_optional(monkeypatch):
    response = await _response(monkeypatch, _completion(token_ids=False), {})

    assert response.tokens is None


@pytest.mark.asyncio
async def test_requested_exact_tokens_are_accepted_when_aligned(monkeypatch):
    response = await _response(
        monkeypatch,
        _completion(logprobs=[-0.1]),
        {"return_token_ids": True, "logprobs": True},
    )

    assert response.tokens is not None
    assert response.tokens.prompt_ids == [10, 11]
    assert response.tokens.completion_ids == [20]
    assert response.tokens.completion_logprobs == [-0.1]


@pytest.mark.asyncio
async def test_requested_token_ids_are_accepted_without_logprobs(monkeypatch):
    response = await _response(
        monkeypatch,
        _completion(),
        {"return_token_ids": True, "logprobs": False},
    )

    assert response.tokens is not None
    assert response.tokens.prompt_ids == [10, 11]
    assert response.tokens.completion_ids == [20]
    assert response.tokens.completion_logprobs == []


@pytest.mark.asyncio
async def test_requested_logprobs_fail_when_absent(monkeypatch):
    with pytest.raises(ProviderError, match="missing, misaligned"):
        await _response(
            monkeypatch,
            _completion(),
            {"return_token_ids": True, "logprobs": True},
        )


@pytest.mark.asyncio
async def test_session_id_sets_generic_and_litellm_affinity_headers():
    client = EvalClient("http://provider/v1", "key")
    try:
        headers = client._headers(ChatDialect(), None, "trajectory-123")
    finally:
        await client.close()

    assert headers["X-Session-ID"] == "trajectory-123"
    assert headers["X-LiteLLM-Session-ID"] == "trajectory-123"


@pytest.mark.asyncio
async def test_outbound_body_denylist_is_final_and_top_level(monkeypatch):
    sent: list[dict] = []
    client = EvalClient(
        "http://provider/v1",
        "secret-key",
        headers={"X-Provider-Secret": "header-secret"},
        outbound_body_denylist=[
            "logprobs",
            "prompt_logprobs",
            "top_logprobs",
            "return_token_ids",
        ],
        capture_model_io=True,
    )

    async def request(url, body, headers, **kwargs):
        sent.append(body)
        return httpx.Response(
            200,
            content=json.dumps(_completion(token_ids=False)).encode(),
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(client, "_request", request)
    try:
        response = await client.get_response(
            ChatDialect(),
            {
                "messages": [{"role": "user", "content": "test"}],
                "logprobs": True,
                "prompt_logprobs": 1,
                "top_logprobs": 9,
                "return_token_ids": True,
                "metadata": {
                    "logprobs": True,
                    "prompt_logprobs": 7,
                    "return_token_ids": True,
                },
            },
            "model",
            # These overrides reintroduce fields from sampling before the final denylist.
            SamplingConfig.model_validate(
                {
                    "logprobs": True,
                    "prompt_logprobs": 2,
                    "top_logprobs": 4,
                    "return_token_ids": True,
                }
            ),
        )
    finally:
        await client.close()

    assert len(sent) == 1
    assert (
        not {
            "logprobs",
            "prompt_logprobs",
            "top_logprobs",
            "return_token_ids",
        }
        & sent[0].keys()
    )
    assert sent[0]["metadata"] == {
        "logprobs": True,
        "prompt_logprobs": 7,
        "return_token_ids": True,
    }
    # Validation follows what was actually sent, so removed token requirements do not reject a
    # response without provider token extensions.
    assert response.tokens is None
    assert response.pending_model_io is not None
    assert response.pending_model_io.request_body == sent[0]
    assert response.pending_model_io.response_body == _completion(token_ids=False)
    assert response.pending_model_io.response_kind == "exact_provider_json"

    trace = vf.Trace(task=vf.Task(idx=0, prompt="test"))
    graph.prepare_turn(trace, [vf.UserMessage(content="test")]).commit(response)
    model_io = next(node.model_io for node in trace.nodes if node.sampled)
    assert model_io is not None
    persisted = json.dumps(model_io.model_dump(mode="json"))
    assert model_io.provider_route == "/chat/completions"
    assert "secret-key" not in persisted
    assert "header-secret" not in persisted
    assert "http://provider" not in persisted


@pytest.mark.asyncio
async def test_default_relay_preserves_generic_provider_fields(monkeypatch):
    sent: dict = {}
    client = EvalClient("http://provider/v1", "key")

    async def request(url, body, headers, **kwargs):
        sent.update(body)
        return httpx.Response(
            200,
            content=json.dumps(_completion(token_ids=False)).encode(),
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(client, "_request", request)
    try:
        response = await client.get_response(
            ChatDialect(),
            {
                "messages": [{"role": "user", "content": "test"}],
                "provider_extension": {"enabled": True},
                "logprobs": False,
            },
            "model",
            SamplingConfig(),
        )
    finally:
        await client.close()

    assert sent["provider_extension"] == {"enabled": True}
    assert sent["logprobs"] is False
    assert response.pending_model_io is None


@pytest.mark.asyncio
@pytest.mark.parametrize("non_finite", [math.nan, math.inf, -math.inf])
async def test_capture_matches_serialized_json_semantics(monkeypatch, non_finite):
    wire_bodies: list[bytes] = []
    client = EvalClient("http://provider/v1", "key", capture_model_io=True)

    async def request(url, body, headers, **kwargs):
        wire_bodies.append(kwargs["encoded_body"])
        return httpx.Response(
            200,
            content=json.dumps(_completion(token_ids=False)).encode(),
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(client, "_request", request)
    try:
        response = await client.get_response(
            ChatDialect(),
            {
                "messages": [{"role": "user", "content": "test"}],
                "provider_extension": {"score": non_finite},
            },
            "model",
            SamplingConfig(),
        )
    finally:
        await client.close()

    assert response.pending_model_io is not None
    assert json.loads(wire_bodies[0]) == response.pending_model_io.request_body
    assert response.pending_model_io.request_body["provider_extension"] == {"score": None}


@pytest.mark.asyncio
@pytest.mark.parametrize("non_finite", [math.nan, math.inf, -math.inf])
async def test_capture_rejects_non_finite_provider_response(monkeypatch, non_finite):
    client = EvalClient("http://provider/v1", "key", capture_model_io=True)
    payload = {**_completion(token_ids=False), "provider_metrics": {"scores": [non_finite]}}

    async def request(url, body, headers, **kwargs):
        return httpx.Response(
            200,
            content=json.dumps(payload).encode(),
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(client, "_request", request)
    try:
        with pytest.raises(ProviderError, match="upstream response contained a non-finite JSON number") as error:
            await client.get_response(
                ChatDialect(),
                {"messages": [{"role": "user", "content": "test"}]},
                "model",
                SamplingConfig(),
            )
    finally:
        await client.close()

    assert error.value.status_code == 502


@pytest.mark.asyncio
async def test_stream_uses_same_final_denylist_and_captures_request(monkeypatch):
    sent: list[dict] = []
    client = EvalClient(
        "http://provider/v1",
        "key",
        outbound_body_denylist=["logprobs", "prompt_logprobs", "top_logprobs"],
        capture_model_io=True,
    )

    async def request(url, body, headers, **kwargs):
        sent.append(body)
        return httpx.Response(
            200,
            content=b"data: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(client, "_request", request)
    try:
        reply = await client.relay(
            ChatDialect(),
            {
                "stream": True,
                "messages": [{"role": "user", "content": "test"}],
                "logprobs": True,
                "prompt_logprobs": 1,
                "nested": {"top_logprobs": 5},
            },
            "model",
            SamplingConfig.model_validate({"logprobs": True, "prompt_logprobs": 2, "top_logprobs": 3}),
        )
        chunks = [chunk async for chunk in reply.chunks]
        await reply.close()
    finally:
        await client.close()

    assert chunks == [b"data: [DONE]\n\n"]
    assert not {"logprobs", "prompt_logprobs", "top_logprobs"} & sent[0].keys()
    assert sent[0]["nested"] == {"top_logprobs": 5}
    streamed = vf.Response(
        id="stream",
        created=1,
        model="model",
        message=vf.AssistantMessage(content="ok"),
        finish_reason="stop",
    )
    assert reply.finalize_response is not None
    reply.finalize_response(streamed)
    assert streamed.pending_model_io is not None
    assert streamed.pending_model_io.request_body == sent[0]
    assert streamed.pending_model_io.response_kind == "normalized_stream_response"
    assert streamed.pending_model_io.response_body == streamed.model_dump(mode="json")


@pytest.mark.asyncio
@pytest.mark.parametrize("non_finite", [math.nan, math.inf, -math.inf])
async def test_stream_capture_rejects_non_finite_normalized_response(monkeypatch, non_finite):
    client = EvalClient("http://provider/v1", "key", capture_model_io=True)

    async def request(url, body, headers, **kwargs):
        return httpx.Response(
            200,
            content=b"data: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(client, "_request", request)
    try:
        reply = await client.relay(
            ChatDialect(),
            {"stream": True, "messages": [{"role": "user", "content": "test"}]},
            "model",
            SamplingConfig(),
        )
        await reply.close()
    finally:
        await client.close()

    streamed = vf.Response(
        id="stream",
        created=1,
        model="model",
        message=vf.AssistantMessage(content="ok"),
        finish_reason="stop",
        usage=vf.Usage(prompt_tokens=1, completion_tokens=1, cost=non_finite),
    )
    assert reply.finalize_response is not None
    with pytest.raises(ProviderError, match="normalized streamed response contained a non-finite JSON number"):
        reply.finalize_response(streamed)
    assert streamed.pending_model_io is None


@pytest.mark.asyncio
async def test_stream_validates_token_requirements_from_final_body(monkeypatch):
    client = EvalClient("http://provider/v1", "key")

    async def request(url, body, headers, **kwargs):
        return httpx.Response(
            200,
            content=b"data: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(client, "_request", request)
    try:
        reply = await client.relay(
            ChatDialect(),
            {
                "stream": True,
                "messages": [{"role": "user", "content": "test"}],
                "return_token_ids": True,
            },
            "model",
            SamplingConfig(),
        )
        await reply.close()
    finally:
        await client.close()

    streamed = vf.Response(
        id="stream",
        created=1,
        model="model",
        message=vf.AssistantMessage(content="ok"),
        finish_reason="stop",
    )
    assert reply.finalize_response is not None
    with pytest.raises(ProviderError, match="token IDs/logprobs"):
        reply.finalize_response(streamed)


@pytest.mark.asyncio
async def test_eval_client_config_resolves_capture_options(monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_KEY", "secret")
    config = EvalClientConfig(
        base_url="http://provider/v1",
        api_key_var="TEST_PROVIDER_KEY",
        outbound_body_denylist=["logprobs"],
        capture_model_io=True,
    )
    client = resolve_client(config)
    try:
        assert isinstance(client, EvalClient)
        assert client.outbound_body_denylist == frozenset({"logprobs"})
        assert client.capture_model_io is True
    finally:
        await client.close()


def test_outbound_body_denylist_rejects_transport_field():
    with pytest.raises(ValueError, match="cannot remove 'stream'"):
        EvalClientConfig(outbound_body_denylist=["stream"])
