import json
import math

import httpx
import pytest
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
