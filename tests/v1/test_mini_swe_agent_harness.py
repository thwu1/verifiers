import ast
import contextvars
import importlib.metadata
import os
import re
import secrets
import subprocess
from types import SimpleNamespace

import litellm
import pytest
from litellm.types.utils import Delta, ModelResponse, StreamingChoices
from verifiers.v1.harnesses.mini_swe_agent.harness import PROGRAM_SOURCE


def _bash_subprocess_shim():
    tree = ast.parse(PROGRAM_SOURCE)
    shim = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_BashSubprocess")
    namespace = {"subprocess": subprocess}
    exec(compile(ast.Module(body=[shim], type_ignores=[]), "program.py", "exec"), namespace)
    return namespace["_BashSubprocess"]


def _streaming_query_shim(original_query, stream_chunk_builder):
    tree = ast.parse(PROGRAM_SOURCE)
    query = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_streaming_litellm_query"
    )
    namespace = {
        "_LOGICAL_REQUEST_HEADER": "X-VF-Logical-Request-ID",
        "_LOGICAL_REQUEST_ID": contextvars.ContextVar("test_logical_request_id", default="a" * 32),
        "_ORIGINAL_LITELLM_QUERY": original_query,
        "litellm": SimpleNamespace(stream_chunk_builder=stream_chunk_builder),
    }
    exec(compile(ast.Module(body=[query], type_ignores=[]), "program.py", "exec"), namespace)
    return namespace["_streaming_litellm_query"]


def _logical_query_shim(original_query):
    tree = ast.parse(PROGRAM_SOURCE)
    query = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_logical_litellm_query"
    )
    request_id = contextvars.ContextVar("test_logical_request_id", default=None)
    namespace = {
        "_LOGICAL_REQUEST_ID": request_id,
        "_ORIGINAL_LITELLM_PUBLIC_QUERY": original_query,
        "secrets": secrets,
    }
    exec(compile(ast.Module(body=[query], type_ignores=[]), "program.py", "exec"), namespace)
    return namespace["_logical_litellm_query"], request_id


def test_mini_swe_agent_rebuilds_streaming_response() -> None:
    calls = []
    expected_response = object()

    def original_query(model, messages, **kwargs):
        calls.append((model, messages, kwargs))
        return iter(["first", "second"])

    def stream_chunk_builder(chunks, *, messages):
        calls.append((chunks, messages))
        return expected_response

    query = _streaming_query_shim(original_query, stream_chunk_builder)
    model = object()
    messages = [{"role": "user", "content": "test"}]

    assert query(model, messages, temperature=0) is expected_response
    assert calls == [
        (
            model,
            messages,
            {
                "temperature": 0,
                "stream": True,
                "extra_headers": {"X-VF-Logical-Request-ID": "a" * 32},
            },
        ),
        (["first", "second"], messages),
    ]


def test_mini_swe_agent_logical_request_id_is_stable_within_query_and_rotates() -> None:
    seen: list[str | None] = []

    def original_query(_model, _messages, **_kwargs):
        seen.extend([request_id.get(), request_id.get()])
        return "response"

    query, request_id = _logical_query_shim(original_query)
    assert query(object(), []) == "response"
    assert query(object(), []) == "response"

    assert len(seen) == 4
    assert seen[0] == seen[1]
    assert seen[2] == seen[3]
    assert seen[0] != seen[2]
    assert all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{32}", value) for value in seen)
    assert request_id.get() is None


def test_mini_swe_agent_streaming_query_replaces_private_identity_header() -> None:
    seen: list[dict] = []

    def original_query(_model, _messages, **kwargs):
        seen.append(kwargs)
        return iter(["chunk"])

    query = _streaming_query_shim(original_query, lambda _chunks, *, messages: messages)
    assert (
        query(
            object(),
            [],
            extra_headers={"x-vf-logical-request-id": "untrusted", "X-Keep": "yes"},
        )
        == []
    )
    assert seen == [
        {
            "stream": True,
            "extra_headers": {
                "X-Keep": "yes",
                "X-VF-Logical-Request-ID": "a" * 32,
            },
        }
    ]


def test_mini_swe_agent_model_retry_reuses_logical_request_id() -> None:
    tree = ast.parse(PROGRAM_SOURCE)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_logical_litellm_query", "_streaming_litellm_query"}
    ]
    seen: list[str] = []
    namespace = {
        "_LOGICAL_REQUEST_HEADER": "X-VF-Logical-Request-ID",
        "_LOGICAL_REQUEST_ID": contextvars.ContextVar("test_retry_request_id", default=None),
        "litellm": SimpleNamespace(stream_chunk_builder=lambda chunks, *, messages: list(chunks)),
        "secrets": secrets,
    }

    def transport(_model, _messages, **kwargs):
        seen.append(kwargs["extra_headers"]["X-VF-Logical-Request-ID"])
        if len(seen) == 1:
            raise ConnectionError("retry")
        return iter(["response"])

    def public_query(model, messages, **kwargs):
        for _ in range(2):
            try:
                return namespace["_streaming_litellm_query"](model, messages, **kwargs)
            except ConnectionError:
                continue
        raise AssertionError("retry did not return")

    namespace["_ORIGINAL_LITELLM_QUERY"] = transport
    namespace["_ORIGINAL_LITELLM_PUBLIC_QUERY"] = public_query
    exec(compile(ast.Module(body=functions, type_ignores=[]), "program.py", "exec"), namespace)

    assert namespace["_logical_litellm_query"](object(), []) == ["response"]
    assert len(seen) == 2
    assert seen[0] == seen[1]
    assert re.fullmatch(r"[0-9a-f]{32}", seen[0])


def test_mini_swe_agent_rejects_empty_stream() -> None:
    query = _streaming_query_shim(
        lambda _model, _messages, **_kwargs: iter(()),
        lambda _chunks, *, messages: None,
    )

    with pytest.raises(RuntimeError, match="no response"):
        query(object(), [])


def test_litellm_stream_rebuild_preserves_reasoning_and_tool_calls() -> None:
    chunks = [
        ModelResponse(
            id="response",
            model="model",
            stream=True,
            choices=[StreamingChoices(index=0, delta=Delta(role="assistant", reasoning_content="reason "))],
        ),
        ModelResponse(
            id="response",
            model="model",
            stream=True,
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        reasoning_content="retained",
                        tool_calls=[
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "bash", "arguments": '{"command":"echo ok"}'},
                            }
                        ],
                    ),
                )
            ],
        ),
        ModelResponse(
            id="response",
            model="model",
            stream=True,
            choices=[StreamingChoices(index=0, finish_reason="tool_calls", delta=Delta())],
        ),
    ]

    response = litellm.stream_chunk_builder(chunks, messages=[{"role": "user", "content": "test"}])

    assert response is not None
    assert response.choices[0].message.reasoning_content == "reason retained"
    assert response.choices[0].message.tool_calls[0].function.arguments == '{"command":"echo ok"}'
    assert response.choices[0].finish_reason == "tool_calls"


def test_mini_swe_agent_246_local_run_and_native_submit(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("minisweagent")
    if importlib.metadata.version("mini-swe-agent") != "2.4.6":
        pytest.skip("compatibility contract is specific to mini-swe-agent 2.4.6")

    from minisweagent.environments import local
    from minisweagent.exceptions import Submitted

    monkeypatch.setattr(local, "subprocess", _bash_subprocess_shim())
    result = local._run(
        "printf 'bash:%s' \"$BASH_VERSION\"",
        str(tmp_path),
        dict(os.environ),
        5,
    )

    assert result.returncode == 0
    assert result.stdout.startswith("bash:")
    with pytest.raises(Submitted):
        local.LocalEnvironment(cwd=str(tmp_path), timeout=5).execute(
            {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}
        )
