import base64

import numpy as np
import pytest
import verifiers.v1 as vf
from verifiers.v1 import graph
from verifiers.v1.model_io import DeltaModelRequest, FullModelRequest, json_sha256
from verifiers.v1.types import PendingModelIO, TurnTokens


def _response(message: vf.AssistantMessage) -> vf.Response:
    return vf.Response(
        id="",
        created=0,
        model="test",
        message=message,
        finish_reason="stop",
    )


def _captured_response(
    message: vf.AssistantMessage,
    request_body: dict,
    response_body: dict,
    *,
    response_kind="exact_provider_json",
) -> vf.Response:
    return vf.Response(
        id=response_body.get("id", "response"),
        created=0,
        model="test",
        message=message,
        finish_reason="stop",
        pending_model_io=PendingModelIO(
            provider_route="/chat/completions",
            request_body=request_body,
            response_body=response_body,
            response_kind=response_kind,
        ),
    )


def test_model_io_two_turn_delta_reconstruction_and_round_trip():
    trace = vf.Trace(task=vf.Task(idx=0, prompt="use a tool"))
    user = vf.UserMessage(content="use a tool")
    call = vf.ToolCall(id="call_1", name="shell", arguments='{"cmd":"pwd"}')
    assistant = vf.AssistantMessage(reasoning_content="I should inspect the directory.", tool_calls=[call])
    tool = vf.ToolMessage(tool_call_id="call_1", name="shell", content="/workspace")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "shell",
                "description": "Run a command",
                "parameters": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                },
            },
        }
    ]
    first_request = {
        "model": "kimi",
        "messages": [{"role": "user", "content": "use a tool"}],
        "tools": tools,
        "temperature": 0.6,
        "seed": 123,
    }
    first_response = {
        "id": "one",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "reasoning_content": "I should inspect the directory.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "shell", "arguments": '{"cmd":"pwd"}'},
                        }
                    ],
                }
            }
        ],
    }
    graph.prepare_turn(trace, [user]).commit(_captured_response(assistant, first_request, first_response))

    second_request = {
        **{key: value for key, value in first_request.items() if key != "seed"},
        "temperature": 0.7,
        "messages": [
            *first_request["messages"],
            first_response["choices"][0]["message"],
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "shell",
                "content": "/workspace",
            },
        ],
    }
    second_response = {
        "id": "two",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "reasoning_content": "The directory is correct.",
                    "content": "done",
                }
            }
        ],
    }
    graph.prepare_turn(trace, [user, assistant, tool]).commit(
        _captured_response(
            vf.AssistantMessage(content="done", reasoning_content="The directory is correct."),
            second_request,
            second_response,
        )
    )

    sampled = [i for i, node in enumerate(trace.nodes) if node.sampled]
    first_id, second_id = sampled
    assert isinstance(trace.nodes[first_id].model_io.request, FullModelRequest)
    second_capture = trace.nodes[second_id].model_io
    assert second_capture is not None
    assert second_capture.provider_route == "/chat/completions"
    assert isinstance(second_capture.request, DeltaModelRequest)
    assert second_capture.request.base_node == first_id
    assert second_capture.request.set_fields == {"temperature": 0.7}
    assert second_capture.request.remove_fields == ["seed"]
    assert second_capture.request.append_fields["messages"] == second_request["messages"][1:]
    assert second_capture.response.kind == "exact_provider_json"
    assert second_capture.response.body == second_response
    assert vf.validate_model_io(trace.nodes, second_id) == second_request

    restored = vf.WireTrace.model_validate(trace.model_dump())
    assert vf.validate_model_io(restored.nodes, second_id) == second_request
    assert restored.nodes[second_id].model_io == trace.nodes[second_id].model_io


def test_model_io_rewritten_branch_uses_full_fallback():
    trace = vf.Trace(task=vf.Task(idx=0, prompt="original"))
    original = vf.UserMessage(content="original")
    graph.prepare_turn(trace, [original]).commit(
        _captured_response(
            vf.AssistantMessage(content="first"),
            {"model": "m", "messages": [{"role": "user", "content": "original"}]},
            {"id": "one", "output": "first"},
        )
    )

    # This request forks from the root instead of continuing through the sampled assistant. It
    # must not delta against the linear cache's unrelated latest request.
    rewritten = vf.UserMessage(content="rewritten")
    rewritten_request = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "original"},
            {"role": "user", "content": "rewritten"},
        ],
        "tools": [{"type": "function", "function": {"name": "new"}}],
    }
    graph.prepare_turn(trace, [original, rewritten]).commit(
        _captured_response(
            vf.AssistantMessage(content="branch"),
            rewritten_request,
            {"id": "branch", "output": "branch"},
        )
    )

    branch_id = max(i for i, node in enumerate(trace.nodes) if node.sampled)
    capture = trace.nodes[branch_id].model_io
    assert capture is not None
    assert isinstance(capture.request, FullModelRequest)
    assert vf.reconstruct_model_request(trace.nodes, branch_id) == rewritten_request
    assert trace.num_branches == 2


def test_model_io_branch_deltas_against_its_ancestor_not_latest_cache():
    trace = vf.Trace(task=vf.Task(idx=0, prompt="root"))
    user = vf.UserMessage(content="root")
    first = vf.AssistantMessage(content="first")
    first_request = {
        "model": "m",
        "messages": [{"role": "user", "content": "root"}],
        "tools": [{"type": "function", "function": {"name": "large-tool-name"}}],
    }
    graph.prepare_turn(trace, [user]).commit(
        _captured_response(first, first_request, {"id": "first", "output": "first"})
    )
    first_id = next(i for i, node in enumerate(trace.nodes) if node.sampled)

    left_user = vf.UserMessage(content="left")
    left_request = {
        **first_request,
        "messages": [
            *first_request["messages"],
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "left"},
        ],
    }
    graph.prepare_turn(trace, [user, first, left_user]).commit(
        _captured_response(
            vf.AssistantMessage(content="left answer"),
            left_request,
            {"id": "left", "output": "left answer"},
        )
    )

    right_user = vf.UserMessage(content="right")
    right_request = {
        **first_request,
        "messages": [
            *first_request["messages"],
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "right"},
        ],
    }
    graph.prepare_turn(trace, [user, first, right_user]).commit(
        _captured_response(
            vf.AssistantMessage(content="right answer"),
            right_request,
            {"id": "right", "output": "right answer"},
        )
    )

    right_id = max(i for i, node in enumerate(trace.nodes) if node.sampled)
    right_capture = trace.nodes[right_id].model_io
    assert right_capture is not None
    assert isinstance(right_capture.request, DeltaModelRequest)
    assert right_capture.request.base_node == first_id
    assert vf.validate_model_io(trace.nodes, right_id) == right_request


def test_model_io_detects_request_and_response_hash_corruption():
    trace = vf.Trace(task=vf.Task(idx=0, prompt="q"))
    graph.prepare_turn(trace, [vf.UserMessage(content="q")]).commit(
        _captured_response(
            vf.AssistantMessage(content="a"),
            {"model": "m", "messages": [{"role": "user", "content": "q"}]},
            {"id": "one", "output": "a"},
        )
    )
    node_id = next(i for i, node in enumerate(trace.nodes) if node.sampled)
    model_io = trace.nodes[node_id].model_io
    assert model_io is not None

    model_io.request.sha256 = "0" * 64
    with pytest.raises(ValueError, match="request hash mismatch"):
        vf.validate_model_io(trace.nodes, node_id)

    model_io.request.sha256 = json_sha256(vf.reconstruct_model_request(trace.nodes, node_id, validate_hash=False))
    model_io.response.body["output"] = "tampered"
    with pytest.raises(ValueError, match="response hash mismatch"):
        vf.validate_model_io(trace.nodes, node_id)


def test_model_io_normalized_stream_response_and_old_trace_compatibility():
    trace = vf.Trace(task=vf.Task(idx=0, prompt="q"))
    normalized = {
        "id": "stream",
        "message": {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "thinking",
        },
        "finish_reason": "stop",
    }
    graph.prepare_turn(trace, [vf.UserMessage(content="q")]).commit(
        _captured_response(
            vf.AssistantMessage(content="answer", reasoning_content="thinking"),
            {
                "model": "m",
                "stream": True,
                "messages": [{"role": "user", "content": "q"}],
            },
            normalized,
            response_kind="normalized_stream_response",
        )
    )
    node = next(node for node in trace.nodes if node.sampled)
    assert node.model_io is not None
    assert node.model_io.response.kind == "normalized_stream_response"
    assert node.model_io.response.body == normalized

    old_dump = trace.model_dump()
    for raw_node in old_dump["nodes"]:
        raw_node.pop("model_io", None)
    restored_old = vf.WireTrace.model_validate(old_dump)
    assert all(node.model_io is None for node in restored_old.nodes)


def _routed_payload(num_tokens: int, start: int, base: int, layers: int = 2, top_k: int = 1):
    """A fake `generate` router-replay sidecar (uint8 `[num_tokens, layers, top_k]`, base64)."""
    arr = np.arange(num_tokens * layers * top_k).reshape(num_tokens, layers, top_k).astype(np.uint8) + base
    return {
        "data": base64.b64encode(arr.tobytes()).decode(),
        "shape": list(arr.shape),
        "start": start,
    }


def test_routed_experts_attributed_and_aligned_across_turns():
    """Each turn's full routing (start=0) is attributed to the nodes it created; the new turn's
    nodes get this turn's slice and reused nodes keep theirs, so `Branch.routed_experts`
    concatenates back to a `[tokens, layers, top_k]` array aligned 1:1 with `branch.token_ids` —
    and survives the base64 wire round-trip."""
    trace = vf.Trace(task=vf.Task(idx=0, prompt="x"))
    user = vf.UserMessage(content="u1")
    graph.prepare_turn(trace, [user]).commit(
        vf.Response(
            id="a",
            created=0,
            model="t",
            message=vf.AssistantMessage(content="a1"),
            finish_reason="stop",
            tokens=TurnTokens(
                prompt_ids=[10, 11, 12],
                completion_ids=[20, 21],
                message_spans=[(0, 2)],
                routed_experts=_routed_payload(5, 0, 0),
            ),
        )
    )
    graph.prepare_turn(
        trace,
        [user, vf.AssistantMessage(content="a1"), vf.UserMessage(content="u2")],
    ).commit(
        vf.Response(
            id="b",
            created=0,
            model="t",
            message=vf.AssistantMessage(content="a2"),
            finish_reason="stop",
            tokens=TurnTokens(
                prompt_ids=[10, 11, 12, 20, 21, 30, 31],
                completion_ids=[40, 41],
                message_spans=[(0, 2), None, (5, 7)],
                routed_experts=_routed_payload(9, 0, 100),
            ),
        )
    )
    branch = trace.branches[-1]
    re = branch.routed_experts
    assert re is not None
    assert re.shape[0] == len(branch.token_ids)

    restored = type(trace).model_validate(trace.model_dump())
    re2 = restored.branches[-1].routed_experts
    assert re2 is not None and re2.shape == re.shape and bool((re2 == re).all())
    assert all(node.routed_experts is None or node.routed_experts.flags.owndata for node in trace.nodes)


def test_routed_experts_none_when_absent():
    """No routing captured (engine ran without `enable_return_routed_experts`) -> the branch
    reports None and the trainer simply skips replay."""
    trace = vf.Trace(task=vf.Task(idx=0, prompt="x"))
    graph.prepare_turn(trace, [vf.UserMessage(content="u1")]).commit(
        vf.Response(
            id="a",
            created=0,
            model="t",
            message=vf.AssistantMessage(content="a1"),
            finish_reason="stop",
            tokens=TurnTokens(prompt_ids=[1, 2], completion_ids=[3], message_spans=[(0, 2)]),
        )
    )
    assert trace.branches[-1].routed_experts is None


def test_tool_call_hash_matches_v0_content_and_arguments_normalization():
    left = vf.AssistantMessage(
        content=None,
        tool_calls=[vf.ToolCall(id="call_0", name="lookup", arguments='{"b": 2, "a": 1}')],
    )
    right = vf.AssistantMessage(
        content="",
        tool_calls=[vf.ToolCall(id="call_0", name="lookup", arguments='{"a":1,"b":2}')],
    )

    assert graph.message_hash(left) == graph.message_hash(right)


def test_reasoning_content_participates_in_graph_prefix_matching():
    task = vf.Task(idx=0, prompt="use a tool")
    trace = vf.Trace(task=task)
    user = vf.UserMessage(content="use a tool")
    call = vf.ToolCall(id="call_0", name="lookup", arguments="{}")

    graph.prepare_turn(trace, [user]).commit(
        _response(
            vf.AssistantMessage(
                content=None,
                reasoning_content="plan A",
                tool_calls=[call],
            )
        )
    )
    graph.prepare_turn(
        trace,
        [
            user,
            vf.AssistantMessage(
                content=None,
                reasoning_content="plan B",
                tool_calls=[call],
            ),
            vf.ToolMessage(content="result", tool_call_id="call_0"),
        ],
    ).commit(_response(vf.AssistantMessage(content="done")))

    tool_call_nodes = [
        node for node in trace.nodes if isinstance(node.message, vf.AssistantMessage) and node.message.tool_calls
    ]
    assert len(tool_call_nodes) == 2


def test_renderer_level_break_forks_by_token_id():
    """Two turns with the *same* message sequence and identical message hashes, but the prior
    assistant turn is retokenized (renderer drift — e.g. a chat template dropping a `<think>`
    block on re-render): the stored prefix tokens no longer match this turn's `prompt_ids`.
    Message-hash dedup alone would silently reuse the stale prefix; token-identity prefix reuse
    must fork at the diverging node. Each branch's leaf→root token concatenation still equals
    its own `prompt_ids + completion_ids`."""
    user = vf.UserMessage(content="u1")
    a1 = vf.AssistantMessage(content="a1")
    u2 = vf.UserMessage(content="u2")

    def first_turn(trace):
        graph.prepare_turn(trace, [user]).commit(
            vf.Response(
                id="a",
                created=0,
                model="t",
                message=a1,
                finish_reason="stop",
                tokens=TurnTokens(prompt_ids=[1, 2, 3], completion_ids=[4, 5], message_spans=[(0, 2)]),
            )
        )

    def second_turn(trace, prompt_ids):
        graph.prepare_turn(trace, [user, a1, u2]).commit(
            vf.Response(
                id="b",
                created=0,
                model="t",
                message=vf.AssistantMessage(content="a2"),
                finish_reason="stop",
                tokens=TurnTokens(
                    prompt_ids=prompt_ids,
                    completion_ids=[8],
                    message_spans=[(0, 2), (2, 5), (5, 7)],
                ),
            )
        )

    # Control: the prior turn re-renders to the same tokens -> stays one linear branch.
    linear = vf.Trace(task=vf.Task(idx=0, prompt="x"))
    first_turn(linear)
    second_turn(linear, [1, 2, 3, 4, 5, 6, 7])
    assert linear.num_branches == 1
    assert linear.branches[0].token_ids == [1, 2, 3, 4, 5, 6, 7, 8]

    # Break: the assistant turn retokenizes (4 -> 99), so prompt_ids diverge at that node.
    broken = vf.Trace(task=vf.Task(idx=0, prompt="x"))
    first_turn(broken)
    second_turn(broken, [1, 2, 3, 99, 5, 6, 7])
    assert broken.num_branches == 2
    assert sorted(b.token_ids for b in broken.branches) == [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 99, 5, 6, 7, 8],
    ]


def test_prompt_supplied_assistant_messages_are_not_sampled_turns():
    task = vf.Task(idx=0, prompt="few-shot")
    trace = vf.Trace(task=task)
    fabricated = vf.AssistantMessage(
        content=None,
        tool_calls=[vf.ToolCall(id="call_0", name="lookup", arguments="{}")],
    )
    response = vf.AssistantMessage(content="real answer")

    graph.prepare_turn(
        trace,
        [
            vf.UserMessage(content="question"),
            fabricated,
            vf.ToolMessage(content="fabricated result", tool_call_id="call_0"),
        ],
    ).commit(_response(response))

    assert [n.sampled for n in trace.nodes] == [False, False, False, True]
    assert trace.num_turns == 1
    assert trace.assistant_messages == [response]
