from types import SimpleNamespace

import pytest
from verifiers.v1 import graph
from verifiers.v1.clients import RolloutContext
from verifiers.v1.clients.client import RelayReply
from verifiers.v1.errors import ProviderError
from verifiers.v1.graph import MessageNode
from verifiers.v1.interception import InterceptionServer, RolloutLimits, RolloutSession
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace
from verifiers.v1.types import (
    AssistantMessage,
    Response,
    SamplingConfig,
    TurnTokens,
    Usage,
    UserMessage,
)


def _response(
    prompt: int,
    completion: int,
    *,
    content: str = "done",
    usage: Usage | None = None,
) -> Response:
    return Response(
        id="response",
        created=0,
        model="model",
        message=AssistantMessage(content=content),
        finish_reason="stop",
        usage=usage,
        tokens=TurnTokens(
            prompt_ids=list(range(prompt)),
            completion_ids=list(range(completion)),
            completion_logprobs=[-0.1] * completion,
        ),
    )


def _usage_response(
    prompt: int,
    completion: int,
    *,
    content: str = "done",
    cached_input: int | None = None,
) -> Response:
    return Response(
        id="response",
        created=0,
        model="model",
        message=AssistantMessage(content=content),
        finish_reason="stop",
        usage=Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            cached_input_tokens=cached_input,
        ),
    )


def _session(trace: Trace, limits: RolloutLimits, max_tokens: int = 100):
    return RolloutSession(
        ctx=RolloutContext(
            client=None,  # type: ignore[arg-type]
            model="model",
            sampling=SamplingConfig(max_tokens=max_tokens),
        ),
        trace=trace,
        limits=limits,
    )


def test_generation_budget_is_clamped_to_smallest_remaining_cap():
    trace = Trace(
        task=Task(idx=0, prompt="test"),
        nodes=[
            MessageNode(
                message=AssistantMessage(content="prior"),
                sampled=True,
                token_ids=list(range(7)),
                mask=[False, False, False, True, True, True, True],
                logprobs=[-0.1] * 4,
            )
        ],
    )
    limits = RolloutLimits(max_output_tokens=6, max_total_tokens=10)

    sampling, stopped = limits.constrain_sampling(trace, SamplingConfig(max_tokens=100), prompt_prefix_tokens=7)

    assert stopped is None
    assert sampling.max_tokens == 2


def test_no_token_cap_leaves_sampling_unchanged():
    trace = Trace(task=Task(idx=0, prompt="test"))
    original = SamplingConfig(max_tokens=100)

    sampling, stopped = RolloutLimits().constrain_sampling(trace, original, prompt_prefix_tokens=0)

    assert stopped is None
    assert sampling is original


def test_empty_generation_budget_stops_before_request():
    trace = Trace(task=Task(idx=0, prompt="test"))
    session = _session(trace, RolloutLimits(max_total_tokens=8))

    _, stopped = session.sampling_for(prompt_prefix_tokens=8)

    assert stopped == "max_total_tokens"
    assert trace.stop_condition == "max_total_tokens"


def test_oversized_response_is_not_committed():
    trace = Trace(task=Task(idx=0, prompt="test"))
    session = _session(trace, RolloutLimits(max_total_tokens=10))
    turn = graph.prepare_turn(trace, [UserMessage(content="test")])

    stopped = session.commit(turn, _response(prompt=8, completion=3))

    assert stopped == "max_total_tokens"
    assert trace.stop_condition == "max_total_tokens"
    assert trace.nodes == []


def test_response_at_hard_total_cap_is_committed():
    trace = Trace(task=Task(idx=0, prompt="test"))
    session = _session(trace, RolloutLimits(max_total_tokens=10))
    turn = graph.prepare_turn(trace, [UserMessage(content="test")])

    stopped = session.commit(turn, _response(prompt=7, completion=3))

    assert stopped is None
    assert trace.branches[0].total_tokens == 10


def test_provider_usage_clamps_next_request_without_token_ids():
    trace = Trace(task=Task(idx=0, prompt="test"))
    session = _session(trace, RolloutLimits(max_output_tokens=6, max_total_tokens=12))
    prompt = [UserMessage(content="test")]
    session.commit(graph.prepare_turn(trace, prompt), _usage_response(prompt=7, completion=3))

    next_turn = graph.prepare_turn(
        trace,
        [*prompt, AssistantMessage(content="done"), UserMessage(content="next")],
    )
    sampling, stopped = session.sampling_for(prompt_prefix_tokens=next_turn.accounted_path_len)

    assert next_turn.path_len == 0
    assert next_turn.accounted_path_len == 10
    assert trace.prompt_len == 7
    assert trace.completion_len == 3
    assert trace.total_tokens == 10
    assert stopped is None
    assert sampling.max_tokens == 2


def test_provider_usage_stops_before_request_at_total_cap():
    trace = Trace(task=Task(idx=0, prompt="test"))
    session = _session(trace, RolloutLimits(max_total_tokens=10))
    prompt = [UserMessage(content="test")]
    session.commit(graph.prepare_turn(trace, prompt), _usage_response(prompt=7, completion=3))
    next_turn = graph.prepare_turn(
        trace,
        [*prompt, AssistantMessage(content="done"), UserMessage(content="next")],
    )

    _, stopped = session.sampling_for(prompt_prefix_tokens=next_turn.accounted_path_len)

    assert stopped == "max_total_tokens"
    assert trace.stop_condition == "max_total_tokens"


def test_exact_tokens_override_disagreeing_provider_usage():
    trace = Trace(task=Task(idx=0, prompt="test"))
    prompt = [UserMessage(content="test")]
    response = _response(
        prompt=7,
        completion=3,
        usage=Usage(prompt_tokens=80, completion_tokens=30),
    )
    graph.prepare_turn(trace, prompt).commit(response)

    next_turn = graph.prepare_turn(
        trace,
        [*prompt, AssistantMessage(content="done")],
    )

    assert trace.prompt_len == 7
    assert trace.completion_len == 3
    assert trace.total_tokens == 10
    assert next_turn.path_len == 10
    assert next_turn.accounted_path_len == 10


def test_mixed_exact_then_usage_only_uses_latest_usage():
    trace = Trace(task=Task(idx=0, prompt="test"))
    prompt = [UserMessage(content="test")]
    first = AssistantMessage(content="first")
    second_user = UserMessage(content="next")
    second = AssistantMessage(content="second")
    graph.prepare_turn(trace, prompt).commit(_response(prompt=7, completion=3, content="first"))
    graph.prepare_turn(trace, [*prompt, first, second_user]).commit(
        _usage_response(prompt=12, completion=2, content="second")
    )

    next_turn = graph.prepare_turn(
        trace,
        [*prompt, first, second_user, second],
    )

    assert trace.prompt_len == 12
    assert trace.completion_len == 5
    assert trace.total_tokens == 14
    assert next_turn.path_len == 10
    assert next_turn.accounted_path_len == 14


def test_mixed_usage_only_then_exact_uses_latest_exact_tokens():
    trace = Trace(task=Task(idx=0, prompt="test"))
    prompt = [UserMessage(content="test")]
    first = AssistantMessage(content="first")
    second_user = UserMessage(content="next")
    second = AssistantMessage(content="second")
    graph.prepare_turn(trace, prompt).commit(_usage_response(prompt=7, completion=3, content="first"))
    graph.prepare_turn(trace, [*prompt, first, second_user]).commit(
        _response(
            prompt=12,
            completion=2,
            content="second",
            usage=Usage(prompt_tokens=80, completion_tokens=30),
        )
    )

    next_turn = graph.prepare_turn(
        trace,
        [*prompt, first, second_user, second],
    )

    assert trace.prompt_len == 12
    assert trace.completion_len == 5
    assert trace.total_tokens == 14
    assert next_turn.path_len == 14
    assert next_turn.accounted_path_len == 14


def test_provider_usage_includes_cached_input_in_budget():
    trace = Trace(task=Task(idx=0, prompt="test"))
    session = _session(trace, RolloutLimits(max_total_tokens=12))
    prompt = [UserMessage(content="test")]
    session.commit(
        graph.prepare_turn(trace, prompt),
        _usage_response(prompt=4, completion=3, cached_input=3),
    )
    next_turn = graph.prepare_turn(
        trace,
        [*prompt, AssistantMessage(content="done")],
    )

    sampling, stopped = session.sampling_for(prompt_prefix_tokens=next_turn.accounted_path_len)

    assert trace.prompt_len == 7
    assert trace.total_tokens == 10
    assert next_turn.accounted_path_len == 10
    assert stopped is None
    assert sampling.max_tokens == 2


def test_usage_only_branches_keep_path_specific_accounting():
    trace = Trace(task=Task(idx=0, prompt="test"))
    limits = RolloutLimits(max_total_tokens=40)
    prompt = [UserMessage(content="test")]
    first = AssistantMessage(content="first")
    left_user = UserMessage(content="left")
    left = AssistantMessage(content="left answer")
    right_user = UserMessage(content="right")
    right = AssistantMessage(content="right answer")
    graph.prepare_turn(trace, prompt).commit(_usage_response(prompt=7, completion=3, content="first"))
    graph.prepare_turn(trace, [*prompt, first, left_user]).commit(
        _usage_response(prompt=12, completion=2, content="left answer")
    )
    graph.prepare_turn(trace, [*prompt, first, right_user]).commit(
        _usage_response(prompt=13, completion=4, content="right answer")
    )

    left_turn = graph.prepare_turn(
        trace,
        [*prompt, first, left_user, left, UserMessage(content="left again")],
    )
    right_turn = graph.prepare_turn(
        trace,
        [*prompt, first, right_user, right, UserMessage(content="right again")],
    )
    left_sampling, left_stopped = limits.constrain_sampling(
        trace,
        SamplingConfig(max_tokens=100),
        prompt_prefix_tokens=left_turn.accounted_path_len,
    )
    right_sampling, right_stopped = limits.constrain_sampling(
        trace,
        SamplingConfig(max_tokens=100),
        prompt_prefix_tokens=right_turn.accounted_path_len,
    )

    assert [(branch.prompt_len, branch.completion_len, branch.total_tokens) for branch in trace.branches] == [
        (12, 5, 14),
        (13, 7, 17),
    ]
    assert (trace.prompt_len, trace.completion_len, trace.total_tokens) == (
        25,
        12,
        31,
    )
    assert left_turn.accounted_path_len == 14
    assert right_turn.accounted_path_len == 17
    assert left_stopped is right_stopped is None
    assert left_sampling.max_tokens == 26
    assert right_sampling.max_tokens == 23


@pytest.mark.asyncio
async def test_post_stream_validation_error_is_preserved_on_session(monkeypatch):
    response = Response(
        id="stream",
        created=0,
        model="model",
        message=AssistantMessage(content="done"),
        finish_reason="stop",
    )

    class Parser:
        on_done = None

        def feed(self, chunk):
            pass

        def finish(self):
            return response

    class Dialect:
        def stream_parser(self):
            return Parser()

    async def chunks():
        yield b'data: {"chunk":true}\n\n'

    async def close():
        pass

    def reject_missing_tokens(parsed):
        raise ProviderError("missing requested token IDs")

    class Client:
        async def relay(self, *args, **kwargs):
            return RelayReply(
                content_type="text/event-stream",
                chunks=chunks(),
                close=close,
                finalize_response=reject_missing_tokens,
            )

    class StreamResponse:
        def __init__(self, **kwargs):
            self.content_type = ""
            self.writes = []
            self.eof = False

        async def prepare(self, request):
            pass

        async def write(self, chunk):
            self.writes.append(chunk)

        async def write_eof(self):
            self.eof = True

    monkeypatch.setattr("verifiers.v1.interception.server.web.StreamResponse", StreamResponse)
    trace = Trace(task=Task(idx=0, prompt="test"))
    session = RolloutSession(
        ctx=RolloutContext(
            client=Client(),  # type: ignore[arg-type]
            model="model",
            sampling=SamplingConfig(),
        ),
        trace=trace,
    )
    request = SimpleNamespace(headers={})

    streamed = await InterceptionServer()._stream(
        request,
        session,
        Dialect(),
        {"stream": True},
        [UserMessage(content="test")],
    )

    assert isinstance(session.error, ProviderError)
    assert trace.nodes == []
    assert streamed.eof is True
