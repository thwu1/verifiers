from verifiers.v1 import graph
from verifiers.v1.clients import RolloutContext
from verifiers.v1.graph import MessageNode
from verifiers.v1.interception import RolloutLimits, RolloutSession
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace
from verifiers.v1.types import (
    AssistantMessage,
    Response,
    SamplingConfig,
    TurnTokens,
    UserMessage,
)


def _response(prompt: int, completion: int) -> Response:
    return Response(
        id="response",
        created=0,
        model="model",
        message=AssistantMessage(content="done"),
        finish_reason="stop",
        tokens=TurnTokens(
            prompt_ids=list(range(prompt)),
            completion_ids=list(range(completion)),
            completion_logprobs=[-0.1] * completion,
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
