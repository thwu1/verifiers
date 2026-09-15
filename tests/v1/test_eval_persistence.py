import json
import math
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from verifiers.v1.cli.eval import resume, runner
from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace


def _result_row(idx: int, *, valid_tokens: bool) -> dict:
    return {
        "task": {"idx": idx},
        "errors": [],
        "nodes": [
            {
                "sampled": True,
                "token_ids": [10, 11] if valid_tokens else [],
                "mask": [False, True] if valid_tokens else [],
                "logprobs": [-0.25] if valid_tokens else [],
            }
        ],
    }


def test_resume_retries_token_invalid_success_only_when_requested(tmp_path):
    (tmp_path / "results.jsonl").write_text(json.dumps(_result_row(7, valid_tokens=False)) + "\n")

    keep, owed = resume.plan(tmp_path, [7], 1, group=False)
    assert keep == [0]
    assert owed == {}

    keep, owed = resume.plan(tmp_path, [7], 1, group=False, require_exact_tokens=True)
    assert keep == []
    assert owed == {7: 1}


def test_resume_keeps_valid_exact_token_row(tmp_path):
    (tmp_path / "results.jsonl").write_text(json.dumps(_result_row(7, valid_tokens=True)) + "\n")

    keep, owed = resume.plan(tmp_path, [7], 1, group=False, require_exact_tokens=True)
    assert keep == [0]
    assert owed == {}


def test_resume_ignores_only_truncated_final_row(tmp_path):
    complete = json.dumps(_result_row(7, valid_tokens=True)).encode()
    (tmp_path / "results.jsonl").write_bytes(complete + b'\n{"task":{"idx":8},"nodes":[')

    keep, owed = resume.plan(tmp_path, [7, 8], 1, group=False)

    assert keep == [0]
    assert owed == {8: 1}


@pytest.mark.parametrize(
    "contents",
    [
        b'{"task":{"idx":7}\n',
        b'{"task":{"idx":7}\n' + json.dumps(_result_row(8, valid_tokens=True)).encode() + b"\n",
    ],
    ids=["newline-terminated-final", "interior"],
)
def test_resume_raises_for_malformed_complete_rows(tmp_path, contents):
    (tmp_path / "results.jsonl").write_bytes(contents)

    with pytest.raises(ValueError):
        resume.plan(tmp_path, [7, 8], 1, group=False)


@pytest.mark.parametrize("logprob", [math.nan, math.inf, -math.inf])
def test_resume_retries_non_finite_logprobs(tmp_path, logprob):
    row = _result_row(7, valid_tokens=True)
    row["nodes"][0]["logprobs"] = [logprob]
    (tmp_path / "results.jsonl").write_text(json.dumps(row) + "\n")

    keep, owed = resume.plan(tmp_path, [7], 1, group=False, require_exact_tokens=True)

    assert keep == []
    assert owed == {7: 1}


def test_resume_token_validation_is_enabled_by_provider_request():
    config = EvalConfig(
        rich=False,
        sampling={"return_token_ids": True, "logprobs": True},
    )

    assert resume.exact_tokens_requested(config)


@pytest.mark.asyncio
@pytest.mark.parametrize("retain_traces", [True, False])
async def test_run_eval_can_release_durably_persisted_traces(monkeypatch, tmp_path, retain_traces):
    trace = Trace(task=Task(idx=0, prompt="test"))
    rollout = SimpleNamespace(trace=None)

    class FakeEpisode:
        rollouts = [rollout]

        async def run(self, semaphore, on_complete, retain_traces=True):
            rollout.trace = trace
            await on_complete(trace)
            if not retain_traces:
                rollout.trace = None
            return [trace] if retain_traces else []

    class FakeTaskset:
        def load_tasks(self):
            return [trace.task]

    class FakeEnv:
        taskset = FakeTaskset()

        @asynccontextmanager
        async def serving(self, tasks):
            yield

        def episode(self, task, ctx, n):
            return FakeEpisode()

    class FakeClient:
        async def close(self):
            pass

    monkeypatch.setattr(runner, "resolve_client", lambda config: FakeClient())
    config = EvalConfig(
        rich=False,
        retain_traces=retain_traces,
        output_dir=tmp_path,
    )

    traces = await runner.run_eval(FakeEnv(), config)

    assert len((tmp_path / "results.jsonl").read_text().splitlines()) == 1
    assert traces == ([trace] if retain_traces else [])
    assert rollout.trace is (trace if retain_traces else None)


def test_no_retain_requires_non_rich_mode():
    with pytest.raises(ValueError, match="--no-rich"):
        EvalConfig(retain_traces=False)
