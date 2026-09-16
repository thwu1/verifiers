import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from verifiers.v1.env import Environment
from verifiers.v1.episode import Episode
from verifiers.v1.rollout import Rollout
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import Task
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.trace import Trace

pytestmark = pytest.mark.asyncio


class LifecycleRuntime(Runtime):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    async def start(self) -> None:
        self.events.append("runtime-start")

    async def stop(self) -> None:
        self.events.append("runtime-stop")

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        return ProgramResult(exit_code=0, stdout="", stderr="")

    async def read(self, path: str) -> bytes:
        return b""

    async def write(self, path: str, data: bytes) -> None:
        return None


class LifecycleTaskset(Taskset):
    def __init__(self, events: list[str], failure_stage: str, *, cleanup_failure: bool = False) -> None:
        super().__init__(TasksetConfig(id="lifecycle-test"))
        self.events = events
        self.failure_stage = failure_stage
        self.cleanup_failure = cleanup_failure

    def load_tasks(self) -> list[Task]:
        return []

    async def setup(self, task: Task, runtime: Runtime) -> None:
        self.events.append("taskset-setup")
        if self.failure_stage == "setup":
            raise RuntimeError("setup failure")

    async def finalize(self, task: Task, trace, runtime: Runtime) -> None:
        self.events.append("taskset-finalize")
        if self.failure_stage == "finalize":
            raise RuntimeError("finalize failure")

    async def score(self, trace, runtime: Runtime) -> None:
        self.events.append("taskset-score")
        if self.failure_stage == "scoring":
            raise RuntimeError("scoring failure")

    async def cleanup(self, task: Task, trace, runtime: Runtime) -> None:
        self.events.append("taskset-cleanup")
        if self.cleanup_failure:
            raise RuntimeError("cleanup failure")

    async def close(self) -> None:
        self.events.append("taskset-close")


class LifecycleHarness:
    def __init__(self, events: list[str], *, block: bool, failure_stage: str) -> None:
        self.config = SimpleNamespace(id="lifecycle-test", name="lifecycle-test")
        self.events = events
        self.block = block
        self.failure_stage = failure_stage
        self.started = asyncio.Event()

    async def setup(self, runtime: Runtime) -> None:
        self.events.append("harness-setup")

    async def run(self, *args, **kwargs) -> None:
        self.events.append("harness-run")
        self.started.set()
        if self.block:
            await asyncio.Event().wait()
        if self.failure_stage == "harness":
            raise RuntimeError("harness failure")

    async def score(self, trace, runtime: Runtime) -> None:
        self.events.append("harness-score")


@asynccontextmanager
async def fake_interception(self, pool, runtime, session):
    yield "http://127.0.0.1:1/v1", "secret", 1, "http://127.0.0.1:1"


def make_rollout(
    monkeypatch: pytest.MonkeyPatch,
    *,
    block: bool,
    failure_stage: str = "harness",
    cleanup_failure: bool = False,
) -> tuple[Rollout, LifecycleHarness, list[str]]:
    events: list[str] = []
    runtime = LifecycleRuntime(events)
    taskset = LifecycleTaskset(events, failure_stage, cleanup_failure=cleanup_failure)
    harness = LifecycleHarness(events, block=block, failure_stage=failure_stage)
    monkeypatch.setattr("verifiers.v1.rollout.make_runtime", lambda config, name: runtime)
    monkeypatch.setattr(Rollout, "_serve_interception", fake_interception)
    rollout = Rollout(
        task=Task(idx=0, prompt="test"),
        taskset=taskset,
        harness=harness,
        ctx=SimpleNamespace(),
        runtime_config=SimpleNamespace(type="test"),
    )
    return rollout, harness, events


@pytest.mark.parametrize("failure_stage", ["setup", "harness", "finalize", "scoring"])
async def test_taskset_cleanup_runs_before_runtime_stop_on_failure(monkeypatch, failure_stage) -> None:
    rollout, _, events = make_rollout(monkeypatch, block=False, failure_stage=failure_stage)

    trace = await rollout.run()

    assert trace.error is not None
    assert events[-2:] == ["taskset-cleanup", "runtime-stop"]
    assert events.count("taskset-cleanup") == 1


async def test_taskset_cleanup_runs_on_success(monkeypatch) -> None:
    rollout, _, events = make_rollout(monkeypatch, block=False, failure_stage="none")

    trace = await rollout.run()

    assert trace.error is None
    assert events[-2:] == ["taskset-cleanup", "runtime-stop"]


async def test_cleanup_failure_does_not_replace_existing_rollout_error(monkeypatch) -> None:
    rollout, _, events = make_rollout(
        monkeypatch,
        block=False,
        failure_stage="harness",
        cleanup_failure=True,
    )

    trace = await rollout.run()

    assert trace.error is not None
    assert "harness failure" in trace.error.message
    assert "cleanup failure" not in trace.error.message
    assert events[-2:] == ["taskset-cleanup", "runtime-stop"]


async def test_taskset_cleanup_runs_before_runtime_stop_on_cancellation(monkeypatch) -> None:
    rollout, harness, events = make_rollout(monkeypatch, block=True)
    running = asyncio.create_task(rollout.run())
    await harness.started.wait()

    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    assert events[-2:] == ["taskset-cleanup", "runtime-stop"]
    assert events.count("taskset-cleanup") == 1


async def test_environment_serving_closes_taskset_after_resources_on_cancellation() -> None:
    events: list[str] = []
    taskset = LifecycleTaskset(events, failure_stage="none")

    @asynccontextmanager
    async def shared_tools(tasks):
        try:
            yield {}
        finally:
            events.append("shared-tools-stop")

    @asynccontextmanager
    async def interception_pool():
        try:
            yield SimpleNamespace()
        finally:
            events.append("interception-stop")

    environment = SimpleNamespace(
        taskset=taskset,
        shared_tools=shared_tools,
        interception_pool=interception_pool,
        _shared_urls={},
        _interception=None,
    )
    entered = asyncio.Event()

    async def serve() -> None:
        async with Environment.serving(environment, []):
            entered.set()
            await asyncio.Event().wait()

    running = asyncio.create_task(serve())
    await entered.wait()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    assert events == ["interception-stop", "shared-tools-stop", "taskset-close"]
    assert environment._shared_urls == {}
    assert environment._interception is None


async def test_environment_serving_closes_taskset_when_resource_enter_fails() -> None:
    events: list[str] = []
    taskset = LifecycleTaskset(events, failure_stage="none")

    @asynccontextmanager
    async def failing_shared_tools(tasks):
        raise RuntimeError("shared tool startup failed")
        yield {}

    environment = SimpleNamespace(
        taskset=taskset,
        shared_tools=failing_shared_tools,
        interception_pool=lambda: None,
        _shared_urls={},
        _interception=None,
    )

    with pytest.raises(RuntimeError, match="shared tool startup failed"):
        async with Environment.serving(environment, []):
            raise AssertionError("unreachable")

    assert events == ["taskset-close"]


async def test_episode_quiesces_sibling_rollouts_when_persistence_fails(monkeypatch) -> None:
    events: list[str] = []
    both_started = asyncio.Event()
    started = 0

    class EpisodeRollout:
        def __init__(self, *, slow: bool) -> None:
            self.slow = slow
            self.phase = None
            self.trace = None

        async def run(self) -> Trace:
            nonlocal started
            started += 1
            if started == 2:
                both_started.set()
            await both_started.wait()
            if self.slow:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    events.append("slow-cancelled")
                    raise
            return Trace(task=Task(idx=0, prompt="test"))

    async def run_direct(rollout, retry):
        return await rollout.run()

    async def persist(trace: Trace) -> None:
        raise RuntimeError("persistence failure")

    monkeypatch.setattr("verifiers.v1.episode.run_with_retry", run_direct)
    taskset = LifecycleTaskset(events, failure_stage="none")
    episode = Episode(
        [EpisodeRollout(slow=False), EpisodeRollout(slow=True)],
        taskset,
        retry=SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="persistence failure"):
        await episode.run(on_complete=persist)

    assert events == ["slow-cancelled"]
