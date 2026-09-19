import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from verifiers.v1.env import Environment
from verifiers.v1.episode import Episode
from verifiers.v1.errors import SandboxError
from verifiers.v1.rollout import Rollout
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import Task
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.trace import Trace

pytestmark = pytest.mark.asyncio


class LifecycleRuntime(Runtime):
    def __init__(
        self,
        events: list[str],
        *,
        cleanup_must_succeed: bool = False,
        stop_failure: bool = False,
    ) -> None:
        super().__init__()
        self.events = events
        self.cleanup_must_succeed = cleanup_must_succeed
        self.stop_failure = stop_failure
        self.unusable_error: SandboxError | None = None

    async def start(self) -> None:
        self.events.append("runtime-start")

    async def stop(self) -> None:
        self.events.append("runtime-stop")
        if self.stop_failure:
            raise RuntimeError("runtime stop failure")

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        return ProgramResult(exit_code=0, stdout="", stderr="")

    async def read(self, path: str) -> bytes:
        return b""

    async def write(self, path: str, data: bytes) -> None:
        return None

    def ensure_usable(self) -> None:
        if self.unusable_error is not None:
            raise self.unusable_error


class LifecycleTaskset(Taskset):
    def __init__(
        self, events: list[str], failure_stage: str, *, cleanup_failure: bool = False
    ) -> None:
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
    runtime_cleanup_must_succeed: bool = False,
    runtime_stop_failure: bool = False,
) -> tuple[Rollout, LifecycleHarness, list[str]]:
    events: list[str] = []
    runtime = LifecycleRuntime(
        events,
        cleanup_must_succeed=runtime_cleanup_must_succeed,
        stop_failure=runtime_stop_failure,
    )
    taskset = LifecycleTaskset(events, failure_stage, cleanup_failure=cleanup_failure)
    harness = LifecycleHarness(events, block=block, failure_stage=failure_stage)
    monkeypatch.setattr(
        "verifiers.v1.rollout.make_runtime", lambda config, name: runtime
    )
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
async def test_taskset_cleanup_runs_before_runtime_stop_on_failure(
    monkeypatch, failure_stage
) -> None:
    rollout, _, events = make_rollout(
        monkeypatch, block=False, failure_stage=failure_stage
    )

    trace = await rollout.run()

    assert trace.error is not None
    assert events[-2:] == ["taskset-cleanup", "runtime-stop"]
    assert events.count("taskset-cleanup") == 1


async def test_taskset_cleanup_runs_on_success(monkeypatch) -> None:
    rollout, _, events = make_rollout(monkeypatch, block=False, failure_stage="none")

    trace = await rollout.run()

    assert trace.error is None
    assert events[-2:] == ["taskset-cleanup", "runtime-stop"]


async def test_integrity_critical_runtime_stop_failure_is_captured(monkeypatch) -> None:
    rollout, _, events = make_rollout(
        monkeypatch,
        block=False,
        failure_stage="none",
        runtime_cleanup_must_succeed=True,
        runtime_stop_failure=True,
    )

    trace = await rollout.run()

    assert trace.error is not None
    assert trace.error.type == "SandboxError"
    assert "runtime stop failure" in trace.error.message
    assert events[-2:] == ["taskset-cleanup", "runtime-stop"]


async def test_integrity_critical_stop_failure_preserves_existing_error(
    monkeypatch,
) -> None:
    rollout, _, events = make_rollout(
        monkeypatch,
        block=False,
        failure_stage="harness",
        runtime_cleanup_must_succeed=True,
        runtime_stop_failure=True,
    )

    trace = await rollout.run()

    assert trace.error is not None
    assert trace.error.type == "RuntimeError"
    assert "harness failure" in trace.error.message
    assert "runtime stop failure" not in trace.error.message
    assert events[-2:] == ["taskset-cleanup", "runtime-stop"]


async def test_sandbox_cancellation_failure_skips_finalize_and_scoring(
    monkeypatch,
) -> None:
    rollout, harness, events = make_rollout(
        monkeypatch,
        block=False,
        failure_stage="none",
    )

    async def fail_cancel(*args, **kwargs) -> None:
        events.append("harness-run")
        raise SandboxError("command cancellation failed")

    harness.run = fail_cancel
    trace = await rollout.run()

    assert trace.error is not None
    assert trace.error.type == "SandboxError"
    assert trace.stop_condition == "error"
    assert "taskset-finalize" not in events
    assert "taskset-score" not in events
    assert "harness-score" not in events
    assert events[-2:] == ["taskset-cleanup", "runtime-stop"]


async def test_harness_timeout_waits_for_failed_cancellation_and_skips_scoring(
    monkeypatch,
) -> None:
    rollout, harness, events = make_rollout(
        monkeypatch,
        block=False,
        failure_stage="none",
    )

    async def fail_while_cancelling(*args, **kwargs) -> None:
        events.append("harness-run")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            events.append("command-drained")
            assert isinstance(rollout.runtime, LifecycleRuntime)
            rollout.runtime.unusable_error = SandboxError("command cancellation failed")
            raise

    harness.run = fail_while_cancelling
    rollout.harness_timeout = 0.01
    trace = await rollout.run()

    assert trace.error is not None
    assert trace.error.type == "SandboxError"
    assert trace.stop_condition == "error"
    assert "taskset-finalize" not in events
    assert "taskset-score" not in events
    assert "harness-score" not in events
    assert "taskset-cleanup" not in events
    assert events[-2:] == ["command-drained", "runtime-stop"]


async def test_unusable_runtime_quiesces_before_taskset_cleanup(monkeypatch) -> None:
    rollout, harness, events = make_rollout(
        monkeypatch,
        block=False,
        failure_stage="none",
    )
    stop_started = asyncio.Event()
    worker_finished = asyncio.Event()

    async def fail_while_cancelling(*args, **kwargs) -> None:
        events.append("harness-run")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            assert isinstance(rollout.runtime, LifecycleRuntime)
            rollout.runtime.unusable_error = SandboxError("command cancellation failed")
            rollout.runtime.stop = stop_after_quarantine
            raise

    async def stop_after_quarantine() -> None:
        events.append("runtime-stop-start")
        stop_started.set()
        await worker_finished.wait()
        events.append("worker-finished")

    harness.run = fail_while_cancelling
    rollout.harness_timeout = 0.01
    running = asyncio.create_task(rollout.run())

    await asyncio.wait_for(stop_started.wait(), timeout=1)
    assert "taskset-cleanup" not in events
    worker_finished.set()
    trace = await asyncio.wait_for(running, timeout=1)

    assert trace.error is not None
    assert trace.error.type == "SandboxError"
    assert "taskset-cleanup" not in events
    assert events[-2:] == ["runtime-stop-start", "worker-finished"]


async def test_suppressed_timeout_still_checks_runtime_before_scoring(
    monkeypatch,
) -> None:
    rollout, harness, events = make_rollout(
        monkeypatch,
        block=False,
        failure_stage="none",
    )

    async def suppress_cancellation(*args, **kwargs) -> None:
        events.append("harness-run")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            events.append("command-drained")
            assert isinstance(rollout.runtime, LifecycleRuntime)
            rollout.runtime.unusable_error = SandboxError("command cancellation failed")

    harness.run = suppress_cancellation
    rollout.harness_timeout = 0.01
    trace = await rollout.run()

    assert trace.error is not None
    assert trace.error.type == "SandboxError"
    assert trace.stop_condition == "error"
    assert "taskset-finalize" not in events
    assert "taskset-score" not in events
    assert "taskset-cleanup" not in events
    assert events[-2:] == ["command-drained", "runtime-stop"]


async def test_parent_cancellation_propagates_even_when_runtime_becomes_unusable(
    monkeypatch,
) -> None:
    rollout, harness, events = make_rollout(
        monkeypatch,
        block=False,
        failure_stage="none",
    )

    async def invalidate_while_cancelling(*args, **kwargs) -> None:
        events.append("harness-run")
        harness.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            events.append("command-drained")
            assert isinstance(rollout.runtime, LifecycleRuntime)
            rollout.runtime.unusable_error = SandboxError("command cancellation failed")
            raise

    harness.run = invalidate_while_cancelling
    rollout.harness_timeout = 60
    running = asyncio.create_task(rollout.run())
    await harness.started.wait()
    running.cancel()

    with pytest.raises(asyncio.CancelledError):
        await running

    assert "taskset-finalize" not in events
    assert "taskset-score" not in events
    assert "taskset-cleanup" not in events
    assert events[-2:] == ["command-drained", "runtime-stop"]


async def test_harness_timeout_drains_before_finalize_and_scoring(monkeypatch) -> None:
    rollout, harness, events = make_rollout(
        monkeypatch,
        block=False,
        failure_stage="none",
    )

    async def drain_while_cancelling(*args, **kwargs) -> None:
        events.append("harness-run")
        try:
            await asyncio.Event().wait()
        finally:
            events.append("command-drained")

    harness.run = drain_while_cancelling
    rollout.harness_timeout = 0.01
    trace = await rollout.run()

    assert trace.error is None
    assert trace.stop_condition == "harness_timeout"
    assert events.index("command-drained") < events.index("taskset-finalize")
    assert events.index("taskset-finalize") < events.index("taskset-score")


async def test_cleanup_failure_does_not_replace_existing_rollout_error(
    monkeypatch,
) -> None:
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


async def test_taskset_cleanup_runs_before_runtime_stop_on_cancellation(
    monkeypatch,
) -> None:
    rollout, harness, events = make_rollout(monkeypatch, block=True)
    running = asyncio.create_task(rollout.run())
    await harness.started.wait()

    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    assert events[-2:] == ["taskset-cleanup", "runtime-stop"]
    assert events.count("taskset-cleanup") == 1


async def test_environment_serving_closes_taskset_after_resources_on_cancellation() -> (
    None
):
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


async def test_episode_quiesces_sibling_rollouts_when_persistence_fails(
    monkeypatch,
) -> None:
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
