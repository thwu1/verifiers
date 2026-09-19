from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from verifiers.v1.env import EnvConfig, Environment
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.interception import InterceptionPool
from verifiers.v1.rollout import Rollout
from verifiers.v1.runtimes import ProgramResult, SandoqConfig
from verifiers.v1.taskset import Taskset, TasksetConfig


class FakeRemoteRuntime:
    def __init__(self) -> None:
        self.host_endpoint_calls: list[int] = []

    @asynccontextmanager
    async def host_endpoint(self, port: int):
        self.host_endpoint_calls.append(port)
        yield f"https://remote.example/{port}"


class EmptyTaskset(Taskset):
    def load_tasks(self):
        return []


class HostHarness(Harness):
    RUNS_ON_HOST = True
    SUPPORTS_MCP = False
    SUPPORTS_USER_SIM = False

    async def launch(self, *_args, **_kwargs):
        return ProgramResult(exit_code=0, stdout="", stderr="")


class RuntimeHarness(HostHarness):
    RUNS_ON_HOST = False


class ToolTaskset(EmptyTaskset):
    def tools(self, task):
        return []


class UserTaskset(EmptyTaskset):
    def user(self, task):
        return None


def host_sandoq_config() -> SandoqConfig:
    return SandoqConfig(
        network_access=False,
        host_tunnel="none",
        expected_environment="oci-runner-firecracker",
        ecr_token_file=Path("/run/secrets/ecr-token"),
    )


async def test_host_side_harness_uses_local_interception_without_runtime_tunnel() -> (
    None
):
    runtime = FakeRemoteRuntime()
    pool = InterceptionPool(
        host_sandoq_config(),
        multiplex=1,
        consumer_runs_on_host=True,
    )

    async with pool:
        async with pool.acquire(SimpleNamespace(closed=False), runtime) as (
            endpoint,
            _secret,
            port,
            base,
        ):
            assert endpoint == f"http://127.0.0.1:{port}/v1"
            assert base == f"http://127.0.0.1:{port}"

    assert runtime.host_endpoint_calls == []


async def test_remote_harness_keeps_instance_host_endpoint() -> None:
    runtime = FakeRemoteRuntime()
    pool = InterceptionPool(
        SandoqConfig(host_tunnel="sandoq"),
        multiplex=1,
    )

    async with pool:
        async with pool.acquire(SimpleNamespace(closed=False), runtime) as (
            endpoint,
            _secret,
            port,
            base,
        ):
            assert endpoint == f"https://remote.example/{port}/v1"
            assert base == f"https://remote.example/{port}"

    assert len(runtime.host_endpoint_calls) == 1


def test_environment_propagates_host_harness_topology_to_pool() -> None:
    environment = object.__new__(Environment)
    environment.harness = SimpleNamespace(
        RUNS_ON_HOST=True,
        config=SimpleNamespace(runtime=host_sandoq_config()),
    )
    environment.config = SimpleNamespace(multiplex=7)

    pool = environment.interception_pool()

    assert pool.is_local is True
    assert pool.instance_host_endpoint is False
    assert pool.multiplex == 7


async def test_nonpooled_host_harness_uses_local_interception() -> None:
    runtime = FakeRemoteRuntime()
    rollout = Rollout(
        task=object(),
        taskset=object(),
        harness=SimpleNamespace(RUNS_ON_HOST=True),
        ctx=object(),
        runtime_config=host_sandoq_config(),
    )

    async with rollout._serve_interception(
        None, runtime, SimpleNamespace(closed=False)
    ) as (
        endpoint,
        _secret,
        _state_port,
        state_base,
    ):
        assert endpoint.startswith("http://127.0.0.1:")
        assert endpoint == f"{state_base}/v1"

    assert runtime.host_endpoint_calls == []


@pytest.mark.parametrize("host_tunnel", ["sandoq", "modal", "prime"])
def test_environment_rejects_host_harness_with_remote_sandoq_tunnel(
    monkeypatch, host_tunnel: str
) -> None:
    taskset = EmptyTaskset(TasksetConfig(id="fake-taskset"))
    harness = HostHarness(
        HarnessConfig(
            id="fake-host-harness",
            runtime=SandoqConfig(host_tunnel=host_tunnel),
        )
    )
    monkeypatch.setattr("verifiers.v1.loaders.load_taskset", lambda _config: taskset)
    monkeypatch.setattr("verifiers.v1.loaders.load_harness", lambda _config: harness)
    config = EnvConfig.model_construct(
        taskset=taskset.config,
        harness=harness.config,
    )

    with pytest.raises(ValueError, match="Sandoq host-side harness"):
        Environment(config)


def test_environment_accepts_host_harness_with_no_sandoq_tunnel(
    monkeypatch,
) -> None:
    taskset = EmptyTaskset(TasksetConfig(id="fake-taskset"))
    harness = HostHarness(
        HarnessConfig(
            id="fake-host-harness",
            runtime=host_sandoq_config(),
        )
    )
    monkeypatch.setattr("verifiers.v1.loaders.load_taskset", lambda _config: taskset)
    monkeypatch.setattr("verifiers.v1.loaders.load_harness", lambda _config: harness)
    config = EnvConfig.model_construct(
        taskset=taskset.config,
        harness=harness.config,
    )

    environment = Environment(config)

    assert environment.harness is harness
    assert environment.taskset is taskset


def test_environment_rejects_runtime_harness_without_sandoq_tunnel(
    monkeypatch,
) -> None:
    taskset = EmptyTaskset(TasksetConfig(id="fake-taskset"))
    harness = RuntimeHarness(
        HarnessConfig(
            id="fake-runtime-harness",
            runtime=host_sandoq_config(),
        )
    )
    monkeypatch.setattr("verifiers.v1.loaders.load_taskset", lambda _config: taskset)
    monkeypatch.setattr("verifiers.v1.loaders.load_harness", lambda _config: harness)
    config = EnvConfig.model_construct(
        taskset=taskset.config,
        harness=harness.config,
    )

    with pytest.raises(ValueError, match="runtime-side harness"):
        Environment(config)


@pytest.mark.parametrize("taskset_type", [ToolTaskset, UserTaskset])
def test_host_harness_rejects_unplumbed_tool_or_user_consumers(
    monkeypatch, taskset_type
) -> None:
    taskset = taskset_type(TasksetConfig(id="fake-taskset"))
    harness = HostHarness(
        HarnessConfig(
            id="fake-host-harness",
            runtime=host_sandoq_config(),
        )
    )
    monkeypatch.setattr("verifiers.v1.loaders.load_taskset", lambda _config: taskset)
    monkeypatch.setattr("verifiers.v1.loaders.load_harness", lambda _config: harness)
    config = EnvConfig.model_construct(
        taskset=taskset.config,
        harness=harness.config,
    )

    with pytest.raises(ValueError, match="without MCP tools or a user simulator"):
        Environment(config)
