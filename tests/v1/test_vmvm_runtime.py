import asyncio
import concurrent.futures
import threading

import pytest
from verifiers.v1.errors import ProviderError, SandboxError, TunnelError
from verifiers.v1.runtimes import (
    VMVMConfig,
    VMVMRuntime,
    make_runtime,
    runtime_has_instance_host_endpoint,
    runtime_is_local,
    vmvm,
)


class FakeBackend:
    def __init__(self) -> None:
        self.commands: list[tuple[str, float]] = []
        self.files: dict[str, bytes] = {}
        self.open_tunnels: list[object] = []
        self.destroyed = False
        self.destroy_calls = 0
        self.restart_calls = 0
        self.restart_result = True
        self.network_prepare_calls = 0
        self.network_activate_calls = 0
        self.recovery_results: list[vmvm.VMVMBashResult | None] = []
        self.result: vmvm.VMVMBashResult = {
            "status": "success",
            "output": "ok",
            "error_type": "none",
            "exit_code": 0,
        }

    def run_bash(self, command: str, timeout: float = 60.0) -> vmvm.VMVMBashResult:
        self.commands.append((command, timeout))
        return self.result

    def restart_session(self) -> bool:
        self.restart_calls += 1
        return self.restart_result

    def recover_last(self) -> vmvm.VMVMBashResult | None:
        return self.recovery_results.pop(0)

    def transfer_file(self, file_content: str | bytes, remote_path: str) -> None:
        self.files[remote_path] = file_content.encode() if isinstance(file_content, str) else file_content

    def read_file(self, remote_path: str) -> bytes:
        return self.files[remote_path]

    def prepare_network_isolation(self) -> None:
        self.network_prepare_calls += 1

    def activate_network_isolation(self) -> None:
        self.network_activate_calls += 1

    def open_host_tunnel(self, local_port: int) -> tuple[object, str]:
        tunnel = object()
        self.open_tunnels.append(tunnel)
        return tunnel, f"http://10.88.0.1:{local_port}"

    def close_host_tunnel(self, tunnel: object) -> None:
        self.open_tunnels.remove(tunnel)

    def destroy(self) -> None:
        self.destroyed = True
        self.destroy_calls += 1

    def get_debugging_info(self) -> dict[str, object]:
        return {"container_id": "abc123"}


async def test_vmvm_runtime_lifecycle(monkeypatch) -> None:
    backend = FakeBackend()
    monkeypatch.setattr(vmvm, "create_backend", lambda config: backend)
    config = VMVMConfig(
        image="swebench/image",
        workdir="/testbed",
        session_timeout=123,
    )
    runtime = make_runtime(config, "rollout-1")

    assert isinstance(runtime, VMVMRuntime)
    assert runtime_is_local(config) is False
    assert runtime_has_instance_host_endpoint(config) is True
    await runtime.start()
    assert runtime.descriptor == "abc123"

    result = await runtime.run(["sh", "-c", "printf ok"], {"MESSAGE": "value with spaces"})
    assert result.exit_code == 0
    assert result.stdout == "ok"
    assert backend.commands == [
        ("mkdir -p /testbed", 123),
        (
            "cd /testbed && env 'MESSAGE=value with spaces' sh -c 'printf ok'",
            123,
        ),
    ]

    await runtime.write("artifact.bin", b"\x00\x01")
    assert await runtime.read("artifact.bin") == b"\x00\x01"
    assert backend.files == {"/testbed/artifact.bin": b"\x00\x01"}

    async with runtime.host_endpoint(4321) as url:
        assert url == "http://10.88.0.1:4321"
        assert len(backend.open_tunnels) == 1
    assert backend.open_tunnels == []

    await runtime.stop()
    assert backend.destroyed is True


async def test_vmvm_backend_init_pool_does_not_starve_default_executor(monkeypatch) -> None:
    loop = asyncio.get_running_loop()
    default_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="vmvm-default-test",
    )
    init_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="vmvm-init-test",
    )
    loop.set_default_executor(default_executor)
    first_backend = FakeBackend()
    second_backend = FakeBackend()
    first_constructor_entered = threading.Event()
    second_constructor_entered = threading.Event()
    release_second_constructor = threading.Event()
    first_probe_ran = threading.Event()
    constructor_threads: list[str] = []
    probe_threads: list[str] = []

    def create(config: VMVMConfig) -> FakeBackend:
        constructor_threads.append(threading.current_thread().name)
        if config.image == "first-image":
            first_constructor_entered.set()
            assert second_constructor_entered.wait(2)
            return first_backend
        second_constructor_entered.set()
        assert release_second_constructor.wait(5)
        return second_backend

    def first_run_bash(command: str, timeout: float = 60.0) -> vmvm.VMVMBashResult:
        probe_threads.append(threading.current_thread().name)
        first_probe_ran.set()
        return FakeBackend.run_bash(first_backend, command, timeout)

    first_backend.run_bash = first_run_bash
    monkeypatch.setattr(vmvm, "create_backend", create)
    monkeypatch.setattr(vmvm, "_get_backend_init_executor", lambda: init_executor)
    first = VMVMRuntime(VMVMConfig(image="first-image"))
    second = VMVMRuntime(VMVMConfig(image="second-image"))
    first_start = asyncio.create_task(first.start())
    second_start = asyncio.create_task(second.start())
    try:
        for _ in range(200):
            if first_constructor_entered.is_set() and second_constructor_entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert first_constructor_entered.is_set()
        assert second_constructor_entered.is_set()
        await asyncio.wait_for(asyncio.shield(first_start), timeout=2)
        assert first_probe_ran.is_set()
        assert not second_start.done()
        assert all(name.startswith("vmvm-init-test") for name in constructor_threads)
        assert probe_threads and all(name.startswith("vmvm-default-test") for name in probe_threads)
        release_second_constructor.set()
        await asyncio.wait_for(second_start, timeout=2)
        await first.stop()
        await second.stop()
    finally:
        release_second_constructor.set()
        await asyncio.gather(first_start, second_start, return_exceptions=True)
        init_executor.shutdown(wait=True, cancel_futures=True)


async def test_vmvm_runtime_submits_all_backend_init_to_dedicated_executor(monkeypatch) -> None:
    class RecordingExecutor(concurrent.futures.Executor):
        def __init__(self) -> None:
            self.submitted: list[object] = []

        def submit(self, function, /, *args, **kwargs):
            self.submitted.append(function)
            future: concurrent.futures.Future[object] = concurrent.futures.Future()
            future.set_result(function(*args, **kwargs))
            return future

    executor = RecordingExecutor()
    backends: list[FakeBackend] = []
    default_calls: list[object] = []

    def create(_config: VMVMConfig) -> FakeBackend:
        backend = FakeBackend()
        backends.append(backend)
        return backend

    async def to_thread(function, /, *args, **kwargs):
        default_calls.append(function)
        return function(*args, **kwargs)

    monkeypatch.setattr(vmvm, "create_backend", create)
    monkeypatch.setattr(vmvm, "_get_backend_init_executor", lambda: executor)
    monkeypatch.setattr(vmvm.asyncio, "to_thread", to_thread)
    runtimes = [VMVMRuntime(VMVMConfig(), name=f"rollout-{index}") for index in range(96)]

    await asyncio.gather(*(runtime.start() for runtime in runtimes))

    assert len(executor.submitted) == 96
    assert all(function is create for function in executor.submitted)
    assert len(default_calls) == 96
    assert all(getattr(function, "__name__", "") == "run_bash" for function in default_calls)
    assert len(backends) == 96


async def test_vmvm_runtime_cancellation_destroys_late_backend(monkeypatch) -> None:
    init_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    backend = FakeBackend()
    constructor_entered = threading.Event()
    release_constructor = threading.Event()

    def create(_config: VMVMConfig) -> FakeBackend:
        constructor_entered.set()
        assert release_constructor.wait(5)
        return backend

    monkeypatch.setattr(vmvm, "create_backend", create)
    monkeypatch.setattr(vmvm, "_get_backend_init_executor", lambda: init_executor)
    runtime = VMVMRuntime(VMVMConfig())
    start = asyncio.create_task(runtime.start())
    try:
        for _ in range(200):
            if constructor_entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert constructor_entered.is_set()
        start.cancel()
        with pytest.raises(asyncio.CancelledError):
            await start
        release_constructor.set()
        for _ in range(200):
            if backend.destroyed:
                break
            await asyncio.sleep(0.01)
        assert backend.destroy_calls == 1
        assert not vmvm._pending_backend_cleanups
    finally:
        release_constructor.set()
        init_executor.shutdown(wait=True, cancel_futures=True)


async def test_vmvm_runtime_cancellation_waits_for_initial_probe_before_destroy(monkeypatch) -> None:
    init_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    backend = FakeBackend()
    probe_entered = threading.Event()
    release_probe = threading.Event()

    def run_bash(command: str, timeout: float = 60.0) -> vmvm.VMVMBashResult:
        probe_entered.set()
        assert release_probe.wait(5)
        return FakeBackend.run_bash(backend, command, timeout)

    backend.run_bash = run_bash
    monkeypatch.setattr(vmvm, "create_backend", lambda _config: backend)
    monkeypatch.setattr(vmvm, "_get_backend_init_executor", lambda: init_executor)
    runtime = VMVMRuntime(VMVMConfig())
    start = asyncio.create_task(runtime.start())
    try:
        for _ in range(200):
            if probe_entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert probe_entered.is_set()
        start.cancel()
        with pytest.raises(asyncio.CancelledError):
            await start
        await asyncio.sleep(0.05)
        assert backend.destroy_calls == 0
        release_probe.set()
        for _ in range(200):
            if backend.destroyed:
                break
            await asyncio.sleep(0.01)
        assert backend.destroy_calls == 1
        assert not vmvm._pending_backend_cleanups
    finally:
        release_probe.set()
        init_executor.shutdown(wait=True, cancel_futures=True)


async def test_vmvm_runtime_destroys_backend_once_when_initial_probe_fails(monkeypatch) -> None:
    backend = FakeBackend()
    backend.result = {
        "status": "error",
        "output": "probe failed",
        "error_type": "exit",
        "exit_code": 7,
    }
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(vmvm, "create_backend", lambda _config: backend)
    monkeypatch.setattr(vmvm, "_get_backend_init_executor", lambda: executor)
    try:
        with pytest.raises(SandboxError, match="VMVM provisioning failed"):
            await VMVMRuntime(VMVMConfig()).start()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    assert backend.destroy_calls == 1


async def test_vmvm_runtime_maps_backend_init_failure_without_cleanup(monkeypatch) -> None:
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def fail(_config: VMVMConfig) -> FakeBackend:
        raise RuntimeError("synthetic init failure")

    monkeypatch.setattr(vmvm, "create_backend", fail)
    monkeypatch.setattr(vmvm, "_get_backend_init_executor", lambda: executor)
    try:
        with pytest.raises(SandboxError, match="VMVM provisioning failed"):
            await VMVMRuntime(VMVMConfig()).start()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    assert not vmvm._pending_backend_cleanups


async def test_vmvm_deferred_cleanup_log_redacts_backend_error(caplog) -> None:
    backend = FakeBackend()
    secret = "opaque_session_auth_and_task_payload"

    def fail_destroy() -> None:
        raise RuntimeError(secret)

    backend.destroy = fail_destroy
    with caplog.at_level("WARNING", logger=vmvm.__name__):
        await vmvm._destroy_backend_safely(backend)

    assert caplog.messages == ["vmvm: deferred backend cleanup failed"]
    assert secret not in caplog.text


def test_vmvm_backend_init_executor_shutdown_and_fork_reset_are_idempotent(monkeypatch) -> None:
    monkeypatch.setattr(vmvm, "_backend_init_executor", None)
    monkeypatch.setattr(vmvm, "_backend_init_executor_pid", None)
    monkeypatch.setattr(vmvm, "_backend_init_lock", threading.Lock())
    first = vmvm._get_backend_init_executor()

    assert vmvm._get_backend_init_executor() is first
    vmvm._shutdown_backend_init_executor()
    vmvm._shutdown_backend_init_executor()
    assert vmvm._backend_init_executor is None
    second = vmvm._get_backend_init_executor()
    assert second is not first

    vmvm._reset_backend_init_executor_after_fork()
    third = vmvm._get_backend_init_executor()
    assert third is not second

    second.shutdown(wait=True, cancel_futures=True)
    vmvm._shutdown_backend_init_executor()


async def test_vmvm_runtime_activates_network_before_deferred_startup_and_program(
    monkeypatch,
) -> None:
    backend = FakeBackend()
    events: list[str] = []

    def activate() -> None:
        backend.network_activate_calls += 1
        events.append("activate")

    def run_bash(command: str, timeout: float = 60.0) -> vmvm.VMVMBashResult:
        backend.commands.append((command, timeout))
        if "start-service" in command:
            events.append("startup")
        if "run-agent" in command:
            events.append("program")
        return backend.result

    backend.activate_network_isolation = activate
    backend.run_bash = run_bash
    monkeypatch.setattr(vmvm, "create_backend", lambda config: backend)
    runtime = VMVMRuntime(VMVMConfig(session_timeout=10))
    await runtime.start()

    await runtime.configure_network_policy("no-network")
    runtime.defer_until_network_isolated(["start-service"])
    async with runtime.host_endpoint(4321):
        result = await runtime.run_program(["run-agent"], {})

    assert result.exit_code == 0
    assert backend.network_prepare_calls == 1
    assert events == ["activate", "startup", "program"]
    isolated_commands = [command for command, _ in backend.commands if "NO_PROXY=*" in command]
    assert len(isolated_commands) == 3
    assert any("socket.create_connection" in command for command in isolated_commands)
    assert any("curl --noproxy" in command for command in isolated_commands)
    assert any("wget --no-proxy" in command for command in isolated_commands)
    assert any("start-service" in command for command in isolated_commands)
    assert any("run-agent" in command for command in isolated_commands)
    assert backend.open_tunnels == []

    with pytest.raises(SandboxError, match="cannot relax"):
        await runtime.configure_network_policy("public")

    await runtime.stop()


async def test_vmvm_runtime_closes_tunnel_when_post_isolation_http_probe_fails(
    monkeypatch,
) -> None:
    backend = FakeBackend()

    def run_bash(command: str, timeout: float = 60.0) -> vmvm.VMVMBashResult:
        backend.commands.append((command, timeout))
        if "socket.create_connection" in command:
            return {
                "status": "error",
                "output": "probe failed",
                "error_type": "exit",
                "exit_code": 7,
            }
        return backend.result

    backend.run_bash = run_bash
    monkeypatch.setattr(vmvm, "create_backend", lambda config: backend)
    runtime = VMVMRuntime(VMVMConfig(session_timeout=10))
    await runtime.start()
    await runtime.configure_network_policy("no-network")

    with pytest.raises(TunnelError, match="unreachable after no-network activation"):
        async with runtime.host_endpoint(4321):
            pytest.fail("an unreachable isolated tunnel must not be yielded")

    assert backend.network_activate_calls == 1
    assert backend.open_tunnels == []
    await runtime.stop()


@pytest.mark.parametrize(
    "body_error",
    [ProviderError("provider failed"), RuntimeError("harness failed")],
)
async def test_vmvm_runtime_preserves_body_error_when_tunnel_remains_reachable(
    monkeypatch,
    body_error: Exception,
) -> None:
    backend = FakeBackend()
    monkeypatch.setattr(vmvm, "create_backend", lambda config: backend)
    runtime = VMVMRuntime(VMVMConfig(session_timeout=10))
    await runtime.start()

    with pytest.raises(type(body_error)) as caught:
        async with runtime.host_endpoint(4321):
            raise body_error

    assert caught.value is body_error
    probes = [command for command, _ in backend.commands if "socket.create_connection" in command]
    assert len(probes) == 1
    assert "NO_PROXY=*" in probes[0]
    assert "curl --noproxy" in probes[0]
    assert "wget --no-proxy" in probes[0]
    assert backend.open_tunnels == []
    await runtime.stop()


async def test_vmvm_runtime_reclassifies_body_error_when_tunnel_became_unreachable(
    monkeypatch,
) -> None:
    backend = FakeBackend()
    monkeypatch.setattr(vmvm, "create_backend", lambda config: backend)
    runtime = VMVMRuntime(VMVMConfig(session_timeout=10))
    await runtime.start()
    body_error = ProviderError("provider failed")

    with pytest.raises(TunnelError, match="became unreachable") as caught:
        async with runtime.host_endpoint(4321):
            backend.result = {
                "status": "error",
                "output": "probe failed",
                "error_type": "exit",
                "exit_code": 7,
            }
            raise body_error

    assert caught.value.__cause__ is body_error
    assert backend.open_tunnels == []
    await runtime.stop()


async def test_vmvm_runtime_does_not_recover_command_timeout(monkeypatch) -> None:
    backend = FakeBackend()
    monkeypatch.setattr(vmvm, "create_backend", lambda config: backend)
    runtime = VMVMRuntime(VMVMConfig(session_timeout=10))
    await runtime.start()
    backend.result = {
        "status": "error",
        "output": "command timed out",
        "error_type": "timeout",
        "exit_code": -1,
    }

    with pytest.raises(SandboxError, match="timeout"):
        await runtime.run(["true"], {})

    assert backend.restart_calls == 0

    await runtime.stop()


async def test_vmvm_runtime_recovers_in_flight_command_exactly_once(monkeypatch) -> None:
    backend = FakeBackend()
    monkeypatch.setattr(vmvm, "create_backend", lambda config: backend)
    runtime = VMVMRuntime(VMVMConfig(session_timeout=10))
    await runtime.start()
    backend.result = {
        "status": "error",
        "output": "connection dropped",
        "error_type": "broken_pipe",
        "exit_code": -1,
    }
    backend.recovery_results = [
        {
            "status": "success",
            "output": "finished once",
            "error_type": "none",
            "exit_code": 0,
        }
    ]

    result = await runtime.run(["do-work"], {})

    assert result == vmvm.ProgramResult(exit_code=0, stdout="finished once", stderr="")
    assert backend.restart_calls == 1
    assert backend.commands.count(("cd /app && do-work", 10)) == 1

    await runtime.stop()


async def test_vmvm_runtime_recovers_across_repeated_transport_drops(monkeypatch) -> None:
    backend = FakeBackend()
    monkeypatch.setattr(vmvm, "create_backend", lambda config: backend)
    runtime = VMVMRuntime(VMVMConfig(session_timeout=10))
    await runtime.start()
    backend.result = {
        "status": "error",
        "output": "first drop",
        "error_type": "broken_pipe",
        "exit_code": -1,
    }
    backend.recovery_results = [
        {
            "status": "error",
            "output": "second drop",
            "error_type": "broken_pipe",
            "exit_code": -1,
        },
        {
            "status": "error",
            "output": "command failed",
            "error_type": "exit",
            "exit_code": 7,
        },
    ]

    result = await runtime.run(["do-work"], {})

    assert result == vmvm.ProgramResult(exit_code=7, stdout="command failed", stderr="")
    assert backend.restart_calls == 2
    assert backend.commands.count(("cd /app && do-work", 10)) == 1

    await runtime.stop()


async def test_vmvm_runtime_bounds_repeated_transport_recovery(monkeypatch) -> None:
    backend = FakeBackend()
    monkeypatch.setattr(vmvm, "create_backend", lambda config: backend)
    runtime = VMVMRuntime(VMVMConfig(session_timeout=10))
    await runtime.start()
    dropped: vmvm.VMVMBashResult = {
        "status": "error",
        "output": "connection dropped",
        "error_type": "broken_pipe",
        "exit_code": -1,
    }
    backend.result = dropped
    backend.recovery_results = [dropped] * vmvm.MAX_TRANSPORT_RECOVERY_ATTEMPTS

    with pytest.raises(SandboxError, match="transport remained unavailable"):
        await runtime.run(["do-work"], {})

    assert backend.restart_calls == vmvm.MAX_TRANSPORT_RECOVERY_ATTEMPTS
    assert backend.commands.count(("cd /app && do-work", 10)) == 1

    await runtime.stop()


@pytest.mark.parametrize(
    "restart_result,recovery_result,match",
    [
        (False, None, "sandbox state is unavailable"),
        (True, None, "exact-once execution cannot be proven"),
    ],
)
async def test_vmvm_runtime_rejects_unrecoverable_command(
    monkeypatch,
    restart_result: bool,
    recovery_result: vmvm.VMVMBashResult | None,
    match: str,
) -> None:
    backend = FakeBackend()
    monkeypatch.setattr(vmvm, "create_backend", lambda config: backend)
    runtime = VMVMRuntime(VMVMConfig(session_timeout=10))
    await runtime.start()
    backend.result = {
        "status": "error",
        "output": "connection dropped",
        "error_type": "broken_pipe",
        "exit_code": -1,
    }
    backend.restart_result = restart_result
    backend.recovery_results = [recovery_result]

    with pytest.raises(SandboxError, match=match):
        await runtime.run(["do-work"], {})

    await runtime.stop()
