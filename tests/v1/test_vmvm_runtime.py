import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

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

    def run_bash_with_recovery(
        self,
        command: str,
        timeout: float,
        max_attempts: int,
    ) -> vmvm.VMVMBashResult:
        result = self.run_bash(command, timeout)
        for _ in range(max_attempts):
            if result["exit_code"] >= 0 or result["error_type"] != "broken_pipe":
                break
            if not self.restart_session():
                raise RuntimeError("VMVM reconnect failed: sandbox state is unavailable")
            recovered = self.recover_last()
            if recovered is None:
                raise RuntimeError("VMVM command recovery failed: exact-once execution cannot be proven")
            result = recovered
        return result

    def cancel_active_command(self, timeout: float) -> bool:
        return True

    def restart_session(self) -> bool:
        self.restart_calls += 1
        return self.restart_result

    def recover_last(self) -> vmvm.VMVMBashResult | None:
        return self.recovery_results.pop(0)

    def transfer_file(self, file_content: str | bytes, remote_path: str) -> None:
        self.files[remote_path] = file_content.encode() if isinstance(file_content, str) else file_content

    def read_file(self, remote_path: str) -> bytes:
        return self.files[remote_path]

    def start_compose(self, compose_yaml: bytes) -> str:
        return "compose-container"

    def run_service_bash(
        self,
        service: str,
        command: str,
        timeout: float = 60.0,
        env: dict[str, str] | None = None,
        user: str | int | None = None,
    ) -> vmvm.VMVMBashResult:
        self.commands.append((f"{service}:{command}", timeout))
        return self.result

    def read_service_file(self, service: str, remote_path: str) -> bytes:
        return self.files[f"{service}:{remote_path}"]

    def run_root_bash(self, command: str, timeout: float = 60.0) -> vmvm.VMVMBashResult:
        self.commands.append((f"root:{command}", timeout))
        return self.result

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

    def get_debugging_info(self) -> dict[str, object]:
        return {"container_id": "abc123"}


async def test_vmvm_runtime_lifecycle(monkeypatch) -> None:
    backend = FakeBackend()
    monkeypatch.setattr(vmvm, "create_backend", lambda config, cancel_event=None: backend)
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


async def test_vmvm_runtime_cancellation_joins_command_before_reuse(monkeypatch) -> None:
    backend = FakeBackend()
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    events: list[str] = []

    def run_bash(command: str, timeout: float = 60.0) -> vmvm.VMVMBashResult:
        backend.commands.append((command, timeout))
        if "score" in command:
            events.append("follow-up")
            assert finished.is_set()
            return backend.result
        if "run-agent" not in command:
            return backend.result
        events.append("agent-started")
        started.set()
        assert release.wait(timeout=2)
        events.append("agent-finished")
        finished.set()
        return backend.result

    def cancel_active_command(timeout: float) -> bool:
        events.append("cancel")
        release.set()
        return finished.wait(timeout=timeout)

    backend.run_bash = run_bash
    backend.cancel_active_command = cancel_active_command
    monkeypatch.setattr(vmvm, "create_backend", lambda config, cancel_event=None: backend)
    monkeypatch.setattr(vmvm, "COMMAND_CANCELLATION_GRACE_SECONDS", 1.0)
    runtime = VMVMRuntime(VMVMConfig(session_timeout=10))
    await runtime.start()

    operation = asyncio.create_task(runtime.run_program(["run-agent"], {}))
    assert await asyncio.to_thread(started.wait, 1)
    operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(operation, timeout=2)

    assert finished.is_set()
    runtime.ensure_usable()
    follow_up = await runtime.run(["score"], {})
    assert follow_up.exit_code == 0
    assert events == ["agent-started", "cancel", "agent-finished", "follow-up"]
    await runtime.stop()


async def test_vmvm_runtime_cancellation_during_transport_recovery_joins_before_reuse(
    monkeypatch,
) -> None:
    backend = FakeBackend()
    recovery_started = threading.Event()
    recovery_release = threading.Event()
    recovery_finished = threading.Event()
    events: list[str] = []

    def recover_last() -> vmvm.VMVMBashResult:
        events.append("recovery-started")
        recovery_started.set()
        assert recovery_release.wait(timeout=2)
        events.append("recovery-finished")
        recovery_finished.set()
        return {
            "status": "error",
            "output": "",
            "error_type": "exit",
            "exit_code": 130,
        }

    def cancel_active_command(timeout: float) -> bool:
        events.append("cancel")
        recovery_release.set()
        return recovery_finished.wait(timeout=timeout)

    backend.recover_last = recover_last
    backend.cancel_active_command = cancel_active_command
    monkeypatch.setattr(vmvm, "create_backend", lambda config, cancel_event=None: backend)
    monkeypatch.setattr(vmvm, "COMMAND_CANCELLATION_GRACE_SECONDS", 1.0)
    runtime = VMVMRuntime(VMVMConfig(session_timeout=10))
    await runtime.start()
    backend.result = {
        "status": "error",
        "output": "",
        "error_type": "broken_pipe",
        "exit_code": -1,
    }

    operation = asyncio.create_task(runtime.run_program(["run-agent"], {}))
    assert await asyncio.to_thread(recovery_started.wait, 1)
    operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(operation, timeout=2)

    runtime.ensure_usable()
    backend.result = {
        "status": "success",
        "output": "ok",
        "error_type": "none",
        "exit_code": 0,
    }
    follow_up = await runtime.run(["score"], {})
    assert follow_up.exit_code == 0
    assert events == ["recovery-started", "cancel", "recovery-finished"]
    assert backend.restart_calls == 1
    await runtime.stop()


async def test_vmvm_runtime_cancellation_failure_destroys_before_return(monkeypatch) -> None:
    backend = FakeBackend()
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def run_bash(command: str, timeout: float = 60.0) -> vmvm.VMVMBashResult:
        backend.commands.append((command, timeout))
        if "run-agent" not in command:
            return backend.result
        started.set()
        assert release.wait(timeout=2)
        finished.set()
        return backend.result

    def cancel_active_command(timeout: float) -> bool:
        raise RuntimeError("interrupt failed")

    def destroy() -> None:
        backend.destroyed = True
        release.set()
        raise RuntimeError("teardown failed")

    backend.run_bash = run_bash
    backend.cancel_active_command = cancel_active_command
    backend.destroy = destroy
    monkeypatch.setattr(vmvm, "create_backend", lambda config, cancel_event=None: backend)
    monkeypatch.setattr(vmvm, "COMMAND_CANCELLATION_GRACE_SECONDS", 1.0)
    runtime = VMVMRuntime(VMVMConfig(session_timeout=10))
    await runtime.start()

    operation = asyncio.create_task(runtime.run_program(["run-agent"], {}))
    assert await asyncio.to_thread(started.wait, 1)
    operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(operation, timeout=2)

    assert backend.destroyed is True
    assert finished.is_set()
    assert runtime._backend is None
    with pytest.raises(SandboxError, match="could not restore a safe runtime"):
        runtime.ensure_usable()


async def test_vmvm_runtime_cancellation_has_one_total_deadline(monkeypatch) -> None:
    backend = FakeBackend()
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def run_bash(command: str, timeout: float = 60.0) -> vmvm.VMVMBashResult:
        backend.commands.append((command, timeout))
        if "run-agent" not in command:
            return backend.result
        started.set()
        assert release.wait(timeout=2)
        finished.set()
        return backend.result

    def cancel_active_command(timeout: float) -> bool:
        time.sleep(timeout * 0.9)
        return False

    def destroy() -> None:
        time.sleep(0.18)
        backend.destroyed = True
        release.set()

    backend.run_bash = run_bash
    backend.cancel_active_command = cancel_active_command
    backend.destroy = destroy
    monkeypatch.setattr(vmvm, "create_backend", lambda config, cancel_event=None: backend)
    monkeypatch.setattr(vmvm, "COMMAND_CANCELLATION_GRACE_SECONDS", 0.4)
    runtime = VMVMRuntime(VMVMConfig(session_timeout=10))
    await runtime.start()

    operation = asyncio.create_task(runtime.run_program(["run-agent"], {}))
    assert await asyncio.to_thread(started.wait, 1)
    begin = time.monotonic()
    operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await operation
    elapsed = time.monotonic() - begin

    assert elapsed < 0.48
    assert backend.destroyed is True
    assert finished.is_set()
    with pytest.raises(SandboxError, match="could not restore a safe runtime"):
        runtime.ensure_usable()


async def test_vmvm_runtime_retains_quarantined_backend_until_workers_finish(
    monkeypatch,
) -> None:
    backend = FakeBackend()
    command_started = threading.Event()
    command_release = threading.Event()
    destroy_started = threading.Event()
    destroy_release = threading.Event()

    def run_bash(command: str, timeout: float = 60.0) -> vmvm.VMVMBashResult:
        backend.commands.append((command, timeout))
        if "run-agent" not in command:
            return backend.result
        command_started.set()
        assert command_release.wait(timeout=2)
        return backend.result

    def cancel_active_command(timeout: float) -> bool:
        return False

    def destroy() -> None:
        destroy_started.set()
        assert destroy_release.wait(timeout=2)
        backend.destroyed = True

    backend.run_bash = run_bash
    backend.cancel_active_command = cancel_active_command
    backend.destroy = destroy
    monkeypatch.setattr(vmvm, "create_backend", lambda config, cancel_event=None: backend)
    monkeypatch.setattr(vmvm, "COMMAND_CANCELLATION_GRACE_SECONDS", 0.05)
    runtime = VMVMRuntime(VMVMConfig(session_timeout=10))
    await runtime.start()

    operation = asyncio.create_task(runtime.run_program(["run-agent"], {}))
    async with asyncio.timeout(1):
        while not command_started.is_set():
            await asyncio.sleep(0.001)
    operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(operation, timeout=1)

    assert destroy_started.is_set()
    assert runtime._backend is None
    assert runtime._quarantined_backend is backend
    assert runtime._quarantine_tasks
    begin = time.monotonic()
    with pytest.raises(SandboxError, match="teardown exceeded"):
        await runtime.stop()
    assert time.monotonic() - begin < 0.1
    assert runtime._quarantined_backend is backend

    command_release.set()
    destroy_release.set()
    async with asyncio.timeout(2):
        while runtime._quarantine_tasks:
            await asyncio.sleep(0.001)
    assert runtime._quarantined_backend is None
    assert not runtime._quarantine_tasks
    assert not runtime._worker_tasks
    assert not runtime._worker_threads


async def test_vmvm_runtime_cancellation_bypasses_saturated_default_executor(
    monkeypatch,
) -> None:
    backend = FakeBackend()
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    control_started = threading.Event()

    def run_bash(command: str, timeout: float = 60.0) -> vmvm.VMVMBashResult:
        backend.commands.append((command, timeout))
        if "run-agent" not in command:
            return backend.result
        started.set()
        assert release.wait(timeout=2)
        finished.set()
        return backend.result

    def cancel_active_command(timeout: float) -> bool:
        control_started.set()
        release.set()
        return finished.wait(timeout=timeout)

    backend.run_bash = run_bash
    backend.cancel_active_command = cancel_active_command
    monkeypatch.setattr(vmvm, "create_backend", lambda config, cancel_event=None: backend)
    monkeypatch.setattr(vmvm, "COMMAND_CANCELLATION_GRACE_SECONDS", 1.0)
    runtime = VMVMRuntime(VMVMConfig(session_timeout=10))
    await runtime.start()

    loop = asyncio.get_running_loop()
    await asyncio.to_thread(lambda: None)
    original_executor = loop._default_executor
    assert original_executor is not None
    saturated_executor = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(saturated_executor)
    executor_started = threading.Event()
    executor_release = threading.Event()
    executor_blocker = loop.run_in_executor(
        None,
        lambda: (executor_started.set(), executor_release.wait(timeout=2)),
    )
    try:
        async with asyncio.timeout(1):
            while not executor_started.is_set():
                await asyncio.sleep(0.001)
        operation = asyncio.create_task(runtime.run_program(["run-agent"], {}))
        async with asyncio.timeout(1):
            while not started.is_set():
                await asyncio.sleep(0.001)
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(operation, timeout=2)
    finally:
        release.set()
        executor_release.set()
        await executor_blocker
        loop.set_default_executor(original_executor)
        saturated_executor.shutdown(wait=True)

    assert control_started.is_set()
    assert finished.is_set()
    assert not runtime._worker_tasks
    assert not runtime._worker_threads
    await runtime.stop()
    assert not runtime._worker_tasks
    assert not runtime._worker_threads


async def test_vmvm_runtime_simultaneous_cancellations_have_reserved_control_capacity(
    monkeypatch,
) -> None:
    count = 6
    barrier = threading.Barrier(count)
    backends = [FakeBackend() for _ in range(count)]
    runtimes = [VMVMRuntime(VMVMConfig(session_timeout=10)) for _ in range(count)]
    started = [threading.Event() for _ in range(count)]
    release = [threading.Event() for _ in range(count)]
    finished = [threading.Event() for _ in range(count)]

    for index, (runtime, backend) in enumerate(zip(runtimes, backends, strict=True)):
        runtime._backend = backend

        def run_bash(
            command: str,
            timeout: float = 60.0,
            *,
            index: int = index,
            backend: FakeBackend = backend,
        ) -> vmvm.VMVMBashResult:
            backend.commands.append((command, timeout))
            started[index].set()
            assert release[index].wait(timeout=2)
            finished[index].set()
            return backend.result

        def cancel_active_command(timeout: float, *, index: int = index) -> bool:
            barrier.wait(timeout=timeout)
            release[index].set()
            return finished[index].wait(timeout=timeout)

        backend.run_bash = run_bash
        backend.cancel_active_command = cancel_active_command

    monkeypatch.setattr(vmvm, "COMMAND_CANCELLATION_GRACE_SECONDS", 1.0)
    loop = asyncio.get_running_loop()
    await asyncio.to_thread(lambda: None)
    original_executor = loop._default_executor
    assert original_executor is not None
    saturated_executor = ThreadPoolExecutor(max_workers=count)
    loop.set_default_executor(saturated_executor)
    executor_started = [threading.Event() for _ in range(count)]
    executor_release = threading.Event()
    executor_blockers = [
        loop.run_in_executor(
            None,
            lambda event=event: (event.set(), executor_release.wait(timeout=2)),
        )
        for event in executor_started
    ]
    try:
        async with asyncio.timeout(1):
            while not all(event.is_set() for event in executor_started):
                await asyncio.sleep(0.001)
        operations = [asyncio.create_task(runtime.run_program(["run-agent"], {})) for runtime in runtimes]
        async with asyncio.timeout(1):
            while not all(event.is_set() for event in started):
                await asyncio.sleep(0.001)
        for operation in operations:
            operation.cancel()
        results = await asyncio.wait_for(
            asyncio.gather(*operations, return_exceptions=True),
            timeout=2,
        )
    finally:
        for event in release:
            event.set()
        executor_release.set()
        await asyncio.gather(*executor_blockers)
        loop.set_default_executor(original_executor)
        saturated_executor.shutdown(wait=True)

    assert all(isinstance(result, asyncio.CancelledError) for result in results)
    assert all(event.is_set() for event in finished)
    assert all(not runtime._worker_tasks for runtime in runtimes)
    assert all(not runtime._worker_threads for runtime in runtimes)
    await asyncio.gather(*(runtime.stop() for runtime in runtimes))
    assert all(not runtime._worker_tasks for runtime in runtimes)
    assert all(not runtime._worker_threads for runtime in runtimes)


async def test_vmvm_runtime_cancelled_provisioning_destroys_late_backend(monkeypatch) -> None:
    backend = FakeBackend()
    create_started = threading.Event()
    create_release = threading.Event()

    def create(config, cancel_event=None) -> FakeBackend:
        create_started.set()
        assert create_release.wait(timeout=2)
        return backend

    monkeypatch.setattr(vmvm, "create_backend", create)
    monkeypatch.setattr(vmvm, "COMMAND_CANCELLATION_GRACE_SECONDS", 1.0)
    runtime = VMVMRuntime(VMVMConfig(session_timeout=10))
    starting = asyncio.create_task(runtime.start())
    async with asyncio.timeout(1):
        while not create_started.is_set():
            await asyncio.sleep(0.001)

    starting.cancel()
    await asyncio.sleep(0.01)
    assert not starting.done()
    create_release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(starting, timeout=2)

    assert backend.destroyed is True
    assert runtime._backend is None
    assert runtime._quarantined_backend is None
    assert not runtime._quarantine_tasks
    assert not runtime._worker_tasks
    assert not runtime._worker_threads


async def test_vmvm_runtime_over_grace_provisioning_thread_retains_cleanup_ownership(
    monkeypatch,
) -> None:
    backend = FakeBackend()
    create_started = threading.Event()
    create_release = threading.Event()

    def create(config, cancel_event=None) -> FakeBackend:
        create_started.set()
        assert create_release.wait(timeout=2)
        return backend

    monkeypatch.setattr(vmvm, "create_backend", create)
    monkeypatch.setattr(vmvm, "COMMAND_CANCELLATION_GRACE_SECONDS", 0.02)
    runtime = VMVMRuntime(VMVMConfig(session_timeout=10))
    starting = asyncio.create_task(runtime.start())
    async with asyncio.timeout(1):
        while not create_started.is_set():
            await asyncio.sleep(0.001)

    begin = time.monotonic()
    starting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(starting, timeout=1)
    assert time.monotonic() - begin < 0.1
    assert backend.destroyed is False
    assert len(runtime._worker_threads) == 1
    assert next(iter(runtime._worker_threads)).daemon is True
    # The loop remains free to finish unrelated evaluator work while the late
    # worker retains synchronous cleanup ownership.
    await asyncio.wait_for(asyncio.sleep(0), timeout=0.1)

    create_release.set()
    async with asyncio.timeout(1):
        while not backend.destroyed or runtime._worker_tasks or runtime._worker_threads:
            await asyncio.sleep(0.001)
    assert runtime._backend is None


async def test_vmvm_runtime_provisioning_cancel_signal_unblocks_factory(monkeypatch) -> None:
    started = threading.Event()
    cancellation_seen = threading.Event()

    def create(config, cancel_event) -> FakeBackend:
        started.set()
        assert cancel_event.wait(timeout=2)
        cancellation_seen.set()
        raise SandboxError("provisioning cancelled")

    monkeypatch.setattr(vmvm, "create_backend", create)
    monkeypatch.setattr(vmvm, "COMMAND_CANCELLATION_GRACE_SECONDS", 1.0)
    runtime = VMVMRuntime(VMVMConfig(session_timeout=10))
    starting = asyncio.create_task(runtime.start())
    async with asyncio.timeout(1):
        while not started.is_set():
            await asyncio.sleep(0.001)

    starting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(starting, timeout=1)

    assert cancellation_seen.is_set()
    assert not runtime._worker_tasks
    assert not runtime._worker_threads


async def test_vmvm_runtime_cancelled_initial_command_drains_and_destroys(monkeypatch) -> None:
    backend = FakeBackend()
    mkdir_started = threading.Event()
    mkdir_release = threading.Event()
    mkdir_finished = threading.Event()

    def run_bash(command: str, timeout: float = 60.0) -> vmvm.VMVMBashResult:
        backend.commands.append((command, timeout))
        mkdir_started.set()
        assert mkdir_release.wait(timeout=2)
        mkdir_finished.set()
        return backend.result

    def cancel_active_command(timeout: float) -> bool:
        mkdir_release.set()
        return mkdir_finished.wait(timeout=timeout)

    backend.run_bash = run_bash
    backend.cancel_active_command = cancel_active_command
    monkeypatch.setattr(vmvm, "create_backend", lambda config, cancel_event=None: backend)
    monkeypatch.setattr(vmvm, "COMMAND_CANCELLATION_GRACE_SECONDS", 1.0)
    runtime = VMVMRuntime(VMVMConfig(session_timeout=10))
    starting = asyncio.create_task(runtime.start())
    async with asyncio.timeout(1):
        while not mkdir_started.is_set():
            await asyncio.sleep(0.001)

    starting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(starting, timeout=2)

    assert mkdir_finished.is_set()
    assert backend.destroyed is True
    assert runtime._backend is None
    assert runtime._quarantined_backend is None
    assert not runtime._worker_tasks
    assert not runtime._worker_threads


async def test_vmvm_runtime_final_destroy_removes_resource_created_after_first_destroy(
    monkeypatch,
) -> None:
    backend = FakeBackend()
    operation_started = threading.Event()
    first_destroy_finished = threading.Event()
    late_resource = threading.Event()
    destroy_calls = 0

    def start_compose(compose_yaml: bytes) -> str:
        operation_started.set()
        assert first_destroy_finished.wait(timeout=2)
        late_resource.set()
        return "late-container"

    def destroy() -> None:
        nonlocal destroy_calls
        destroy_calls += 1
        if destroy_calls == 1:
            first_destroy_finished.set()
        else:
            late_resource.clear()
        backend.destroyed = True

    backend.start_compose = start_compose
    backend.destroy = destroy
    monkeypatch.setattr(vmvm, "create_backend", lambda config, cancel_event=None: backend)
    monkeypatch.setattr(vmvm, "COMMAND_CANCELLATION_GRACE_SECONDS", 1.0)
    runtime = VMVMRuntime(VMVMConfig(session_timeout=10))
    await runtime.start()

    operation = asyncio.create_task(runtime.start_compose(b"services: {}"))
    async with asyncio.timeout(1):
        while not operation_started.is_set():
            await asyncio.sleep(0.001)
    operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(operation, timeout=2)

    assert destroy_calls >= 2
    assert not late_resource.is_set()
    assert runtime._backend is None
    assert runtime._quarantined_backend is None
    assert not runtime._worker_tasks
    assert not runtime._worker_threads


async def test_vmvm_runtime_cancelled_write_drains_before_invalidating(monkeypatch) -> None:
    backend = FakeBackend()
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    events: list[str] = []

    def transfer_file(file_content: str | bytes, remote_path: str) -> None:
        events.append("write-started")
        started.set()
        assert release.wait(timeout=2)
        events.append("write-finished")
        finished.set()

    def destroy() -> None:
        events.append("destroy")
        backend.destroyed = True
        release.set()

    backend.transfer_file = transfer_file
    backend.destroy = destroy
    monkeypatch.setattr(vmvm, "create_backend", lambda config, cancel_event=None: backend)
    monkeypatch.setattr(vmvm, "COMMAND_CANCELLATION_GRACE_SECONDS", 1.0)
    runtime = VMVMRuntime(VMVMConfig(session_timeout=10))
    await runtime.start()

    operation = asyncio.create_task(runtime.write("mcp.json", b"{}"))
    assert await asyncio.to_thread(started.wait, 1)
    operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(operation, timeout=2)

    assert finished.is_set()
    assert backend.destroyed is True
    assert runtime._backend is None
    assert events == ["write-started", "destroy", "write-finished", "destroy"]
    with pytest.raises(SandboxError, match="file write was cancelled"):
        runtime.ensure_usable()


@pytest.mark.parametrize(
    "operation",
    ["read", "root", "service", "service-read", "compose"],
)
async def test_vmvm_runtime_cancelled_backend_operations_fail_closed(
    monkeypatch,
    operation: str,
) -> None:
    backend = FakeBackend()
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocked(*args, **kwargs):
        started.set()
        assert release.wait(timeout=2)
        finished.set()
        if operation in ("root", "service"):
            return backend.result
        if operation == "service-read":
            return b"payload"
        if operation == "compose":
            return "compose-container"
        return b"payload"

    target = {
        "read": "read_file",
        "root": "run_root_bash",
        "service": "run_service_bash",
        "service-read": "read_service_file",
        "compose": "start_compose",
    }[operation]
    setattr(backend, target, blocked)

    def destroy() -> None:
        backend.destroyed = True
        release.set()

    backend.destroy = destroy
    monkeypatch.setattr(vmvm, "create_backend", lambda config, cancel_event=None: backend)
    monkeypatch.setattr(vmvm, "COMMAND_CANCELLATION_GRACE_SECONDS", 1.0)
    runtime = VMVMRuntime(VMVMConfig(session_timeout=10))
    await runtime.start()

    operations = {
        "read": lambda: runtime.read("artifact.bin"),
        "root": lambda: runtime.run_root("true"),
        "service": lambda: runtime.run_service("sidecar", ["true"], {}),
        "service-read": lambda: runtime.read_service("sidecar", "/artifact.bin"),
        "compose": lambda: runtime.start_compose(b"services: {}"),
    }
    task = asyncio.create_task(operations[operation]())
    assert await asyncio.to_thread(started.wait, 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2)

    assert finished.is_set()
    assert backend.destroyed is True
    assert runtime._backend is None
    with pytest.raises(SandboxError, match="cancelled"):
        runtime.ensure_usable()


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
    monkeypatch.setattr(vmvm, "create_backend", lambda config, cancel_event=None: backend)
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
    monkeypatch.setattr(vmvm, "create_backend", lambda config, cancel_event=None: backend)
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
    monkeypatch.setattr(vmvm, "create_backend", lambda config, cancel_event=None: backend)
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
    monkeypatch.setattr(vmvm, "create_backend", lambda config, cancel_event=None: backend)
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
    monkeypatch.setattr(vmvm, "create_backend", lambda config, cancel_event=None: backend)
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
    monkeypatch.setattr(vmvm, "create_backend", lambda config, cancel_event=None: backend)
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
    monkeypatch.setattr(vmvm, "create_backend", lambda config, cancel_event=None: backend)
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
    monkeypatch.setattr(vmvm, "create_backend", lambda config, cancel_event=None: backend)
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
    monkeypatch.setattr(vmvm, "create_backend", lambda config, cancel_event=None: backend)
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
