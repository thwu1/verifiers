import asyncio
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
        self.files[remote_path] = (
            file_content.encode() if isinstance(file_content, str) else file_content
        )

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
    assert runtime.cleanup_must_succeed is True
    assert runtime_is_local(config) is False
    assert runtime_has_instance_host_endpoint(config) is True
    await runtime.start()
    assert runtime.descriptor == "abc123"

    result = await runtime.run(
        ["sh", "-c", "printf ok"], {"MESSAGE": "value with spaces"}
    )
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


async def test_vmvm_runtime_propagates_destroy_failure(monkeypatch) -> None:
    backend = FakeBackend()
    monkeypatch.setattr(vmvm, "create_backend", lambda config: backend)
    runtime = VMVMRuntime(VMVMConfig(session_timeout=10))
    await runtime.start()

    def fail_destroy() -> None:
        raise RuntimeError("release receipt unavailable")

    backend.destroy = fail_destroy
    with pytest.raises(
        SandboxError, match="VMVM cleanup failed: release receipt unavailable"
    ):
        await runtime.stop()


async def test_vmvm_runtime_stop_finishes_cleanup_under_repeated_cancellation(
    monkeypatch,
) -> None:
    backend = FakeBackend()
    entered = threading.Event()
    release = threading.Event()

    def blocking_destroy() -> None:
        entered.set()
        assert release.wait(timeout=5)
        backend.destroyed = True

    backend.destroy = blocking_destroy
    monkeypatch.setattr(vmvm, "create_backend", lambda config: backend)
    runtime = VMVMRuntime(VMVMConfig(session_timeout=10))
    await runtime.start()

    stopping = asyncio.create_task(runtime.stop())
    assert await asyncio.to_thread(entered.wait, 5)
    stopping.cancel()
    await asyncio.sleep(0)
    assert not stopping.done()
    stopping.cancel()
    await asyncio.sleep(0)
    assert not stopping.done()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await stopping
    assert backend.destroyed is True


async def test_vmvm_runtime_cancelled_start_destroys_late_backend(monkeypatch) -> None:
    backend = FakeBackend()
    entered = threading.Event()
    release = threading.Event()

    def delayed_create(_config) -> FakeBackend:
        entered.set()
        assert release.wait(timeout=5)
        return backend

    monkeypatch.setattr(vmvm, "create_backend", delayed_create)
    runtime = VMVMRuntime(VMVMConfig(session_timeout=10))
    starting = asyncio.create_task(runtime.start())
    assert await asyncio.to_thread(entered.wait, 5)
    starting.cancel()
    await asyncio.sleep(0)
    assert not starting.done()
    starting.cancel()
    await asyncio.sleep(0)
    assert not starting.done()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await starting
    assert backend.destroyed is True
    assert runtime._backend is None


async def test_vmvm_runtime_cancelled_stop_preserves_cleanup_failure(
    monkeypatch,
) -> None:
    backend = FakeBackend()
    entered = threading.Event()
    release = threading.Event()

    def blocking_failed_destroy() -> None:
        entered.set()
        assert release.wait(timeout=5)
        raise RuntimeError("release failed")

    backend.destroy = blocking_failed_destroy
    monkeypatch.setattr(vmvm, "create_backend", lambda config: backend)
    runtime = VMVMRuntime(VMVMConfig(session_timeout=10))
    await runtime.start()

    stopping = asyncio.create_task(runtime.stop())
    assert await asyncio.to_thread(entered.wait, 5)
    stopping.cancel()
    await asyncio.sleep(0)
    stopping.cancel()
    await asyncio.sleep(0)
    release.set()

    with pytest.raises(asyncio.CancelledError) as caught:
        await stopping
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert "release failed" in "\n".join(caught.value.__notes__)


async def test_vmvm_runtime_cancelled_constructor_preserves_failure(
    monkeypatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    def delayed_failed_create(_config) -> FakeBackend:
        entered.set()
        assert release.wait(timeout=5)
        raise RuntimeError("constructor failed")

    monkeypatch.setattr(vmvm, "create_backend", delayed_failed_create)
    runtime = VMVMRuntime(VMVMConfig(session_timeout=10))
    starting = asyncio.create_task(runtime.start())
    assert await asyncio.to_thread(entered.wait, 5)
    starting.cancel()
    await asyncio.sleep(0)
    starting.cancel()
    await asyncio.sleep(0)
    release.set()

    with pytest.raises(asyncio.CancelledError) as caught:
        await starting
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert "constructor failed" in "\n".join(caught.value.__notes__)


async def test_vmvm_runtime_preserves_provisioning_and_cleanup_failures(
    monkeypatch,
) -> None:
    backend = FakeBackend()
    backend.result = {
        "status": "error",
        "output": "mkdir failed",
        "error_type": "exit",
        "exit_code": 1,
    }

    def fail_destroy() -> None:
        raise RuntimeError("release failed")

    backend.destroy = fail_destroy
    monkeypatch.setattr(vmvm, "create_backend", lambda config: backend)
    runtime = VMVMRuntime(VMVMConfig(session_timeout=10))

    with pytest.raises(
        SandboxError, match="provisioning failed and cleanup failed"
    ) as caught:
        await runtime.start()

    assert "mkdir failed" in str(caught.value)
    assert "release failed" in str(caught.value)
    assert isinstance(caught.value.__cause__, RuntimeError)


async def test_vmvm_runtime_malformed_provisioning_result_destroys_backend(
    monkeypatch,
) -> None:
    backend = FakeBackend()
    backend.run_bash = lambda *_args, **_kwargs: None
    monkeypatch.setattr(vmvm, "create_backend", lambda config: backend)
    runtime = VMVMRuntime(VMVMConfig(session_timeout=10))

    with pytest.raises(SandboxError, match="VMVM provisioning failed"):
        await runtime.start()

    assert backend.destroyed is True
    assert runtime._backend is None


async def test_vmvm_runtime_debug_identity_failure_destroys_backend(
    monkeypatch,
) -> None:
    backend = FakeBackend()

    def fail_debugging_info() -> dict[str, object]:
        raise RuntimeError("debug identity failed")

    backend.get_debugging_info = fail_debugging_info
    monkeypatch.setattr(vmvm, "create_backend", lambda config: backend)
    runtime = VMVMRuntime(VMVMConfig(session_timeout=10))

    with pytest.raises(SandboxError, match="debug identity failed"):
        await runtime.start()

    assert backend.destroyed is True
    assert runtime._backend is None


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
    isolated_commands = [
        command for command, _ in backend.commands if "NO_PROXY=*" in command
    ]
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
    probes = [
        command
        for command, _ in backend.commands
        if "socket.create_connection" in command
    ]
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


async def test_vmvm_runtime_recovers_in_flight_command_exactly_once(
    monkeypatch,
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


async def test_vmvm_runtime_recovers_across_repeated_transport_drops(
    monkeypatch,
) -> None:
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
