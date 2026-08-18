import pytest
from verifiers.v1.errors import SandboxError
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
        self.result: vmvm.VMVMBashResult = {
            "status": "success",
            "output": "ok",
            "error_type": "none",
            "exit_code": 0,
        }

    def run_bash(self, command: str, timeout: float = 60.0) -> vmvm.VMVMBashResult:
        self.commands.append((command, timeout))
        return self.result

    def transfer_file(self, file_content: str | bytes, remote_path: str) -> None:
        self.files[remote_path] = file_content.encode() if isinstance(file_content, str) else file_content

    def read_file(self, remote_path: str) -> bytes:
        return self.files[remote_path]

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


async def test_vmvm_runtime_surfaces_transport_failure(monkeypatch) -> None:
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

    with pytest.raises(SandboxError, match="broken_pipe"):
        await runtime.run(["true"], {})

    await runtime.stop()
