from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import (
    SandoqConfig,
    SandoqRuntime,
    make_runtime,
    runtime_has_instance_host_endpoint,
    runtime_is_local,
    sandoq,
)


class FakeSandoqClient:
    def __init__(self) -> None:
        self.request = None
        self.commands: list[tuple[str, str | None, dict[str, str], int]] = []
        self.files: dict[str, bytes] = {}
        self.deleted: list[str] = []
        self.closed = False
        self.error: Exception | None = None

    async def create(self, request):
        self.request = request
        return SimpleNamespace(id="assignment-123")

    async def wait_for_creation(self, sandbox_id: str) -> None:
        assert sandbox_id == "assignment-123"

    async def execute_command(
        self,
        sandbox_id: str,
        command: str,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ):
        assert sandbox_id == "assignment-123"
        self.commands.append((command, working_dir, env or {}, timeout or 0))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(exit_code=0, stdout="ok", stderr="")

    async def upload_bytes(
        self,
        sandbox_id: str,
        path: str,
        data: bytes,
        **kwargs,
    ) -> None:
        assert sandbox_id == "assignment-123"
        self.files[path] = data

    async def download_file(
        self,
        sandbox_id: str,
        path: str,
        local_path: str,
        **kwargs,
    ) -> None:
        assert sandbox_id == "assignment-123"
        Path(local_path).write_bytes(self.files[path])

    async def delete(self, sandbox_id: str) -> None:
        self.deleted.append(sandbox_id)

    async def aclose(self) -> None:
        self.closed = True


async def test_sandoq_runtime_lifecycle(monkeypatch) -> None:
    client = FakeSandoqClient()
    opened_tunnels: list[int] = []

    @asynccontextmanager
    async def fake_host_endpoint(port: int, *, name: str | None = None):
        opened_tunnels.append(port)
        assert name is None
        yield f"https://relay.example/{port}"

    monkeypatch.setattr(sandoq, "create_client", lambda config: client)
    monkeypatch.setattr(sandoq, "modal_host_endpoint", fake_host_endpoint)
    config = SandoqConfig(
        image="swebench/image",
        workdir="/testbed",
        session_timeout=123,
    )
    runtime = make_runtime(config, "rollout-1")

    assert isinstance(runtime, SandoqRuntime)
    assert runtime_is_local(config) is False
    assert runtime_has_instance_host_endpoint(config) is True
    await runtime.start()
    assert runtime.descriptor == "assignment-123"
    assert client.request.docker_image == "swebench/image"
    assert client.request.environment_vars == {"OCI_EXPECTED_WORKDIR": "/testbed"}

    result = await runtime.run(["sh", "-c", "printf ok"], {"MESSAGE": "value with spaces"})
    assert result.exit_code == 0
    assert result.stdout == "ok"
    assert client.commands[:2] == [
        ("mkdir -p /testbed", "/", {}, 123),
        ("sh -c 'printf ok'", "/testbed", {"MESSAGE": "value with spaces"}, 123),
    ]

    await runtime.write("artifacts/payload.bin", b"\x00\x01")
    assert await runtime.read("artifacts/payload.bin") == b"\x00\x01"
    assert client.files == {"/testbed/artifacts/payload.bin": b"\x00\x01"}

    async with runtime.host_endpoint(4321) as url:
        assert url == "https://relay.example/4321"
    assert opened_tunnels == [4321]

    await runtime.stop()
    assert client.deleted == ["assignment-123"]
    assert client.closed is True


async def test_sandoq_runtime_never_replays_uncertain_command(monkeypatch) -> None:
    client = FakeSandoqClient()
    monkeypatch.setattr(sandoq, "create_client", lambda config: client)
    runtime = SandoqRuntime(SandoqConfig(session_timeout=10))
    await runtime.start()
    client.commands.clear()
    client.error = RuntimeError("transport unavailable after request")

    with pytest.raises(SandboxError, match="transport unavailable"):
        await runtime.run(["touch", "ambiguous-side-effect"], {})

    assert len(client.commands) == 1
    client.error = None
    await runtime.stop()


async def test_sandoq_runtime_surfaces_unknown_gateway_result(monkeypatch) -> None:
    client = FakeSandoqClient()
    monkeypatch.setattr(sandoq, "create_client", lambda config: client)
    runtime = SandoqRuntime(SandoqConfig(session_timeout=10))
    await runtime.start()
    client.commands.clear()

    async def unknown_result(*args, **kwargs):
        client.commands.append((args[1], kwargs.get("working_dir"), kwargs.get("env", {}), 10))
        return SimpleNamespace(
            exit_code=75,
            stdout="",
            stderr=(
                "OCI runner gateway temporarily unavailable; command execution status "
                "is unknown and the command was not replayed"
            ),
        )

    client.execute_command = unknown_result
    with pytest.raises(SandboxError, match="status is unknown"):
        await runtime.run(["touch", "ambiguous-side-effect"], {})

    assert len(client.commands) == 1
    await runtime.stop()
