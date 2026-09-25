import asyncio
import os
import stat
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from verifiers.v1.errors import SandboxError, TunnelError
from verifiers.v1.runtimes import (
    ProgramResult,
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
        self.background_commands: list[tuple[str, str | None, dict[str, str], int, int]] = []
        self.files: dict[str, bytes] = {}
        self.deleted: list[str] = []
        self.closed = False
        self.error: Exception | None = None
        self.close_error: Exception | None = None
        self.delete_error: Exception | None = None
        self.wait_error: Exception | None = None

    async def create(self, request):
        self.request = request
        return SimpleNamespace(id="assignment-123")

    async def wait_for_creation(self, sandbox_id: str) -> None:
        assert sandbox_id == "assignment-123"
        if self.wait_error is not None:
            raise self.wait_error

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

    async def run_background_job(
        self,
        sandbox_id: str,
        command: str,
        timeout: int | None = None,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
        poll_interval: int = 3,
    ):
        assert sandbox_id == "assignment-123"
        self.background_commands.append((command, working_dir, env or {}, timeout or 0, poll_interval))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(exit_code=0, stdout="program-ok", stderr="")

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

    async def delete(self, sandbox_id: str):
        if self.delete_error is not None:
            raise self.delete_error
        self.deleted.append(sandbox_id)
        return {"status": "deleted", "verified_http_status": 404}

    async def aclose(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


@pytest.mark.parametrize(
    ("environment", "task_network"),
    [
        ("oci-runner-firecracker", "host"),
        ("oci-runner-firecracker-small", "none"),
        ("oci-runner-firecracker-tunnel-pull", "none"),
    ],
)
def test_sandoq_no_network_rejects_unapproved_precreate_config(
    monkeypatch, environment: str, task_network: str
) -> None:
    constructed = []
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda _path: SimpleNamespace(
            st_mode=stat.S_IFREG | 0o600,
            st_uid=os.getuid(),
            st_nlink=1,
        ),
    )
    monkeypatch.setattr(Path, "is_symlink", lambda _path: False)
    module = ModuleType("sandoq_provider.oci_client")
    module.get_oci_config = lambda: SimpleNamespace(
        environment=environment,
        task_network=task_network,
        token_file=Path("/opaque/token"),
        allow_dockerhub_fallback=False,
        create_deadline_s=1800,
        pull_timeout_s=1200,
        pull_poll_max_errors=10,
        gateway_retry_attempts=15,
        gateway_retry_interval_s=2,
        podman_ignore_chown_errors=True,
        require_resource_limits=True,
        session_reuse=True,
        ecr=SimpleNamespace(
            enabled=True,
            registry="168653207203.dkr.ecr.us-east-2.amazonaws.com",
            region="us-east-2",
            pull_through_prefix="pt_dockerio",
            token_file=Path("/run/secrets/ecr-token"),
        ),
    )
    module.read_token_file = lambda _path: pytest.fail("token read before policy gate")
    module.OCIRunnerAsyncSandboxClient = lambda: constructed.append(True)
    monkeypatch.setitem(sys.modules, "sandoq_provider.oci_client", module)
    secrets_module = ModuleType("sandoq_provider.secrets")
    secrets_module.read_secret_file = lambda *_args: "opaque-secret"
    monkeypatch.setitem(sys.modules, "sandoq_provider.secrets", secrets_module)

    with pytest.raises(SandboxError, match="Sandoq"):
        sandoq.create_client(
            SandoqConfig(
                network_access=False,
                host_tunnel="none",
                expected_environment="oci-runner-firecracker",
                ecr_token_file=Path("/run/secrets/ecr-token"),
            )
        )
    assert constructed == []


def test_sandoq_no_network_accepts_exact_precreate_config(monkeypatch) -> None:
    client = object()
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda _path: SimpleNamespace(
            st_mode=stat.S_IFREG | 0o600,
            st_uid=os.getuid(),
            st_nlink=1,
        ),
    )
    monkeypatch.setattr(Path, "is_symlink", lambda _path: False)
    module = ModuleType("sandoq_provider.oci_client")
    module.get_oci_config = lambda: SimpleNamespace(
        environment="oci-runner-firecracker",
        task_network="none",
        token_file=Path("/opaque/token"),
        allow_dockerhub_fallback=False,
        create_deadline_s=1800,
        pull_timeout_s=1200,
        pull_poll_max_errors=10,
        gateway_retry_attempts=15,
        gateway_retry_interval_s=2,
        podman_ignore_chown_errors=True,
        require_resource_limits=True,
        session_reuse=True,
        ecr=SimpleNamespace(
            enabled=True,
            registry="168653207203.dkr.ecr.us-east-2.amazonaws.com",
            region="us-east-2",
            pull_through_prefix="pt_dockerio",
            token_file=Path("/run/secrets/ecr-token"),
        ),
    )
    module.read_token_file = lambda path: None
    module.OCIRunnerAsyncSandboxClient = lambda: client
    monkeypatch.setitem(sys.modules, "sandoq_provider.oci_client", module)
    secrets_module = ModuleType("sandoq_provider.secrets")
    secrets_module.read_secret_file = lambda *_args: "opaque-secret"
    monkeypatch.setitem(sys.modules, "sandoq_provider.secrets", secrets_module)

    assert (
        sandoq.create_client(
            SandoqConfig(
                network_access=False,
                host_tunnel="none",
                expected_environment="oci-runner-firecracker",
                ecr_token_file=Path("/run/secrets/ecr-token"),
            )
        )
        is client
    )


def test_sandoq_public_host_harness_accepts_official_provider(monkeypatch, tmp_path: Path) -> None:
    client = object()
    ecr_token_file = tmp_path / "ecr-token"
    ecr_token_file.write_text("opaque-token\n")
    ecr_token_file.chmod(0o600)
    reads: list[tuple[str, Path]] = []
    module = ModuleType("sandoq_provider.oci_client")
    module.get_oci_config = lambda: SimpleNamespace(
        environment="oci-runner",
        task_network="none",
        token_file=Path("/opaque/sandoq-token"),
        ecr=SimpleNamespace(
            enabled=True,
            registry="168653207203.dkr.ecr.us-east-2.amazonaws.com",
            region="us-east-2",
            pull_through_prefix="pt_dockerio",
            token_file=ecr_token_file,
        ),
    )
    module.read_token_file = lambda path: reads.append(("sandoq", path))
    module.OCIRunnerAsyncSandboxClient = lambda: client
    monkeypatch.setitem(sys.modules, "sandoq_provider.oci_client", module)
    secrets_module = ModuleType("sandoq_provider.secrets")
    secrets_module.read_secret_file = lambda path, *_args: reads.append(("ecr", path))
    monkeypatch.setitem(sys.modules, "sandoq_provider.secrets", secrets_module)

    config = SandoqConfig(
        network_access=True,
        host_tunnel="none",
        expected_environment="oci-runner",
        ecr_token_file=ecr_token_file,
    )

    assert sandoq.create_client(config) is client
    assert reads == [
        ("ecr", ecr_token_file),
        ("sandoq", Path("/opaque/sandoq-token")),
    ]


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
        host_tunnel="modal",
    )
    runtime = make_runtime(config, "rollout-1")

    assert isinstance(runtime, SandoqRuntime)
    assert runtime_is_local(config) is False
    assert runtime_has_instance_host_endpoint(config) is True
    await runtime.start()
    assert runtime.descriptor == "assignment-123"
    assert client.request.docker_image == "swebench/image"
    assert client.request.start_command is None
    assert getattr(client.request, "vm", False) is False
    assert client.request.environment_vars == {"OCI_EXPECTED_WORKDIR": "/testbed"}

    result = await runtime.run(["sh", "-c", "printf ok"], {"MESSAGE": "value with spaces"})
    assert result.exit_code == 0
    assert result.stdout == "ok"
    assert client.commands == [
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


async def test_sandoq_runtime_builds_prime_sandboxes_042_container_request(
    monkeypatch,
) -> None:
    client = FakeSandoqClient()
    monkeypatch.setattr(sandoq, "create_client", lambda config: client)
    import prime_sandboxes

    def strict_request(**values):
        assert "network_access" not in values
        assert values["vm"] is False
        return SimpleNamespace(**values)

    monkeypatch.setattr(prime_sandboxes, "CreateSandboxRequest", strict_request)
    runtime = SandoqRuntime(SandoqConfig(), name="rollout-042")

    await runtime.start()

    assert client.request.name == "rollout-042"
    assert client.request.vm is False
    await runtime.stop()


async def test_sandoq_environment_mode_creates_configured_workdir(monkeypatch) -> None:
    client = FakeSandoqClient()
    monkeypatch.setattr(sandoq, "create_client", lambda config: client)
    runtime = SandoqRuntime(SandoqConfig(mode="environment", workdir="/workspace", host_tunnel="modal"))

    await runtime.start()

    assert client.request.environment_vars is None
    assert client.commands == [("mkdir -p /workspace", "/", {}, 3600)]
    await runtime.stop()


@pytest.mark.parametrize(
    "updates",
    [
        {"mode": "environment"},
        {"network_access": True},
        {"expected_environment": None},
        {"expected_environment": "oci-runner-firecracker-small"},
        {"ecr_token_file": None},
        {"ecr_token_file": Path("relative-token")},
    ],
)
def test_sandoq_host_side_config_rejects_incomplete_policy(updates) -> None:
    values = {
        "mode": "oci-runner",
        "network_access": False,
        "host_tunnel": "none",
        "expected_environment": "oci-runner-firecracker",
        "ecr_token_file": Path("/run/secrets/ecr-token"),
        **updates,
    }

    with pytest.raises(ValueError, match="Sandoq host-side execution"):
        SandoqConfig(**values)


@pytest.mark.parametrize(
    "updates",
    [
        {"mode": "environment"},
        {"network_access": True},
        {"expected_environment": None},
        {"expected_environment": "oci-runner-firecracker-small"},
        {"ecr_token_file": None},
        {"ecr_token_file": Path("relative-token")},
    ],
)
def test_sandoq_create_client_rechecks_host_policy_after_unvalidated_copy(
    updates,
) -> None:
    values = {
        "type": "sandoq",
        "mode": "oci-runner",
        "network_access": False,
        "host_tunnel": "none",
        "expected_environment": "oci-runner-firecracker",
        "ecr_token_file": Path("/run/secrets/ecr-token"),
        **updates,
    }
    config = SandoqConfig.model_construct(**values)

    with pytest.raises(SandboxError, match="host-side configuration"):
        sandoq.create_client(config)


async def test_sandoq_no_tunnel_mode_rejects_host_endpoint_calls() -> None:
    runtime = SandoqRuntime(
        SandoqConfig(
            network_access=False,
            host_tunnel="none",
            expected_environment="oci-runner-firecracker",
            ecr_token_file=Path("/run/secrets/ecr-token"),
        )
    )

    with pytest.raises(SandboxError, match="must not be called"):
        async with runtime.host_endpoint(4321):
            pass


async def test_sandoq_runtime_ignores_close_error_after_verified_delete(
    monkeypatch,
) -> None:
    client = FakeSandoqClient()
    client.close_error = TimeoutError("telemetry exporter did not stop")
    monkeypatch.setattr(sandoq, "create_client", lambda config: client)
    runtime = SandoqRuntime(SandoqConfig(host_tunnel="modal"))
    await runtime.start()

    await runtime.stop()

    assert client.deleted == ["assignment-123"]
    assert client.closed is True


async def test_sandoq_runtime_surfaces_unverified_delete(monkeypatch) -> None:
    client = FakeSandoqClient()
    client.delete_error = TimeoutError("typed 404 was not observed")
    monkeypatch.setattr(sandoq, "create_client", lambda config: client)
    runtime = SandoqRuntime(SandoqConfig(host_tunnel="modal"))
    await runtime.start()

    with pytest.raises(SandboxError, match="deletion was not verified"):
        await runtime.stop()

    assert client.deleted == []
    assert client.closed is False
    assert runtime._active is True


async def test_sandoq_runtime_rejects_semantically_unverified_delete(
    monkeypatch,
) -> None:
    client = FakeSandoqClient()

    async def unverified_delete(sandbox_id: str):
        client.deleted.append(sandbox_id)
        return {"cleanup_verified": False, "poisoned": True}

    client.delete = unverified_delete
    monkeypatch.setattr(sandoq, "create_client", lambda config: client)
    runtime = SandoqRuntime(SandoqConfig(host_tunnel="modal"))
    await runtime.start()

    with pytest.raises(SandboxError, match="deletion was not verified"):
        await runtime.stop()
    assert runtime._active is False
    assert client.closed is True
    await runtime.stop()
    assert client.deleted == ["assignment-123"]


@pytest.mark.parametrize(
    "response",
    [
        {"poisoned": False, "outer_deletion_verified_http_status": 404},
        {"cleanup_verified": False, "verified_http_status": 404},
        {"nested_recycle_verified": True, "status": "release_in_progress"},
        {"nested_recycle_verified": True, "status": "already_released"},
        {"nested_recycle_verified": True, "error": "cleanup incomplete"},
    ],
)
async def test_sandoq_runtime_rejects_ambiguous_cleanup_receipts(monkeypatch, response) -> None:
    client = FakeSandoqClient()

    async def ambiguous_delete(_sandbox_id: str):
        return response

    client.delete = ambiguous_delete
    monkeypatch.setattr(sandoq, "create_client", lambda config: client)
    runtime = SandoqRuntime(SandoqConfig(host_tunnel="modal"))
    await runtime.start()

    with pytest.raises(SandboxError, match="deletion was not verified"):
        await runtime.stop()
    assert runtime._active is False
    assert client.closed is True


async def test_sandoq_runtime_accepts_poisoned_but_verified_cleanup(
    monkeypatch,
) -> None:
    client = FakeSandoqClient()

    async def poisoned_delete(_sandbox_id: str):
        return {
            "status": "poisoned",
            "poisoned": True,
            "outer_deletion_verified_http_status": 404,
            "error": "nested recycle failed before verified outer deletion",
        }

    client.delete = poisoned_delete
    monkeypatch.setattr(sandoq, "create_client", lambda config: client)
    runtime = SandoqRuntime(SandoqConfig(host_tunnel="modal"))
    await runtime.start()

    await runtime.stop()
    assert runtime._active is False
    assert client.closed is True


async def test_sandoq_runtime_preserves_cleanup_retry_after_provisioning_failure(
    monkeypatch,
) -> None:
    client = FakeSandoqClient()
    client.wait_error = RuntimeError("readiness failed")
    client.delete_error = TimeoutError("typed 404 was not observed")
    monkeypatch.setattr(sandoq, "create_client", lambda config: client)
    runtime = SandoqRuntime(SandoqConfig(host_tunnel="modal"))

    with pytest.raises(SandboxError, match="deletion was not verified"):
        await runtime.start()

    assert runtime._active is True
    assert runtime._client is client
    assert client.closed is False


async def test_sandoq_runtime_preserves_cleanup_retry_after_cancelled_provisioning(
    monkeypatch,
) -> None:
    client = FakeSandoqClient()
    waiting = asyncio.Event()

    async def wait_for_creation(_sandbox_id: str) -> None:
        waiting.set()
        await asyncio.Event().wait()

    client.wait_for_creation = wait_for_creation
    client.delete_error = TimeoutError("typed 404 was not observed")
    monkeypatch.setattr(sandoq, "create_client", lambda config: client)
    runtime = SandoqRuntime(SandoqConfig(host_tunnel="modal"))
    starting = asyncio.create_task(runtime.start())
    await waiting.wait()

    starting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await starting

    assert runtime._active is True
    assert runtime._client is client
    assert client.closed is False


async def test_sandoq_runtime_finishes_delete_after_repeated_provisioning_cancellation(
    monkeypatch,
) -> None:
    client = FakeSandoqClient()
    waiting = asyncio.Event()
    deleting = asyncio.Event()
    allow_delete = asyncio.Event()

    async def wait_for_creation(_sandbox_id: str) -> None:
        waiting.set()
        await asyncio.Event().wait()

    async def delete(sandbox_id: str):
        deleting.set()
        await allow_delete.wait()
        client.deleted.append(sandbox_id)
        return {"status": "deleted", "verified_http_status": 404}

    client.wait_for_creation = wait_for_creation
    client.delete = delete
    monkeypatch.setattr(sandoq, "create_client", lambda config: client)
    runtime = SandoqRuntime(SandoqConfig(host_tunnel="modal"))
    starting = asyncio.create_task(runtime.start())
    await waiting.wait()

    starting.cancel()
    await deleting.wait()
    starting.cancel()
    await asyncio.sleep(0)
    assert client.deleted == []
    allow_delete.set()

    with pytest.raises(asyncio.CancelledError):
        await starting
    assert client.deleted == ["assignment-123"]
    assert runtime._active is False
    assert client.closed is True


async def test_sandoq_runtime_finishes_delete_before_stop_propagates_cancellation(
    monkeypatch,
) -> None:
    client = FakeSandoqClient()
    deleting = asyncio.Event()
    allow_delete = asyncio.Event()

    async def delete(sandbox_id: str):
        deleting.set()
        await allow_delete.wait()
        client.deleted.append(sandbox_id)
        return {"status": "deleted", "verified_http_status": 404}

    client.delete = delete
    monkeypatch.setattr(sandoq, "create_client", lambda config: client)
    runtime = SandoqRuntime(SandoqConfig(host_tunnel="modal"))
    await runtime.start()
    stopping = asyncio.create_task(runtime.stop())
    await deleting.wait()

    stopping.cancel()
    await asyncio.sleep(0)
    assert client.deleted == []
    allow_delete.set()

    with pytest.raises(asyncio.CancelledError):
        await stopping
    assert client.deleted == ["assignment-123"]
    assert runtime._active is False
    assert client.closed is True


async def test_sandoq_runtime_uses_native_reverse_tunnel(monkeypatch) -> None:
    events: list[object] = []

    class FakeRelayTunnel:
        def __init__(self, local_port: int, **kwargs) -> None:
            events.append(("init", local_port, kwargs))

        def start(self) -> None:
            events.append("start")

        def stop(self) -> None:
            events.append("stop")

    info = SimpleNamespace(
        environment="oci-runner-firecracker",
        port_urls={"tunnel": "https://tunnel.example/session"},
        metadata={"task_network": "host"},
    )
    registry = SimpleNamespace(get=lambda sandbox_id: info if sandbox_id == "assignment-123" else None)
    provider = ModuleType("sandoq_provider")
    provider.registry = registry
    tunnel_module = ModuleType("sandoq_provider.tunnel")
    tunnel_module.SandoqRelayTunnel = FakeRelayTunnel
    monkeypatch.setitem(sys.modules, "sandoq_provider", provider)
    monkeypatch.setitem(sys.modules, "sandoq_provider.tunnel", tunnel_module)
    monkeypatch.setenv("OCI_RUNNER_TASK_NETWORK", "host")

    runtime = SandoqRuntime(
        SandoqConfig(
            host_tunnel="sandoq",
            guest_tunnel_url="http://127.0.0.1:8485",
            tunnel_pool_size=4,
            tunnel_ready_timeout=12,
        )
    )
    runtime._sandbox_id = "assignment-123"

    async with runtime.host_endpoint(4321) as url:
        assert url == "http://127.0.0.1:8485"

    assert events == [
        (
            "init",
            4321,
            {
                "tunnel_url": "https://tunnel.example/session",
                "pool_size": 4,
                "ready_timeout": 12,
            },
        ),
        "start",
        "stop",
    ]


async def test_sandoq_runtime_buffers_interception_before_native_tunnel(monkeypatch) -> None:
    events: list[object] = []
    logs: list[str] = []
    monkeypatch.setattr(sandoq.logger, "info", lambda message, *args: logs.append(message % args))

    class FakeBufferedProxy:
        def __init__(self, endpoint: str, secret: str) -> None:
            events.append(("proxy-init", endpoint, secret))
            self.port = 5678
            self.stats = SimpleNamespace(
                snapshot=lambda: {
                    "requests": 2,
                    "upstream_attempts": 1,
                    "paths": {"/v1/chat/completions": 2, "/private/task-derived-path": 1},
                    "errors": ["private failure detail"],
                }
            )

        async def start(self) -> None:
            events.append("proxy-start")

        async def close(self) -> None:
            events.append("proxy-close")

    buffered_module = ModuleType("sandoq_provider.buffered_chat")
    buffered_module.BufferedChatCompletionsProxy = FakeBufferedProxy
    monkeypatch.setitem(sys.modules, "sandoq_provider.buffered_chat", buffered_module)
    runtime = SandoqRuntime(
        SandoqConfig(
            host_tunnel="sandoq",
            buffered_chat_completions=True,
        )
    )

    @asynccontextmanager
    async def fake_host_endpoint(port: int):
        events.append(("tunnel", port))
        yield "http://127.0.0.1:8485"

    monkeypatch.setattr(runtime, "host_endpoint", fake_host_endpoint)

    async with runtime.interception_endpoint(4321, "rollout-secret") as url:
        assert url == "http://127.0.0.1:8485"
        events.append("yield")

    assert events == [
        ("proxy-init", "http://127.0.0.1:4321", "rollout-secret"),
        "proxy-start",
        ("tunnel", 5678),
        "yield",
        "proxy-close",
    ]
    log = "\n".join(logs)
    assert 'upstream_attempts":1' in log
    assert 'error_count":1' in log
    assert 'unknown_path_requests":1' in log
    assert "private failure detail" not in log
    assert "private/task-derived-path" not in log


@pytest.mark.parametrize("updates", [{"host_tunnel": "modal"}, {"mode": "environment"}])
def test_sandoq_buffered_interception_requires_native_tunnel(updates) -> None:
    config = {"buffered_chat_completions": True, "host_tunnel": "sandoq", **updates}
    with pytest.raises(ValueError, match="buffered_chat_completions"):
        SandoqConfig(**config)


async def test_sandoq_runtime_attributes_native_tunnel_start_failure(
    monkeypatch,
) -> None:
    class FailingRelayTunnel:
        def __init__(self, _local_port: int, **_kwargs) -> None:
            pass

        def start(self) -> None:
            raise TimeoutError("relay was not ready")

        def stop(self) -> None:
            pass

    info = SimpleNamespace(
        environment="oci-runner-firecracker",
        port_urls={"tunnel": "https://tunnel.example/session"},
        metadata={"task_network": "host"},
    )
    provider = ModuleType("sandoq_provider")
    provider.registry = SimpleNamespace(get=lambda _sandbox_id: info)
    tunnel_module = ModuleType("sandoq_provider.tunnel")
    tunnel_module.SandoqRelayTunnel = FailingRelayTunnel
    monkeypatch.setitem(sys.modules, "sandoq_provider", provider)
    monkeypatch.setitem(sys.modules, "sandoq_provider.tunnel", tunnel_module)
    runtime = SandoqRuntime(SandoqConfig(host_tunnel="sandoq"))
    runtime._sandbox_id = "assignment-123"

    with pytest.raises(TunnelError, match="failed to start"):
        async with runtime.host_endpoint(4321):
            pytest.fail("an unavailable tunnel must not yield")


async def test_sandoq_runtime_waits_for_cancelled_tunnel_start_before_stop(
    monkeypatch,
) -> None:
    start_entered = asyncio.Event()
    allow_start = threading.Event()
    events: list[str] = []
    loop = asyncio.get_running_loop()

    class BlockingRelayTunnel:
        def __init__(self, _local_port: int, **_kwargs) -> None:
            pass

        def start(self) -> None:
            events.append("start-entered")
            loop.call_soon_threadsafe(start_entered.set)
            allow_start.wait()
            events.append("start-finished")

        def stop(self) -> None:
            events.append("stop")

    info = SimpleNamespace(
        environment="oci-runner-firecracker",
        port_urls={"tunnel": "https://tunnel.example/session"},
        metadata={"task_network": "host"},
    )
    provider = ModuleType("sandoq_provider")
    provider.registry = SimpleNamespace(get=lambda _sandbox_id: info)
    tunnel_module = ModuleType("sandoq_provider.tunnel")
    tunnel_module.SandoqRelayTunnel = BlockingRelayTunnel
    monkeypatch.setitem(sys.modules, "sandoq_provider", provider)
    monkeypatch.setitem(sys.modules, "sandoq_provider.tunnel", tunnel_module)
    runtime = SandoqRuntime(SandoqConfig(host_tunnel="sandoq"))
    runtime._sandbox_id = "assignment-123"

    async def use_tunnel() -> None:
        async with runtime.host_endpoint(4321):
            pytest.fail("cancelled tunnel start must not yield")

    running = asyncio.create_task(use_tunnel())
    await start_entered.wait()
    running.cancel()
    await asyncio.sleep(0)
    assert events == ["start-entered"]
    allow_start.set()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert events == ["start-entered", "start-finished", "stop"]


async def test_sandoq_runtime_native_tunnel_fails_closed(monkeypatch) -> None:
    info = SimpleNamespace(
        environment="oci-runner-firecracker",
        port_urls={"tunnel": "https://tunnel.example/session"},
        metadata={"task_network": "none"},
    )
    provider = ModuleType("sandoq_provider")
    provider.registry = SimpleNamespace(get=lambda _sandbox_id: info)
    tunnel_module = ModuleType("sandoq_provider.tunnel")
    tunnel_module.SandoqRelayTunnel = object
    monkeypatch.setitem(sys.modules, "sandoq_provider", provider)
    monkeypatch.setitem(sys.modules, "sandoq_provider.tunnel", tunnel_module)
    runtime = SandoqRuntime(SandoqConfig(host_tunnel="sandoq"))
    runtime._sandbox_id = "assignment-123"

    with pytest.raises(SandboxError, match="provisioned task network to be host"):
        async with runtime.host_endpoint(4321):
            pass


async def test_sandoq_runtime_tunnel_cleanup_preserves_cancellation(
    monkeypatch,
) -> None:
    started = asyncio.Event()

    class FailingStopTunnel:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> None:
            raise RuntimeError("tunnel stop failed")

    info = SimpleNamespace(
        environment="oci-runner-firecracker",
        port_urls={"tunnel": "https://tunnel.example/session"},
        metadata={"task_network": "host"},
    )
    provider = ModuleType("sandoq_provider")
    provider.registry = SimpleNamespace(get=lambda _sandbox_id: info)
    tunnel_module = ModuleType("sandoq_provider.tunnel")
    tunnel_module.SandoqRelayTunnel = FailingStopTunnel
    monkeypatch.setitem(sys.modules, "sandoq_provider", provider)
    monkeypatch.setitem(sys.modules, "sandoq_provider.tunnel", tunnel_module)
    monkeypatch.setenv("OCI_RUNNER_TASK_NETWORK", "host")
    runtime = SandoqRuntime(SandoqConfig(host_tunnel="sandoq"))
    runtime._sandbox_id = "assignment-123"

    with pytest.raises(TunnelError, match="cleanup failed"):
        async with runtime.host_endpoint(4321):
            pass

    async def use_tunnel() -> None:
        async with runtime.host_endpoint(4321):
            started.set()
            await asyncio.Event().wait()

    running = asyncio.create_task(use_tunnel())
    await started.wait()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:8485",
        "http://example.com:8485",
        "http://127.0.0.1",
        "http://127.0.0.1:8485/v1",
    ],
)
def test_sandoq_config_rejects_non_loopback_tunnel_url(url: str) -> None:
    with pytest.raises(ValueError, match="explicit HTTP loopback URL"):
        SandoqConfig(guest_tunnel_url=url)


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


async def test_sandoq_runtime_runs_long_program_as_one_background_job(
    monkeypatch,
) -> None:
    client = FakeSandoqClient()
    monkeypatch.setattr(sandoq, "create_client", lambda config: client)
    runtime = SandoqRuntime(SandoqConfig(workdir="/testbed", session_timeout=7200))
    await runtime.start()

    result = await runtime.run_program(["agent", "--task", "value with spaces"], {"MODEL": "kimi"})

    assert result == ProgramResult(exit_code=0, stdout="program-ok", stderr="")
    assert client.background_commands == [
        (
            "agent --task 'value with spaces'",
            "/testbed",
            {"MODEL": "kimi"},
            7200,
            3,
        )
    ]
    await runtime.stop()


async def test_sandoq_runtime_never_replays_uncertain_background_launch(
    monkeypatch,
) -> None:
    client = FakeSandoqClient()
    monkeypatch.setattr(sandoq, "create_client", lambda config: client)
    runtime = SandoqRuntime(SandoqConfig(session_timeout=7200))
    await runtime.start()
    client.error = RuntimeError("transport unavailable after background launch")

    with pytest.raises(SandboxError, match="transport unavailable"):
        await runtime.run_program(["agent", "--task", "opaque"], {})

    assert len(client.background_commands) == 1
    client.error = None
    await runtime.stop()


async def test_sandoq_environment_cancellation_prevents_scoring(monkeypatch) -> None:
    client = FakeSandoqClient()
    running = asyncio.Event()

    async def background(*_args, **_kwargs):
        running.set()
        await asyncio.Event().wait()

    client.run_background_job = background
    monkeypatch.setattr(sandoq, "create_client", lambda config: client)
    runtime = SandoqRuntime(SandoqConfig(mode="environment"))
    await runtime.start()
    program = asyncio.create_task(runtime.run_program(["agent"], {}))
    await running.wait()

    program.cancel()
    with pytest.raises(asyncio.CancelledError):
        await program
    with pytest.raises(SandboxError, match="was not joined"):
        runtime.ensure_usable()

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
