"""Remote Sandoq runtime backed by the pinned OCI runner provider."""

import asyncio
import contextlib
import logging
import shlex
import stat
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Literal
from urllib.parse import urlsplit

from pydantic import Field, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.errors import SandboxError, TunnelError
from verifiers.v1.runtimes.base import (
    ProgramResult,
    Runtime,
    parse_gpu,
)
from verifiers.v1.runtimes.base import (
    host_endpoint as shared_host_endpoint,
)
from verifiers.v1.runtimes.limiters import creation_limiter
from verifiers.v1.runtimes.modal_tunnel import modal_host_endpoint

logger = logging.getLogger(__name__)
_OCI_BOOTSTRAP_WORKDIR = "/tmp"


async def _finish_thread_task(
    task: asyncio.Task[None],
) -> tuple[asyncio.CancelledError | None, Exception | None]:
    """Wait for a thread-backed operation without detaching it on cancellation."""
    cancelled = None
    error = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as caught:
            cancelled = caught
        except Exception as caught:  # noqa: BLE001 - returned to lifecycle caller
            error = caught
            break
    if error is None:
        try:
            task.result()
        except Exception as caught:  # noqa: BLE001 - returned to lifecycle caller
            error = caught
    return cancelled, error


class SandoqConfig(BaseConfig):
    type: Literal["sandoq"] = "sandoq"
    image: str = "python:3.11-slim"
    workdir: str = "/app"
    network_access: bool = True
    mode: Literal["oci-runner", "environment"] = "oci-runner"
    """OCI runner accepts per-task images; environment uses a predeployed Sandoq image map."""
    session_timeout: float = Field(3600.0, gt=0)
    """Maximum duration of one command inside the sandbox, in seconds."""
    cpu: float = 1.0
    memory: float = 2.0
    gpu: str | None = None
    disk: float = 5.0
    creates_per_sec: float | None = None
    """Optional process-wide creation pacing; the OCI runner also applies its own pool limits."""
    host_tunnel: Literal["none", "sandoq", "modal", "prime"] = "modal"
    """How a sandbox reaches host interception services."""
    guest_tunnel_url: str = "http://127.0.0.1:8485"
    """Loopback endpoint exposed by the Sandoq Firecracker environment."""
    tunnel_pool_size: int = Field(8, ge=1, le=64)
    tunnel_ready_timeout: float = Field(30.0, gt=0, le=120)
    expected_environment: str | None = None
    """Exact registry environment required by policy-sensitive callers."""
    ecr_token_file: Path | None = None
    """Expected private ECR credential path; contents are never retained."""

    @model_validator(mode="after")
    def validate_sandoq_tunnel(self) -> "SandoqConfig":
        parsed = urlsplit(self.guest_tunnel_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost"}
            or parsed.port is None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "guest_tunnel_url must be an explicit HTTP loopback URL with a port"
            )
        if self.mode == "oci-runner" and self.host_tunnel == "none":
            if (
                self.network_access is not False
                or self.expected_environment != "oci-runner-firecracker"
                or self.ecr_token_file is None
                or not self.ecr_token_file.is_absolute()
            ):
                raise ValueError(
                    "Sandoq host-side execution requires network_access=false, the "
                    "production Firecracker environment, and an absolute ECR token path"
                )
        elif self.mode == "oci-runner" and not self.network_access:
            raise ValueError(
                "Sandoq OCI no-network execution requires a host-side harness and no host tunnel"
            )
        if self.host_tunnel == "none" and self.mode != "oci-runner":
            raise ValueError(
                "Sandoq host-side execution is supported only by OCI runner mode"
            )
        return self


def create_client(config: SandoqConfig) -> Any:
    """Construct the PR-pinned provider lazily so other runtimes need no internal dependency."""
    if config.host_tunnel == "none" and (
        config.mode != "oci-runner"
        or config.network_access is not False
        or config.expected_environment != "oci-runner-firecracker"
        or config.ecr_token_file is None
        or not config.ecr_token_file.is_absolute()
    ):
        raise SandboxError("Sandoq host-side no-network configuration is incomplete")
    if (
        config.mode == "oci-runner"
        and not config.network_access
        and config.host_tunnel != "none"
    ):
        raise SandboxError(
            "Sandoq OCI no-network execution requires a host-side harness and no host tunnel"
        )
    try:
        if config.mode == "oci-runner":
            from sandoq_provider.oci_client import (
                OCIRunnerAsyncSandboxClient,
                get_oci_config,
                read_token_file,
            )
            from sandoq_provider.secrets import read_secret_file

            oci_config = get_oci_config()
            if config.host_tunnel == "none" and oci_config.task_network != "none":
                raise SandboxError(
                    "Sandoq host-side harnesses require nested task networking to be disabled"
                )
            if config.host_tunnel == "sandoq" and oci_config.task_network != "host":
                raise SandboxError(
                    "Sandoq native reverse tunnels require nested host networking"
                )
            if not config.network_access:
                expected_environment = "oci-runner-firecracker"
                ecr_token_file = oci_config.ecr.token_file
                try:
                    ecr_token_stat = ecr_token_file.lstat()
                except (AttributeError, OSError) as error:
                    raise SandboxError(
                        "Sandoq no-network requires an available ECR token file"
                    ) from error
                if (
                    config.expected_environment != expected_environment
                    or oci_config.environment != expected_environment
                    or config.host_tunnel != "none"
                    or oci_config.task_network != "none"
                    or not oci_config.ecr.enabled
                    or oci_config.ecr.registry
                    != "168653207203.dkr.ecr.us-east-2.amazonaws.com"
                    or oci_config.ecr.region != "us-east-2"
                    or oci_config.ecr.pull_through_prefix != "pt_dockerio"
                    or config.ecr_token_file is None
                    or not config.ecr_token_file.is_absolute()
                    or not ecr_token_file.is_absolute()
                    or ecr_token_file != config.ecr_token_file.expanduser()
                    or ecr_token_file.is_symlink()
                    or not stat.S_ISREG(ecr_token_stat.st_mode)
                    or stat.S_IMODE(ecr_token_stat.st_mode) != 0o600
                    or oci_config.allow_dockerhub_fallback
                    or oci_config.create_deadline_s != 1800
                    or oci_config.pull_timeout_s != 1200
                    or oci_config.pull_poll_max_errors != 10
                    or oci_config.gateway_retry_attempts != 15
                    or oci_config.gateway_retry_interval_s != 2
                    or not oci_config.podman_ignore_chown_errors
                    or not oci_config.require_resource_limits
                    or not oci_config.session_reuse
                ):
                    raise SandboxError(
                        "Sandoq no-network requires a host-side harness, the production "
                        "Firecracker environment, disabled nested task networking, "
                        "authenticated production ECR pull-through, and disabled "
                        "direct-Docker-Hub fallback"
                    )
                read_secret_file(ecr_token_file, "ECR token", SandboxError)
            read_token_file(oci_config.token_file)
            return OCIRunnerAsyncSandboxClient()
        from sandoq_provider.client import SandoqAsyncSandboxClient

        return SandoqAsyncSandboxClient()
    except ModuleNotFoundError as error:
        if error.name not in {"sandoq_provider", "sandoq_client"}:
            raise
        raise ModuleNotFoundError(
            "SandoqRuntime requires the pinned provider from deps/sandoq-provider/"
            "extensions/sandoq on PYTHONPATH and its sandoq-client dependency"
        ) from error


class SandoqRuntime(Runtime):
    """Runs a harness in Sandoq while preserving the Verifiers Runtime contract."""

    is_local: ClassVar[bool] = False
    instance_host_endpoint: ClassVar[bool] = True
    cleanup_must_succeed: ClassVar[bool] = True

    def __init__(self, config: SandoqConfig, name: str | None = None) -> None:
        super().__init__(name)
        self.config = config
        self._client: Any = None
        self._sandbox_id: str | None = None
        self._active = False

    @property
    def descriptor(self) -> str | None:
        return self._sandbox_id

    @property
    def sandbox_id(self) -> str:
        if self._sandbox_id is None:
            raise RuntimeError("Sandoq runtime is not running")
        return self._sandbox_id

    async def start(self) -> None:
        from prime_sandboxes import CreateSandboxRequest

        client = create_client(self.config)
        self._client = client
        gpu_type, gpu_count = parse_gpu(self.config.gpu)
        request = CreateSandboxRequest(
            name=self.name,
            docker_image=self.config.image,
            start_command=None,
            cpu_cores=self.config.cpu,
            memory_gb=self.config.memory,
            disk_size_gb=self.config.disk,
            gpu_count=gpu_count,
            gpu_type=gpu_type,
            network_access=self.config.network_access,
            vm=False,
            timeout_minutes=24 * 60,
            environment_vars={"OCI_EXPECTED_WORKDIR": _OCI_BOOTSTRAP_WORKDIR}
            if self.config.mode == "oci-runner"
            else None,
        )
        try:
            async with (
                creation_limiter(self.config.creates_per_sec, "sandoq-sandbox")
                or contextlib.nullcontext()
            ):
                sandbox = await client.create(request)
                self._sandbox_id = str(sandbox.id)
                self._active = True
                await client.wait_for_creation(self._sandbox_id)
            result = await self._run(
                client,
                ["mkdir", "-p", self.config.workdir],
                {},
                working_dir=_OCI_BOOTSTRAP_WORKDIR
                if self.config.mode == "oci-runner"
                else "/",
            )
            if result.exit_code != 0:
                raise RuntimeError(result.stderr or result.stdout)
        except asyncio.CancelledError:
            _, delete_error = await self._delete_active(client)
            if delete_error is not None:
                logger.warning(
                    "sandoq: deletion was not verified after cancelled provisioning "
                    "for %s: %s",
                    self._sandbox_id,
                    delete_error,
                )
            if not self._active:
                self._client = None
                with contextlib.suppress(Exception):
                    await client.aclose()
            raise
        except Exception as error:
            cancelled, cleanup_error = await self._delete_active(client)
            if not self._active:
                self._client = None
                with contextlib.suppress(Exception):
                    await client.aclose()
            if cancelled is not None:
                raise cancelled
            message = f"Sandoq provisioning failed: {error}"
            if cleanup_error is not None:
                message += f"; deletion was not verified: {cleanup_error}"
            raise SandboxError(message) from error
        logger.info(
            "sandoq: sandbox %s up (mode=%s image=%s)",
            self._sandbox_id,
            self.config.mode,
            self.config.image,
        )

    async def _delete_active(
        self, client: Any
    ) -> tuple[asyncio.CancelledError | None, Exception | None]:
        """Finish one delete attempt even when this task receives repeated cancellation."""
        if not self._active or self._sandbox_id is None:
            return None, None
        delete_task = asyncio.create_task(client.delete(self._sandbox_id))
        cancelled = None
        while not delete_task.done():
            try:
                await asyncio.shield(delete_task)
            except asyncio.CancelledError as error:
                cancelled = error
            except Exception:  # provider error is read from the completed task below
                break
        try:
            response = delete_task.result()
        except Exception as error:  # noqa: BLE001 - caller converts provider failures
            return cancelled, error
        if not isinstance(response, dict):
            return cancelled, RuntimeError("provider returned no cleanup verification")
        # A normal provider response consumes and unregisters the assignment,
        # even when its cleanup receipt is semantically unverified. Retrying the
        # assignment id would be misinterpreted as an outer-session id.
        self._active = False
        status = response.get("status")
        semantic_cleanup_verified = (
            (status == "deleted" and response.get("verified_http_status") == 404)
            or (
                status == "recycled"
                and response.get("nested_recycle_verified") is True
                and response.get("poisoned") is not True
            )
            or (
                status == "retired"
                and response.get("nested_recycle_verified") is True
                and response.get("outer_deletion_verified_http_status") == 404
            )
            or (
                status == "poisoned"
                and response.get("poisoned") is True
                and response.get("outer_deletion_verified_http_status") == 404
            )
        )
        cleanup_verified = (
            semantic_cleanup_verified
            and response.get("cleanup_verified", True) is True
            and (not response.get("error") or status == "poisoned")
        )
        if not cleanup_verified:
            return cancelled, RuntimeError("provider reported unverified cleanup")
        return cancelled, None

    async def _run(
        self,
        client: Any,
        argv: list[str],
        env: dict[str, str],
        *,
        working_dir: str | None = None,
    ) -> ProgramResult:
        result = await client.execute_command(
            self.sandbox_id,
            shlex.join(argv),
            working_dir=working_dir or self.config.workdir,
            env=env,
            timeout=int(self.config.session_timeout),
        )
        stderr = result.stderr or ""
        if (
            result.exit_code == 75
            and "execution status is unknown and the command was not replayed" in stderr
        ):
            raise SandboxError(stderr)
        return ProgramResult(
            exit_code=result.exit_code if result.exit_code is not None else -1,
            stdout=result.stdout or "",
            stderr=stderr,
        )

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        if self._client is None:
            raise SandboxError("Sandoq runtime is not running")
        try:
            return await self._run(self._client, argv, env)
        except Exception as error:
            # Model/agent commands are intentionally single-attempt. An uncertain response may
            # have side effects in the task container, so the Runtime must never replay it.
            raise SandboxError(f"Sandoq exec failed: {error}") from error

    async def run_background(
        self, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        inner = f"nohup {shlex.join(argv)} > {shlex.quote(log)} 2>&1 < /dev/null &"
        result = await self.run(["sh", "-c", inner], env)
        if result.exit_code != 0:
            raise SandboxError(
                f"Sandoq background launch failed: {(result.stderr or result.stdout).strip()}"
            )

    def _abs(self, path: str) -> str:
        if path.startswith("/"):
            return path
        return f"{self.config.workdir.rstrip('/')}/{path}"

    async def read(self, path: str) -> bytes:
        if self._client is None:
            raise SandboxError("Sandoq runtime is not running")
        target = self._abs(path)
        try:
            with tempfile.TemporaryDirectory() as directory:
                download = Path(directory) / "download"
                await self._client.download_file(
                    self.sandbox_id,
                    target,
                    str(download),
                    timeout=int(self.config.session_timeout),
                )
                return await asyncio.to_thread(download.read_bytes)
        except Exception as error:
            raise SandboxError(f"read {path!r}: {error}") from error

    async def write(self, path: str, data: bytes) -> None:
        if self._client is None:
            raise SandboxError("Sandoq runtime is not running")
        target = self._abs(path)
        parent = str(PurePosixPath(target).parent)
        mkdir = await self.run(["mkdir", "-p", parent], {})
        if mkdir.exit_code != 0:
            raise SandboxError(
                f"write {path!r}: could not create {parent}: {mkdir.stderr}"
            )
        try:
            await self._client.upload_bytes(
                self.sandbox_id,
                target,
                data,
                filename=PurePosixPath(target).name,
                timeout=int(self.config.session_timeout),
            )
        except Exception as error:
            raise SandboxError(f"write {path!r}: {error}") from error

    async def expose(self, port: int) -> str | None:
        raise SandboxError(
            "Sandoq OCI runner does not support arbitrary nested-container port exposure; "
            f"place the server with the harness or on the host instead (port {port})"
        )

    @contextlib.asynccontextmanager
    async def host_endpoint(self, port: int):
        if self.config.host_tunnel == "none":
            raise SandboxError(
                "host_endpoint must not be called for a host-side Sandoq harness"
            )
        if self.config.host_tunnel == "prime":
            async with shared_host_endpoint(port, is_local=False) as url:
                yield url
            return
        if self.config.host_tunnel == "modal":
            async with modal_host_endpoint(port) as url:
                yield url
            return
        if self.config.mode != "oci-runner":
            raise SandboxError(
                "the native Sandoq host tunnel requires mode='oci-runner'"
            )
        try:
            from sandoq_provider import registry
            from sandoq_provider.tunnel import SandoqRelayTunnel
        except (ImportError, ModuleNotFoundError) as error:
            raise SandboxError(
                "the native Sandoq host tunnel requires the authoritative provider "
                "with sandoq_provider.tunnel on PYTHONPATH"
            ) from error

        info = registry.get(self.sandbox_id)
        if info is None:
            raise SandboxError(
                f"Sandoq session metadata is unavailable for {self.sandbox_id!r}"
            )
        if self.config.expected_environment is not None:
            environment_matches = info.environment == self.config.expected_environment
        else:
            environment_matches = info.environment.startswith("oci-runner-firecracker")
        if not environment_matches:
            raise SandboxError(
                "the native Sandoq host tunnel received an unexpected environment"
            )
        if info.metadata.get("task_network") != "host":
            raise SandboxError(
                "the native Sandoq host tunnel requires the provisioned task network to be host"
            )
        tunnel_url = info.port_urls.get("tunnel")
        if not tunnel_url:
            raise SandboxError(
                f"Sandoq environment {info.environment!r} has no named 'tunnel' port"
            )
        tunnel = SandoqRelayTunnel(
            port,
            tunnel_url=tunnel_url,
            pool_size=self.config.tunnel_pool_size,
            ready_timeout=self.config.tunnel_ready_timeout,
        )
        try:
            start_task = asyncio.create_task(asyncio.to_thread(tunnel.start))
            start_cancelled, start_error = await _finish_thread_task(start_task)
            if start_error is not None:
                raise TunnelError(
                    "Sandoq host tunnel failed to start: "
                    f"{type(start_error).__name__}: {start_error}"
                ) from start_error
            if start_cancelled is not None:
                raise start_cancelled
            yield self.config.guest_tunnel_url.rstrip("/")
        finally:
            original_error = sys.exception()
            stop_task = asyncio.create_task(asyncio.to_thread(tunnel.stop))
            cancelled, cleanup_error = await _finish_thread_task(stop_task)
            if cleanup_error is not None:
                if original_error is None and cancelled is None:
                    raise TunnelError(
                        "Sandoq host tunnel cleanup failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    ) from cleanup_error
                logger.warning(
                    "sandoq: tunnel cleanup failed while preserving %s: %s",
                    type(original_error or cancelled).__name__,
                    cleanup_error,
                )
            if cancelled is not None and original_error is None:
                raise cancelled

    def cleanup(self) -> None:
        if not self._active:
            return
        self._active = False
        if self.config.mode == "oci-runner":
            from sandoq_provider.oci_client import delete_registered_sessions_sync

            result = delete_registered_sessions_sync()
            if result.get("failed"):
                logger.warning(
                    "sandoq: synchronous cleanup failures: %s", result["failed"]
                )
            return
        from sandoq_provider.sync_client import SandoqSandboxClient

        if self._sandbox_id is not None:
            SandoqSandboxClient().delete(self._sandbox_id)

    async def stop(self) -> None:
        client = self._client
        if client is None:
            return
        cancelled, provider_error = await self._delete_active(client)
        delete_error = (
            SandboxError(
                f"Sandoq deletion was not verified for {self._sandbox_id}: "
                f"{provider_error}"
            )
            if provider_error is not None
            else None
        )
        if not self._active:
            self._client = None
            with contextlib.suppress(Exception):
                await client.aclose()
        if cancelled is not None:
            if delete_error is not None:
                logger.warning(
                    "sandoq: deletion was not verified while preserving cancellation "
                    "for %s: %s",
                    self._sandbox_id,
                    provider_error,
                )
            raise cancelled
        if delete_error is not None:
            raise delete_error
