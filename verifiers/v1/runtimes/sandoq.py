"""Remote Sandoq runtime backed by the pinned OCI runner provider."""

import asyncio
import contextlib
import json
import logging
import os
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

_PUBLIC_OCI_ENVIRONMENT = "oci-runner"
_ISOLATED_OCI_ENVIRONMENT = "oci-runner-firecracker"
_PRODUCTION_ECR_REGISTRY = "168653207203.dkr.ecr.us-east-2.amazonaws.com"
_PRODUCTION_ECR_REGION = "us-east-2"
_PRODUCTION_ECR_PULL_THROUGH_PREFIX = "pt_dockerio"


def _host_harness_environment(config: "SandoqConfig") -> str | None:
    """Return the one environment approved for an explicit host-harness profile."""
    if (
        config.mode != "oci-runner"
        or config.host_tunnel != "none"
        or config.ecr_token_file is None
        or not config.ecr_token_file.is_absolute()
    ):
        return None
    if config.network_access:
        return _PUBLIC_OCI_ENVIRONMENT
    return _ISOLATED_OCI_ENVIRONMENT


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
    provisioning_retries: int = Field(1, ge=0, le=3)
    """Retries after a failed sandbox assignment has been conclusively cleaned up."""
    host_tunnel: Literal["none", "sandoq", "modal", "prime"] = "modal"
    """How a sandbox reaches host interception services."""
    guest_tunnel_url: str = "http://127.0.0.1:8485"
    """Loopback endpoint exposed by the Sandoq Firecracker environment."""
    tunnel_pool_size: int = Field(8, ge=1, le=64)
    tunnel_ready_timeout: float = Field(30.0, gt=0, le=120)
    buffered_chat_completions: bool = False
    """Buffer guest SSE calls into exact non-streaming provider responses."""
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
            raise ValueError("guest_tunnel_url must be an explicit HTTP loopback URL with a port")
        if self.mode == "oci-runner" and self.host_tunnel == "none":
            expected_environment = _host_harness_environment(self)
            if expected_environment is None or self.expected_environment != expected_environment:
                raise ValueError(
                    "Sandoq host-side execution requires an approved network/environment "
                    "profile and an absolute ECR token path"
                )
        elif self.mode == "oci-runner" and not self.network_access:
            raise ValueError("Sandoq OCI no-network execution requires a host-side harness and no host tunnel")
        if self.host_tunnel == "none" and self.mode != "oci-runner":
            raise ValueError("Sandoq host-side execution is supported only by OCI runner mode")
        if self.buffered_chat_completions and (self.mode != "oci-runner" or self.host_tunnel != "sandoq"):
            raise ValueError("buffered_chat_completions requires the native Sandoq host tunnel")
        return self


def create_client(config: SandoqConfig) -> Any:
    """Construct the maintained Sandoq provider without loading it for other runtimes."""
    host_harness_environment = _host_harness_environment(config)
    if config.host_tunnel == "none" and (
        host_harness_environment is None or config.expected_environment != host_harness_environment
    ):
        raise SandboxError("Sandoq host-side configuration is incomplete")
    if config.mode == "oci-runner" and not config.network_access and config.host_tunnel != "none":
        raise SandboxError("Sandoq OCI no-network execution requires a host-side harness and no host tunnel")
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
                raise SandboxError("Sandoq host-side harnesses require the provider's default task-network setting")
            if config.host_tunnel == "sandoq" and oci_config.task_network != "host":
                raise SandboxError("Sandoq native reverse tunnels require nested host networking")
            if config.host_tunnel == "none":
                assert host_harness_environment is not None
                ecr_token_file = oci_config.ecr.token_file
                try:
                    ecr_token_stat = ecr_token_file.lstat()
                except (AttributeError, OSError) as error:
                    raise SandboxError("Sandoq host-side execution requires an available ECR token file") from error
                if (
                    config.expected_environment != host_harness_environment
                    or oci_config.environment != host_harness_environment
                    or oci_config.task_network != "none"
                    or not oci_config.ecr.enabled
                    or oci_config.ecr.registry != _PRODUCTION_ECR_REGISTRY
                    or oci_config.ecr.region != _PRODUCTION_ECR_REGION
                    or oci_config.ecr.pull_through_prefix != _PRODUCTION_ECR_PULL_THROUGH_PREFIX
                    or config.ecr_token_file is None
                    or not config.ecr_token_file.is_absolute()
                    or not ecr_token_file.is_absolute()
                    or ecr_token_file != config.ecr_token_file.expanduser()
                    or ecr_token_file.is_symlink()
                    or not stat.S_ISREG(ecr_token_stat.st_mode)
                    or stat.S_IMODE(ecr_token_stat.st_mode) != 0o600
                    or ecr_token_stat.st_uid != os.getuid()
                    or ecr_token_stat.st_nlink != 1
                ):
                    raise SandboxError(
                        "Sandoq host-side execution requires the selected OCI environment "
                        "and authenticated production ECR pull-through"
                    )
                read_secret_file(ecr_token_file, "ECR token", SandboxError)
            if not config.network_access:
                if (
                    getattr(oci_config, "allow_dockerhub_fallback", True)
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
            read_token_file(oci_config.token_file)
            return OCIRunnerAsyncSandboxClient()
        from sandoq_provider.client import SandoqAsyncSandboxClient

        return SandoqAsyncSandboxClient()
    except ModuleNotFoundError as error:
        if error.name not in {"sandoq_provider", "sandoq_client"}:
            raise
        raise ModuleNotFoundError(
            "SandoqRuntime requires Prime-RL extensions/sandoq on PYTHONPATH and its official sandoq-client dependency"
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
        self._program_cleanup_error: str | None = None

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
            # prime-sandboxes 0.2.42 represents container mode explicitly via
            # ``vm=False``. Network policy is owned and prevalidated by the
            # maintained Sandoq OCI provider, not by this request model.
            vm=False,
            timeout_minutes=24 * 60,
            # The authoritative OCI provider validates this directory before it
            # exposes Bash.  Passing a generic bootstrap directory and creating
            # the task workdir afterwards would bypass the image/workdir
            # integrity check and could run a task in an empty directory.
            environment_vars={"OCI_EXPECTED_WORKDIR": self.config.workdir}
            if self.config.mode == "oci-runner"
            else None,
        )
        total_attempts = self.config.provisioning_retries + 1
        for attempt in range(1, total_attempts + 1):
            self._sandbox_id = None
            self._active = False
            try:
                async with creation_limiter(self.config.creates_per_sec, "sandoq-sandbox") or contextlib.nullcontext():
                    sandbox = await client.create(request)
                    self._sandbox_id = str(sandbox.id)
                    self._active = True
                    await client.wait_for_creation(self._sandbox_id)
                if self.config.mode == "environment":
                    result = await self._run(
                        client,
                        ["mkdir", "-p", self.config.workdir],
                        {},
                        working_dir="/",
                    )
                    if result.exit_code != 0:
                        raise RuntimeError(result.stderr or result.stdout)
            except asyncio.CancelledError as error:
                _, delete_error, _ = await self._cleanup_failed_provisioning(client, error)
                if delete_error is not None:
                    logger.warning("sandoq: deletion was not verified after cancelled provisioning")
                await self._close_inactive_client(client)
                raise
            except Exception as error:
                cancelled, cleanup_error, retry_safe = await self._cleanup_failed_provisioning(client, error)
                if cancelled is not None:
                    await self._close_inactive_client(client)
                    raise cancelled
                if retry_safe and attempt < total_attempts:
                    logger.warning(
                        "sandoq: retrying sandbox provisioning after verified cleanup (retry %d/%d)",
                        attempt,
                        self.config.provisioning_retries,
                    )
                    continue
                await self._close_inactive_client(client)
                if cleanup_error is not None:
                    raise SandboxError("Sandoq provisioning failed; deletion was not verified") from None
                raise SandboxError(
                    f"Sandoq provisioning failed after {attempt} attempt{'s' if attempt != 1 else ''}"
                ) from None
            break
        logger.info(
            "sandoq: sandbox up (mode=%s image=%s)",
            self.config.mode,
            self.config.image,
        )

    async def _close_inactive_client(self, client: Any) -> None:
        if self._active:
            return
        self._sandbox_id = None
        self._client = None
        with contextlib.suppress(Exception):
            await client.aclose()

    async def _cleanup_failed_provisioning(
        self,
        client: Any,
        error: BaseException,
    ) -> tuple[asyncio.CancelledError | None, Exception | None, bool]:
        """Clean one pre-agent attempt and report whether another attempt is safe."""
        had_assignment = self._active and self._sandbox_id is not None
        provisioning_cancelled = self._find_cancellation(error)
        attached = getattr(error, "cleanup_result", None)
        if attached is not None:
            if not isinstance(attached, dict) or not self._cleanup_receipt_matches_active(attached):
                return (
                    provisioning_cancelled,
                    RuntimeError("provider cleanup receipt did not match the active assignment"),
                    False,
                )
            # The provider consumes and unregisters an assignment whenever it
            # attaches a terminal cleanup receipt, even if that receipt says the
            # underlying cleanup could not be verified. Never delete the same
            # assignment id again as though it were an outer-session id.
            self._active = False
            cleanup_error = (
                None
                if self._cleanup_response_verified(attached)
                else RuntimeError("provider reported unverified cleanup")
            )
            self._sandbox_id = None
            return (
                provisioning_cancelled,
                cleanup_error,
                had_assignment and cleanup_error is None and provisioning_cancelled is None,
            )

        cleanup_cancelled, cleanup_error = await self._delete_active(client)
        if not self._active:
            self._sandbox_id = None
        cancelled = provisioning_cancelled or cleanup_cancelled
        return cancelled, cleanup_error, had_assignment and cleanup_error is None and cancelled is None

    @staticmethod
    def _find_cancellation(error: BaseException) -> asyncio.CancelledError | None:
        """Recover cancellation wrapped by a provider while it reports cleanup failure."""
        current: BaseException | None = error
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, asyncio.CancelledError):
                return current
            current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
        return None

    def _cleanup_receipt_matches_active(self, response: dict[str, object]) -> bool:
        if self._sandbox_id is None:
            return False
        receipt_id = response.get("assignment_id", response.get("sandbox_id"))
        return isinstance(receipt_id, str) and receipt_id == self._sandbox_id

    @staticmethod
    def _cleanup_response_verified(response: dict[str, object]) -> bool:
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
        return bool(
            semantic_cleanup_verified
            and response.get("cleanup_verified", True) is True
            and (not response.get("error") or status == "poisoned")
        )

    async def _delete_active(self, client: Any) -> tuple[asyncio.CancelledError | None, Exception | None]:
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
        if not self._cleanup_response_verified(response):
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
        if result.exit_code == 75 and "execution status is unknown and the command was not replayed" in stderr:
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

    async def run_program(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        """Run one long-lived harness without holding a gateway request open.

        OCI runner shell requests have a proxy deadline below five minutes, while an
        agent rollout can legitimately run for hours.  The provider's background-job
        protocol launches the command exactly once, then observes it through
        idempotent status polls.  In particular, do not fall back to ``run`` after an
        ambiguous launch: replaying the agent would fork the rollout trace.
        """
        if self._client is None:
            raise SandboxError("Sandoq runtime is not running")
        try:
            result = await self._client.run_background_job(
                self.sandbox_id,
                shlex.join(argv),
                timeout=int(self.config.session_timeout),
                working_dir=self.config.workdir,
                env=env,
                poll_interval=3,
            )
        except asyncio.CancelledError:
            # OCI runner joins or quarantines its background job before propagating
            # cancellation. The generic environment API has no equivalent stop/join
            # primitive, so a cancelled program must never proceed to task scoring.
            if self.config.mode != "oci-runner":
                self._program_cleanup_error = "cancelled Sandoq environment program was not joined"
            raise
        except Exception as error:
            raise SandboxError(f"Sandoq program failed: {error}") from error
        return ProgramResult(
            exit_code=result.exit_code if result.exit_code is not None else -1,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
        )

    def ensure_usable(self) -> None:
        if self._program_cleanup_error is not None:
            raise SandboxError(self._program_cleanup_error)

    async def run_background(self, argv: list[str], env: dict[str, str], log: str) -> None:
        inner = f"nohup {shlex.join(argv)} > {shlex.quote(log)} 2>&1 < /dev/null &"
        result = await self.run(["sh", "-c", inner], env)
        if result.exit_code != 0:
            raise SandboxError(f"Sandoq background launch failed: {(result.stderr or result.stdout).strip()}")

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
            raise SandboxError(f"write {path!r}: could not create {parent}: {mkdir.stderr}")
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
            raise SandboxError("host_endpoint must not be called for a host-side Sandoq harness")
        if self.config.host_tunnel == "prime":
            async with shared_host_endpoint(port, is_local=False) as url:
                yield url
            return
        if self.config.host_tunnel == "modal":
            async with modal_host_endpoint(port) as url:
                yield url
            return
        if self.config.mode != "oci-runner":
            raise SandboxError("the native Sandoq host tunnel requires mode='oci-runner'")
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
            raise SandboxError(f"Sandoq session metadata is unavailable for {self.sandbox_id!r}")
        if self.config.expected_environment is not None:
            environment_matches = info.environment == self.config.expected_environment
        else:
            environment_matches = info.environment.startswith("oci-runner-firecracker")
        if not environment_matches:
            raise SandboxError("the native Sandoq host tunnel received an unexpected environment")
        if info.metadata.get("task_network") != "host":
            raise SandboxError("the native Sandoq host tunnel requires the provisioned task network to be host")
        tunnel_url = info.port_urls.get("tunnel")
        if not tunnel_url:
            raise SandboxError(f"Sandoq environment {info.environment!r} has no named 'tunnel' port")
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
                    f"Sandoq host tunnel failed to start: {type(start_error).__name__}: {start_error}"
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
                        f"Sandoq host tunnel cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}"
                    ) from cleanup_error
                logger.warning(
                    "sandoq: tunnel cleanup failed while preserving %s: %s",
                    type(original_error or cancelled).__name__,
                    cleanup_error,
                )
            if cancelled is not None and original_error is None:
                raise cancelled

    @contextlib.asynccontextmanager
    async def interception_endpoint(self, port: int, secret: str):
        if not self.config.buffered_chat_completions:
            async with self.host_endpoint(port) as url:
                yield url
            return
        try:
            from sandoq_provider.buffered_chat import BufferedChatCompletionsProxy
        except (ImportError, ModuleNotFoundError) as error:
            raise SandboxError(
                "buffered Sandoq interception requires sandoq_provider.buffered_chat on PYTHONPATH"
            ) from error

        proxy = BufferedChatCompletionsProxy(f"http://127.0.0.1:{port}", secret)
        await proxy.start()
        try:
            async with self.host_endpoint(proxy.port) as url:
                yield url
        finally:
            try:
                await proxy.close()
            finally:
                snapshot = proxy.stats.snapshot()
                errors = snapshot.pop("errors", [])
                paths = snapshot.pop("paths", {})
                snapshot["error_count"] = len(errors)
                snapshot["path_counts"] = {
                    path: int(paths.get(path, 0))
                    for path in ("/muse-code/models", "/v1/chat/completions", "/v1/responses")
                }
                snapshot["unknown_path_requests"] = sum(
                    int(count) for path, count in paths.items() if path not in snapshot["path_counts"]
                )
                logger.info(
                    "sandoq: buffered model proxy summary %s",
                    json.dumps(snapshot, sort_keys=True, separators=(",", ":")),
                )

    def cleanup(self) -> None:
        if not self._active:
            return
        self._active = False
        if self.config.mode == "oci-runner":
            from sandoq_provider.oci_client import delete_registered_sessions_sync

            result = delete_registered_sessions_sync()
            if result.get("failed"):
                logger.warning("sandoq: synchronous cleanup failures: %s", result["failed"])
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
            SandboxError(f"Sandoq deletion was not verified for {self._sandbox_id}: {provider_error}")
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
                    "sandoq: deletion was not verified while preserving cancellation for %s: %s",
                    self._sandbox_id,
                    provider_error,
                )
            raise cancelled
        if delete_error is not None:
            raise delete_error
