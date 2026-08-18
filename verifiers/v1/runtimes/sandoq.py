"""Remote Sandoq runtime backed by the pinned OCI runner provider."""

import asyncio
import contextlib
import logging
import shlex
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Literal

from pydantic import Field
from pydantic_config import BaseConfig

from verifiers.v1.errors import SandboxError
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
    host_tunnel: Literal["modal", "prime"] = "modal"
    """How a sandbox reaches host interception services. Modal needs no Prime credentials."""


def create_client(config: SandoqConfig) -> Any:
    """Construct the PR-pinned provider lazily so other runtimes need no internal dependency."""
    try:
        if config.mode == "oci-runner":
            from sandoq_provider.oci_client import (
                OCIRunnerAsyncSandboxClient,
                get_oci_config,
                read_token_file,
            )

            read_token_file(get_oci_config().token_file)
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
        gpu_type, gpu_count = parse_gpu(self.config.gpu)
        request = CreateSandboxRequest(
            name=self.name,
            docker_image=self.config.image,
            start_command="tail -f /dev/null",
            cpu_cores=self.config.cpu,
            memory_gb=self.config.memory,
            disk_size_gb=self.config.disk,
            gpu_count=gpu_count,
            gpu_type=gpu_type,
            network_access=self.config.network_access,
            timeout_minutes=24 * 60,
            environment_vars={"OCI_EXPECTED_WORKDIR": _OCI_BOOTSTRAP_WORKDIR}
            if self.config.mode == "oci-runner"
            else None,
        )
        try:
            async with creation_limiter(self.config.creates_per_sec, "sandoq-sandbox") or contextlib.nullcontext():
                sandbox = await client.create(request)
                self._sandbox_id = str(sandbox.id)
                self._active = True
                await client.wait_for_creation(self._sandbox_id)
            result = await self._run(
                client,
                ["mkdir", "-p", self.config.workdir],
                {},
                working_dir=_OCI_BOOTSTRAP_WORKDIR if self.config.mode == "oci-runner" else "/",
            )
            if result.exit_code != 0:
                raise RuntimeError(result.stderr or result.stdout)
        except asyncio.CancelledError:
            if self._active and self._sandbox_id is not None:
                with contextlib.suppress(Exception):
                    await asyncio.shield(client.delete(self._sandbox_id))
                self._active = False
            with contextlib.suppress(Exception):
                await client.aclose()
            raise
        except Exception as error:
            if self._active and self._sandbox_id is not None:
                with contextlib.suppress(Exception):
                    await asyncio.shield(client.delete(self._sandbox_id))
                self._active = False
            with contextlib.suppress(Exception):
                await client.aclose()
            raise SandboxError(f"Sandoq provisioning failed: {error}") from error
        self._client = client
        logger.info(
            "sandoq: sandbox %s up (mode=%s image=%s)",
            self._sandbox_id,
            self.config.mode,
            self.config.image,
        )

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
        if self.config.host_tunnel == "prime":
            async with shared_host_endpoint(port, is_local=False) as url:
                yield url
            return
        async with modal_host_endpoint(port) as url:
            yield url

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
        client, self._client = self._client, None
        if client is None:
            return
        try:
            if self._active and self._sandbox_id is not None:
                await client.delete(self._sandbox_id)
                self._active = False
        except Exception as error:
            logger.warning(
                "sandoq: failed to release sandbox %s: %s",
                self._sandbox_id,
                error,
            )
        finally:
            with contextlib.suppress(Exception):
                await client.aclose()
