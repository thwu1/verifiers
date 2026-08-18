"""Remote VMVM runtime backed by a vacli-leased container."""

import asyncio
import contextlib
import logging
import shlex
from pathlib import PurePosixPath
from typing import ClassVar, Literal, Protocol, TypedDict

from pydantic import Field
from pydantic_config import BaseConfig

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import ProgramResult, Runtime, open_tunnel

logger = logging.getLogger(__name__)


class VMVMBashResult(TypedDict):
    status: Literal["success", "error"]
    output: str
    error_type: Literal["none", "timeout", "too_long", "exit", "broken_pipe", "other"]
    exit_code: int


class VMVMBackend(Protocol):
    def run_bash(self, command: str, timeout: float = 60.0) -> VMVMBashResult: ...

    def transfer_file(self, file_content: str | bytes, remote_path: str) -> None: ...

    def read_file(self, remote_path: str) -> bytes: ...

    def open_host_tunnel(self, local_port: int) -> tuple[object, str]: ...

    def close_host_tunnel(self, tunnel: object) -> None: ...

    def destroy(self) -> None: ...

    def get_debugging_info(self) -> dict[str, object]: ...


class VMVMConfig(BaseConfig):
    type: Literal["vmvm"] = "vmvm"
    image: str = "python:3.11-slim"
    workdir: str = "/app"
    session_timeout: float = Field(3600.0, gt=0)
    """Maximum duration of one command inside the VMVM container, in seconds."""
    fallback_image: str | None = None
    """Secondary image to pull when the primary image is unavailable."""
    tenant_id: str = "async_2347641"
    lease_ttl: str = "60s"
    """Lease expiry after missed auto-renewals; vacli renews while the runtime is active."""
    tunnel_ready_timeout: float = Field(120.0, gt=0)
    sshd_ready_timeout: float = Field(180.0, gt=0)
    max_session_buffer_size: int | None = Field(16 * 1024 * 1024, gt=0)
    """Maximum captured command output. None uses the backend default."""
    # VMVM leases have provider-selected resources. Accept task resource declarations so
    # tasksets compose without warnings, but do not claim that the lease enforces them.
    cpu: float | None = None
    memory: float | None = None
    gpu: str | None = None
    disk: float | None = None


def create_backend(config: VMVMConfig) -> VMVMBackend:
    """Construct the internal vacli backend only when VMVM is selected."""
    try:
        from vmvm_tb_v2._vacli.backend import VacliVMVMBackend, VacliVMVMConfig
    except ModuleNotFoundError as error:
        if error.name != "vmvm_tb_v2":
            raise
        raise ModuleNotFoundError(
            "VMVMRuntime requires the local vmvm_tb_v2 package; add "
            "environments/vmvm_tb_v2 to PYTHONPATH before launching the evaluator"
        ) from error

    backend_config = VacliVMVMConfig(
        image_url=config.image,
        fallback_image_url=config.fallback_image,
        work_dir=config.workdir,
        session_timeout=config.session_timeout,
        tenant_id=config.tenant_id,
        lease_ttl=config.lease_ttl,
        tunnel_ready_timeout=config.tunnel_ready_timeout,
        sshd_ready_timeout=config.sshd_ready_timeout,
        max_session_buffer_size=config.max_session_buffer_size,
    )
    return VacliVMVMBackend(backend_config)


def shell_command(argv: list[str], env: dict[str, str], workdir: str) -> str:
    """Render one argv invocation for the backend's persistent Bash session."""
    command = shlex.join(argv)
    if env:
        assignments = " ".join(shlex.quote(f"{key}={value}") for key, value in env.items())
        command = f"env {assignments} {command}"
    return f"cd {shlex.quote(workdir)} && {command}"


class VMVMRuntime(Runtime):
    """Runs a harness inside a podman container on a vacli-leased VMVM."""

    is_local: ClassVar[bool] = False
    instance_host_endpoint: ClassVar[bool] = True

    def __init__(self, config: VMVMConfig, name: str | None = None) -> None:
        super().__init__(name)
        self.config = config
        self._backend: VMVMBackend | None = None
        self._descriptor: str | None = None
        self._run_lock = asyncio.Lock()

    @property
    def descriptor(self) -> str | None:
        return self._descriptor

    @property
    def backend(self) -> VMVMBackend:
        if self._backend is None:
            raise RuntimeError("VMVM runtime is not running")
        return self._backend

    async def start(self) -> None:
        backend: VMVMBackend | None = None
        try:
            backend = await asyncio.to_thread(create_backend, self.config)
            result = await asyncio.to_thread(
                backend.run_bash,
                f"mkdir -p {shlex.quote(self.config.workdir)}",
                self.config.session_timeout,
            )
            if result["exit_code"] != 0:
                raise RuntimeError(result["output"])
        except Exception as error:
            if backend is not None:
                await asyncio.to_thread(backend.destroy)
            raise SandboxError(f"VMVM provisioning failed: {error}") from error
        self._backend = backend
        info = backend.get_debugging_info()
        container_id = info.get("container_id")
        self._descriptor = str(container_id) if container_id is not None else self.name
        logger.info("vmvm: container %s up (image=%s)", self._descriptor, self.config.image)

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        command = shell_command(argv, env, self.config.workdir)
        try:
            async with self._run_lock:
                result = await asyncio.to_thread(self.backend.run_bash, command, self.config.session_timeout)
        except Exception as error:
            raise SandboxError(f"VMVM exec failed: {error}") from error
        if result["exit_code"] < 0:
            raise SandboxError(f"VMVM exec failed ({result['error_type']}): {result['output']}")
        return ProgramResult(exit_code=result["exit_code"], stdout=result["output"], stderr="")

    async def run_background(self, argv: list[str], env: dict[str, str], log: str) -> None:
        inner = f"nohup {shlex.join(argv)} > {shlex.quote(log)} 2>&1 < /dev/null &"
        result = await self.run(["sh", "-c", inner], env)
        if result.exit_code != 0:
            raise SandboxError(f"VMVM background launch failed: {result.stdout.strip()}")

    def absolute_path(self, path: str) -> str:
        target = PurePosixPath(path)
        if target.is_absolute():
            return str(target)
        return str(PurePosixPath(self.config.workdir) / target)

    async def read(self, path: str) -> bytes:
        target = self.absolute_path(path)
        try:
            return await asyncio.to_thread(self.backend.read_file, target)
        except Exception as error:
            raise SandboxError(f"read {path!r}: {error}") from error

    async def write(self, path: str, data: bytes) -> None:
        target = self.absolute_path(path)
        try:
            await asyncio.to_thread(self.backend.transfer_file, data, target)
        except Exception as error:
            raise SandboxError(f"write {path!r}: {error}") from error

    @contextlib.asynccontextmanager
    async def host_endpoint(self, port: int):
        async def start() -> tuple[object, str]:
            return await asyncio.to_thread(self.backend.open_host_tunnel, port)

        tunnel, url = await open_tunnel(start, f"VMVM host tunnel (port {port})")
        try:
            yield url
        finally:
            await asyncio.to_thread(self.backend.close_host_tunnel, tunnel)

    def cleanup(self) -> None:
        backend, self._backend = self._backend, None
        if backend is not None:
            backend.destroy()
