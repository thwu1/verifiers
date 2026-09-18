"""Remote VMVM runtime backed by a vacli-leased container."""

import asyncio
import atexit
import concurrent.futures
import contextlib
import logging
import os
import shlex
import threading
from pathlib import PurePosixPath
from typing import ClassVar, Literal, Protocol, TypedDict
from urllib.parse import urlsplit

from pydantic import Field
from pydantic_config import BaseConfig

from verifiers.v1.errors import SandboxError, TunnelError
from verifiers.v1.runtimes.base import ProgramResult, Runtime, open_tunnel

logger = logging.getLogger(__name__)
MAX_TRANSPORT_RECOVERY_ATTEMPTS = 5
MAX_BACKEND_INIT_WORKERS = min(32, (os.cpu_count() or 1) + 4)
ISOLATED_PROGRAM_ENV = {
    "HTTP_PROXY": "",
    "HTTPS_PROXY": "",
    "ALL_PROXY": "",
    "http_proxy": "",
    "https_proxy": "",
    "all_proxy": "",
    "NO_PROXY": "*",
    "no_proxy": "*",
}

_backend_init_executor: concurrent.futures.ThreadPoolExecutor | None = None
_backend_init_executor_pid: int | None = None
_backend_init_lock = threading.Lock()
_pending_backend_cleanups: set[asyncio.Task[None]] = set()


def _get_backend_init_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Return the lazy process-local pool used only for blocking VMVM setup."""
    global _backend_init_executor, _backend_init_executor_pid
    process_id = os.getpid()
    with _backend_init_lock:
        if _backend_init_executor is None or _backend_init_executor_pid != process_id:
            # An executor inherited across fork has no live worker threads in the
            # child.  Drop the inherited object without touching its locks and
            # create a process-local pool on first use.
            _backend_init_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=MAX_BACKEND_INIT_WORKERS,
                thread_name_prefix="vmvm-backend-init",
            )
            _backend_init_executor_pid = process_id
        return _backend_init_executor


def _shutdown_backend_init_executor() -> None:
    """Idempotently stop the current process's lazy setup pool."""
    global _backend_init_executor, _backend_init_executor_pid
    process_id = os.getpid()
    with _backend_init_lock:
        executor = _backend_init_executor
        owner = _backend_init_executor_pid
        _backend_init_executor = None
        _backend_init_executor_pid = None
    if executor is not None and owner == process_id:
        executor.shutdown(wait=False, cancel_futures=True)


def _reset_backend_init_executor_after_fork() -> None:
    """Discard parent-only executor and lock state in a forked child."""
    global _backend_init_executor, _backend_init_executor_pid, _backend_init_lock
    global _pending_backend_cleanups
    _backend_init_executor = None
    _backend_init_executor_pid = None
    _backend_init_lock = threading.Lock()
    _pending_backend_cleanups = set()


atexit.register(_shutdown_backend_init_executor)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_backend_init_executor_after_fork)


class VMVMBashResult(TypedDict):
    status: Literal["success", "error"]
    output: str
    error_type: Literal["none", "timeout", "too_long", "exit", "broken_pipe", "other"]
    exit_code: int


class VMVMBackend(Protocol):
    def run_bash(self, command: str, timeout: float = 60.0) -> VMVMBashResult: ...

    def restart_session(self) -> bool: ...

    def recover_last(self) -> VMVMBashResult | None: ...

    def transfer_file(self, file_content: str | bytes, remote_path: str) -> None: ...

    def read_file(self, remote_path: str) -> bytes: ...

    def prepare_network_isolation(self) -> None: ...

    def activate_network_isolation(self) -> None: ...

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
    # The VM tier remains provider-selected. CPU and memory declarations are applied as
    # Podman cgroup limits; they cannot increase the leased VM's physical capacity.
    cpu: float | None = Field(None, gt=0)
    memory: float | None = Field(None, gt=0)
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
        cpu=config.cpu,
        memory_gb=config.memory,
    )
    return VacliVMVMBackend(backend_config)


async def _destroy_backend_safely(backend: VMVMBackend) -> None:
    try:
        await asyncio.to_thread(backend.destroy)
    except Exception:
        logger.exception("vmvm: deferred backend cleanup failed")


async def _destroy_backend_when_ready(future: asyncio.Future[VMVMBackend]) -> None:
    try:
        backend = await future
    except asyncio.CancelledError:
        return
    except Exception:  # noqa: BLE001 - a failed constructor owns no backend to clean up
        return
    await _destroy_backend_safely(backend)


async def _destroy_backend_after_probe(
    probe: asyncio.Task[VMVMBashResult],
    backend: VMVMBackend,
) -> None:
    try:
        await probe
    except asyncio.CancelledError:
        pass
    except Exception:  # noqa: BLE001 - probe failure must not skip lease cleanup
        logger.warning("vmvm: initial probe failed during deferred cleanup")
    await _destroy_backend_safely(backend)


def _track_backend_cleanup(task: asyncio.Task[None]) -> None:
    _pending_backend_cleanups.add(task)
    task.add_done_callback(_pending_backend_cleanups.discard)


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
        self._network_lock = asyncio.Lock()
        self._network_mode: Literal["public", "no-network"] = "public"
        self._network_active = False
        self._deferred_network_commands: list[tuple[list[str], dict[str, str]]] = []

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
        init_future: asyncio.Future[VMVMBackend] | None = None
        probe: asyncio.Task[VMVMBashResult] | None = None
        try:
            init_future = asyncio.wrap_future(
                _get_backend_init_executor().submit(create_backend, self.config)
            )
            backend = await asyncio.shield(init_future)
            probe = asyncio.create_task(
                asyncio.to_thread(
                    backend.run_bash,
                    f"mkdir -p {shlex.quote(self.config.workdir)}",
                    self.config.session_timeout,
                )
            )
            result = await asyncio.shield(probe)
            if result["exit_code"] != 0:
                raise RuntimeError(result["output"])
        except asyncio.CancelledError:
            if backend is None and init_future is not None:
                _track_backend_cleanup(asyncio.create_task(_destroy_backend_when_ready(init_future)))
            elif backend is not None and probe is not None:
                _track_backend_cleanup(asyncio.create_task(_destroy_backend_after_probe(probe, backend)))
            elif backend is not None:
                _track_backend_cleanup(asyncio.create_task(_destroy_backend_safely(backend)))
            raise
        except Exception as error:
            if backend is not None:
                await _destroy_backend_safely(backend)
            raise SandboxError(f"VMVM provisioning failed: {error}") from error
        self._backend = backend
        info = backend.get_debugging_info()
        container_id = info.get("container_id")
        self._descriptor = str(container_id) if container_id is not None else self.name
        logger.info("vmvm: container %s up (image=%s)", self._descriptor, self.config.image)

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        if self._network_active:
            env = {**env, **ISOLATED_PROGRAM_ENV}
        command = shell_command(argv, env, self.config.workdir)
        async with self._run_lock:
            try:
                result = await asyncio.to_thread(self.backend.run_bash, command, self.config.session_timeout)
            except Exception as error:
                raise SandboxError(f"VMVM exec failed: {error}") from error

            for attempt in range(1, MAX_TRANSPORT_RECOVERY_ATTEMPTS + 1):
                if result["exit_code"] >= 0 or result["error_type"] != "broken_pipe":
                    break
                logger.warning(
                    "vmvm: transport dropped; recovering the in-flight command (%d/%d)",
                    attempt,
                    MAX_TRANSPORT_RECOVERY_ATTEMPTS,
                )
                try:
                    restarted = await asyncio.to_thread(self.backend.restart_session)
                except Exception as error:
                    raise SandboxError(f"VMVM reconnect failed: {error}") from error
                if not restarted:
                    raise SandboxError("VMVM reconnect failed: sandbox state is unavailable")
                try:
                    recovered = await asyncio.to_thread(self.backend.recover_last)
                except Exception as error:
                    raise SandboxError(f"VMVM command recovery failed: {error}") from error
                if recovered is None:
                    raise SandboxError("VMVM command recovery failed: exact-once execution cannot be proven")
                result = recovered
            if result["exit_code"] < 0 and result["error_type"] == "broken_pipe":
                raise SandboxError(
                    "VMVM exec failed: transport remained unavailable after "
                    f"{MAX_TRANSPORT_RECOVERY_ATTEMPTS} recovery attempts"
                )
        if result["exit_code"] < 0:
            raise SandboxError(f"VMVM exec failed ({result['error_type']}): {result['output']}")
        return ProgramResult(exit_code=result["exit_code"], stdout=result["output"], stderr="")

    async def run_background(self, argv: list[str], env: dict[str, str], log: str) -> None:
        inner = f"nohup {shlex.join(argv)} > {shlex.quote(log)} 2>&1 < /dev/null &"
        result = await self.run(["sh", "-c", inner], env)
        if result.exit_code != 0:
            raise SandboxError(f"VMVM background launch failed: {result.stdout.strip()}")

    async def configure_network_policy(
        self,
        mode: Literal["public", "no-network"],
    ) -> None:
        """Prepare a monotonic public-to-isolated network policy transition."""
        if mode == self._network_mode:
            return
        if self._network_mode == "no-network":
            raise SandboxError("VMVM cannot relax an active no-network policy back to public")
        try:
            await asyncio.to_thread(self.backend.prepare_network_isolation)
        except Exception as error:
            raise SandboxError(f"VMVM network-isolation preparation failed: {error}") from error
        self._network_mode = "no-network"

    def defer_until_network_isolated(
        self,
        argv: list[str],
        env: dict[str, str] | None = None,
    ) -> None:
        if self._network_mode != "no-network":
            raise RuntimeError("deferred startup requires a no-network policy")
        self._deferred_network_commands.append((list(argv), dict(env or {})))

    async def activate_network_policy(self) -> None:
        """Tighten the prepared workload network before untrusted execution."""
        if self._network_mode != "no-network":
            return
        async with self._network_lock:
            if not self._network_active:
                try:
                    await asyncio.to_thread(self.backend.activate_network_isolation)
                except Exception as error:
                    raise SandboxError(f"VMVM network-isolation activation failed: {error}") from error
                self._network_active = True
            commands, self._deferred_network_commands = (
                self._deferred_network_commands,
                [],
            )
            for argv, env in commands:
                result = await self.run(argv, env)
                if result.exit_code != 0:
                    raise SandboxError(f"VMVM deferred isolated startup failed: {result.stdout[-2000:]}")

    async def run_program(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        await self.activate_network_policy()
        return await self.run(argv, env)

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

    async def host_endpoint_is_reachable(self, url: str) -> bool:
        """Check the workload-to-host HTTP path without consulting proxy settings."""
        endpoint = urlsplit(url)
        if endpoint.hostname is None or endpoint.port is None:
            return False
        python_probe = (
            "import socket; "
            f"s=socket.create_connection(({endpoint.hostname!r}, {endpoint.port}), timeout=5); "
            "s.settimeout(5); "
            "s.sendall(b'GET / HTTP/1.0\\r\\nHost: localhost\\r\\n\\r\\n'); "
            "data=s.recv(16); s.close(); assert data.startswith(b'HTTP/')"
        )
        shell_probe = shlex.quote("printf 'GET / HTTP/1.0\\r\\nHost: localhost\\r\\n\\r\\n'")
        probe_url = shlex.quote(f"http://{endpoint.hostname}:{endpoint.port}/")
        probe = (
            "if command -v python3 >/dev/null 2>&1; then "
            f"exec python3 -c {shlex.quote(python_probe)}; "
            "elif command -v python >/dev/null 2>&1; then "
            f"exec python -c {shlex.quote(python_probe)}; "
            "elif command -v nc >/dev/null 2>&1; then "
            f"{shell_probe} | nc -w 5 {shlex.quote(endpoint.hostname)} {endpoint.port} | head -c 5 | grep -q '^HTTP/'; "
            "elif command -v busybox >/dev/null 2>&1; then "
            f"{shell_probe} | busybox nc -w 5 {shlex.quote(endpoint.hostname)} {endpoint.port} | "
            "head -c 5 | grep -q '^HTTP/'; "
            "elif command -v curl >/dev/null 2>&1; then "
            f"exec curl --noproxy '*' --silent --show-error --connect-timeout 5 --max-time 5 "
            f"--output /dev/null {probe_url}; "
            "elif command -v wget >/dev/null 2>&1; then "
            f"headers=$(wget --no-proxy --server-response --timeout=5 --tries=1 "
            f"--output-document=/dev/null {probe_url} 2>&1 || true); "
            "printf '%s\\n' \"$headers\" | grep -q 'HTTP/'; "
            "else exit 125; fi"
        )
        result = await self.run(["sh", "-c", probe], ISOLATED_PROGRAM_ENV)
        return result.exit_code == 0

    @contextlib.asynccontextmanager
    async def host_endpoint(self, port: int):
        async def start() -> tuple[object, str]:
            return await asyncio.to_thread(self.backend.open_host_tunnel, port)

        tunnel, url = await open_tunnel(start, f"VMVM host tunnel (port {port})")
        try:
            await self.activate_network_policy()
            if self._network_active and not await self.host_endpoint_is_reachable(url):
                raise TunnelError("VMVM host tunnel was unreachable after no-network activation")
            try:
                yield url
            except Exception as body_error:
                try:
                    reachable = await self.host_endpoint_is_reachable(url)
                except Exception:  # noqa: BLE001 - any probe failure means the tunnel is unavailable
                    reachable = False
                if not reachable:
                    raise TunnelError("VMVM host tunnel became unreachable during the rollout") from body_error
                raise
        finally:
            await asyncio.to_thread(self.backend.close_host_tunnel, tunnel)

    def cleanup(self) -> None:
        backend, self._backend = self._backend, None
        if backend is not None:
            backend.destroy()
