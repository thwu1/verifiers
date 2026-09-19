"""Remote VMVM runtime backed by a vacli-leased container."""

import asyncio
import contextlib
import logging
import shlex
import threading
from collections.abc import Awaitable, Callable
from pathlib import PurePosixPath
from typing import ClassVar, Literal, Protocol, TypedDict, TypeVar
from urllib.parse import urlsplit

from pydantic import Field
from pydantic_config import BaseConfig

from verifiers.v1.errors import SandboxError, TunnelError
from verifiers.v1.runtimes.base import ProgramResult, Runtime, open_tunnel

logger = logging.getLogger(__name__)
MAX_TRANSPORT_RECOVERY_ATTEMPTS = 5
COMMAND_CANCELLATION_GRACE_SECONDS = 600.0
_BackendResult = TypeVar("_BackendResult")
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


class VMVMBashResult(TypedDict):
    status: Literal["success", "error"]
    output: str
    error_type: Literal["none", "timeout", "too_long", "exit", "broken_pipe", "other"]
    exit_code: int


class VMVMBackend(Protocol):
    def run_bash(self, command: str, timeout: float = 60.0) -> VMVMBashResult: ...

    def run_bash_with_recovery(
        self,
        command: str,
        timeout: float,
        max_attempts: int,
    ) -> VMVMBashResult: ...

    def cancel_active_command(self, timeout: float) -> bool: ...

    def restart_session(self) -> bool: ...

    def recover_last(self) -> VMVMBashResult | None: ...

    def transfer_file(self, file_content: str | bytes, remote_path: str) -> None: ...

    def read_file(self, remote_path: str) -> bytes: ...

    def start_compose(self, compose_yaml: bytes) -> str: ...

    def run_service_bash(
        self,
        service: str,
        command: str,
        timeout: float = 60.0,
        env: dict[str, str] | None = None,
        user: str | int | None = None,
    ) -> VMVMBashResult: ...

    def read_service_file(self, service: str, remote_path: str) -> bytes: ...

    def run_root_bash(self, command: str, timeout: float = 60.0) -> VMVMBashResult: ...

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


def create_backend(
    config: VMVMConfig,
    cancel_event: threading.Event | None = None,
) -> VMVMBackend:
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
        provisioning_cancel_event=cancel_event,
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
        self._quarantined_backend: VMVMBackend | None = None
        self._descriptor: str | None = None
        self._cancellation_error: SandboxError | None = None
        self._run_lock = asyncio.Lock()
        self._network_lock = asyncio.Lock()
        self._network_mode: Literal["public", "no-network"] = "public"
        self._network_active = False
        self._deferred_network_commands: list[tuple[list[str], dict[str, str]]] = []
        self._worker_tasks: set[asyncio.Task[object]] = set()
        self._worker_threads: set[threading.Thread] = set()
        self._quarantine_tasks: set[asyncio.Task[None]] = set()
        self._quarantine_deadline: float | None = None

    @property
    def descriptor(self) -> str | None:
        return self._descriptor

    @property
    def backend(self) -> VMVMBackend:
        if self._backend is None:
            raise RuntimeError("VMVM runtime is not running")
        return self._backend

    def ensure_usable(self) -> None:
        if self._cancellation_error is not None:
            raise self._cancellation_error

    def _start_backend_call(
        self,
        action: str,
        function: Callable[..., _BackendResult],
        *args: object,
        daemon: bool = True,
    ) -> asyncio.Task[_BackendResult]:
        """Run cancellation/teardown on transient capacity reserved for this runtime.

        A transient thread makes both command and control submission independent of the
        event loop's shared executor.  Control therefore cannot queue behind commands,
        including a command that had not started when its asyncio task was cancelled.
        Unlike a global control pool, simultaneous rollout cancellations cannot starve
        one another; unlike a per-runtime executor, idle rollouts retain no threads.  The
        coroutine does not complete until the transient thread itself has exited.
        """

        async def call() -> _BackendResult:
            loop = asyncio.get_running_loop()
            finished = asyncio.Event()
            outcome: list[tuple[bool, object]] = []

            def invoke() -> None:
                try:
                    outcome.append((True, function(*args)))
                except BaseException as error:
                    outcome.append((False, error))
                finally:
                    with contextlib.suppress(RuntimeError):
                        loop.call_soon_threadsafe(finished.set)

            thread = threading.Thread(
                target=invoke,
                name=f"vmvm-worker-{id(self):x}-{action}",
                daemon=daemon,
            )
            self._worker_threads.add(thread)
            thread.start()
            try:
                await finished.wait()
                while thread.is_alive():
                    await asyncio.sleep(0)
                thread.join(timeout=0)
                succeeded, value = outcome[0]
                if succeeded:
                    return value  # type: ignore[return-value]
                assert isinstance(value, BaseException)
                raise value
            finally:
                if not thread.is_alive():
                    self._worker_threads.discard(thread)

        task = asyncio.create_task(call())
        self._worker_tasks.add(task)  # type: ignore[arg-type]

        def discard(completed: asyncio.Task[_BackendResult]) -> None:
            self._worker_tasks.discard(completed)  # type: ignore[arg-type]
            if not completed.cancelled():
                with contextlib.suppress(BaseException):
                    completed.exception()

        task.add_done_callback(discard)
        return task

    def _track_quarantine(self, coroutine: Awaitable[None]) -> None:
        task = asyncio.create_task(coroutine)
        self._quarantine_tasks.add(task)
        task.add_done_callback(self._quarantine_tasks.discard)

    async def _finish_quarantine(
        self,
        backend: VMVMBackend,
        workers: tuple[asyncio.Task[object], ...],
    ) -> None:
        """Retain backend ownership until every detached worker has really stopped."""
        for worker in workers:
            with contextlib.suppress(BaseException):
                await asyncio.shield(worker)
        final_destroy = self._start_backend_call("final-destroy", backend.destroy)
        with contextlib.suppress(BaseException):
            await asyncio.shield(final_destroy)
        if self._quarantined_backend is backend:
            self._quarantined_backend = None
            self._quarantine_deadline = None

    async def start(self) -> None:
        backend: VMVMBackend | None = None
        ownership_lock = threading.Lock()
        provisioning_cancelled = False
        provisioning_cancel_event = threading.Event()
        available_backend: VMVMBackend | None = None

        def create_owned_backend() -> VMVMBackend:
            nonlocal available_backend
            created = create_backend(self.config, provisioning_cancel_event)
            with ownership_lock:
                worker_must_destroy = provisioning_cancelled
                if not worker_must_destroy:
                    available_backend = created
            if worker_must_destroy:
                # This thread remains the synchronous owner even if cancellation
                # outlives the event loop: a late-created lease is torn down here,
                # rather than relying on an asyncio continuation to run.
                created.destroy()
                raise SandboxError("VMVM provisioning completed after cancellation")
            return created

        create_worker = self._start_backend_call(
            "provision",
            create_owned_backend,
        )
        try:
            try:
                backend = await asyncio.shield(create_worker)
                with ownership_lock:
                    available_backend = None
            except asyncio.CancelledError as cancellation:
                with ownership_lock:
                    provisioning_cancelled = True
                    backend = available_backend
                    available_backend = None
                provisioning_cancel_event.set()
                deadline = asyncio.get_running_loop().time() + COMMAND_CANCELLATION_GRACE_SECONDS
                if backend is not None:
                    self._backend = backend
                    try:
                        await self._destroy_and_drain(
                            backend,
                            (create_worker,),
                            deadline,
                            "VMVM provisioning was cancelled and teardown failed",
                        )
                    except SandboxError as error:
                        self._cancellation_error = error
                    raise cancellation
                remaining = deadline - asyncio.get_running_loop().time()
                try:
                    await asyncio.wait_for(asyncio.shield(create_worker), timeout=remaining)
                except BaseException as error:
                    self._cancellation_error = SandboxError(
                        "VMVM provisioning cancellation exceeded the cancellation grace"
                    )
                    self._cancellation_error.__cause__ = error
                    raise cancellation
                raise cancellation

            self._backend = backend
            mkdir_worker = self._start_backend_call(
                "provision-command",
                backend.run_bash,
                f"mkdir -p {shlex.quote(self.config.workdir)}",
                self.config.session_timeout,
            )
            try:
                result = await asyncio.shield(mkdir_worker)
            except asyncio.CancelledError as cancellation:
                try:
                    await self._cancel_command(backend, mkdir_worker)
                except SandboxError as error:
                    self._cancellation_error = error
                if self._backend is backend:
                    try:
                        await self.stop()
                    except BaseException as error:
                        self._cancellation_error = SandboxError(
                            "VMVM provisioning command was cancelled and teardown failed"
                        )
                        self._cancellation_error.__cause__ = error
                raise cancellation
            if result["exit_code"] != 0:
                raise RuntimeError(result["output"])
        except Exception as error:
            if backend is not None:
                try:
                    await self.stop()
                except Exception:
                    logger.warning("VMVM provisioning teardown failed", exc_info=True)
            raise SandboxError(f"VMVM provisioning failed: {error}") from error
        info = backend.get_debugging_info()
        container_id = info.get("container_id")
        self._descriptor = str(container_id) if container_id is not None else self.name
        logger.info("vmvm: container %s up (image=%s)", self._descriptor, self.config.image)

    async def _cancel_command(
        self,
        backend: VMVMBackend,
        worker: asyncio.Task[VMVMBashResult],
    ) -> None:
        """Drain a cancelled backend command before this runtime can be reused."""

        loop = asyncio.get_running_loop()
        deadline = loop.time() + COMMAND_CANCELLATION_GRACE_SECONDS
        interrupt_deadline = loop.time() + COMMAND_CANCELLATION_GRACE_SECONDS / 2

        async def wait_bounded(task, until: float = deadline):
            remaining = until - loop.time()
            if remaining <= 0:
                raise TimeoutError
            return await asyncio.wait_for(asyncio.shield(task), timeout=remaining)

        reusable = False
        cancellation_error: BaseException | None = None

        def cancel_with_remaining_budget() -> bool:
            remaining = interrupt_deadline - loop.time()
            if remaining <= 0:
                return False
            return backend.cancel_active_command(remaining)

        cancel_worker = self._start_backend_call(
            "cancel",
            cancel_with_remaining_budget,
        )
        try:
            reusable = await wait_bounded(cancel_worker, interrupt_deadline)
        except (Exception, asyncio.CancelledError) as error:
            cancellation_error = error

        if reusable:
            try:
                await wait_bounded(worker, interrupt_deadline)
            except Exception:
                # The cancelled command's result is intentionally discarded.  The
                # backend has joined it and restored a fresh shell for scoring.
                pass
            except asyncio.CancelledError:
                pass
            else:
                return
            reusable = worker.done()
            if reusable:
                return

        await self._destroy_and_drain(
            backend,
            (cancel_worker, worker),
            deadline,
            "VMVM command cancellation could not restore a safe runtime",
            cancellation_error,
        )

    async def _destroy_and_drain(
        self,
        backend: VMVMBackend,
        workers: tuple[asyncio.Task[object], ...],
        deadline: float,
        message: str,
        cause: BaseException | None = None,
    ) -> None:
        """Invalidate one backend, tear it down, and join every worker by one deadline."""

        if self._backend is backend:
            self._backend = None
        self._quarantined_backend = backend
        self._quarantine_deadline = deadline
        destroy_worker = self._start_backend_call("destroy", backend.destroy)
        all_workers = (destroy_worker, *workers)
        loop = asyncio.get_running_loop()
        first_error = cause
        for worker in all_workers:
            if not worker.done():
                remaining = deadline - loop.time()
                if remaining > 0:
                    try:
                        await asyncio.wait_for(asyncio.shield(worker), timeout=remaining)
                    except TimeoutError:
                        pass
                    except BaseException as error:
                        first_error = first_error or error
            if worker.done() and not worker.cancelled():
                try:
                    worker.result()
                except BaseException as error:
                    first_error = first_error or error
        if any(not worker.done() for worker in all_workers):
            self._track_quarantine(self._finish_quarantine(backend, all_workers))
            raise SandboxError(f"{message}; teardown workers exceeded the cancellation grace") from first_error

        # A non-command backend operation may create a resource after the first
        # destroy passed that resource's cleanup site.  Once every data worker is
        # drained, make one final idempotent teardown pass before releasing ownership.
        final_destroy = self._start_backend_call("final-destroy", backend.destroy)
        remaining = deadline - loop.time()
        if remaining > 0:
            try:
                await asyncio.wait_for(asyncio.shield(final_destroy), timeout=remaining)
            except TimeoutError as error:
                first_error = first_error or error
            except BaseException as error:
                first_error = first_error or error
        if not final_destroy.done():
            self._track_quarantine(self._finish_quarantine(backend, (final_destroy,)))
            raise SandboxError(f"{message}; final teardown exceeded the cancellation grace") from first_error
        if not final_destroy.cancelled():
            try:
                final_destroy.result()
            except BaseException as error:
                first_error = first_error or error
        if self._quarantined_backend is backend:
            self._quarantined_backend = None
            self._quarantine_deadline = None
        raise SandboxError(message) from first_error

    async def _call_backend(
        self,
        action: str,
        function: Callable[..., _BackendResult],
        *args: object,
    ) -> _BackendResult:
        """Run one synchronous backend operation with fail-closed cancellation."""

        backend = self.backend
        worker = self._start_backend_call(action, function, *args)
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError as error:
            deadline = asyncio.get_running_loop().time() + COMMAND_CANCELLATION_GRACE_SECONDS
            try:
                await self._destroy_and_drain(
                    backend,
                    (worker,),
                    deadline,
                    f"VMVM {action} was cancelled and the runtime was invalidated",
                    error,
                )
            except SandboxError as cancellation_error:
                self._cancellation_error = cancellation_error
            raise

    def _run_command(self, backend: VMVMBackend, command: str) -> VMVMBashResult:
        """Run and recover one command in the same cancellable worker thread."""
        result = backend.run_bash_with_recovery(
            command,
            self.config.session_timeout,
            MAX_TRANSPORT_RECOVERY_ATTEMPTS,
        )
        if result["exit_code"] < 0 and result["error_type"] == "broken_pipe":
            raise SandboxError(
                "VMVM exec failed: transport remained unavailable after "
                f"{MAX_TRANSPORT_RECOVERY_ATTEMPTS} recovery attempts"
            )
        return result

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        if self._network_active:
            env = {**env, **ISOLATED_PROGRAM_ENV}
        command = shell_command(argv, env, self.config.workdir)
        async with self._run_lock:
            backend = self.backend
            worker = self._start_backend_call("command", self._run_command, backend, command)
            try:
                result = await asyncio.shield(worker)
            except asyncio.CancelledError:
                try:
                    await self._cancel_command(backend, worker)
                except SandboxError as error:
                    self._cancellation_error = error
                raise
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
            async with self._run_lock:
                await self._call_backend(
                    "network-isolation preparation",
                    self.backend.prepare_network_isolation,
                )
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
                    async with self._run_lock:
                        await self._call_backend(
                            "network-isolation activation",
                            self.backend.activate_network_isolation,
                        )
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
            async with self._run_lock:
                return await self._call_backend("file read", self.backend.read_file, target)
        except Exception as error:
            raise SandboxError(f"read {path!r}: {error}") from error

    async def write(self, path: str, data: bytes) -> None:
        target = self.absolute_path(path)
        try:
            async with self._run_lock:
                await self._call_backend("file write", self.backend.transfer_file, data, target)
        except Exception as error:
            raise SandboxError(f"write {path!r}: {error}") from error

    async def start_compose(self, compose_yaml: bytes) -> str:
        try:
            async with self._run_lock:
                descriptor = await self._call_backend(
                    "Compose provisioning",
                    self.backend.start_compose,
                    compose_yaml,
                )
        except Exception as error:
            raise SandboxError(f"VMVM Compose provisioning failed: {error}") from error
        self._descriptor = descriptor
        return descriptor

    async def run_root(self, command: str) -> ProgramResult:
        try:
            async with self._run_lock:
                result = await self._call_backend(
                    "root command",
                    self.backend.run_root_bash,
                    command,
                    self.config.session_timeout,
                )
        except Exception as error:
            raise SandboxError(f"VMVM root command failed: {error}") from error
        if result["exit_code"] < 0:
            raise SandboxError(f"VMVM root command failed ({result['error_type']}): {result['output']}")
        return ProgramResult(exit_code=result["exit_code"], stdout=result["output"], stderr="")

    async def run_service(
        self,
        service: str,
        argv: list[str],
        env: dict[str, str],
        *,
        user: str | int | None = None,
    ) -> ProgramResult:
        if service in ("", "main"):
            return await self.run(argv, env)
        command = shlex.join(argv)
        try:
            async with self._run_lock:
                result = await self._call_backend(
                    "service command",
                    self.backend.run_service_bash,
                    service,
                    command,
                    self.config.session_timeout,
                    env,
                    user,
                )
        except Exception as error:
            raise SandboxError(f"VMVM service exec failed: {error}") from error
        if result["exit_code"] < 0:
            raise SandboxError(f"VMVM service exec failed ({result['error_type']}): {result['output']}")
        return ProgramResult(exit_code=result["exit_code"], stdout=result["output"], stderr="")

    async def read_service(self, service: str, path: str) -> bytes:
        if service in ("", "main"):
            return await self.read(path)
        try:
            async with self._run_lock:
                return await self._call_backend(
                    "service file read",
                    self.backend.read_service_file,
                    service,
                    path,
                )
        except Exception as error:
            raise SandboxError(f"read {path!r} from service {service!r}: {error}") from error

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
            async with self._run_lock:
                return await self._call_backend(
                    "host-tunnel open",
                    self.backend.open_host_tunnel,
                    port,
                )

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
                except Exception:
                    reachable = False
                if not reachable:
                    raise TunnelError("VMVM host tunnel became unreachable during the rollout") from body_error
                raise
        finally:
            if self._backend is not None:
                async with self._run_lock:
                    await self._call_backend(
                        "host-tunnel close",
                        self.backend.close_host_tunnel,
                        tunnel,
                    )

    async def stop(self) -> None:
        """Destroy the backend without depending on command-executor capacity."""
        async with self._run_lock:
            loop = asyncio.get_running_loop()
            deadline = self._quarantine_deadline or (loop.time() + COMMAND_CANCELLATION_GRACE_SECONDS)
            cancellation: asyncio.CancelledError | None = None
            teardown_error: BaseException | None = None

            async def drain(worker: asyncio.Future[object]) -> None:
                nonlocal cancellation, teardown_error
                while not worker.done():
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        teardown_error = teardown_error or SandboxError("VMVM teardown exceeded the cancellation grace")
                        return
                    try:
                        await asyncio.wait_for(asyncio.shield(worker), timeout=remaining)
                    except TimeoutError as error:
                        teardown_error = teardown_error or SandboxError("VMVM teardown exceeded the cancellation grace")
                        teardown_error.__cause__ = error
                        return
                    except asyncio.CancelledError as error:
                        cancellation = cancellation or error
                    except BaseException as error:
                        teardown_error = teardown_error or error
                if not worker.cancelled():
                    try:
                        worker.result()
                    except BaseException as error:
                        teardown_error = teardown_error or error

            while self._quarantine_tasks and loop.time() < deadline:
                pending = tuple(self._quarantine_tasks)
                await drain(asyncio.gather(*pending))
                for task in pending:
                    if task.done():
                        self._quarantine_tasks.discard(task)
            if self._quarantine_tasks:
                teardown_error = teardown_error or SandboxError("VMVM teardown exceeded the cancellation grace")

            backend, self._backend = self._backend, None
            if backend is not None:
                self._quarantined_backend = backend
                destroy_worker = self._start_backend_call("stop", backend.destroy)
                await drain(destroy_worker)
                if destroy_worker.done() and self._quarantined_backend is backend:
                    self._quarantined_backend = None
                    self._quarantine_deadline = None
                elif not destroy_worker.done():
                    self._quarantine_deadline = deadline
                    self._track_quarantine(self._finish_quarantine(backend, (destroy_worker,)))

            if cancellation is not None:
                raise cancellation
            if teardown_error is not None:
                raise teardown_error

    def cleanup(self) -> None:
        backend, self._backend = self._backend, None
        quarantined, self._quarantined_backend = self._quarantined_backend, None
        for owned in (backend, quarantined):
            if owned is not None:
                owned.destroy()
