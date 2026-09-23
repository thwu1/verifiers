"""Execution runtimes for harnesses.

Each runtime decides WHERE the program runs and HOW it reaches the host
interception server: subprocess (local), docker (local container), or prime /
modal / Sandoq / VMVM (remote sandbox). They share the `Runtime` contract, so the Environment is
runtime-agnostic. `RuntimeConfig` is the discriminated config union and
`make_runtime` builds the runtime matching a config.
"""

from typing import Annotated

from pydantic import Field

from verifiers.v1.runtimes.base import (
    HOST,
    ProgramResult,
    Runtime,
    host_endpoint,
    reachable_url,
    register,
)
from verifiers.v1.runtimes.docker import DockerConfig, DockerRuntime
from verifiers.v1.runtimes.modal import ModalConfig, ModalRuntime
from verifiers.v1.runtimes.prime import PrimeConfig, PrimeRuntime
from verifiers.v1.runtimes.sandoq import SandoqConfig, SandoqRuntime
from verifiers.v1.runtimes.subprocess import SubprocessConfig, SubprocessRuntime
from verifiers.v1.runtimes.vmvm import VMVMConfig, VMVMRuntime

RuntimeConfig = Annotated[
    SubprocessConfig | DockerConfig | PrimeConfig | ModalConfig | SandoqConfig | VMVMConfig,
    Field(discriminator="type"),
]


def _runtime_cls(config: RuntimeConfig) -> type[Runtime]:
    if isinstance(config, PrimeConfig):
        return PrimeRuntime
    if isinstance(config, ModalConfig):
        return ModalRuntime
    if isinstance(config, DockerConfig):
        return DockerRuntime
    if isinstance(config, SandoqConfig):
        return SandoqRuntime
    if isinstance(config, VMVMConfig):
        return VMVMRuntime
    return SubprocessRuntime


def make_runtime(config: RuntimeConfig, name: str | None = None) -> Runtime:
    runtime = _runtime_cls(config)(config, name)
    register(runtime)
    return runtime


def runtime_is_local(config: RuntimeConfig) -> bool:
    """Whether a runtime of this config shares the host network (so a program inside it reaches a
    host service at localhost, no tunnel) — read off the runtime class, without provisioning one.
    The interception pool / rollout use it to decide whether to tunnel their host port via
    `host_endpoint`."""
    return _runtime_cls(config).is_local


def runtime_has_instance_host_endpoint(config: RuntimeConfig) -> bool:
    """Whether host reachability depends on an already-started runtime instance."""
    return _runtime_cls(config).instance_host_endpoint


__all__ = [
    "ProgramResult",
    "Runtime",
    "RuntimeConfig",
    "make_runtime",
    "runtime_is_local",
    "runtime_has_instance_host_endpoint",
    "host_endpoint",
    "reachable_url",
    "HOST",
    "SubprocessConfig",
    "SubprocessRuntime",
    "DockerConfig",
    "DockerRuntime",
    "PrimeConfig",
    "PrimeRuntime",
    "ModalConfig",
    "ModalRuntime",
    "SandoqConfig",
    "SandoqRuntime",
    "VMVMConfig",
    "VMVMRuntime",
]
