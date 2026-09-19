"""A pool of shared interception servers, grown on demand, so N concurrent rollouts need
~N/multiplex servers + tunnels rather than one each.

Behind most remote runtimes each rollout's interception endpoint needs a rate-capped public
tunnel. Each shared `InterceptionServer` serves up to `multiplex` rollouts behind one such
tunnel. A runtime with instance-scoped host reachability, such as VMVM or Sandoq, still shares servers
but opens an SSH route from each provisioned instance. The pool is elastic: `acquire` reuses
a server with a free slot, else brings up a new one. The harness authenticates with a
per-rollout secret, which is what the server routes by.
"""

import asyncio
import logging
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

from verifiers.v1.interception.server import InterceptionServer, RolloutSession
from verifiers.v1.runtimes import (
    HOST,
    Runtime,
    RuntimeConfig,
    reachable_url,
    runtime_has_instance_host_endpoint,
    runtime_is_local,
)

logger = logging.getLogger(__name__)


@dataclass
class PooledServer:
    server: InterceptionServer
    # The shared reachable base for provider-independent tunnels. An instance-scoped runtime
    # (VMVM) opens its own route to this server while it holds a pool slot.
    base_url: str | None
    load: int = 0
    healthy: bool = True
    draining: int = 0


class InterceptionPool:
    """Shared interception servers. `multiplex` rollouts share one
    server (one tunnel behind a remote runtime); `acquire` hands a rollout a slot on one,
    bringing up a new server when all are at capacity."""

    def __init__(self, runtime_config: RuntimeConfig, multiplex: int) -> None:
        # The harness runtime's topology decides reachability: a remote one needs a host tunnel
        # to the interception port, a local one is reached at localhost. Read off the runtime
        # class (no provisioning) — the pool never runs a sandbox.
        self.runtime_type = runtime_config.type
        self.is_local = runtime_is_local(runtime_config)
        self.instance_host_endpoint = runtime_has_instance_host_endpoint(runtime_config)
        self.multiplex = max(1, multiplex)
        self._servers: list[PooledServer] = []
        self._lock = asyncio.Lock()
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> "InterceptionPool":
        await self._stack.__aenter__()
        return self

    async def __aexit__(self, *exc) -> None:
        # The stack tears down every server + its host tunnel (LIFO), even if one teardown fails.
        await self._stack.aclose()

    async def _entry(self) -> PooledServer:
        """A server with spare capacity — reuse one under `multiplex`, else bring up a new one
        (its own host endpoint). The caller holds `_lock`."""
        for entry in self._servers:
            if entry.healthy and entry.draining == 0 and entry.load < self.multiplex:
                return entry
        server = InterceptionServer()
        await self._stack.enter_async_context(server)
        # The interception server is a HOST service the harness reaches: localhost for a local
        # harness runtime, a tunnel for a remote one. Owned by the pool's stack, torn down with it.
        url = None
        if not self.instance_host_endpoint:
            url = await self._stack.enter_async_context(
                reachable_url(HOST, server.port, consumer_is_local=self.is_local)
            )
        entry = PooledServer(server, url)
        self._servers.append(entry)
        logger.info(
            "interception pool: %d server(s), multiplex=%d (%s)",
            len(self._servers),
            self.multiplex,
            self.runtime_type,
        )
        return entry

    @asynccontextmanager
    async def acquire(self, session: RolloutSession, runtime: Runtime):
        """Register `session` on a server with spare capacity (bringing one up if needed) and yield
        its `(endpoint, secret, port, base_url)` — `endpoint` is the model route (`{base_url}/v1`),
        `port` the interception server's host port (a per-rollout tool server's own channel), and
        `base_url` its reachable URL (how a `shared` server reaches this rollout's `/state` + `/task`);
        free the slot on exit."""
        async with self._lock:
            entry = await self._entry()
            secret = entry.server.register(session)
            entry.load += 1

        released = False
        primary_error: BaseException | None = None

        async def release() -> None:
            nonlocal released
            async with self._lock:
                if released:
                    return
                released = True
                # Exclude new admissions before unregister starts.  Multiple
                # sessions can drain concurrently, hence a count rather than a
                # boolean; the entry is reusable only after every drain succeeds.
                entry.draining += 1
            try:
                await entry.server.unregister(secret)
            except BaseException:
                async with self._lock:
                    entry.healthy = False
                raise
            finally:
                async with self._lock:
                    entry.draining -= 1
                    entry.load -= 1

        try:
            if entry.base_url is not None:
                try:
                    yield f"{entry.base_url}/v1", secret, entry.server.port, entry.base_url
                except BaseException as error:
                    primary_error = error
                    raise
                finally:
                    try:
                        await release()
                    except BaseException:
                        if not isinstance(primary_error, asyncio.CancelledError):
                            raise
            else:
                async with runtime.host_endpoint(entry.server.port) as base_url:
                    try:
                        yield f"{base_url}/v1", secret, entry.server.port, base_url
                    except BaseException as error:
                        primary_error = error
                        raise
                    finally:
                        try:
                            await release()
                        except BaseException:
                            if not isinstance(primary_error, asyncio.CancelledError):
                                raise
        except BaseException as error:
            primary_error = primary_error or error
            raise
        finally:
            if not released:
                try:
                    await release()
                except BaseException:
                    if not isinstance(primary_error, asyncio.CancelledError):
                        raise
