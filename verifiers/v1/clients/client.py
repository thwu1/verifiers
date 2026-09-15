"""The client abstraction: turn a prompt into a `Response`.

Collapsed from v1's 4-typevar generic ABC with five conversion hooks to a single
abstract method. Each concrete client owns its own wire translation internally.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass

from verifiers.v1.dialects import Dialect
from verifiers.v1.graph import PendingTurn
from verifiers.v1.types import Response, Sampling, SamplingConfig

logger = logging.getLogger(__name__)

SESSION_ID_HEADER = "X-Session-ID"
LITELLM_SESSION_ID_HEADER = "X-LiteLLM-Session-ID"
"""Per-rollout routing headers.

Every turn of one rollout sends the same trace id under both the generic session header and
LiteLLM's deployment-affinity header.  The former supports consistent-hash routers configured
with ``request_id_headers``; the latter is what LiteLLM's ``session_affinity`` pre-call check
actually consumes.  Sending both keeps the growing cross-turn prefix on one engine across the
two router implementations instead of repeatedly discarding its KV cache.
"""


def session_id_headers(session_id: str | None) -> dict[str, str] | None:
    if session_id is None:
        return None
    return {
        SESSION_ID_HEADER: session_id,
        LITELLM_SESSION_ID_HEADER: session_id,
    }


@dataclass
class RelayReply:
    """A relayed upstream response: content type, complete SSE events, and connection cleanup."""

    content_type: str
    chunks: AsyncIterator[bytes]
    close: Callable[[], Awaitable[None]]


class Client(ABC):
    @abstractmethod
    async def get_response(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
        session_id: str | None = None,
        turn: PendingTurn | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        """Run one completion -> a vf `Response`. The eval client forwards the native JSON and
        eligible end-to-end headers, then parses a copy via `dialect`; the train client derives
        the typed prompt from `body` and tokenizes it.

        `session_id` is the rollout's stable id (the trace id); when set, the client sends it
        as the `SESSION_ID_HEADER` so a session-affinity router keeps the rollout's turns on
        one engine for cross-turn prefix-cache reuse. `turn` is the graph-resolved prompt
        prefix; train clients may use it for renderer bridging, while relay clients ignore it."""

    async def relay(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
        session_id: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> RelayReply:
        """Stream a (possibly SSE) response back, relaying the provider's bytes — the proxy's
        path for a streaming request. Only the relay (eval) client supports it; the renderer
        generates and cannot stream."""
        raise NotImplementedError(f"{type(self).__name__} does not support streaming")

    async def relay_aux(self, dialect: Dialect, route: str, body: dict) -> dict:
        """Relay a non-model-turn side request (an `aux_route`, e.g. Anthropic's `count_tokens`)
        verbatim to the provider and return its JSON. Only the relay (eval) client supports it."""
        raise NotImplementedError(f"{type(self).__name__} does not relay aux routes")

    async def close(self) -> None:
        """Release any underlying resources. Default no-op."""


@dataclass(frozen=True)
class RolloutContext:
    """The collaborators a single rollout needs (client + model + sampling), bundled
    so harnesses hold no rollout state. Built by the Environment."""

    model: str
    client: Client
    sampling: Sampling
