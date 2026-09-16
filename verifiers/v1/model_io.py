"""Compact, integrity-checked capture of model-provider JSON I/O.

Provider-bound requests in an agent rollout grow by appending messages. Persisting every full
request would therefore make a linear trace quadratic in its number of turns.
``ModelRequestDelta`` stores the top-level changes from the nearest captured ancestor and gives
growing lists an append operation; requests that do not benefit from a delta use an independent
full snapshot instead.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import Field, model_validator

from verifiers.v1.types import StrictBaseModel

if TYPE_CHECKING:
    from verifiers.v1.graph import MessageNode


JsonObject = dict[str, Any]


class ModelIOBase(StrictBaseModel):
    """Closed base for persisted capture records."""


class FullModelRequest(ModelIOBase):
    """An independent exact request snapshot (the first capture or a delta fallback)."""

    kind: Literal["full"] = "full"
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    body: JsonObject


class DeltaModelRequest(ModelIOBase):
    """Top-level changes against the request captured on ``base_node``."""

    kind: Literal["delta"] = "delta"
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_node: int = Field(ge=0)
    set_fields: JsonObject = Field(default_factory=dict)
    remove_fields: list[str] = Field(default_factory=list)
    append_fields: dict[str, list[Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_operations(self) -> "DeltaModelRequest":
        removed = set(self.remove_fields)
        set_keys = set(self.set_fields)
        append_keys = set(self.append_fields)
        if len(removed) != len(self.remove_fields):
            raise ValueError("model request delta contains duplicate remove fields")
        if overlap := (removed & set_keys) | (removed & append_keys) | (set_keys & append_keys):
            raise ValueError(f"model request delta operations overlap: {sorted(overlap)!r}")
        if any(not values for values in self.append_fields.values()):
            raise ValueError("model request delta append fields must be non-empty")
        return self


ModelRequestCapture = Annotated[FullModelRequest | DeltaModelRequest, Field(discriminator="kind")]


class ModelResponseCapture(ModelIOBase):
    """A provider response, exact for JSON responses and normalized for SSE streams."""

    kind: Literal["exact_provider_json", "normalized_stream_response"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    body: JsonObject


class ModelIO(ModelIOBase):
    """JSON sent to and received from the configured provider route for one sampled turn.

    Bodies retain exact parsed JSON semantics, not original whitespace or duplicate object keys.
    Headers, credentials, and the configured base URL are deliberately outside this record.
    """

    provider_route: str = Field(pattern=r"^/")
    request: ModelRequestCapture
    response: ModelResponseCapture


def clone_json_object(body: JsonObject) -> JsonObject:
    """Own a JSON object without changing its parsed JSON values or key order."""
    return copy.deepcopy(body)


def json_sha256(body: JsonObject) -> str:
    """SHA256 of a deterministic JSON representation (independent of mapping key order)."""
    encoded = json.dumps(
        body,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def list_extends(previous: list[Any], current: list[Any]) -> bool:
    """Whether ``current`` is ``previous`` followed by one or more values, without slicing."""
    return len(current) > len(previous) and all(old == new for old, new in zip(previous, current, strict=False))


def capture_model_request(
    body: JsonObject,
    *,
    base_node: int | None = None,
    base_body: JsonObject | None = None,
) -> ModelRequestCapture:
    """Encode an exact request as the smaller of a top-level ancestor delta and a full copy."""
    current = clone_json_object(body)
    digest = json_sha256(current)
    full = FullModelRequest(sha256=digest, body=current)
    if base_node is None or base_body is None:
        return full

    set_fields: JsonObject = {}
    append_fields: dict[str, list[Any]] = {}
    for key in sorted(current.keys()):
        value = current[key]
        if key in base_body and base_body[key] == value:
            continue
        previous = base_body.get(key)
        if isinstance(previous, list) and isinstance(value, list) and list_extends(previous, value):
            append_fields[key] = clone_json_object({"value": value[len(previous) :]})["value"]
        else:
            set_fields[key] = clone_json_object({"value": value})["value"]
    remove_fields = sorted(set(base_body) - set(current))
    delta = DeltaModelRequest(
        sha256=digest,
        base_node=base_node,
        set_fields=set_fields,
        remove_fields=remove_fields,
        append_fields=append_fields,
    )
    # A rewritten large list generally makes the delta larger than the full request. Falling back
    # bounds that exceptional branch to one snapshot; ordinary growing message lists use append.
    if len(delta.model_dump_json()) >= len(full.model_dump_json()):
        return full
    return delta


def capture_model_response(
    body: JsonObject,
    *,
    kind: Literal["exact_provider_json", "normalized_stream_response"],
) -> ModelResponseCapture:
    """Own and hash one exact or explicitly normalized response JSON object."""
    captured = clone_json_object(body)
    return ModelResponseCapture(kind=kind, sha256=json_sha256(captured), body=captured)


def reconstruct_model_request(nodes: list[MessageNode], node_id: int, *, validate_hash: bool = True) -> JsonObject:
    """Reconstruct one sampled node's exact request, validating every delta in its chain."""
    memo: dict[int, JsonObject] = {}
    visiting: set[int] = set()

    def reconstruct(current_id: int) -> JsonObject:
        if current_id in memo:
            return clone_json_object(memo[current_id])
        if current_id in visiting:
            raise ValueError(f"model I/O request delta cycle at node {current_id}")
        if current_id < 0 or current_id >= len(nodes):
            raise ValueError(f"model I/O request references missing node {current_id}")
        model_io = nodes[current_id].model_io
        if model_io is None:
            raise ValueError(f"node {current_id} has no captured model I/O")
        if not nodes[current_id].sampled:
            raise ValueError(f"non-sampled node {current_id} has captured model I/O")
        visiting.add(current_id)
        request = model_io.request
        if isinstance(request, FullModelRequest):
            body = clone_json_object(request.body)
        else:
            if request.base_node >= current_id:
                raise ValueError(
                    f"model I/O request at node {current_id} references non-prior base node {request.base_node}"
                )
            if not nodes[request.base_node].sampled:
                raise ValueError(
                    f"model I/O request at node {current_id} references non-sampled base node {request.base_node}"
                )
            ancestor = nodes[current_id].parent
            while ancestor is not None and ancestor != request.base_node:
                ancestor = nodes[ancestor].parent
            if ancestor is None:
                raise ValueError(
                    f"model I/O request at node {current_id} references non-ancestor base node {request.base_node}"
                )
            body = reconstruct(request.base_node)
            for key in request.remove_fields:
                body.pop(key, None)
            for key, value in request.set_fields.items():
                body[key] = clone_json_object({"value": value})["value"]
            for key, suffix in request.append_fields.items():
                previous = body.get(key)
                if not isinstance(previous, list):
                    raise ValueError(f"model I/O request delta appends to non-list field {key!r}")
                body[key] = [*previous, *clone_json_object({"value": suffix})["value"]]
        if validate_hash and json_sha256(body) != request.sha256:
            raise ValueError(f"model I/O request hash mismatch at node {current_id}")
        visiting.remove(current_id)
        memo[current_id] = body
        return clone_json_object(body)

    return reconstruct(node_id)


def validate_model_io(nodes: list[MessageNode], node_id: int) -> JsonObject:
    """Validate a node's request chain and response hash; return its reconstructed request."""
    request = reconstruct_model_request(nodes, node_id, validate_hash=True)
    model_io = nodes[node_id].model_io
    if model_io is None:  # narrowed by reconstruction, retained for static type checkers
        raise ValueError(f"node {node_id} has no captured model I/O")
    if json_sha256(model_io.response.body) != model_io.response.sha256:
        raise ValueError(f"model I/O response hash mismatch at node {node_id}")
    return request
