"""Resume an interrupted eval: re-run only the rollouts a previous run didn't finish.

A run writes `config.toml` + `results.jsonl` into its output dir. `--resume <dir>` reloads
that config verbatim (so it takes no other flags) and writes back into the same dir, running
only the rollouts still owed: the *missing* ones (never written — the run was interrupted) and
the *errored* ones (written with an error). Good rollouts are kept; errored ones are dropped
and redone. A group-scored taskset is resumed a whole task at a time (its rollouts are scored
together), so any task that isn't fully complete is redone from scratch.
"""

import json
import math
import tomllib
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from pydantic_core import from_json

from verifiers.v1.configs.eval import EvalConfig


def split_resume(argv: list[str]) -> tuple[Path | None, list[str]]:
    """Pull `--resume <dir>` / `--resume=<dir>` out of argv, returning (dir, the other args).
    The caller rejects any leftover args, since resume re-runs the saved config verbatim."""
    for i, arg in enumerate(argv):
        if arg == "--resume":
            if i + 1 >= len(argv):
                raise SystemExit(
                    "--resume needs an output dir: uv run eval --resume <dir>"
                )
            return Path(argv[i + 1]), argv[:i] + argv[i + 2 :]
        if arg.startswith("--resume="):
            return Path(arg.split("=", 1)[1]), argv[:i] + argv[i + 1 :]
    return None, argv


def load_resume_config(resume_dir: Path) -> EvalConfig:
    """Rebuild the run's `EvalConfig` from its saved `config.toml`, pointed back at its own
    output dir so the resumed rollouts append to the same `results.jsonl`."""
    config_path = resume_dir / "config.toml"
    if not config_path.exists():
        raise SystemExit(
            f"--resume: no config.toml in {resume_dir} - not an eval output dir"
        )
    config = EvalConfig.model_validate(tomllib.loads(config_path.read_text()))
    config.resume = resume_dir
    config.output_dir = resume_dir
    return config


def exact_tokens_requested(config: EvalConfig) -> bool:
    """Whether this run explicitly asked the provider for exact token-level training data."""
    sampling = config.sampling.model_dump(exclude_none=True)
    return sampling.get("return_token_ids") is True or sampling.get("logprobs") is True


def logprobs_requested(config: EvalConfig) -> bool:
    """Whether persisted sampled-token logprobs are required for this run."""
    return config.sampling.model_dump(exclude_none=True).get("logprobs") is True


def _is_finite_number(value: object) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)) or (
        isinstance(value, float) and math.isfinite(value)
    )


def _has_exact_tokens(row: dict, *, require_logprobs: bool = False) -> bool:
    """Check the persisted graph's token-training invariants without building a ``Trace``."""
    nodes = row.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return False
    sampled_tokens = 0
    for node in nodes:
        if not isinstance(node, dict):
            return False
        token_ids = node.get("token_ids")
        mask = node.get("mask")
        logprobs = node.get("logprobs")
        if not (
            isinstance(token_ids, list)
            and isinstance(mask, list)
            and isinstance(logprobs, list)
            and len(token_ids) == len(mask)
            and all(isinstance(token, int) and not isinstance(token, bool) for token in token_ids)
            and all(isinstance(sampled, bool) for sampled in mask)
            and all(_is_finite_number(logprob) for logprob in logprobs)
        ):
            return False
        expected_logprobs = sum(mask)
        if len(logprobs) not in ({expected_logprobs} if require_logprobs else {0, expected_logprobs}):
            return False
        if node.get("sampled") is True:
            sampled_tokens += expected_logprobs
        elif expected_logprobs:
            return False
    return sampled_tokens > 0


def _read_results(
    results_path: Path,
    *,
    require_exact_tokens: bool = False,
    require_logprobs: bool = False,
) -> Iterator[tuple[int, int, bool]]:
    """Stream `(file reference, task idx, unusable)` without retaining decoded traces."""
    if not results_path.exists():
        return
    with results_path.open("rb") as results:
        while True:
            offset = results.tell()
            line = results.readline()
            if not line:
                break
            if line.strip():
                try:
                    row = from_json(line)
                except ValueError:
                    try:
                        row = json.loads(line)
                    except (UnicodeDecodeError, ValueError):
                        # A killed append can leave only the final JSON object incomplete.
                        # `readline()` lacks a newline only at EOF; ignore that fragment so its
                        # rollout remains owed. Complete/interior bad rows are corruption and
                        # must still fail loudly rather than silently discard durable results.
                        if not line.endswith(b"\n"):
                            break
                        raise
                unusable = bool(row.get("errors")) or (
                    require_exact_tokens
                    and not _has_exact_tokens(row, require_logprobs=require_logprobs)
                )
                yield offset, row["task"]["idx"], unusable


def plan(
    resume_dir: Path,
    selected_idxs: list[int],
    num_rollouts: int,
    group: bool,
    *,
    require_exact_tokens: bool = False,
    require_logprobs: bool = False,
) -> tuple[list[int], dict[int, int]]:
    """Diff the saved results against the run's target (`num_rollouts` per selected task).
    Returns (byte offsets of rows to keep, rollouts owed per task idx). An errored trace is
    dropped and re-run; a group-scored task is kept only if fully complete, else its whole group
    is redone."""
    # Retain only the offsets resume can reuse; trace payloads stay on disk.
    selected = set(selected_idxs)
    by_idx: dict[int, list[int]] = defaultdict(list)
    for offset, idx, unusable in _read_results(
        resume_dir / "results.jsonl",
        require_exact_tokens=require_exact_tokens,
        require_logprobs=require_logprobs,
    ):
        if idx in selected and not unusable and len(by_idx[idx]) < num_rollouts:
            by_idx[idx].append(offset)
    keep: list[int] = []
    owed: dict[int, int] = {}
    for idx in selected_idxs:
        good = by_idx.get(idx, [])
        if group:
            if len(good) >= num_rollouts:
                keep.extend(good)
            else:
                owed[idx] = num_rollouts  # re-run the whole group; keep none of it
        else:
            keep.extend(good)
            missing = num_rollouts - len(good)
            if missing:
                owed[idx] = missing
    return keep, owed


def rewrite_results(resume_dir: Path, keep: list[int]) -> None:
    """Replace `results.jsonl` with just the kept (good) traces; resumed rollouts append. Via a
    temp file + atomic rename, so an interrupted resume can't corrupt the prior good results."""
    path = resume_dir / "results.jsonl"
    tmp = path.with_suffix(".jsonl.tmp")
    if not keep:
        tmp.write_bytes(b"")
        tmp.replace(path)
        return
    # Re-read retained rows by offset while building the atomic replacement.
    with path.open("rb") as results, tmp.open("wb") as output:
        for offset in keep:
            results.seek(offset)
            raw = results.readline()
            output.write(raw)
            if not raw.endswith(b"\n"):
                output.write(b"\n")
    tmp.replace(path)


def nothing_to_resume_msg(resume_dir: Path, num_tasks: int, num_rollouts: int) -> str:
    """The message shown (and then exit 0 - the run is already complete) when every selected
    rollout already completed without error."""
    return (
        f"nothing to resume in {resume_dir}: all {num_tasks}x{num_rollouts} rollouts "
        f"already completed without error"
    )
