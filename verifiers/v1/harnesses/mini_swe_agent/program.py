# /// script
# requires-python = ">=3.10"
# dependencies = ["mini-swe-agent=={version}"]
# ///

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from minisweagent.agents.default import DefaultAgent
from minisweagent.agents.interactive import InteractiveAgent
from minisweagent.config import get_config_path
from minisweagent.environments import local as local_environment
from minisweagent.run.mini import app


def _merged_config_argv(argv: list[str]) -> list[str]:
    """Merge config specs before invoking mini-swe-agent's versioned CLI.

    Releases before 2.0 accept one config path, while newer releases accept a
    sequence of files and dotted overrides. Supplying one merged YAML file keeps
    the harness behavior identical across both interfaces.
    """
    output = [argv[0]]
    config_file: str | None = None
    overrides: list[str] = []
    index = 1
    while index < len(argv):
        argument = argv[index]
        if argument in {"--vf-config-file", "--vf-config-override"}:
            if index + 1 >= len(argv):
                raise ValueError(f"missing value for {argument}")
            value = argv[index + 1]
            if argument == "--vf-config-file":
                config_file = value
            else:
                overrides.append(value)
            index += 2
            continue
        output.append(argument)
        index += 1

    if config_file is None:
        return output
    config = yaml.safe_load(get_config_path(config_file).read_text())
    for spec in overrides:
        key, separator, raw_value = spec.partition("=")
        if not separator:
            raise ValueError(f"invalid config override: {spec!r}")
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        target = config
        keys = key.split(".")
        for part in keys[:-1]:
            child = target.setdefault(part, {})
            if not isinstance(child, dict):
                raise TypeError(f"config path is not a mapping: {key!r}")
            target = child
        target[keys[-1]] = value

    descriptor, merged_path = tempfile.mkstemp(prefix="vf-mini-swe-", suffix=".yaml")
    os.close(descriptor)
    Path(merged_path).write_text(yaml.safe_dump(config, sort_keys=False))
    return [*output, "-c", merged_path]


class _BashSubprocess:
    """Run LocalEnvironment shell actions with SWE-bench's Bash login-shell semantics."""

    PIPE = subprocess.PIPE
    STDOUT = subprocess.STDOUT

    @staticmethod
    def run(command, *args, **kwargs):
        if kwargs.pop("shell", False):
            command = ["bash", "-lc", command]
        return subprocess.run(command, *args, **kwargs)


# The 1.x mini CLI hard-codes InteractiveAgent, whose limit handler prompts on
# stdin. The harness is unattended, so retain its yolo execution behavior but
# use DefaultAgent's benchmark-compatible terminal handling for step limits.
InteractiveAgent.query = DefaultAgent.query
local_environment.subprocess = _BashSubprocess
sys.argv = _merged_config_argv(sys.argv)
app()
