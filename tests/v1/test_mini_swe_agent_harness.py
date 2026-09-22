import ast
import importlib.metadata
import os
import subprocess

import pytest
from verifiers.v1.harnesses.mini_swe_agent.harness import PROGRAM_SOURCE


def _bash_subprocess_shim():
    tree = ast.parse(PROGRAM_SOURCE)
    shim = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_BashSubprocess")
    namespace = {"subprocess": subprocess}
    exec(compile(ast.Module(body=[shim], type_ignores=[]), "program.py", "exec"), namespace)
    return namespace["_BashSubprocess"]


def test_mini_swe_agent_246_local_run_and_native_submit(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("minisweagent")
    if importlib.metadata.version("mini-swe-agent") != "2.4.6":
        pytest.skip("compatibility contract is specific to mini-swe-agent 2.4.6")

    from minisweagent.environments import local
    from minisweagent.exceptions import Submitted

    monkeypatch.setattr(local, "subprocess", _bash_subprocess_shim())
    result = local._run(
        "printf 'bash:%s' \"$BASH_VERSION\"",
        str(tmp_path),
        dict(os.environ),
        5,
    )

    assert result.returncode == 0
    assert result.stdout.startswith("bash:")
    with pytest.raises(Submitted):
        local.LocalEnvironment(cwd=str(tmp_path), timeout=5).execute(
            {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}
        )
