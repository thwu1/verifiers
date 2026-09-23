"""The mini-swe-agent harness: runs the native bash-tool agent through LiteLLM."""

from pathlib import Path

from pydantic import Field

from verifiers.v1.clients import RolloutContext
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()


class MiniSWEAgentHarnessConfig(HarnessConfig):
    """The mini-swe-agent CLI harness."""

    version: str = "2.2.8"
    """mini-swe-agent release to install, pinned for reproducibility."""
    config_file: str = "mini"
    """Built-in mini-swe-agent config filename or path."""
    config_overrides: list[str] = Field(default_factory=list)
    """Additional mini-swe-agent config specs, applied after harness defaults."""


class MiniSWEAgentHarness(Harness[MiniSWEAgentHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = False
    SUPPORTS_MCP = False

    async def setup(self, runtime: Runtime) -> None:
        source = PROGRAM_SOURCE.replace("{version}", self.config.version)
        await runtime.prepare_uv_script(source, self.config.env)

    async def launch(
        self,
        ctx: RolloutContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        if self.config.disabled_tools:
            raise ValueError("mini-swe-agent does not support disabling tools")
        _, prompt = self.resolve_prompt(trace.task)
        source = PROGRAM_SOURCE.replace("{version}", self.config.version)
        args = [
            "--model",
            ctx.model,
            "--model-class",
            "litellm",
            "--task",
            prompt,
            "--exit-immediately",
            "--yolo",
            "--vf-config-file",
            self.config.config_file,
            "--vf-config-override",
            "agent.cost_limit=0",
            # Effectively unlimited; Verifiers owns the rollout timeout.
            "--vf-config-override",
            "environment.timeout=86400",
            "--vf-config-override",
            "model.cost_tracking=ignore_errors",
            "--vf-config-override",
            "model.model_kwargs.custom_llm_provider=openai",
            "--vf-config-override",
            f"model.model_kwargs.api_base={endpoint}",
            "--vf-config-override",
            f"model.model_kwargs.api_key={secret}",
        ]
        for override in self.config.config_overrides:
            args.extend(["--vf-config-override", override])
        env = {
            **self.config.env,
            "MSWEA_CONFIGURED": "true",
            "MSWEA_SILENT_STARTUP": "true",
        }
        program = await runtime.prepare_uv_script(source, self.config.env)
        return await runtime.run_program([*program, *args], env)
