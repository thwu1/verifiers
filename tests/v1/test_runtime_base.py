from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_none
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.runtimes import base as runtime_base


class FakeRuntime(Runtime):
    def __init__(self, preparation_results: list[ProgramResult]) -> None:
        super().__init__()
        self.preparation_results = preparation_results
        self.preparation_calls = 0
        self.writes: list[tuple[str, bytes]] = []

    async def start(self) -> None:
        pass

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        if argv[0:2] == ["sh", "-c"] and argv[2].startswith("mv -f "):
            return ProgramResult(exit_code=0, stdout="", stderr="")
        self.preparation_calls += 1
        index = min(self.preparation_calls - 1, len(self.preparation_results) - 1)
        return self.preparation_results[index]

    async def read(self, path: str) -> bytes:
        raise NotImplementedError

    async def write(self, path: str, data: bytes) -> None:
        self.writes.append((path, data))


def _immediate_retrying(*, retries: int, **kwargs) -> AsyncRetrying:
    return AsyncRetrying(
        stop=stop_after_attempt(retries + 1),
        wait=wait_none(),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )


async def test_prepare_uv_script_retries_then_caches_interpreter(monkeypatch) -> None:
    monkeypatch.setattr(runtime_base, "retrying", _immediate_retrying)
    runtime = FakeRuntime(
        [
            ProgramResult(exit_code=1, stdout="temporary package-index failure", stderr=""),
            ProgramResult(exit_code=0, stdout="installer output\n/venv/bin/python\n", stderr=""),
        ]
    )

    argv = await runtime.prepare_uv_script("print('hello')")

    assert runtime.preparation_calls == 2
    assert argv[-2] == "/venv/bin/python"
    assert argv[-1].startswith("/tmp/vf-scripts/")
    assert await runtime.prepare_uv_script("print('hello')") == argv
    assert runtime.preparation_calls == 2
    assert len(runtime.writes) == 1


async def test_prepare_uv_script_persistent_failure_has_bounded_combined_output(
    monkeypatch,
) -> None:
    monkeypatch.setattr(runtime_base, "retrying", _immediate_retrying)
    runtime = FakeRuntime(
        [
            ProgramResult(
                exit_code=23,
                stdout="x" * 2500 + " stdout-tail",
                stderr="y" * 2500 + " stderr-tail",
            )
        ]
    )

    try:
        await runtime.prepare_uv_script("print('hello')")
    except RuntimeError as error:
        message = str(error)
    else:
        raise AssertionError("persistent preparation failure did not raise")

    assert runtime.preparation_calls == runtime_base._UV_PREPARE_RETRIES + 1
    assert "exit_code=23" in message
    assert "stdout:" in message and "stdout-tail" in message
    assert "stderr:" in message and "stderr-tail" in message
    assert len(message) <= runtime_base._UV_PREPARE_ERROR_OUTPUT_LIMIT + 100
