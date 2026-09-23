"""Public reverse tunnels to host HTTP services using a temporary Modal sandbox."""

import asyncio
import contextlib
import subprocess
import tempfile
import urllib.error
import urllib.request
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from verifiers.v1.errors import TunnelError

_APP_NAME = "verifiers-v1-host-relay"
_REMOTE_PORT = 18000


def _relay_image(modal):
    return (
        modal.Image.debian_slim()
        .apt_install("openssh-server")
        .run_commands(
            "mkdir -p /run/sshd /root/.ssh /etc/ssh/sshd_config.d",
            "chmod 700 /root/.ssh",
            "printf 'PermitRootLogin prohibit-password\\nPasswordAuthentication no\\n"
            "AllowTcpForwarding yes\\nGatewayPorts clientspecified\\n' "
            "> /etc/ssh/sshd_config.d/verifiers-relay.conf",
        )
    )


async def _generate_key(private_key: Path) -> None:
    process = await asyncio.create_subprocess_exec(
        "ssh-keygen",
        "-q",
        "-t",
        "ed25519",
        "-N",
        "",
        "-f",
        str(private_key),
    )
    if await process.wait() != 0:
        raise TunnelError("could not generate the Modal relay SSH key")


def _probe(url: str) -> bool:
    request = urllib.request.Request(url, method="GET")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=5) as response:
            return response.status not in {502, 503, 504}
    except urllib.error.HTTPError as error:
        return error.code not in {502, 503, 504}
    except urllib.error.URLError:
        return False


async def _wait_until_ready(
    url: str,
    ssh_process: asyncio.subprocess.Process,
    ssh_log: Path,
    *,
    timeout: float = 60.0,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if ssh_process.returncode is not None:
            detail = ssh_log.read_text(errors="replace").strip()
            raise TunnelError(f"Modal reverse SSH exited early: {detail}")
        if await asyncio.to_thread(_probe, url):
            return
        await asyncio.sleep(0.5)
    raise TunnelError(f"Modal host relay did not become ready within {timeout:.0f}s")


async def _stop_process(process: asyncio.subprocess.Process | None) -> None:
    if process is None or process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=10)
    except TimeoutError:
        process.kill()
        await process.wait()


@asynccontextmanager
async def modal_host_endpoint(port: int, *, name: str | None = None) -> AsyncIterator[str]:
    """Yield a public HTTPS URL forwarding to ``127.0.0.1:port`` on the host."""
    try:
        import modal
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("Modal host tunneling requires the Modal SDK; install `verifiers[modal]`.") from error

    with tempfile.TemporaryDirectory(prefix="vf-modal-relay-") as directory:
        relay_dir = Path(directory)
        private_key = relay_dir / "relay_ed25519"
        ssh_log = relay_dir / "ssh.log"
        await _generate_key(private_key)
        public_key = private_key.with_suffix(".pub").read_text().strip()

        app = await modal.App.lookup.aio(_APP_NAME, create_if_missing=True)
        sandbox = await modal.Sandbox.create.aio(
            "bash",
            "-lc",
            "printf '%s\\n' \"$RELAY_AUTHORIZED_KEY\" > /root/.ssh/authorized_keys "
            "&& chmod 600 /root/.ssh/authorized_keys && ssh-keygen -A "
            "&& exec /usr/sbin/sshd -D -e",
            app=app,
            image=_relay_image(modal),
            name=name or f"host-relay-{uuid.uuid4().hex[:12]}",
            timeout=24 * 60 * 60,
            encrypted_ports=[22, _REMOTE_PORT],
            secrets=[modal.Secret.from_dict({"RELAY_AUTHORIZED_KEY": public_key})],
            readiness_probe=modal.sandbox.Probe.with_tcp(22),
        )

        ssh_process = None
        log_file = None
        try:
            tunnels = await sandbox.tunnels.aio()
            ssh_tunnel = tunnels.get(22)
            http_tunnel = tunnels.get(_REMOTE_PORT)
            if ssh_tunnel is None or http_tunnel is None:
                raise TunnelError("Modal relay did not publish both required ports")
            relay_host, relay_port = ssh_tunnel.tls_socket
            url = str(http_tunnel.url).rstrip("/")
            proxy_command = f"openssl s_client -quiet -connect {relay_host}:{relay_port} -servername {relay_host}"
            log_file = ssh_log.open("w")
            ssh_process = await asyncio.create_subprocess_exec(
                "ssh",
                "-F",
                "/dev/null",
                "-N",
                "-T",
                "-i",
                str(private_key),
                "-o",
                "BatchMode=yes",
                "-o",
                "ExitOnForwardFailure=yes",
                "-o",
                "ServerAliveInterval=30",
                "-o",
                "ServerAliveCountMax=3",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
                "-o",
                f"ProxyCommand={proxy_command}",
                "-R",
                f"0.0.0.0:{_REMOTE_PORT}:127.0.0.1:{port}",
                "root@verifiers-relay",
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            await _wait_until_ready(url, ssh_process, ssh_log)
            yield url
        finally:
            await _stop_process(ssh_process)
            if log_file is not None:
                log_file.close()
            with contextlib.suppress(Exception):
                await sandbox.terminate.aio()
