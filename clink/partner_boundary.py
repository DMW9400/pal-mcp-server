"""Worker-owned authorization for an attested direct PAL Claude partner.

Minting and activation exist only on ``PartnerBoundaryAuthority`` instances
held in the durable worker process. Descendant hooks receive an opaque id and
nonce and may only ask the worker's private Unix socket to verify them. Merely
importing this module or launching a Claude-looking process cannot mint a
greenlit boundary.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import secrets
import socket
import stat
import struct
import subprocess
import sys
import time
from pathlib import Path

BOUNDARY_KIND = "pal-greenlit-claude-direct-partner-v2"
MAX_LIFETIME_SECONDS = 6 * 60 * 60
PENDING_LIFETIME_SECONDS = 60
MAX_REQUEST_BYTES = 4096
WORKER_SERVICE_LABEL = "com.pal.clink-worker"


def _root() -> Path:
    state = Path(os.environ.get("PAL_STATE_DIR", str(Path.home() / ".pal" / "state"))).expanduser().resolve()
    root = state / "partner-boundaries"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    return root


def _socket_root() -> Path:
    # Darwin limits AF_UNIX paths to roughly 104 bytes. PAL_STATE_DIR may be a
    # deeply nested test or checkout path, so keep only the ephemeral socket in
    # a short per-user 0700 runtime directory; all authorization state remains
    # in worker memory.
    root = Path(f"/private/tmp/pal-boundary-{os.getuid()}")
    if root.is_symlink():
        raise RuntimeError("partner boundary socket root must not be a symlink")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = root.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise RuntimeError("partner boundary socket root has the wrong owner or type")
    os.chmod(root, 0o700)
    return root


def _process_identity(pid: int) -> tuple[str, str, str]:
    result = subprocess.run(
        ["/bin/ps", "-ww", "-o", "lstart=", "-o", "command=", "-p", str(pid)],
        check=True,
        capture_output=True,
        text=True,
        timeout=2,
    )
    line = result.stdout.strip()
    if not line:
        raise RuntimeError(f"process {pid} is unavailable")
    fields = line.split(maxsplit=5)
    if len(fields) < 6:
        raise RuntimeError(f"process {pid} identity is incomplete")
    start, command = " ".join(fields[:5]), fields[5]
    return start.strip(), hashlib.sha256(command.encode()).hexdigest(), command


def _is_exact_direct_claude_command(command: str, expected_executable: str) -> bool:
    executable = command.split(maxsplit=1)[0] if command.strip() else ""
    model_pins = re.findall(r"(?:^|\s)--model\s+fable(?:\s|$)", command)
    effort_pins = re.findall(r"(?:^|\s)--effort\s+xhigh(?:\s|$)", command)
    return (
        secrets.compare_digest(executable, expected_executable)
        and len(model_pins) == 1
        and len(effort_pins) == 1
        and re.search(r"(?:^|\s)--fallback-model(?:[=\s]|$)", command) is None
    )


def _ancestor_pids(pid: int) -> set[int]:
    result = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid="], check=True, capture_output=True, text=True, timeout=2
    )
    parents = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 2:
            parents[int(fields[0])] = int(fields[1])
    ancestors = set()
    current = pid
    for _ in range(128):
        current = parents.get(current, 0)
        if current <= 1 or current in ancestors:
            break
        ancestors.add(current)
    return ancestors


def _socket_peer_pid(peer) -> int:
    """Return the kernel-attested PID at the other end of a Unix socket."""

    if peer is None:
        raise OSError("partner boundary peer socket is unavailable")
    if sys.platform == "darwin":
        # Darwin sys/un.h: SOL_LOCAL=0, LOCAL_PEERPID=0x002.
        return struct.unpack("=i", peer.getsockopt(0, 0x002, 4))[0]
    if sys.platform.startswith("linux") and hasattr(socket, "SO_PEERCRED"):
        return struct.unpack("=iii", peer.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))[0]
    raise OSError(f"partner boundary peer credentials unsupported on {sys.platform}")


def _registered_worker_pid() -> int:
    """Resolve the exact launchd-owned durable worker for this checkout."""

    domain = f"gui/{os.getuid()}/{WORKER_SERVICE_LABEL}"
    result = subprocess.run(
        ["/bin/launchctl", "print", domain], check=True, capture_output=True, text=True, timeout=2
    )
    output = result.stdout
    project_root = str(Path(__file__).resolve().parents[1])
    expected_program = str(Path(project_root, ".pal_venv", "bin", "python"))
    if (
        "state = running" not in output
        or f"program = {expected_program}" not in output
        or f"working directory = {project_root}" not in output
        or "\n\t\t-m\n" not in output
        or "\n\t\tclink.worker\n" not in output
        or "\n\t\tserve\n" not in output
    ):
        raise RuntimeError("registered PAL worker identity does not match this checkout")
    match = re.search(r"^\s*pid = (\d+)\s*$", output, re.MULTILINE)
    if match is None:
        raise RuntimeError("registered PAL worker PID is unavailable")
    return int(match.group(1))


class PartnerBoundaryAuthority:
    """Ephemeral boundary issuer and verifier owned by one live worker."""

    def __init__(self, worker_instance_id: str):
        token = hashlib.sha256(worker_instance_id.encode()).hexdigest()[:16]
        self.socket_path = _socket_root() / f"a-{token}-{secrets.token_hex(6)}.sock"
        self._entries: dict[str, dict[str, object]] = {}
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        if self._server is not None:
            return
        self._server = await asyncio.start_unix_server(self._handle_client, path=str(self.socket_path))
        os.chmod(self.socket_path, 0o600)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self._entries.clear()
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass

    def issue_pending(self, expected_executable: str) -> tuple[str, str]:
        if self._server is None:
            raise RuntimeError("partner boundary authority is not running")
        if not expected_executable or not os.path.isabs(expected_executable):
            raise RuntimeError("partner boundary expected executable must be absolute")
        self._prune()
        boundary_id = secrets.token_hex(24)
        nonce = secrets.token_urlsafe(32)
        self._entries[boundary_id] = {
            "kind": BOUNDARY_KIND,
            "state": "pending",
            "nonce": nonce,
            "expectedExecutable": expected_executable,
            "createdAt": time.time(),
        }
        return boundary_id, nonce

    def activate(self, boundary_id: str, nonce: str, pid: int) -> None:
        entry = self._entries.get(boundary_id)
        if entry is None or entry.get("state") != "pending" or not secrets.compare_digest(
            str(entry.get("nonce", "")), nonce
        ):
            raise RuntimeError("partner boundary pending capability is invalid")
        start, command_hash, command = _process_identity(pid)
        expected_executable = str(entry.get("expectedExecutable", ""))
        if not _is_exact_direct_claude_command(command, expected_executable):
            raise RuntimeError("partner boundary requires an exact direct Claude fable/xhigh process")
        entry.update(
            state="active",
            pid=pid,
            processStart=start,
            commandHash=command_hash,
            expiresAt=time.time() + MAX_LIFETIME_SECONDS,
        )

    def cleanup(self, boundary_id: str | None) -> None:
        if boundary_id is not None:
            self._entries.pop(boundary_id, None)

    def _prune(self) -> None:
        now = time.time()
        expired = [
            boundary_id
            for boundary_id, entry in self._entries.items()
            if (
                entry.get("state") == "pending"
                and float(entry.get("createdAt", 0)) + PENDING_LIFETIME_SECONDS < now
            )
            or (entry.get("state") == "active" and float(entry.get("expiresAt", 0)) < now)
        ]
        for boundary_id in expired:
            self._entries.pop(boundary_id, None)

    def _verify(self, boundary_id: str, nonce: str, verifier_pid: int) -> bool:
        try:
            self._prune()
            entry = self._entries.get(boundary_id)
            if entry is None or entry.get("kind") != BOUNDARY_KIND or entry.get("state") != "active":
                return False
            if not secrets.compare_digest(str(entry.get("nonce", "")), nonce):
                return False
            expiry = float(entry.get("expiresAt", 0))
            now = time.time()
            if expiry < now or expiry > now + MAX_LIFETIME_SECONDS:
                return False
            direct_pid = int(entry["pid"])
            if direct_pid not in _ancestor_pids(verifier_pid):
                return False
            start, command_hash, command = _process_identity(direct_pid)
            if not _is_exact_direct_claude_command(
                command, str(entry.get("expectedExecutable", ""))
            ):
                return False
            return start == entry.get("processStart") and secrets.compare_digest(
                command_hash, str(entry.get("commandHash", ""))
            )
        except (OSError, ValueError, KeyError, subprocess.SubprocessError, RuntimeError):
            return False

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        ok = False
        try:
            raw = await asyncio.wait_for(reader.readline(), timeout=1)
            if 0 < len(raw) <= MAX_REQUEST_BYTES:
                request = json.loads(raw)
                if request.get("op") == "verify":
                    peer_pid = _socket_peer_pid(writer.get_extra_info("socket"))
                    ok = self._verify(
                        str(request.get("boundaryId", "")),
                        str(request.get("nonce", "")),
                        peer_pid,
                    )
        except (asyncio.TimeoutError, OSError, ValueError, TypeError, json.JSONDecodeError, RuntimeError):
            ok = False
        writer.write(json.dumps({"ok": ok}, separators=(",", ":")).encode() + b"\n")
        try:
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()


def verify(socket_text: str, boundary_id: str, nonce: str) -> bool:
    """Ask the live worker authority to verify this exact descendant process."""

    try:
        socket_path = Path(socket_text).expanduser()
        root = _socket_root()
        if socket_path.parent.resolve(strict=True) != root or socket_path.is_symlink():
            return False
        info = socket_path.stat()
        if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
            return False
        request = json.dumps(
            {
                "op": "verify",
                "boundaryId": boundary_id,
                "nonce": nonce,
            },
            separators=(",", ":"),
        ).encode() + b"\n"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(2)
            client.connect(str(socket_path))
            if _socket_peer_pid(client) != _registered_worker_pid():
                return False
            client.sendall(request)
            response = b""
            while not response.endswith(b"\n") and len(response) <= MAX_REQUEST_BYTES:
                chunk = client.recv(1024)
                if not chunk:
                    break
                response += chunk
        return json.loads(response).get("ok") is True
    except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError, RuntimeError):
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("verify", nargs="?")
    parser.add_argument("--socket", required=True)
    parser.add_argument("--boundary-id", required=True)
    parser.add_argument("--nonce", required=True)
    args = parser.parse_args(argv)
    return 0 if verify(args.socket, args.boundary_id, args.nonce) else 1


if __name__ == "__main__":
    raise SystemExit(main())
