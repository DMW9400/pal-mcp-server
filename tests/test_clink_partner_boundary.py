import asyncio
import os

import pytest
import clink.partner_boundary as boundary


EXACT_COMMAND = "/opt/example/claude --model fable --effort xhigh --print"
EXACT_EXECUTABLE = "/opt/example/claude"


async def verify(authority, boundary_id, nonce):
    return await asyncio.to_thread(
        boundary.verify, str(authority.socket_path), boundary_id, nonce
    )


def test_partner_boundary_requires_worker_authority_ancestor_nonce_and_exact_process(monkeypatch, tmp_path):
    monkeypatch.setenv("PAL_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(
        boundary,
        "_process_identity",
        lambda pid: ("Mon Aug 25 12:00:00 2026", "command-hash", EXACT_COMMAND),
    )
    monkeypatch.setattr(boundary, "_ancestor_pids", lambda _pid: {4321})
    authority = boundary.PartnerBoundaryAuthority("worker-one")
    authority._server = object()
    boundary_id, nonce = authority.issue_pending(EXACT_EXECUTABLE)
    authority.activate(boundary_id, nonce, 4321)

    assert authority._verify(boundary_id, nonce, os.getpid())
    assert not authority._verify(boundary_id, "wrong-nonce", os.getpid())
    monkeypatch.setattr(boundary, "_ancestor_pids", lambda _pid: set())
    assert not authority._verify(boundary_id, nonce, os.getpid())
    authority.cleanup(boundary_id)
    assert not authority._verify(boundary_id, nonce, os.getpid())


def test_partner_boundary_has_no_importable_mint_or_activate_functions():
    assert not hasattr(boundary, "create_pending")
    assert not hasattr(boundary, "activate")


def test_partner_boundary_prunes_orphaned_pending_capabilities(monkeypatch, tmp_path):
    monkeypatch.setenv("PAL_STATE_DIR", str(tmp_path / "state"))
    authority = boundary.PartnerBoundaryAuthority("worker-prune")
    authority._server = object()
    orphaned_id, _ = authority.issue_pending(EXACT_EXECUTABLE)
    authority._entries[orphaned_id]["createdAt"] = 0
    authority.issue_pending(EXACT_EXECUTABLE)
    assert orphaned_id not in authority._entries


def test_partner_boundary_rejects_non_claude_or_fallback_process(monkeypatch, tmp_path):
    monkeypatch.setenv("PAL_STATE_DIR", str(tmp_path / "state"))
    authority = boundary.PartnerBoundaryAuthority("worker-two")
    authority._server = object()
    for command in [
        "python worker.py --model fable --effort xhigh",
        "/opt/example/claude --model sonnet --effort xhigh",
        "/opt/example/claude --model fable --effort xhigh --fallback-model opus",
    ]:
        boundary_id, nonce = authority.issue_pending(EXACT_EXECUTABLE)
        monkeypatch.setattr(
            boundary,
            "_process_identity",
            lambda _pid, command=command: ("Mon Aug 25 12:00:00 2026", "command-hash", command),
        )
        with pytest.raises(RuntimeError, match="exact direct Claude"):
            authority.activate(boundary_id, nonce, 4321)
        authority.cleanup(boundary_id)


@pytest.mark.asyncio
async def test_verification_socket_accepts_only_registered_worker_pid(monkeypatch, tmp_path):
    monkeypatch.setenv("PAL_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(
        boundary,
        "_process_identity",
        lambda _pid: ("Mon Aug 25 12:00:00 2026", "command-hash", EXACT_COMMAND),
    )
    monkeypatch.setattr(boundary, "_ancestor_pids", lambda _pid: {4321})
    authority = boundary.PartnerBoundaryAuthority("worker-rpc")
    await authority.start()
    try:
        boundary_id, nonce = authority.issue_pending(EXACT_EXECUTABLE)
        authority.activate(boundary_id, nonce, 4321)

        monkeypatch.setattr(boundary, "_registered_worker_pid", lambda: os.getpid())
        assert await verify(authority, boundary_id, nonce)

        monkeypatch.setattr(boundary, "_registered_worker_pid", lambda: os.getpid() + 1)
        assert not await verify(authority, boundary_id, nonce), (
            "a same-user self-minted authority must fail when it is not the registered LaunchAgent worker"
        )
    finally:
        authority.cleanup(boundary_id)
        await authority.stop()


def test_partner_boundary_accepts_attested_resolved_claude_version_path(monkeypatch, tmp_path):
    monkeypatch.setenv("PAL_STATE_DIR", str(tmp_path / "state"))
    executable = "/Users/example/.local/share/claude/versions/2.1.243"
    command = f"{executable} --model fable --effort xhigh --print"
    monkeypatch.setattr(
        boundary,
        "_process_identity",
        lambda _pid: ("Mon Aug 25 12:00:00 2026", "command-hash", command),
    )
    authority = boundary.PartnerBoundaryAuthority("worker-version-path")
    authority._server = object()
    boundary_id, nonce = authority.issue_pending(executable)

    authority.activate(boundary_id, nonce, 4321)

    assert authority._entries[boundary_id]["state"] == "active"


def test_partner_boundary_rejects_different_executable_with_exact_pins(monkeypatch, tmp_path):
    monkeypatch.setenv("PAL_STATE_DIR", str(tmp_path / "state"))
    command = "/opt/other/claude --model fable --effort xhigh --print"
    monkeypatch.setattr(
        boundary,
        "_process_identity",
        lambda _pid: ("Mon Aug 25 12:00:00 2026", "command-hash", command),
    )
    authority = boundary.PartnerBoundaryAuthority("worker-wrong-executable")
    authority._server = object()
    boundary_id, nonce = authority.issue_pending(EXACT_EXECUTABLE)

    with pytest.raises(RuntimeError, match="exact direct Claude"):
        authority.activate(boundary_id, nonce, 4321)


def test_registered_worker_pid_requires_exact_launchagent_identity(monkeypatch):
    project_root = str(boundary.Path(boundary.__file__).resolve().parents[1])
    python = f"{project_root}/.pal_venv/bin/python"
    output = f"""gui/{os.getuid()}/{boundary.WORKER_SERVICE_LABEL} = {{
\tstate = running
\tprogram = {python}
\targuments = {{
\t\t{python}
\t\t-m
\t\tclink.worker
\t\tserve
\t}}
\tworking directory = {project_root}
\tpid = 4242
}}"""

    class Result:
        stdout = output

    monkeypatch.setattr(boundary.subprocess, "run", lambda *_args, **_kwargs: Result())
    assert boundary._registered_worker_pid() == 4242
    Result.stdout = output.replace("clink.worker", "other.worker")
    with pytest.raises(RuntimeError, match="does not match this checkout"):
        boundary._registered_worker_pid()
