import os
import subprocess
import time
import uuid
from pathlib import Path

from utils.sqlite_conversation_storage import SQLiteConversationStorage


PAL_ROOT = Path(__file__).parents[1]
SERVICE_SCRIPT = PAL_ROOT / "scripts" / "clink-worker-service.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o700)


def test_install_recovers_from_nonzero_bootout_and_transient_bootstrap(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    loaded = state_dir / "loaded"
    loaded.touch()

    _write_executable(
        fake_bin / "launchctl",
        """#!/bin/sh
set -eu
state="$PAL_TEST_STATE"
case "$1" in
  print) test -f "$state/loaded" ;;
  bootout)
    rm -f "$state/loaded"
    test -z "${PAL_TEST_BOOTOUT_DELAY:-}" || sleep "$PAL_TEST_BOOTOUT_DELAY"
    exit 37
    ;;
  bootstrap)
    count=0
    test -f "$state/bootstrap-count" && count=$(cat "$state/bootstrap-count")
    count=$((count + 1))
    printf '%s' "$count" > "$state/bootstrap-count"
    test "$count" -gt 1 || exit 5
    : > "$state/loaded"
    ;;
  kickstart) test -f "$state/loaded" ;;
  *) exit 64 ;;
esac
""",
    )
    _write_executable(fake_bin / "python", "#!/bin/sh\nexit 0\n")

    fake_home = tmp_path / "home"
    env = os.environ | {
        "HOME": str(fake_home),
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "PAL_PYTHON": str(fake_bin / "python"),
        "PAL_TEST_STATE": str(state_dir),
    }
    result = subprocess.run(
        ["bash", str(SERVICE_SCRIPT), "install"],
        cwd=PAL_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "installed and started gui/501/com.pal.clink-worker" in result.stdout
    assert loaded.is_file()
    assert (state_dir / "bootstrap-count").read_text(encoding="utf-8") == "2"


def test_concurrent_installs_serialize_worker_lifecycle(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    loaded = state_dir / "loaded"
    loaded.touch()

    _write_executable(
        fake_bin / "launchctl",
        """#!/bin/sh
set -eu
state="$PAL_TEST_STATE"
case "$1" in
  print) test -f "$state/loaded" ;;
  bootout)
    rm -f "$state/loaded"
    sleep "$PAL_TEST_BOOTOUT_DELAY"
    exit 37
    ;;
  bootstrap) : > "$state/loaded" ;;
  kickstart) test -f "$state/loaded" ;;
  *) exit 64 ;;
esac
""",
    )
    _write_executable(fake_bin / "python", "#!/bin/sh\nexit 0\n")

    fake_home = tmp_path / "home"
    env = os.environ | {
        "HOME": str(fake_home),
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "PAL_PYTHON": str(fake_bin / "python"),
        "PAL_TEST_STATE": str(state_dir),
        "PAL_TEST_BOOTOUT_DELAY": "0.5",
    }
    first = subprocess.Popen(
        ["bash", str(SERVICE_SCRIPT), "install"],
        cwd=PAL_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(0.1)
    second = subprocess.run(
        ["bash", str(SERVICE_SCRIPT), "install"],
        cwd=PAL_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    first_stdout, first_stderr = first.communicate(timeout=10)

    assert first.returncode == 0, first_stderr
    assert second.returncode == 0, second.stderr
    assert "installed and started" in first_stdout
    assert "installed and started" in second.stdout
    assert loaded.is_file()


def test_heartbeat_queued_runs_refreshes_only_healthy_assigned_public_runs(tmp_path):
    store = SQLiteConversationStorage(tmp_path / "state", tmp_path / "keys" / "state.key")
    worker_digests = {"worker-1": "worker-1-digest", "worker-2": "worker-2-digest"}
    for worker, digest in worker_digests.items():
        store.register_process(worker, "clink_worker")
        store.publish_capability(
            cli_name="codex",
            role="default",
            config_digest=digest,
            executable_identity="test-executable",
            model="gpt-5.6-sol",
            reasoning_effort="high",
            owner_instance_id=worker,
            owner_mode="clink_worker",
        )

    def queue_run(worker: str) -> str:
        run_id = str(uuid.uuid4())
        store.create_queued_worker_run(
            run_id=run_id,
            continuation_id=None,
            exchange_id=None,
            cli_name="codex",
            role="default",
            envelope={
                "request": run_id,
                "cli_name": "codex",
                "role": "default",
                "capability_digest": worker_digests[worker],
            },
            assigned_worker_instance_id=worker,
            capability_digest=worker_digests[worker],
        )
        return run_id

    healthy_run = queue_run("worker-1")
    other_owner_run = queue_run("worker-2")
    expired_run = queue_run("worker-1")
    with store._write() as connection:
        connection.execute("UPDATE clink_runs SET updated_at_us=1")
        connection.execute("UPDATE clink_run_queue SET lease_expires_at_us=0 WHERE run_id=?", (expired_run,))
        before = {
            row["run_id"]: (row["updated_at_us"], row["lease_expires_at_us"])
            for row in connection.execute(
                """SELECT r.run_id,r.updated_at_us,q.lease_expires_at_us
                   FROM clink_runs r JOIN clink_run_queue q ON q.run_id=r.run_id"""
            )
        }

    assert store.heartbeat_queued_runs("worker-1", lease_seconds=300) == 1

    after = {
        row["run_id"]: (row["updated_at_us"], row["lease_expires_at_us"])
        for row in store.connection().execute(
            """SELECT r.run_id,r.updated_at_us,q.lease_expires_at_us
               FROM clink_runs r JOIN clink_run_queue q ON q.run_id=r.run_id"""
        )
    }
    assert after[healthy_run][0] > before[healthy_run][0]
    assert after[healthy_run][1] > before[healthy_run][1]
    assert after[other_owner_run] == before[other_owner_run]
    assert after[expired_run] == before[expired_run]
