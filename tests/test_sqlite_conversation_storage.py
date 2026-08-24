import multiprocessing
import uuid

import pytest

from utils.sqlite_conversation_storage import (
    ConversationBusy,
    ConversationCapacityError,
    ExecutionReadinessError,
    SQLiteConversationStorage,
    StorageCorruptionError,
    StorageKeyError,
)


@pytest.fixture
def store(tmp_path):
    return SQLiteConversationStorage(tmp_path / "state", tmp_path / "keys" / "state.key")


def _turn(role, content):
    return {"role": role, "content": content, "timestamp": "2026-08-24T00:00:00+00:00"}


def test_restart_reopens_encrypted_thread_without_plaintext(tmp_path):
    state = tmp_path / "state"
    key = tmp_path / "keys" / "state.key"
    first = SQLiteConversationStorage(state, key)
    thread_id = first.create_thread("clink", {"prompt": "TOP_SECRET"})
    admission = first.begin_exchange(thread_id, "clink", _turn("user", "HIDDEN_USER"))
    first.complete_exchange(admission.exchange_id, _turn("assistant", "HIDDEN_ASSISTANT"))

    raw = (state / "pal_state.sqlite3").read_bytes()
    assert b"TOP_SECRET" not in raw
    assert b"HIDDEN_USER" not in raw
    assert b"HIDDEN_ASSISTANT" not in raw

    reopened = SQLiteConversationStorage(state, key)
    loaded = reopened.load_thread(thread_id)
    assert [turn["content"] for turn in loaded["turns"]] == ["HIDDEN_USER", "HIDDEN_ASSISTANT"]


def test_busy_rejected_before_second_user_turn(store):
    thread_id = store.create_thread("clink", {})
    first = store.begin_exchange(thread_id, "clink", _turn("user", "one"))
    with pytest.raises(ConversationBusy) as error:
        store.begin_exchange(thread_id, "clink", _turn("user", "two"))
    assert error.value.exchange_id == first.exchange_id
    assert store.load_thread(thread_id)["turns"] == []


def test_idempotent_admission_does_not_duplicate(store):
    thread_id = store.create_thread("clink", {})
    first = store.begin_exchange(thread_id, "clink", _turn("user", "one"), "stable-request-1")
    second = store.begin_exchange(thread_id, "clink", _turn("user", "one"), "stable-request-1")
    assert second.exchange_id == first.exchange_id
    assert second.idempotent is True
    count = store.connection().execute("SELECT COUNT(*) FROM conversation_turns").fetchone()[0]
    assert count == 1


def test_initial_idempotency_reuses_thread_and_original_run(store):
    first_thread = store.create_thread("clink", {"prompt": "one"}, initial_idempotency_key="initial-key")
    second_thread = store.create_thread("clink", {"prompt": "two"}, initial_idempotency_key="initial-key")
    assert second_thread == first_thread

    admission = store.begin_exchange(first_thread, "clink", _turn("user", "one"), "initial-key")
    run_id = str(uuid.uuid4())
    store.create_run(
        run_id=run_id,
        continuation_id=first_thread,
        exchange_id=admission.exchange_id,
        cli_name="codex",
        role="default",
        owner_instance_id="pal-1",
    )
    replay = store.begin_exchange(first_thread, "clink", _turn("user", "one"), "initial-key")
    assert replay.idempotent is True
    assert replay.run_id == run_id


def test_capacity_reserves_complete_pair(store):
    thread_id = store.create_thread("clink", {})
    store.append_legacy_turn(thread_id, _turn("user", "legacy"), 300, 3)
    store.append_legacy_turn(thread_id, _turn("assistant", "legacy"), 300, 3)
    with pytest.raises(ConversationCapacityError):
        store.begin_exchange(thread_id, "clink", _turn("user", "cannot-fit"), max_turns=3)
    assert store.connection().execute("SELECT COUNT(*) FROM conversation_turns").fetchone()[0] == 2


def test_capability_required_before_mutation(store):
    thread_id = store.create_thread("clink", {})
    with pytest.raises(ExecutionReadinessError):
        store.begin_exchange(
            thread_id,
            "clink",
            _turn("user", "not admitted"),
            capability_digest="missing",
        )
    assert store.connection().execute("SELECT COUNT(*) FROM conversation_turns").fetchone()[0] == 0

    store.publish_capability(
        cli_name="codex",
        role="default",
        config_digest="ready",
        executable_identity="1:2:3:4",
        model="gpt-5.6-sol",
        reasoning_effort="high",
        owner_instance_id="worker-1",
    )
    admission = store.begin_exchange(
        thread_id,
        "clink",
        _turn("user", "admitted"),
        capability_digest="ready",
    )
    assert admission.exchange_id


def test_atomic_exchange_and_terminal_run_first_terminal_wins(store):
    thread_id = store.create_thread("clink", {})
    admission = store.begin_exchange(thread_id, "clink", _turn("user", "question"))
    run_id = str(uuid.uuid4())
    store.create_run(
        run_id=run_id,
        continuation_id=thread_id,
        exchange_id=admission.exchange_id,
        cli_name="codex",
        role="default",
        owner_instance_id="pal-1",
    )
    store.mark_run_running(run_id)
    store.complete_exchange(
        admission.exchange_id,
        _turn("assistant", "answer"),
        run_terminal={"run_id": run_id, "status": "completed", "result": "RESULT_ONE"},
    )
    store.finalize_run(run_id, status="failed", result="RESULT_TWO")
    assert store.read_run(run_id)["status"] == "completed"
    assert store.read_run(run_id)["result"] == "RESULT_ONE"
    assert [turn["content"] for turn in store.load_thread(thread_id)["turns"]] == ["question", "answer"]


def test_read_run_normalizes_nullable_storage_fields(store):
    run_id = str(uuid.uuid4())
    record = store.create_run(
        run_id=run_id,
        continuation_id=None,
        exchange_id=None,
        cli_name="codex",
        role="default",
        owner_instance_id="pal-1",
    )
    assert record["started_at"] is None
    assert record["finished_at"] is None
    assert record["error"] is None
    assert "started_at_us" not in record
    assert "finished_at_us" not in record
    assert "error_category" not in record


def test_terminal_run_has_no_fk_dependency_on_thread(store):
    thread_id = store.create_thread("clink", {})
    run_id = str(uuid.uuid4())
    store.create_run(
        run_id=run_id,
        continuation_id=thread_id,
        exchange_id=None,
        cli_name="codex",
        role="default",
        owner_instance_id="pal-1",
    )
    store.finalize_run(run_id, status="completed", result="durable")
    with store._write() as connection:
        connection.execute("DELETE FROM conversation_threads WHERE thread_id=?", (thread_id,))
    assert store.read_run(run_id)["result"] == "durable"


def test_cleanup_removes_only_expired_inactive_state(store):
    expired_thread = store.create_thread("clink", {}, ttl_seconds=1)
    live_thread = store.create_thread("clink", {}, ttl_seconds=300)
    active_thread = store.create_thread("clink", {}, ttl_seconds=1)
    store.begin_exchange(active_thread, "clink", _turn("user", "active"), lease_seconds=300)
    old_run = str(uuid.uuid4())
    live_run = str(uuid.uuid4())
    for run_id in (old_run, live_run):
        store.create_run(
            run_id=run_id,
            continuation_id=None,
            exchange_id=None,
            cli_name="codex",
            role="default",
            owner_instance_id="pal-1",
        )
    store.finalize_run(old_run, status="completed", result="old")
    with store._write() as connection:
        connection.execute(
            "UPDATE conversation_threads SET expires_at_us=0 WHERE thread_id IN (?,?)",
            (expired_thread, active_thread),
        )
        connection.execute("UPDATE clink_runs SET finished_at_us=0,updated_at_us=0 WHERE run_id=?", (old_run,))

    assert store.cleanup(run_retention_seconds=1) >= 2
    assert store.load_thread(expired_thread) is None
    assert store.load_thread(live_thread) is not None
    assert (
        store.connection().execute("SELECT 1 FROM conversation_threads WHERE thread_id=?", (active_thread,)).fetchone()
    )
    assert store.read_run(old_run) is None
    assert store.read_run(live_run)["status"] == "queued"


def test_queue_claim_heartbeat_extends_only_owners_lease(store):
    run_id = str(uuid.uuid4())
    store.create_queued_worker_run(
        run_id=run_id,
        continuation_id=None,
        exchange_id=None,
        cli_name="codex",
        role="default",
        envelope={"secret": "prepared"},
    )
    store.claim_next_run("worker-1", lease_seconds=1)
    before = (
        store.connection()
        .execute("SELECT lease_expires_at_us FROM clink_run_queue WHERE run_id=?", (run_id,))
        .fetchone()[0]
    )
    assert store.heartbeat_queue_claim(run_id, "worker-2", lease_seconds=300) is False
    assert store.heartbeat_queue_claim(run_id, "worker-1", lease_seconds=300) is True
    after = (
        store.connection()
        .execute("SELECT lease_expires_at_us FROM clink_run_queue WHERE run_id=?", (run_id,))
        .fetchone()[0]
    )
    assert after > before


def test_run_heartbeat_extends_its_active_exchange(store):
    thread_id = store.create_thread("clink", {})
    admission = store.begin_exchange(thread_id, "clink", _turn("user", "one"), lease_seconds=1)
    run_id = str(uuid.uuid4())
    store.create_run(
        run_id=run_id,
        continuation_id=thread_id,
        exchange_id=admission.exchange_id,
        cli_name="codex",
        role="default",
        owner_instance_id="pal-1",
    )
    before = (
        store.connection()
        .execute("SELECT lease_expires_at_us FROM conversation_exchanges WHERE exchange_id=?", (admission.exchange_id,))
        .fetchone()[0]
    )
    store.heartbeat_run(run_id, lease_seconds=300)
    after = (
        store.connection()
        .execute("SELECT lease_expires_at_us FROM conversation_exchanges WHERE exchange_id=?", (admission.exchange_id,))
        .fetchone()[0]
    )
    assert after > before


def test_expired_unclaimed_queue_is_interrupted_and_never_claimed(store):
    run_id = str(uuid.uuid4())
    store.create_queued_worker_run(
        run_id=run_id,
        continuation_id=None,
        exchange_id=None,
        cli_name="codex",
        role="default",
        envelope={"secret": "do not execute late"},
    )
    with store._write() as connection:
        connection.execute("UPDATE clink_run_queue SET lease_expires_at_us=0 WHERE run_id=?", (run_id,))
    assert store.claim_next_run("late-worker") is None
    assert store.interrupt_stale_claims() == 1
    record = store.read_run(run_id)
    assert record["status"] == "interrupted"
    assert record["error"]["category"] == "worker_queue_expired"


def test_schema_hash_tampering_fails(store):
    store.connection().execute("UPDATE schema_migrations SET migration_sha256='tampered'")
    store.connection().close()
    with pytest.raises(StorageCorruptionError):
        SQLiteConversationStorage(store.state_dir, store.key_file)


def test_valid_v1_database_migrates_to_current(store):
    connection = store.connection()
    connection.execute("DELETE FROM schema_migrations WHERE version>=2")
    connection.execute("DROP INDEX one_initial_idempotency_key_per_tool")
    connection.execute("ALTER TABLE conversation_threads DROP COLUMN initial_idempotency_hash")
    connection.execute("ALTER TABLE worker_capabilities DROP COLUMN owner_mode")
    connection.close()

    reopened = SQLiteConversationStorage(store.state_dir, store.key_file)
    versions = reopened.connection().execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()
    columns = reopened.connection().execute("PRAGMA table_info(worker_capabilities)").fetchall()
    assert [row[0] for row in versions] == [1, 2, 3]
    assert "owner_mode" in {row[1] for row in columns}


def test_key_permissions_fail_closed(tmp_path):
    key_dir = tmp_path / "keys"
    key_dir.mkdir()
    key = key_dir / "state.key"
    key.write_text("not-a-key")
    key.chmod(0o644)
    with pytest.raises(StorageKeyError):
        SQLiteConversationStorage(tmp_path / "state", key)


def _child_append(state: str, key: str, thread_id: str, index: int, queue):
    try:
        storage = SQLiteConversationStorage(state, key)
        result = storage.append_legacy_turn(thread_id, _turn("user", f"child-{index}"), 300, 20)
        queue.put((index, result, None))
    except Exception as exc:
        queue.put((index, False, type(exc).__name__))


def test_multiprocess_distinct_turns_have_no_lost_update(tmp_path):
    state = tmp_path / "state"
    key = tmp_path / "keys" / "state.key"
    storage = SQLiteConversationStorage(state, key)
    thread_id = storage.create_thread("clink", {})
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(target=_child_append, args=(str(state), str(key), thread_id, index, queue))
        for index in range(4)
    ]
    for process in processes:
        process.start()
    results = [queue.get(timeout=20) for _ in processes]
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0
    assert all(result[1] for result in results), results
    assert len(storage.load_thread(thread_id)["turns"]) == 4
