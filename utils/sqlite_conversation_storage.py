"""Encrypted, crash-consistent SQLite state for PAL continuations and clink runs.

The database deliberately keeps structural metadata queryable while encrypting
all user/model content, paths, initial context, execution envelopes and results
with AES-256-GCM. Mutations use ``BEGIN IMMEDIATE`` so cross-process contention
is resolved before application state is read.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import stat
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

SCHEMA_VERSION = 3
TERMINAL_RUN_STATUSES = frozenset({"completed", "failed", "interrupted"})
ACTIVE_EXCHANGE_STATUSES = frozenset({"admitted", "running"})


class DurableStorageError(RuntimeError):
    pass


class StoragePathError(DurableStorageError):
    pass


class StorageKeyError(DurableStorageError):
    pass


class StorageCorruptionError(DurableStorageError):
    pass


class ConversationNotFound(DurableStorageError):
    pass


class ConversationExpired(ConversationNotFound):
    pass


class ConversationBusy(DurableStorageError):
    def __init__(self, exchange_id: str, run_id: str | None = None):
        super().__init__("conversation has an active exchange")
        self.exchange_id = exchange_id
        self.run_id = run_id


class ConversationCapacityError(DurableStorageError):
    pass


class ExecutionReadinessError(DurableStorageError):
    pass


@dataclass(frozen=True)
class ExchangeAdmission:
    exchange_id: str
    pre_user_thread: dict[str, Any]
    idempotent: bool = False
    run_id: str | None = None


def _now_us() -> int:
    return time.time_ns() // 1_000


def _iso(us: int) -> str:
    return datetime.fromtimestamp(us / 1_000_000, tz=timezone.utc).isoformat()


def _canonical_uuid(value: str, label: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, TypeError, AttributeError) as exc:
        raise DurableStorageError(f"{label} must be a canonical UUID") from exc
    if str(parsed) != value:
        raise DurableStorageError(f"{label} must be a canonical UUID")
    return value


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


class _Codec:
    def __init__(self, key: bytes):
        if len(key) != 32:
            raise StorageKeyError("PAL state key must contain exactly 32 bytes")
        self._key = key
        self._aead = AESGCM(key)
        self.key_id = hashlib.sha256(key).hexdigest()[:16]

    def encrypt(self, value: Any, aad: str) -> bytes:
        nonce = secrets.token_bytes(12)
        return nonce + self._aead.encrypt(nonce, _json_bytes(value), aad.encode("utf-8"))

    def decrypt(self, payload: bytes, aad: str) -> Any:
        try:
            raw = self._aead.decrypt(payload[:12], payload[12:], aad.encode("utf-8"))
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise StorageCorruptionError("encrypted PAL state failed authentication") from exc

    def keyed_hash(self, value: str) -> str:
        return hmac.new(self._key, value.encode("utf-8"), hashlib.sha256).hexdigest()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def _reject_symlink_components(path: Path) -> None:
    """Reject any existing symlink in a security-sensitive path."""
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            if current.is_symlink():
                raise StoragePathError(f"PAL state path component cannot be a symlink: {current}")
            if not current.exists():
                break
        except OSError as exc:
            raise StoragePathError(f"PAL state path component could not be inspected: {current}") from exc


def _secure_directory(path: Path) -> Path:
    if not path.is_absolute():
        raise StoragePathError("PAL state paths must be absolute")
    _reject_symlink_components(path)
    if path.exists() and path.is_symlink():
        raise StoragePathError("PAL state directory cannot be a symlink")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    resolved = path.resolve(strict=True)
    info = resolved.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise StoragePathError("PAL state directory has an unexpected type or owner")
    if info.st_mode & 0o077:
        os.chmod(resolved, 0o700)
    return resolved


def _load_or_create_key(path: Path) -> bytes:
    directory = _secure_directory(path.parent)
    if path.exists() and path.is_symlink():
        raise StorageKeyError("PAL state key cannot be a symlink")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        descriptor = -1
    if descriptor >= 0:
        key = secrets.token_bytes(32)
        encoded = base64.urlsafe_b64encode(key)
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(directory)
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
            raise StorageKeyError("PAL state key has an unexpected type, owner, or link count")
        if info.st_mode & 0o077:
            raise StorageKeyError("PAL state key permissions must be 0600")
        key = base64.b64decode(path.read_bytes(), altchars=b"-_", validate=True)
    except StorageKeyError:
        raise
    except Exception as exc:
        raise StorageKeyError("PAL state key is unreadable or invalid") from exc
    if len(key) != 32:
        raise StorageKeyError("PAL state key must decode to exactly 32 bytes")
    return key


_SCHEMA = """
CREATE TABLE schema_migrations (
  version INTEGER PRIMARY KEY,
  migration_sha256 TEXT NOT NULL,
  applied_at_us INTEGER NOT NULL
);
CREATE TABLE conversation_threads (
  thread_id TEXT PRIMARY KEY,
  parent_thread_id TEXT REFERENCES conversation_threads(thread_id) ON DELETE RESTRICT,
  tool_name TEXT NOT NULL,
  created_at_us INTEGER NOT NULL,
  updated_at_us INTEGER NOT NULL,
  expires_at_us INTEGER NOT NULL,
  initial_payload BLOB NOT NULL,
  next_ordinal INTEGER NOT NULL DEFAULT 0,
  turn_count INTEGER NOT NULL DEFAULT 0,
  active_exchange_id TEXT
);
CREATE TABLE conversation_exchanges (
  exchange_id TEXT PRIMARY KEY,
  thread_id TEXT NOT NULL REFERENCES conversation_threads(thread_id) ON DELETE CASCADE,
  tool_name TEXT NOT NULL,
  idempotency_hash TEXT,
  status TEXT NOT NULL CHECK(status IN ('admitted','running','completed','failed','interrupted','abandoned')),
  owner_instance_id TEXT,
  capability_digest TEXT,
  lease_expires_at_us INTEGER,
  created_at_us INTEGER NOT NULL,
  updated_at_us INTEGER NOT NULL,
  failure_category TEXT,
  UNIQUE(thread_id, idempotency_hash)
);
CREATE UNIQUE INDEX one_active_exchange_per_thread
  ON conversation_exchanges(thread_id)
  WHERE status IN ('admitted','running');
CREATE TABLE conversation_turns (
  thread_id TEXT NOT NULL REFERENCES conversation_threads(thread_id) ON DELETE CASCADE,
  ordinal INTEGER NOT NULL,
  exchange_id TEXT REFERENCES conversation_exchanges(exchange_id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK(role IN ('user','assistant')),
  created_at_us INTEGER NOT NULL,
  payload BLOB NOT NULL,
  PRIMARY KEY(thread_id, ordinal),
  UNIQUE(exchange_id, role)
);
CREATE TABLE clink_runs (
  run_id TEXT PRIMARY KEY,
  continuation_id TEXT,
  exchange_id TEXT UNIQUE,
  cli_name TEXT NOT NULL,
  role TEXT,
  status TEXT NOT NULL CHECK(status IN ('queued','running','completed','failed','interrupted')),
  owner_instance_id TEXT,
  owner_type TEXT NOT NULL DEFAULT 'pal' CHECK(owner_type IN ('pal','worker')),
  attachment TEXT NOT NULL DEFAULT 'attached',
  created_at_us INTEGER NOT NULL,
  started_at_us INTEGER,
  updated_at_us INTEGER NOT NULL,
  finished_at_us INTEGER,
  lease_expires_at_us INTEGER,
  duration_seconds REAL,
  result_payload BLOB,
  result_hmac TEXT,
  output_truncated INTEGER NOT NULL DEFAULT 0,
  error_category TEXT,
  error_message BLOB
);
CREATE INDEX clink_runs_continuation ON clink_runs(continuation_id, created_at_us DESC);
CREATE TABLE process_instances (
  instance_id TEXT PRIMARY KEY,
  pid INTEGER NOT NULL,
  mode TEXT NOT NULL CHECK(mode IN ('pal','clink_worker')),
  started_at_us INTEGER NOT NULL,
  heartbeat_at_us INTEGER NOT NULL,
  stopped_at_us INTEGER
);
CREATE TABLE worker_capabilities (
  cli_name TEXT NOT NULL,
  role TEXT NOT NULL,
  config_digest TEXT NOT NULL,
  executable_identity TEXT NOT NULL,
  model TEXT,
  reasoning_effort TEXT,
  owner_instance_id TEXT NOT NULL,
  attested_at_us INTEGER NOT NULL,
  expires_at_us INTEGER NOT NULL,
  PRIMARY KEY(cli_name, role, config_digest)
);
CREATE TABLE clink_run_queue (
  run_id TEXT PRIMARY KEY REFERENCES clink_runs(run_id) ON DELETE CASCADE,
  status TEXT NOT NULL CHECK(status IN ('queued','claimed')),
  available_at_us INTEGER NOT NULL,
  claimed_by TEXT,
  lease_expires_at_us INTEGER,
  envelope_payload BLOB NOT NULL
);
"""
_SCHEMA_HASH = hashlib.sha256(_SCHEMA.encode("utf-8")).hexdigest()
_MIGRATION_2 = """
ALTER TABLE worker_capabilities
  ADD COLUMN owner_mode TEXT NOT NULL DEFAULT 'pal'
  CHECK(owner_mode IN ('pal','clink_worker'));
"""
_MIGRATION_2_HASH = hashlib.sha256(_MIGRATION_2.encode("utf-8")).hexdigest()
_MIGRATION_3 = """
ALTER TABLE conversation_threads
  ADD COLUMN initial_idempotency_hash TEXT;
CREATE UNIQUE INDEX one_initial_idempotency_key_per_tool
  ON conversation_threads(tool_name, initial_idempotency_hash)
  WHERE initial_idempotency_hash IS NOT NULL;
"""
_MIGRATION_3_HASH = hashlib.sha256(_MIGRATION_3.encode("utf-8")).hexdigest()
_MIGRATION_HASHES = {1: _SCHEMA_HASH, 2: _MIGRATION_2_HASH, 3: _MIGRATION_3_HASH}


class SQLiteConversationStorage:
    def __init__(self, state_dir: Path | str, key_file: Path | str, *, busy_timeout_ms: int = 250):
        self.state_dir = _secure_directory(Path(state_dir).expanduser())
        self.key_file = Path(key_file).expanduser()
        if not self.key_file.is_absolute():
            raise StoragePathError("PAL state key path must be absolute")
        self._codec = _Codec(_load_or_create_key(self.key_file))
        self.db_path = self.state_dir / "pal_state.sqlite3"
        self.busy_timeout_ms = max(1, int(busy_timeout_ms))
        self._local = threading.local()
        self._initialize()

    @property
    def key_id(self) -> str:
        return self._codec.key_id

    def _connect(self) -> sqlite3.Connection:
        for candidate in (
            self.db_path,
            Path(str(self.db_path) + "-wal"),
            Path(str(self.db_path) + "-shm"),
        ):
            if candidate.is_symlink():
                raise StoragePathError("PAL state database files cannot be symlinks")
            if candidate.exists():
                info = candidate.lstat()
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
                    raise StoragePathError("PAL state database file has an unexpected type, owner, or link count")
        connection = sqlite3.connect(
            self.db_path,
            timeout=self.busy_timeout_ms / 1000,
            isolation_level=None,
            check_same_thread=True,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA wal_autocheckpoint=1000")
        connection.execute("PRAGMA journal_size_limit=67108864")
        mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if str(mode).lower() != "wal":
            connection.close()
            raise DurableStorageError("PAL state database could not enable WAL mode")
        return connection

    def connection(self) -> sqlite3.Connection:
        pid = os.getpid()
        connection = getattr(self._local, "connection", None)
        if connection is None or getattr(self._local, "pid", None) != pid:
            if connection is not None:
                connection.close()
            connection = self._connect()
            self._local.connection = connection
            self._local.pid = pid
        return connection

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        connection = self.connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise

    def _initialize(self) -> None:
        connection = self.connection()
        connection.execute("PRAGMA auto_vacuum=INCREMENTAL")
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
        ).fetchone()
        if not exists:
            with self._write() as transaction:
                for statement in _SCHEMA.split(";"):
                    if statement.strip():
                        transaction.execute(statement)
                transaction.execute(
                    "INSERT INTO schema_migrations(version,migration_sha256,applied_at_us) VALUES(?,?,?)",
                    (1, _SCHEMA_HASH, _now_us()),
                )
        rows = connection.execute("SELECT version,migration_sha256 FROM schema_migrations ORDER BY version").fetchall()
        versions = [row["version"] for row in rows]
        if versions and versions != list(range(1, versions[-1] + 1)):
            raise StorageCorruptionError("PAL state migration history has a version gap")
        for row in rows:
            if _MIGRATION_HASHES.get(row["version"]) != row["migration_sha256"]:
                raise StorageCorruptionError("PAL state migration history was edited or is unsupported")
        current_version = rows[-1]["version"] if rows else 0
        if current_version > SCHEMA_VERSION:
            raise StorageCorruptionError("PAL state schema is newer than this PAL build")
        if current_version < 2:
            with self._write() as transaction:
                for statement in _MIGRATION_2.split(";"):
                    if statement.strip():
                        transaction.execute(statement)
                transaction.execute(
                    "INSERT INTO schema_migrations(version,migration_sha256,applied_at_us) VALUES(?,?,?)",
                    (2, _MIGRATION_2_HASH, _now_us()),
                )
            current_version = 2
        if current_version < 3:
            with self._write() as transaction:
                for statement in _MIGRATION_3.split(";"):
                    if statement.strip():
                        transaction.execute(statement)
                transaction.execute(
                    "INSERT INTO schema_migrations(version,migration_sha256,applied_at_us) VALUES(?,?,?)",
                    (3, _MIGRATION_3_HASH, _now_us()),
                )
        check = connection.execute("PRAGMA quick_check").fetchone()[0]
        if check != "ok":
            raise StorageCorruptionError("PAL state database quick_check failed")
        try:
            os.chmod(self.db_path, 0o600)
        except OSError as exc:
            raise StoragePathError("PAL state database permissions could not be secured") from exc

    def _thread_row(self, connection: sqlite3.Connection, thread_id: str, *, now: int | None = None):
        row = connection.execute("SELECT * FROM conversation_threads WHERE thread_id=?", (thread_id,)).fetchone()
        if row is None:
            raise ConversationNotFound("conversation thread was not found")
        if row["expires_at_us"] <= (now or _now_us()):
            raise ConversationExpired("conversation thread has expired")
        return row

    def _load_thread(self, connection: sqlite3.Connection, thread_id: str) -> dict[str, Any]:
        row = self._thread_row(connection, thread_id)
        turns = connection.execute(
            """SELECT t.payload FROM conversation_turns t
               LEFT JOIN conversation_exchanges e ON e.exchange_id=t.exchange_id
               WHERE t.thread_id=? AND (t.exchange_id IS NULL OR e.status='completed')
               ORDER BY t.ordinal""",
            (thread_id,),
        ).fetchall()
        return {
            "thread_id": row["thread_id"],
            "parent_thread_id": row["parent_thread_id"],
            "created_at": _iso(row["created_at_us"]),
            "last_updated_at": _iso(row["updated_at_us"]),
            "tool_name": row["tool_name"],
            "turns": [self._codec.decrypt(turn["payload"], f"turn:{thread_id}") for turn in turns],
            "initial_context": self._codec.decrypt(row["initial_payload"], f"thread:{thread_id}"),
        }

    def create_thread(
        self,
        tool_name: str,
        initial_context: dict[str, Any],
        parent_thread_id: str | None = None,
        thread_id: str | None = None,
        ttl_seconds: int = 10_800,
        initial_idempotency_key: str | None = None,
    ) -> str:
        thread_id = _canonical_uuid(thread_id or str(uuid.uuid4()), "thread_id")
        now = _now_us()
        initial_idempotency_hash = self._codec.keyed_hash(initial_idempotency_key) if initial_idempotency_key else None
        with self._write() as connection:
            if initial_idempotency_hash:
                existing = connection.execute(
                    """SELECT thread_id FROM conversation_threads
                       WHERE tool_name=? AND initial_idempotency_hash=?""",
                    (tool_name, initial_idempotency_hash),
                ).fetchone()
                if existing:
                    self._thread_row(connection, existing["thread_id"], now=now)
                    return existing["thread_id"]
            if parent_thread_id:
                _canonical_uuid(parent_thread_id, "parent_thread_id")
                self._thread_row(connection, parent_thread_id, now=now)
                if parent_thread_id == thread_id:
                    raise DurableStorageError("conversation parent cycle rejected")
            connection.execute(
                """INSERT INTO conversation_threads
                   (thread_id,parent_thread_id,tool_name,created_at_us,updated_at_us,expires_at_us,
                    initial_payload,initial_idempotency_hash)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (
                    thread_id,
                    parent_thread_id,
                    tool_name,
                    now,
                    now,
                    now + int(ttl_seconds) * 1_000_000,
                    self._codec.encrypt(initial_context, f"thread:{thread_id}"),
                    initial_idempotency_hash,
                ),
            )
        return thread_id

    def load_thread(self, thread_id: str) -> dict[str, Any] | None:
        _canonical_uuid(thread_id, "thread_id")
        try:
            return self._load_thread(self.connection(), thread_id)
        except ConversationNotFound:
            return None

    def append_legacy_turn(self, thread_id: str, turn: dict[str, Any], ttl_seconds: int, max_turns: int) -> bool:
        _canonical_uuid(thread_id, "thread_id")
        now = _now_us()
        with self._write() as connection:
            row = self._thread_row(connection, thread_id, now=now)
            if row["turn_count"] >= max_turns:
                return False
            ordinal = row["next_ordinal"]
            connection.execute(
                "INSERT INTO conversation_turns(thread_id,ordinal,exchange_id,role,created_at_us,payload) VALUES(?,?,NULL,?,?,?)",
                (thread_id, ordinal, turn["role"], now, self._codec.encrypt(turn, f"turn:{thread_id}")),
            )
            connection.execute(
                """UPDATE conversation_threads SET next_ordinal=?,turn_count=turn_count+1,
                   updated_at_us=?,expires_at_us=? WHERE thread_id=?""",
                (ordinal + 1, now, now + int(ttl_seconds) * 1_000_000, thread_id),
            )
        return True

    def begin_exchange(
        self,
        thread_id: str,
        tool_name: str,
        user_turn: dict[str, Any],
        client_idempotency_key: str | None = None,
        owner_instance_id: str | None = None,
        capability_digest: str | None = None,
        ttl_seconds: int = 10_800,
        max_turns: int = 50,
        lease_seconds: int = 90,
    ) -> ExchangeAdmission:
        _canonical_uuid(thread_id, "thread_id")
        exchange_id = str(uuid.uuid4())
        now = _now_us()
        idem_hash = self._codec.keyed_hash(client_idempotency_key) if client_idempotency_key else None
        with self._write() as connection:
            row = self._thread_row(connection, thread_id, now=now)
            if capability_digest:
                ready = connection.execute(
                    "SELECT 1 FROM worker_capabilities WHERE config_digest=? AND expires_at_us>?",
                    (capability_digest, now),
                ).fetchone()
                if ready is None:
                    raise ExecutionReadinessError("no fresh matching execution capability")
            if idem_hash:
                existing = connection.execute(
                    """SELECT e.exchange_id,r.run_id FROM conversation_exchanges e
                       LEFT JOIN clink_runs r ON r.exchange_id=e.exchange_id
                       WHERE e.thread_id=? AND e.idempotency_hash=?""",
                    (thread_id, idem_hash),
                ).fetchone()
                if existing:
                    snapshot = self._load_thread(connection, thread_id)
                    return ExchangeAdmission(existing["exchange_id"], snapshot, True, existing["run_id"])
            active = connection.execute(
                """SELECT e.exchange_id,e.lease_expires_at_us,r.run_id
                   FROM conversation_exchanges e LEFT JOIN clink_runs r ON r.exchange_id=e.exchange_id
                   WHERE e.thread_id=? AND e.status IN ('admitted','running')""",
                (thread_id,),
            ).fetchone()
            if active:
                if active["run_id"] is None and active["lease_expires_at_us"] and active["lease_expires_at_us"] <= now:
                    connection.execute(
                        "UPDATE conversation_exchanges SET status='interrupted',updated_at_us=? WHERE exchange_id=?",
                        (now, active["exchange_id"]),
                    )
                    connection.execute(
                        "UPDATE conversation_threads SET active_exchange_id=NULL WHERE thread_id=?",
                        (thread_id,),
                    )
                else:
                    raise ConversationBusy(active["exchange_id"], active["run_id"])
            if row["turn_count"] + 2 > max_turns:
                raise ConversationCapacityError("conversation has no complete exchange capacity remaining")
            snapshot = self._load_thread(connection, thread_id)
            ordinal = row["next_ordinal"]
            connection.execute(
                """INSERT INTO conversation_exchanges
                   (exchange_id,thread_id,tool_name,idempotency_hash,status,owner_instance_id,
                    capability_digest,lease_expires_at_us,created_at_us,updated_at_us)
                   VALUES(?,?,?,?, 'admitted',?,?,?,?,?)""",
                (
                    exchange_id,
                    thread_id,
                    tool_name,
                    idem_hash,
                    owner_instance_id,
                    capability_digest,
                    now + int(lease_seconds) * 1_000_000,
                    now,
                    now,
                ),
            )
            connection.execute(
                "INSERT INTO conversation_turns(thread_id,ordinal,exchange_id,role,created_at_us,payload) VALUES(?,?,?,?,?,?)",
                (thread_id, ordinal, exchange_id, "user", now, self._codec.encrypt(user_turn, f"turn:{thread_id}")),
            )
            connection.execute(
                """UPDATE conversation_threads SET active_exchange_id=?,next_ordinal=?,turn_count=turn_count+1,
                   updated_at_us=?,expires_at_us=? WHERE thread_id=?""",
                (exchange_id, ordinal + 1, now, now + int(ttl_seconds) * 1_000_000, thread_id),
            )
        return ExchangeAdmission(exchange_id, snapshot)

    def complete_exchange(
        self,
        exchange_id: str,
        assistant_turn: dict[str, Any],
        owner_instance_id: str | None = None,
        run_terminal: dict[str, Any] | None = None,
        ttl_seconds: int = 10_800,
    ) -> bool:
        _canonical_uuid(exchange_id, "exchange_id")
        now = _now_us()
        with self._write() as connection:
            exchange = connection.execute(
                "SELECT * FROM conversation_exchanges WHERE exchange_id=?", (exchange_id,)
            ).fetchone()
            if exchange is None:
                raise ConversationNotFound("conversation exchange was not found")
            if exchange["status"] == "completed":
                return False
            if exchange["status"] not in ACTIVE_EXCHANGE_STATUSES:
                raise DurableStorageError("conversation exchange is not completable")
            if owner_instance_id and exchange["owner_instance_id"] not in (None, owner_instance_id):
                raise ConversationBusy(exchange_id)
            thread = self._thread_row(connection, exchange["thread_id"], now=now)
            ordinal = thread["next_ordinal"]
            connection.execute(
                "INSERT INTO conversation_turns(thread_id,ordinal,exchange_id,role,created_at_us,payload) VALUES(?,?,?,?,?,?)",
                (
                    exchange["thread_id"],
                    ordinal,
                    exchange_id,
                    "assistant",
                    now,
                    self._codec.encrypt(assistant_turn, f"turn:{exchange['thread_id']}"),
                ),
            )
            connection.execute(
                "UPDATE conversation_exchanges SET status='completed',updated_at_us=?,lease_expires_at_us=NULL WHERE exchange_id=?",
                (now, exchange_id),
            )
            connection.execute(
                """UPDATE conversation_threads SET active_exchange_id=NULL,next_ordinal=?,turn_count=turn_count+1,
                   updated_at_us=?,expires_at_us=? WHERE thread_id=?""",
                (ordinal + 1, now, now + int(ttl_seconds) * 1_000_000, exchange["thread_id"]),
            )
            if run_terminal:
                self._finalize_run_in_transaction(connection, now=now, **run_terminal)
        return True

    def fail_exchange(self, exchange_id: str, category: str, owner_instance_id: str | None = None) -> bool:
        _canonical_uuid(exchange_id, "exchange_id")
        now = _now_us()
        with self._write() as connection:
            row = connection.execute(
                "SELECT thread_id,status,owner_instance_id FROM conversation_exchanges WHERE exchange_id=?",
                (exchange_id,),
            ).fetchone()
            if row is None:
                return False
            if row["status"] not in ACTIVE_EXCHANGE_STATUSES:
                return False
            if owner_instance_id and row["owner_instance_id"] not in (None, owner_instance_id):
                raise ConversationBusy(exchange_id)
            connection.execute(
                "UPDATE conversation_exchanges SET status='failed',failure_category=?,updated_at_us=?,lease_expires_at_us=NULL WHERE exchange_id=?",
                (str(category)[:120], now, exchange_id),
            )
            connection.execute(
                "UPDATE conversation_threads SET active_exchange_id=NULL WHERE thread_id=? AND active_exchange_id=?",
                (row["thread_id"], exchange_id),
            )
        return True

    def publish_capability(
        self,
        *,
        cli_name: str,
        role: str,
        config_digest: str,
        executable_identity: str,
        model: str | None,
        reasoning_effort: str | None,
        owner_instance_id: str,
        owner_mode: str = "pal",
        ttl_seconds: int = 75,
    ) -> None:
        now = _now_us()
        with self._write() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO worker_capabilities
                   (cli_name,role,config_digest,executable_identity,model,reasoning_effort,
                    owner_instance_id,attested_at_us,expires_at_us,owner_mode) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    cli_name,
                    role,
                    config_digest,
                    executable_identity,
                    model,
                    reasoning_effort,
                    owner_instance_id,
                    now,
                    now + int(ttl_seconds) * 1_000_000,
                    owner_mode,
                ),
            )

    def get_fresh_capability(self, cli_name: str, role: str, config_digest: str) -> dict[str, Any] | None:
        row = (
            self.connection()
            .execute(
                """SELECT * FROM worker_capabilities
               WHERE cli_name=? AND role=? AND config_digest=? AND expires_at_us>?""",
                (cli_name, role, config_digest, _now_us()),
            )
            .fetchone()
        )
        return dict(row) if row else None

    def get_fresh_worker_capability(self, cli_name: str, role: str) -> dict[str, Any] | None:
        row = (
            self.connection()
            .execute(
                """SELECT w.*,p.pid,p.heartbeat_at_us FROM worker_capabilities w
                   JOIN process_instances p ON p.instance_id=w.owner_instance_id
                   WHERE w.cli_name=? AND w.role=? AND w.owner_mode='clink_worker'
                     AND w.expires_at_us>? AND p.stopped_at_us IS NULL AND p.heartbeat_at_us>?
                   ORDER BY w.attested_at_us DESC LIMIT 1""",
                (cli_name, role, _now_us(), _now_us() - 50_000_000),
            )
            .fetchone()
        )
        if row is None:
            return None
        try:
            os.kill(row["pid"], 0)
        except (OSError, ProcessLookupError):
            return None
        return dict(row)

    def register_process(self, instance_id: str, mode: str) -> None:
        now = _now_us()
        with self._write() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO process_instances
                   (instance_id,pid,mode,started_at_us,heartbeat_at_us,stopped_at_us)
                   VALUES(?,?,?,?,?,NULL)""",
                (instance_id, os.getpid(), mode, now, now),
            )

    def heartbeat_process(self, instance_id: str) -> None:
        with self._write() as connection:
            connection.execute(
                "UPDATE process_instances SET heartbeat_at_us=? WHERE instance_id=? AND stopped_at_us IS NULL",
                (_now_us(), instance_id),
            )

    def stop_process(self, instance_id: str) -> None:
        with self._write() as connection:
            connection.execute(
                "UPDATE process_instances SET stopped_at_us=? WHERE instance_id=?",
                (_now_us(), instance_id),
            )

    def create_run(
        self,
        *,
        run_id: str,
        continuation_id: str | None,
        exchange_id: str | None,
        cli_name: str,
        role: str | None,
        owner_instance_id: str | None,
        owner_type: str = "pal",
    ) -> dict[str, Any]:
        _canonical_uuid(run_id, "run_id")
        now = _now_us()
        with self._write() as connection:
            connection.execute(
                """INSERT INTO clink_runs
                   (run_id,continuation_id,exchange_id,cli_name,role,status,owner_instance_id,owner_type,
                    created_at_us,updated_at_us,lease_expires_at_us) VALUES(?,?,?,?,?,'queued',?,?,?,?,?)""",
                (
                    run_id,
                    continuation_id,
                    exchange_id,
                    cli_name,
                    role,
                    owner_instance_id,
                    owner_type,
                    now,
                    now,
                    now + 90_000_000,
                ),
            )
        return self.read_run(run_id) or {}

    def read_run(self, run_id: str) -> dict[str, Any] | None:
        _canonical_uuid(run_id, "run_id")
        row = self.connection().execute("SELECT * FROM clink_runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            return None
        record = dict(row)
        record["created_at"] = _iso(record.pop("created_at_us"))
        record["updated_at"] = _iso(record.pop("updated_at_us"))
        started_at_us = record.pop("started_at_us")
        finished_at_us = record.pop("finished_at_us")
        record["started_at"] = _iso(started_at_us) if started_at_us else None
        record["finished_at"] = _iso(finished_at_us) if finished_at_us else None
        payload = record.pop("result_payload")
        record["result"] = self._codec.decrypt(payload, f"run:{run_id}") if payload else None
        error = record.pop("error_message")
        error_category = record.pop("error_category")
        record["error"] = (
            {
                "category": error_category,
                "message": self._codec.decrypt(error, f"run-error:{run_id}") if error else "",
            }
            if error or error_category
            else None
        )
        record["schema_version"] = SCHEMA_VERSION
        record["owner"] = {"instance_id": record.pop("owner_instance_id"), "type": record.pop("owner_type")}
        return record

    def mark_run_running(self, run_id: str, owner_instance_id: str | None = None, lease_seconds: int = 90):
        now = _now_us()
        with self._write() as connection:
            connection.execute(
                """UPDATE clink_runs SET status='running',started_at_us=COALESCE(started_at_us,?),
                   updated_at_us=?,lease_expires_at_us=?,owner_instance_id=COALESCE(?,owner_instance_id)
                   WHERE run_id=? AND status='queued'""",
                (now, now, now + int(lease_seconds) * 1_000_000, owner_instance_id, run_id),
            )
        return self.read_run(run_id)

    def heartbeat_run(self, run_id: str, lease_seconds: int = 90):
        now = _now_us()
        with self._write() as connection:
            connection.execute(
                """UPDATE clink_runs SET updated_at_us=?,lease_expires_at_us=?
                   WHERE run_id=? AND status IN ('queued','running')""",
                (now, now + int(lease_seconds) * 1_000_000, run_id),
            )
            connection.execute(
                """UPDATE conversation_exchanges SET updated_at_us=?,lease_expires_at_us=?
                   WHERE exchange_id=(SELECT exchange_id FROM clink_runs WHERE run_id=?)
                     AND status IN ('admitted','running')""",
                (now, now + int(lease_seconds) * 1_000_000, run_id),
            )
        return self.read_run(run_id)

    def find_run_by_exchange(self, exchange_id: str) -> dict[str, Any] | None:
        _canonical_uuid(exchange_id, "exchange_id")
        row = self.connection().execute("SELECT run_id FROM clink_runs WHERE exchange_id=?", (exchange_id,)).fetchone()
        return self.read_run(row["run_id"]) if row else None

    def set_run_attachment(self, run_id: str, attachment: str):
        if attachment not in {"attached", "detached"}:
            raise DurableStorageError("invalid run attachment")
        with self._write() as connection:
            connection.execute(
                """UPDATE clink_runs SET attachment=?,updated_at_us=?
                   WHERE run_id=? AND status NOT IN ('completed','failed','interrupted')""",
                (attachment, _now_us(), run_id),
            )
        return self.read_run(run_id)

    def _finalize_run_in_transaction(
        self,
        connection: sqlite3.Connection,
        *,
        now: int,
        run_id: str,
        status: str,
        result: str | None = None,
        error: dict[str, Any] | None = None,
        duration_seconds: float | None = None,
        output_truncated: bool = False,
    ) -> None:
        if status not in TERMINAL_RUN_STATUSES:
            raise DurableStorageError("invalid terminal run status")
        row = connection.execute("SELECT status FROM clink_runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None or row["status"] in TERMINAL_RUN_STATUSES:
            return
        result_payload = self._codec.encrypt(result, f"run:{run_id}") if result is not None else None
        error_payload = (
            self._codec.encrypt(str(error.get("message", ""))[:2_000], f"run-error:{run_id}") if error else None
        )
        connection.execute(
            """UPDATE clink_runs SET status=?,updated_at_us=?,finished_at_us=?,lease_expires_at_us=NULL,
               duration_seconds=?,result_payload=?,result_hmac=?,output_truncated=?,error_category=?,error_message=?
               WHERE run_id=? AND status NOT IN ('completed','failed','interrupted')""",
            (
                status,
                now,
                now,
                duration_seconds,
                result_payload,
                self._codec.keyed_hash(result) if result is not None else None,
                int(bool(output_truncated)),
                str(error.get("category", ""))[:120] if error else None,
                error_payload,
                run_id,
            ),
        )

    def finalize_run(self, run_id: str, **terminal: Any):
        _canonical_uuid(run_id, "run_id")
        with self._write() as connection:
            self._finalize_run_in_transaction(connection, now=_now_us(), run_id=run_id, **terminal)
        return self.read_run(run_id)

    def enqueue_run(self, run_id: str, envelope: dict[str, Any], available_at_us: int | None = None) -> None:
        now = _now_us()
        with self._write() as connection:
            connection.execute(
                """INSERT INTO clink_run_queue
                   (run_id,status,available_at_us,lease_expires_at_us,envelope_payload)
                   VALUES(?,'queued',?,?,?)""",
                (
                    run_id,
                    available_at_us or now,
                    now + 90_000_000,
                    self._codec.encrypt(envelope, f"queue:{run_id}"),
                ),
            )

    def create_queued_worker_run(
        self,
        *,
        run_id: str,
        continuation_id: str | None,
        exchange_id: str | None,
        cli_name: str,
        role: str | None,
        envelope: dict[str, Any],
    ) -> dict[str, Any]:
        """Atomically persist the worker-owned run and deterministic envelope."""
        _canonical_uuid(run_id, "run_id")
        now = _now_us()
        with self._write() as connection:
            connection.execute(
                """INSERT INTO clink_runs
                   (run_id,continuation_id,exchange_id,cli_name,role,status,owner_instance_id,owner_type,
                    created_at_us,updated_at_us,lease_expires_at_us)
                   VALUES(?,?,?,?,?,'queued',NULL,'worker',?,?,NULL)""",
                (run_id, continuation_id, exchange_id, cli_name, role, now, now),
            )
            connection.execute(
                """INSERT INTO clink_run_queue
                   (run_id,status,available_at_us,lease_expires_at_us,envelope_payload)
                   VALUES(?,'queued',?,?,?)""",
                (run_id, now, now + 90_000_000, self._codec.encrypt(envelope, f"queue:{run_id}")),
            )
        return self.read_run(run_id) or {}

    def claim_next_run(self, owner_instance_id: str, lease_seconds: int = 90) -> dict[str, Any] | None:
        now = _now_us()
        with self._write() as connection:
            row = connection.execute(
                """SELECT run_id,envelope_payload FROM clink_run_queue
                   WHERE available_at_us<=? AND status='queued' AND lease_expires_at_us>?
                   ORDER BY available_at_us,run_id LIMIT 1""",
                (now, now),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """UPDATE clink_run_queue SET status='claimed',claimed_by=?,lease_expires_at_us=?
                   WHERE run_id=?""",
                (owner_instance_id, now + int(lease_seconds) * 1_000_000, row["run_id"]),
            )
            connection.execute(
                "UPDATE clink_runs SET owner_instance_id=?,updated_at_us=? WHERE run_id=? AND status='queued'",
                (owner_instance_id, now, row["run_id"]),
            )
            return {
                "run_id": row["run_id"],
                "envelope": self._codec.decrypt(row["envelope_payload"], f"queue:{row['run_id']}"),
            }

    def interrupt_stale_claims(self) -> int:
        """Terminalize expired queued/claimed work without ever replaying it."""
        now = _now_us()
        with self._write() as connection:
            rows = connection.execute(
                """SELECT q.run_id,q.status AS queue_status,r.exchange_id,r.continuation_id
                   FROM clink_run_queue q JOIN clink_runs r ON r.run_id=q.run_id
                   WHERE q.status IN ('queued','claimed') AND q.lease_expires_at_us<=?
                     AND r.status IN ('queued','running')""",
                (now,),
            ).fetchall()
            for row in rows:
                connection.execute(
                    """UPDATE clink_runs SET status='interrupted',updated_at_us=?,finished_at_us=?,
                       lease_expires_at_us=NULL,error_category=?
                       WHERE run_id=? AND status IN ('queued','running')""",
                    (
                        now,
                        now,
                        "worker_queue_expired" if row["queue_status"] == "queued" else "worker_lease_expired",
                        row["run_id"],
                    ),
                )
                if row["exchange_id"]:
                    connection.execute(
                        """UPDATE conversation_exchanges SET status='interrupted',updated_at_us=?,lease_expires_at_us=NULL
                           WHERE exchange_id=? AND status IN ('admitted','running')""",
                        (now, row["exchange_id"]),
                    )
                    connection.execute(
                        "UPDATE conversation_threads SET active_exchange_id=NULL WHERE thread_id=? AND active_exchange_id=?",
                        (row["continuation_id"], row["exchange_id"]),
                    )
                connection.execute("DELETE FROM clink_run_queue WHERE run_id=?", (row["run_id"],))
        return len(rows)

    def heartbeat_queue_claim(self, run_id: str, owner_instance_id: str, lease_seconds: int = 90) -> bool:
        now = _now_us()
        with self._write() as connection:
            cursor = connection.execute(
                """UPDATE clink_run_queue SET lease_expires_at_us=?
                   WHERE run_id=? AND status='claimed' AND claimed_by=?""",
                (now + int(lease_seconds) * 1_000_000, run_id, owner_instance_id),
            )
        return cursor.rowcount == 1

    def delete_queue_input(self, run_id: str) -> None:
        with self._write() as connection:
            connection.execute("DELETE FROM clink_run_queue WHERE run_id=?", (run_id,))

    def cleanup(
        self,
        *,
        run_retention_seconds: int = 7 * 24 * 60 * 60,
        process_retention_seconds: int = 7 * 24 * 60 * 60,
        thread_batch_size: int = 100,
    ) -> int:
        """Bound stale state without touching any live exchange or run.

        Threads form a parent/child tree with ``ON DELETE RESTRICT``. Cleanup
        therefore removes expired leaves in bounded waves; an expired parent is
        collected in a later wave after its descendants disappear. Terminal run
        receipts are independent of thread lifetime and retain their own window.
        """
        now = _now_us()
        run_cutoff = now - max(0, int(run_retention_seconds)) * 1_000_000
        process_cutoff = now - max(0, int(process_retention_seconds)) * 1_000_000
        batch_size = max(1, min(int(thread_batch_size), 1_000))
        removed = 0
        with self._write() as connection:
            cursor = connection.execute(
                """DELETE FROM clink_runs
                   WHERE status IN ('completed','failed','interrupted')
                     AND COALESCE(finished_at_us,updated_at_us) < ?""",
                (run_cutoff,),
            )
            removed += cursor.rowcount
            cursor = connection.execute("DELETE FROM worker_capabilities WHERE expires_at_us < ?", (now,))
            removed += cursor.rowcount
            cursor = connection.execute(
                """DELETE FROM process_instances
                   WHERE COALESCE(stopped_at_us,heartbeat_at_us) < ?""",
                (process_cutoff,),
            )
            removed += cursor.rowcount
            expired_leaves = connection.execute(
                """SELECT t.thread_id FROM conversation_threads t
                   WHERE t.expires_at_us < ? AND t.active_exchange_id IS NULL
                     AND NOT EXISTS (
                       SELECT 1 FROM conversation_exchanges e
                       WHERE e.thread_id=t.thread_id AND e.status IN ('admitted','running')
                     )
                     AND NOT EXISTS (
                       SELECT 1 FROM conversation_threads child
                       WHERE child.parent_thread_id=t.thread_id
                     )
                   ORDER BY t.expires_at_us,t.thread_id LIMIT ?""",
                (now, batch_size),
            ).fetchall()
            for row in expired_leaves:
                cursor = connection.execute(
                    """DELETE FROM conversation_threads
                       WHERE thread_id=? AND expires_at_us < ? AND active_exchange_id IS NULL""",
                    (row["thread_id"], now),
                )
                removed += cursor.rowcount
        # Reclaim a bounded number of free pages without a stop-the-world VACUUM.
        self.connection().execute("PRAGMA incremental_vacuum(256)")
        return removed

    def health(self) -> dict[str, Any]:
        connection = self.connection()
        counts = {}
        for table in ("conversation_threads", "conversation_turns", "conversation_exchanges", "clink_runs"):
            counts[table] = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return {
            "backend": "sqlite",
            "schema_version": SCHEMA_VERSION,
            "journal_mode": connection.execute("PRAGMA journal_mode").fetchone()[0],
            "synchronous": connection.execute("PRAGMA synchronous").fetchone()[0],
            "encryption": "AES-256-GCM",
            "key_id": self.key_id,
            "quick_check": connection.execute("PRAGMA quick_check").fetchone()[0],
            "counts": counts,
        }


_default_lock = threading.Lock()
_default_storage: SQLiteConversationStorage | None = None


def get_default_storage() -> SQLiteConversationStorage:
    global _default_storage
    if _default_storage is None:
        with _default_lock:
            if _default_storage is None:
                state_dir = Path(os.environ.get("PAL_STATE_DIR", str(Path.home() / ".pal" / "state"))).expanduser()
                key_file = Path(
                    os.environ.get("PAL_STATE_KEY_FILE", str(Path.home() / ".pal" / "keys" / "state.key"))
                ).expanduser()
                _default_storage = SQLiteConversationStorage(state_dir, key_file)
    return _default_storage


def reset_default_storage_for_tests() -> None:
    global _default_storage
    with _default_lock:
        connection = getattr(getattr(_default_storage, "_local", None), "connection", None)
        if connection is not None:
            connection.close()
        _default_storage = None
