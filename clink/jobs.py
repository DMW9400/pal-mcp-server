"""Durable records and lifecycle helpers for clink runs.

Production runs live in encrypted SQLite and protected Claude/Codex calls execute in
the independently supervised worker, so both active work and terminal results survive
PAL client/process detach. Terminal writes are immutable and a stale owner is marked
interrupted rather than replayed automatically. Legacy JSON sidecars remain a
read-only compatibility path when the explicit memory backend is selected in tests.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clink.constants import PROJECT_ROOT

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_INTERRUPTED = "interrupted"
TERMINAL_STATUSES = frozenset({STATUS_COMPLETED, STATUS_FAILED, STATUS_INTERRUPTED})

ATTACHMENT_ATTACHED = "attached"
ATTACHMENT_DETACHED = "detached"

MAX_ERROR_MESSAGE_CHARS = 2000
RETENTION_SECONDS = 7 * 24 * 60 * 60
SWEEP_INTERVAL_SECONDS = 60 * 60
#: Cadence at which a live run refreshes ``updated_at``.
HEARTBEAT_INTERVAL_SECONDS = 30.0
#: A non-terminal record untouched for longer than this is reported as interrupted.
STALE_AFTER_SECONDS = 90.0

INSTANCE_ID = str(uuid.uuid4())

_STORE_ROOT: Path | None = None
_STORE_LOCK = threading.Lock()
_CLAIMS: dict[str, str] = {}
_CLAIM_LOCK = threading.Lock()
_last_sweep_monotonic: float | None = None


class JobStoreError(RuntimeError):
    """Raised for malformed run identifiers or unusable records."""


def _sqlite_store():
    """Production uses encrypted SQLite; tests may explicitly select memory/sidecars."""
    if os.environ.get("PAL_CONVERSATION_BACKEND", "sqlite").strip().lower() == "memory":
        return None
    from utils.sqlite_conversation_storage import get_default_storage

    return get_default_storage()


def default_store_root() -> Path:
    """Job records live beside the server logs (same directory server.py creates)."""
    return PROJECT_ROOT / "logs" / "clink_results"


def set_store_root(path: Path | str | None) -> None:
    """Point the store at a different directory (used by tests)."""
    global _STORE_ROOT
    _STORE_ROOT = Path(path) if path is not None else None


def store_root() -> Path:
    root = _STORE_ROOT or default_store_root()
    root.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(root, 0o700)
    except OSError:  # pragma: no cover - best effort on exotic filesystems
        pass
    return root


def new_run_id() -> str:
    return str(uuid.uuid4())


def validate_run_id(run_id: Any) -> str:
    """Accept only a canonical UUID string; anything path-like fails closed."""
    if not isinstance(run_id, str):
        raise JobStoreError("run_id must be a string")
    candidate = run_id.strip()
    try:
        parsed = uuid.UUID(candidate)
    except (ValueError, AttributeError, TypeError) as exc:
        raise JobStoreError(f"Invalid run_id '{run_id}': expected a UUID") from exc
    if str(parsed) != candidate:
        raise JobStoreError(f"Invalid run_id '{run_id}': expected a canonical UUID")
    return candidate


def _record_path(run_id: str) -> Path:
    return store_root() / f"{validate_run_id(run_id)}.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bounded(message: Any) -> str:
    text = "" if message is None else str(message)
    if len(text) > MAX_ERROR_MESSAGE_CHARS:
        return text[:MAX_ERROR_MESSAGE_CHARS]
    return text


def _atomic_write(path: Path, record: dict[str, Any]) -> None:
    payload = json.dumps(record, ensure_ascii=False, indent=2)
    directory = path.parent
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".tmp", dir=str(directory))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:  # pragma: no cover - defensive
            pass
        raise


def _read_unlocked(run_id: str) -> dict[str, Any] | None:
    path = _record_path(run_id)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        logger.warning("Failed to read clink job record %s", run_id, exc_info=True)
        return None
    try:
        record = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Ignoring corrupt clink job record %s", run_id)
        return None
    if not isinstance(record, dict) or record.get("schema_version") != SCHEMA_VERSION:
        logger.warning("Ignoring clink job record %s with unsupported schema", run_id)
        return None
    return record


def read(run_id: str) -> dict[str, Any] | None:
    """Return the stored record, or None when unknown/corrupt/expired."""
    store = _sqlite_store()
    if store is not None:
        record = store.read_run(validate_run_id(run_id))
        if record is not None:
            record.pop("result_hmac", None)
            result = record.get("result")
            record["output_sha256"] = hashlib.sha256(result.encode("utf-8")).hexdigest() if result else None
            return record
        # Read-only compatibility for terminal records made before cutover.
    with _STORE_LOCK:
        return _read_unlocked(run_id)


def is_terminal(record: dict[str, Any] | None) -> bool:
    return bool(record) and record.get("status") in TERMINAL_STATUSES


def staleness_seconds(record: dict[str, Any] | None) -> float | None:
    """Seconds since a non-terminal record was last touched, or None when N/A."""
    if not record or is_terminal(record):
        return None
    updated_at = record.get("updated_at")
    if not updated_at:
        return float("inf")
    try:
        stamp = datetime.fromisoformat(updated_at)
    except ValueError:
        return float("inf")
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc).timestamp() - stamp.timestamp()


def is_stale(record: dict[str, Any] | None) -> bool:
    """True when a non-terminal record has not been touched recently."""
    elapsed = staleness_seconds(record)
    return elapsed is not None and elapsed > STALE_AFTER_SECONDS


def create(
    *,
    run_id: str,
    continuation_id: str | None,
    cli_name: str,
    role: str | None,
    owner_type: str = "pal",
) -> dict[str, Any]:
    """Persist a ``queued`` record. Also runs the opportunistic retention sweep."""
    run_id = validate_run_id(run_id)
    store = _sqlite_store()
    if store is not None:
        from utils.conversation_memory import current_exchange_id

        store.register_process(INSTANCE_ID, "pal")
        return store.create_run(
            run_id=run_id,
            continuation_id=continuation_id,
            exchange_id=current_exchange_id.get(),
            cli_name=cli_name,
            role=role,
            owner_instance_id=INSTANCE_ID,
            owner_type=owner_type,
        )
    _maybe_sweep()
    now = _now()
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "continuation_id": continuation_id,
        "cli_name": cli_name,
        "role": role,
        "status": STATUS_QUEUED,
        "created_at": now,
        "started_at": None,
        "updated_at": now,
        "finished_at": None,
        "duration_seconds": None,
        "owner": {"pid": os.getpid(), "instance_id": INSTANCE_ID},
        "attachment": ATTACHMENT_ATTACHED,
        "error": None,
        "result": None,
        "output_sha256": None,
        "output_truncated": False,
    }
    with _STORE_LOCK:
        _atomic_write(_record_path(run_id), record)
    return record


def _update(run_id: str, mutate) -> dict[str, Any] | None:
    with _STORE_LOCK:
        record = _read_unlocked(run_id)
        if record is None:
            return None
        updated = mutate(record)
        if updated is None:
            return record
        updated["updated_at"] = _now()
        _atomic_write(_record_path(run_id), updated)
        return updated


def mark_running(run_id: str) -> dict[str, Any] | None:
    store = _sqlite_store()
    if store is not None:
        return store.mark_run_running(validate_run_id(run_id), INSTANCE_ID)

    def mutate(record: dict[str, Any]) -> dict[str, Any] | None:
        if record.get("status") in TERMINAL_STATUSES:
            return None
        record["status"] = STATUS_RUNNING
        record["started_at"] = record.get("started_at") or _now()
        return record

    return _update(run_id, mutate)


def touch(run_id: str) -> dict[str, Any] | None:
    """Refresh ``updated_at`` so readers do not classify a live run as interrupted."""

    store = _sqlite_store()
    if store is not None:
        store.heartbeat_process(INSTANCE_ID)
        return store.heartbeat_run(validate_run_id(run_id))

    def mutate(record: dict[str, Any]) -> dict[str, Any] | None:
        if record.get("status") in TERMINAL_STATUSES:
            return None
        return record

    return _update(run_id, mutate)


def set_attachment(run_id: str, attachment: str) -> dict[str, Any] | None:
    """Client cancellation detaches the request; it never changes ``status``.

    A cancellation that races completion must not rewrite the terminal record.
    """

    store = _sqlite_store()
    if store is not None:
        return store.set_run_attachment(validate_run_id(run_id), attachment)

    def mutate(record: dict[str, Any]) -> dict[str, Any] | None:
        if record.get("status") in TERMINAL_STATUSES:
            return None
        record["attachment"] = attachment
        return record

    return _update(run_id, mutate)


def finalize(
    run_id: str,
    *,
    status: str,
    result: str | None = None,
    error: dict[str, Any] | None = None,
    duration_seconds: float | None = None,
    output_truncated: bool = False,
) -> dict[str, Any] | None:
    """Write the terminal record. The first terminal write wins and is immutable."""
    if status not in TERMINAL_STATUSES:
        raise JobStoreError(f"'{status}' is not a terminal clink job status")

    store = _sqlite_store()
    if store is not None:
        record = store.finalize_run(
            validate_run_id(run_id),
            status=status,
            result=result,
            error=error,
            duration_seconds=duration_seconds,
            output_truncated=output_truncated,
        )
        if record is not None:
            record.pop("result_hmac", None)
            result_value = record.get("result")
            record["output_sha256"] = hashlib.sha256(result_value.encode("utf-8")).hexdigest() if result_value else None
        return record

    with _STORE_LOCK:
        record = _read_unlocked(run_id)
        if record is None:
            return None
        if record.get("status") in TERMINAL_STATUSES:
            return record
        now = _now()
        record["status"] = status
        record["finished_at"] = now
        record["updated_at"] = now
        record["duration_seconds"] = round(duration_seconds, 3) if duration_seconds is not None else None
        record["result"] = result
        record["output_sha256"] = hashlib.sha256(result.encode("utf-8")).hexdigest() if result else None
        record["output_truncated"] = bool(output_truncated)
        if error:
            record["error"] = {
                "category": _bounded(error.get("category")),
                "message": _bounded(error.get("message")),
            }
        _atomic_write(_record_path(run_id), record)
        return record


def sweep() -> int:
    """Delete terminal records older than the retention window. Never touches live runs."""
    global _last_sweep_monotonic
    sqlite_store = _sqlite_store()
    if sqlite_store is not None:
        removed = sqlite_store.cleanup(run_retention_seconds=RETENTION_SECONDS)
        _last_sweep_monotonic = time.monotonic()
        return removed
    removed = 0
    cutoff = datetime.now(timezone.utc).timestamp() - RETENTION_SECONDS
    try:
        entries = list(store_root().glob("*.json"))
    except OSError:  # pragma: no cover - defensive
        return 0
    for path in entries:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict) or record.get("status") not in TERMINAL_STATUSES:
            continue
        finished_at = record.get("finished_at") or record.get("updated_at")
        try:
            stamp = datetime.fromisoformat(finished_at)
        except (TypeError, ValueError):
            continue
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        if stamp.timestamp() >= cutoff:
            continue
        try:
            path.unlink()
            removed += 1
        except OSError:  # pragma: no cover - best effort
            continue
    _last_sweep_monotonic = time.monotonic()
    return removed


def interrupt_stale_worker_queue() -> int:
    """Terminalize expired worker queue entries and stale PAL-owned runs."""
    store = _sqlite_store()
    if store is None:
        return 0
    return store.interrupt_stale_claims() + store.interrupt_stale_pal_runs()


def _maybe_sweep() -> None:
    global _last_sweep_monotonic
    if _last_sweep_monotonic is not None and (time.monotonic() - _last_sweep_monotonic) < SWEEP_INTERVAL_SECONDS:
        return
    _last_sweep_monotonic = time.monotonic()
    try:
        sweep()
    except Exception:  # pragma: no cover - retention must never break a run
        logger.debug("clink job retention sweep failed", exc_info=True)


def claim(continuation_id: str, run_id: str) -> str:
    """Claim a continuation for ``run_id``; returns the already-active run on contention.

    Conversations are process-local, so an in-process claim is sufficient; the sidecar
    status provides cross-process advisory visibility only.
    """
    if _sqlite_store() is not None:
        from utils.conversation_memory import current_exchange_id

        if current_exchange_id.get():
            return run_id
    with _CLAIM_LOCK:
        active = _CLAIMS.get(continuation_id)
        if active is not None and active != run_id:
            return active
        _CLAIMS[continuation_id] = run_id
        return run_id


def release(continuation_id: str | None, run_id: str) -> None:
    if _sqlite_store() is not None:
        from utils.conversation_memory import current_exchange_id

        if current_exchange_id.get():
            return
    if not continuation_id:
        return
    with _CLAIM_LOCK:
        if _CLAIMS.get(continuation_id) == run_id:
            _CLAIMS.pop(continuation_id, None)
