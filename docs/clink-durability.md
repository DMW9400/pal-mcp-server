# Clink durability contract

PAL stores continuations and clink run records in a local SQLite WAL database
under `PAL_STATE_DIR` (default `~/.pal/state`). User prompts, responses, paths,
initial context, model metadata, queued execution envelopes and results are
encrypted with AES-256-GCM using `PAL_STATE_KEY_FILE` (default
`~/.pal/keys/state.key`). The key file is created as mode 0600 and is required
to restore the database.

## Guarantees

- Completed threads and terminal run results survive MCP client detachment and
  PAL process restart until their retention/TTL boundary.
- Each continuation permits one admitted/running exchange. Admission, the user
  turn, idempotency key and assistant capacity reservation are one SQLite
  transaction.
- A successful clink assistant turn and terminal run result commit together.
- The launchd-owned worker executes queued calls independently of PAL, so a PAL
  restart does not terminate active clink work.
- A claimed call is never automatically executed again. Worker loss marks it
  interrupted after its lease expires; retry requires an explicit caller action.
- Queue entries are assigned to one exact worker instance. A healthy worker
  renews its backlog leases; entries left behind by a stopped worker expire
  after 90 seconds and are terminalized rather than transferred or replayed.
- Idempotency keys bind initial calls to one thread and every call to one
  exchange/run; retries return or follow that original run without launching.
- Claude collaboration is fixed to Fable 5 (`fable`) at `xhigh`. Codex is fixed
  to `gpt-5.6-sol` at `high`. Config, role arguments, executable identity and
  observed CLI metadata are validated fail-closed. Protected clients must use
  the canonical `claude`/`codex` executable, never a configured wrapper, and
  production calls require the independently supervised worker for both
  foreground and background execution.

Idempotency keys are request-bound within this local PAL trust boundary. A key
reused with different semantic request content is rejected; an identical retry
returns the original thread/exchange/run. Callers should still namespace keys
when multiple unrelated local clients share one PAL state directory.

SQLite protects PAL state, not arbitrary effects performed by an external CLI.
The contract is exactly-once state admission/terminalization, not exactly-once
filesystem or network effects inside Claude or Codex.

## Worker service

Install or refresh the per-user launchd service:

```bash
scripts/clink-worker-service.sh install
```

Inspect it with `scripts/clink-worker-service.sh status`. The worker is idle and
uses no model tokens until PAL queues a clink call. Capability heartbeats and
polling are local SQLite operations. Background calls fail before inserting a
turn unless a fresh matching worker capability exists.

## Recovery and backup

Back up the database files and key together while PAL and the worker are
stopped. Losing the key intentionally makes encrypted content unrecoverable.
Never copy an active WAL database without its `-wal` and `-shm` companions; use
SQLite's backup API for online backups. A migration hash mismatch, failed
authenticated decrypt, unsafe key permissions, symlinked state path or failed
`quick_check` prevents startup rather than silently discarding state.

Legacy JSON run sidecars remain read-only compatibility inputs. New production
runs use encrypted SQLite and do not write plaintext sidecars.
