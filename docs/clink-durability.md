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
- Claude's directly addressed partner is fixed to Fable 5 (`fable`) at `xhigh`.
  Codex's directly addressed partner is fixed to `gpt-5.6-sol` at `high`.
  Config, role arguments, executable identity and worker capability are attested
  before admission. Once that direct partner is admitted, the run is greenlit:
  terminal and nested-agent model/effort telemetry is observe-only and can never
  invalidate, cancel, retry, or block it.
- The durable worker alone mints and activates a short-lived in-memory boundary
  bound to the direct Claude process's already-attested resolved executable,
  exact PID, process-start identity, command hash, random nonce, ancestry and
  expiry. The configured command remains the canonical `claude` name; executable
  attestation resolves its versioned target, and boundary activation requires
  the launched process's argv[0] to equal that exact target. It never infers
  model authority from a basename or symlink spelling. Project hooks can only query the
  worker's private verification socket; the client derives the server PID from
  kernel peer credentials and requires the exact launchd-registered worker for
  this checkout before standing down for Claude-owned nested routing. Importing
  PAL code or hosting another same-user authority cannot self-admit. PAL's
  Opus/high preference for substantive nested work is advisory and is excluded
  from capability digests, worker readiness, admission and result validity.
- Protected clients must use the canonical `claude`/`codex` executable, never a
  configured wrapper or direct-model fallback. Production calls require the
  independently supervised worker for foreground and background execution.

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

Inspect it with `scripts/clink-worker-service.sh status`; status validates every
protected role against one exact worker instance. The worker is idle and
uses no model tokens until PAL queues a clink call. Capability heartbeats and
polling are local SQLite operations. Background calls fail before inserting a
turn unless a fresh matching worker capability exists.

Submit long work with `background: true` and a stable request-scoped idempotency
key. Capture `run_id` and `continuation_id`, then poll the same run. A request
timeout renews observation by another poll; it never authorizes a duplicate
submission. If the MCP transport closes, the launchd worker continues and the
durable run/result remains recoverable after the desktop client opens a fresh
transport. Never kill an MCP child or worker as a retry mechanism.

If a direct launch is rejected before admission, first run `status` and
`python -m clink.readiness`. Refresh only with `scripts/clink-worker-service.sh
install` when readiness or worker digests are stale. If readiness is exact but
activation still fails, compare the canonical executable's resolved target with
the launch receipt and run `tests/test_clink_partner_boundary.py`; do not add a
wrapper, alias, fallback model, or post-run telemetry gate. The boundary's
resolved-version-path regression is the owner for symlinked Claude installs.

## Recovery and backup

Back up the database files and key together while PAL and the worker are
stopped. Losing the key intentionally makes encrypted content unrecoverable.
Never copy an active WAL database without its `-wal` and `-shm` companions; use
SQLite's backup API for online backups. A migration hash mismatch, failed
authenticated decrypt, unsafe key permissions, symlinked state path or failed
`quick_check` prevents startup rather than silently discarding state.

Legacy JSON run sidecars remain read-only compatibility inputs. New production
runs use encrypted SQLite and do not write plaintext sidecars.
