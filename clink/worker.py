"""Independently supervised durable queue worker for PAL clink.

The worker is intentionally inert until an operator or service manager starts
``python -m clink.worker serve``. It never retries a claimed model execution:
if its lease dies, recovery marks the run interrupted and requires an explicit
caller retry, preventing accidental duplicate token spend.
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import time
from typing import Any

from clink import jobs
from clink.agents import create_agent
from clink.policy import PartnerModelPolicyError, attest_client
from clink.partner_boundary import PartnerBoundaryAuthority
from clink.registry import ClinkRegistry
from utils.conversation_memory import current_exchange_id, fail_exchange
from utils.sqlite_conversation_storage import get_default_storage

CAPABILITY_TTL_SECONDS = 75
CAPABILITY_REFRESH_SECONDS = 25
QUEUE_POLL_SECONDS = 0.25
CLEANUP_INTERVAL_SECONDS = 60 * 60


class ClinkWorker:
    def __init__(self):
        self.instance_id = jobs.INSTANCE_ID
        self.store = get_default_storage()
        self.registry = ClinkRegistry()
        self.partner_boundary_authority = PartnerBoundaryAuthority(self.instance_id)
        self._last_capability_refresh = 0.0
        self._last_cleanup = 0.0

    def _publish_capabilities(self) -> None:
        for cli_name in self.registry.list_clients():
            client = self.registry.get_client(cli_name)
            for role_name in client.list_roles():
                role = client.get_role(role_name)
                try:
                    capability = attest_client(client, role)
                except (PartnerModelPolicyError, OSError):
                    continue
                self.store.publish_capability(
                    cli_name=capability.cli_name,
                    role=role.name,
                    config_digest=capability.config_digest,
                    executable_identity=capability.executable_identity,
                    model=capability.model,
                    reasoning_effort=capability.reasoning_effort,
                    owner_instance_id=self.instance_id,
                    owner_mode="clink_worker",
                    ttl_seconds=CAPABILITY_TTL_SECONDS,
                )
        self._last_capability_refresh = time.monotonic()

    async def _execute(self, claim: dict[str, Any]) -> None:
        from tools.clink import CLinkRequest, CLinkTool

        run_id = claim["run_id"]
        envelope = claim["envelope"]
        exchange_id = envelope.get("exchange_id")
        token = current_exchange_id.set(exchange_id)
        owner_task = asyncio.current_task()
        heartbeat = asyncio.create_task(self._heartbeat_claim(run_id, owner_task))
        try:
            client = self.registry.get_client(envelope["cli_name"])
            role = client.get_role(envelope.get("role"))
            capability = attest_client(client, role)
            if capability.config_digest != envelope["capability_digest"]:
                raise RuntimeError("queued clink capability no longer matches the worker")
            request_payload = dict(envelope["request"])
            if envelope.get("continuation_id"):
                request_payload["continuation_id"] = envelope["continuation_id"]
            request = CLinkRequest.model_validate(request_payload)
            jobs.mark_running(run_id)
            tool = CLinkTool()
            await tool._run_pipeline(
                agent=create_agent(client, partner_boundary_authority=self.partner_boundary_authority),
                client_config=client,
                role_config=role,
                request=request,
                prompt_text=envelope["prompt_text"],
                system_prompt_text=envelope.get("system_prompt_text") or "",
                absolute_file_paths=list(envelope.get("absolute_file_paths") or []),
                images=list(envelope.get("images") or []),
                on_event=None,
                thread_id=envelope.get("thread_id"),
                run_id=run_id,
                continuation_id=envelope.get("continuation_id"),
                durable=True,
                capability_digest=capability.config_digest,
                exchange_id=exchange_id,
            )
        except asyncio.CancelledError:
            fail_exchange(exchange_id, "worker_interrupted")
            jobs.finalize(
                run_id,
                status=jobs.STATUS_INTERRUPTED,
                error={"category": "worker_interrupted", "message": "worker stopped during execution"},
            )
            raise
        except BaseException as exc:
            fail_exchange(exchange_id, "worker_failed")
            jobs.finalize(
                run_id,
                status=jobs.STATUS_FAILED,
                error={"category": "worker_failed", "message": str(exc)},
            )
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            self.store.delete_queue_input(run_id)
            current_exchange_id.reset(token)

    async def _heartbeat_claim(self, run_id: str, owner_task: asyncio.Task | None) -> None:
        while True:
            await asyncio.sleep(CAPABILITY_REFRESH_SECONDS)
            if not self.store.heartbeat_queue_claim(run_id, self.instance_id):
                if owner_task is not None:
                    owner_task.cancel()
                return
            self.store.heartbeat_process(self.instance_id)
            self.store.heartbeat_queued_runs(self.instance_id)
            self._publish_capabilities()

    async def serve(self) -> None:
        loop = asyncio.get_running_loop()
        stop_requested = asyncio.Event()
        installed_signals = []
        for signum in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(signum, stop_requested.set)
                installed_signals.append(signum)
            except (NotImplementedError, RuntimeError):  # pragma: no cover - non-POSIX fallback
                pass
        self.store.register_process(self.instance_id, "clink_worker")
        await self.partner_boundary_authority.start()
        try:
            self.store.interrupt_stale_claims()
            self.store.cleanup()
            self._last_cleanup = time.monotonic()
            self._publish_capabilities()
            while not stop_requested.is_set():
                if time.monotonic() - self._last_capability_refresh >= CAPABILITY_REFRESH_SECONDS:
                    self.store.heartbeat_process(self.instance_id)
                    self.store.heartbeat_queued_runs(self.instance_id)
                    self.store.interrupt_stale_claims()
                    self._publish_capabilities()
                if time.monotonic() - self._last_cleanup >= CLEANUP_INTERVAL_SECONDS:
                    self.store.cleanup()
                    self._last_cleanup = time.monotonic()
                claim = self.store.claim_next_run(self.instance_id)
                if claim is None:
                    try:
                        await asyncio.wait_for(stop_requested.wait(), timeout=QUEUE_POLL_SECONDS)
                    except asyncio.TimeoutError:
                        pass
                    continue
                execution = asyncio.create_task(self._execute(claim))
                stopping = asyncio.create_task(stop_requested.wait())
                done, _ = await asyncio.wait({execution, stopping}, return_when=asyncio.FIRST_COMPLETED)
                if stopping in done:
                    execution.cancel()
                await asyncio.gather(execution, return_exceptions=True)
                stopping.cancel()
                await asyncio.gather(stopping, return_exceptions=True)
        finally:
            await self.partner_boundary_authority.stop()
            self.store.stop_process(self.instance_id)
            for signum in installed_signals:
                loop.remove_signal_handler(signum)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PAL durable clink worker")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("serve")
    args = parser.parse_args(argv)
    if args.command == "serve":
        asyncio.run(ClinkWorker().serve())
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
