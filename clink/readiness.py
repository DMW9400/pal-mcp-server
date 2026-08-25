"""Exact protected-role readiness for the independently supervised clink worker."""

from __future__ import annotations

import argparse
import json
from typing import Any

from clink.policy import PartnerModelPolicyError, attest_client, get_policy
from clink.registry import ClinkRegistry
from utils.sqlite_conversation_storage import get_default_storage


def collect_worker_readiness(registry, storage) -> dict[str, Any]:
    roles: list[dict[str, Any]] = []
    owner_ids: set[str] = set()
    for cli_name in registry.list_clients():
        client = registry.get_client(cli_name)
        if get_policy(client.name) is None:
            continue
        for role_name in client.list_roles():
            role = client.get_role(role_name)
            try:
                expected = attest_client(client, role)
            except (PartnerModelPolicyError, OSError) as exc:
                roles.append(
                    {
                        "cli_name": client.name,
                        "role": role_name,
                        "ok": False,
                        "mismatches": ["local_attestation_failed"],
                        "error": str(exc),
                    }
                )
                continue
            worker = storage.get_fresh_worker_capability(expected.cli_name, role_name)
            mismatches: list[str] = []
            if worker is None:
                mismatches.append("missing_or_stale")
            else:
                comparisons = {
                    "config_digest": expected.config_digest,
                    "executable_identity": expected.executable_identity,
                    "model": expected.model,
                    "reasoning_effort": expected.reasoning_effort,
                }
                mismatches.extend(key for key, value in comparisons.items() if worker.get(key) != value)
                if worker.get("owner_mode") != "clink_worker":
                    mismatches.append("owner_mode")
                owner_id = worker.get("owner_instance_id")
                if owner_id:
                    owner_ids.add(owner_id)
            roles.append(
                {
                    "cli_name": expected.cli_name,
                    "role": role_name,
                    "ok": not mismatches,
                    "mismatches": mismatches,
                    "expected_digest": expected.config_digest,
                    "worker_digest": worker.get("config_digest") if worker else None,
                    "owner_instance_id": worker.get("owner_instance_id") if worker else None,
                }
            )
    if not roles:
        return {"schema_version": 1, "ok": False, "owner_instance_id": None, "roles": [], "error": "no protected clients"}
    single_owner = len(owner_ids) == 1
    return {
        "schema_version": 1,
        "ok": all(item["ok"] for item in roles) and single_owner,
        "owner_instance_id": next(iter(owner_ids)) if single_owner else None,
        "roles": roles,
        **({"error": "protected roles are not published by one worker instance"} if not single_owner else {}),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check every protected clink role against the current worker")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    result = collect_worker_readiness(ClinkRegistry(), get_default_storage())
    if not args.quiet:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
