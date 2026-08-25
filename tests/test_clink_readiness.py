from types import SimpleNamespace

from clink import readiness


class FakeClient:
    name = "claude"

    def list_roles(self):
        return ["default", "sparring"]

    def get_role(self, name):
        return SimpleNamespace(name=name)


class FakeRegistry:
    def list_clients(self):
        return ["claude"]

    def get_client(self, _name):
        return FakeClient()


class FakeStorage:
    def __init__(self, records):
        self.records = records

    def get_fresh_worker_capability(self, cli_name, role):
        return self.records.get((cli_name, role))


def capability(role):
    return SimpleNamespace(
        cli_name="claude",
        config_digest=f"digest-{role}",
        executable_identity="exec-1",
        model="fable",
        reasoning_effort="xhigh",
    )


def worker(role, owner="worker-1"):
    return {
        "config_digest": f"digest-{role}",
        "executable_identity": "exec-1",
        "model": "fable",
        "reasoning_effort": "xhigh",
        "owner_mode": "clink_worker",
        "owner_instance_id": owner,
    }


def test_readiness_requires_every_protected_role(monkeypatch):
    monkeypatch.setattr(readiness, "attest_client", lambda _client, role: capability(role.name))
    result = readiness.collect_worker_readiness(
        FakeRegistry(),
        FakeStorage({("claude", "default"): worker("default")}),
    )

    assert result["ok"] is False
    assert result["roles"][0]["ok"] is True
    assert result["roles"][1]["mismatches"] == ["missing_or_stale"]


def test_readiness_requires_exact_capabilities_from_one_worker(monkeypatch):
    monkeypatch.setattr(readiness, "attest_client", lambda _client, role: capability(role.name))
    records = {
        ("claude", "default"): worker("default"),
        ("claude", "sparring"): worker("sparring"),
    }

    passing = readiness.collect_worker_readiness(FakeRegistry(), FakeStorage(records))
    assert passing["ok"] is True
    assert passing["owner_instance_id"] == "worker-1"

    records[("claude", "sparring")]["reasoning_effort"] = "low"
    failing = readiness.collect_worker_readiness(FakeRegistry(), FakeStorage(records))
    assert failing["ok"] is False
    assert failing["roles"][1]["mismatches"] == ["reasoning_effort"]
