"""Who may call a shared sdgf server, and what each caller may spend.

A clients file lists each client with the sha256 of its bearer token (never the token
itself) and its limits:

    {"clients": [
      {"id": "team-a", "token_sha256": "9f86d0...", "max_running": 2,
       "max_tokens": 5000000, "max_cost_usd": 50, "allow_hooks": false}
    ]}

    python -m sdgf.clients new team-a     prints a new token, once, and its file entry

A missing limit means no limit beyond the server's own. Over stdio, or HTTP without a
clients file, every call comes from the single LOCAL client under the server's limits.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import secrets
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sdgf.service import ServiceError
from sdgf.service.tasks import NAME_RE

LOCAL = "local"
_FIELDS = {"id", "token_sha256", "max_running", "max_tokens", "max_cost_usd", "allow_hooks"}


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_token() -> str:
    return "sdgf_" + secrets.token_urlsafe(32)


@dataclass(frozen=True)
class ClientPolicy:
    id: str
    token_sha256: str | None = None
    max_running: int | None = None
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    allow_hooks: bool | None = None  # None: the server's setting

    def limits(self) -> dict[str, Any]:
        return {
            "max_running": self.max_running,
            "max_tokens": self.max_tokens,
            "max_cost_usd": self.max_cost_usd,
        }


class ClientRegistry:
    def __init__(self, policies: Sequence[ClientPolicy]):
        ids = [p.id for p in policies]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ServiceError(f"duplicate client ids {dupes}", code="bad_clients_file")
        self.policies = {p.id: p for p in policies}

    @classmethod
    def load(cls, path: str | Path) -> ClientRegistry:
        try:
            entries = json.loads(Path(path).read_text(encoding="utf-8"))["clients"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as e:
            raise ServiceError(f"{path}: not a clients file: {e}", code="bad_clients_file") from e
        policies = []
        for entry in entries:
            unknown = sorted(set(entry) - _FIELDS)
            if unknown or not NAME_RE.fullmatch(str(entry.get("id", ""))):
                raise ServiceError(
                    f"{path}: bad client entry {entry.get('id')!r} (unknown keys {unknown})",
                    code="bad_clients_file",
                )
            if entry["id"] == LOCAL or not entry.get("token_sha256"):
                raise ServiceError(
                    f"{path}: client {entry['id']!r} needs a token_sha256 and another id "
                    f"than {LOCAL!r}",
                    code="bad_clients_file",
                )
            policies.append(ClientPolicy(**entry))
        return cls(policies)

    def verify(self, token: str) -> ClientPolicy | None:
        digest = token_digest(token)
        found = None
        for policy in self.policies.values():  # compare every entry, in constant time
            if hmac.compare_digest(digest, policy.token_sha256 or ""):
                found = policy
        return found

    def get(self, client_id: str) -> ClientPolicy | None:
        return self.policies.get(client_id)


class StaticTokenVerifier:
    """The MCP SDK's TokenVerifier over a ClientRegistry: a known token is its client."""

    def __init__(self, registry: ClientRegistry):
        self.registry = registry

    async def verify_token(self, token: str) -> Any:
        from mcp.server.auth.provider import AccessToken

        policy = self.registry.verify(token)
        if policy is None:
            return None
        return AccessToken(token=token, client_id=policy.id, scopes=[])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sdgf.clients")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("new", help="make a token for a client and print its file entry")
    p.add_argument("id")
    args = parser.parse_args(argv)
    if not NAME_RE.fullmatch(args.id) or args.id == LOCAL:
        print(f"sdgf: error: bad client id {args.id!r}", file=sys.stderr)
        return 2
    token = new_token()
    entry = {"id": args.id, "token_sha256": token_digest(token)}
    print(json.dumps({"token": token, "entry": entry}, indent=2))
    print("sdgf: give the token to the client; only its sha256 goes in the file", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
