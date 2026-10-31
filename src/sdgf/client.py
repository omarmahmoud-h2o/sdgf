"""Call a remote sdgf server from Python: a harness's side of `python -m sdgf.mcp --http`.

    async with SdgfClient("https://sdgf.example/mcp", token=TOKEN) as sdgf:
        job = await sdgf.call("start_run", task="fag", target_size=200)
        done = await sdgf.wait(job["job_id"])

    status = call("https://sdgf.example/mcp", "my_usage", token=TOKEN)   # one call, blocking

The tools and their arguments are the MCP server's (sdgf.mcp.server). A tool's error
raises SdgfToolError with its code, message and hint. A harness on the same machine as
the store can skip the server and use sdgf.service.Service and sdgf.jobs.JobManager
directly; this client is for one that reaches sdgf over the network. Needs the mcp extra.
"""

from __future__ import annotations

import json
from types import TracebackType
from typing import Any, Self

import anyio
import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

DEFAULT_POLL_S = 60.0  # the server's wait_job blocks for at most 120 s per call


class SdgfToolError(Exception):
    def __init__(self, code: str, message: str, hint: str | None = None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.hint = hint


def _tool_error(text: str) -> SdgfToolError:
    try:
        error = json.loads(text[text.index("{") :])
        return SdgfToolError(error["code"], error["message"], error.get("hint"))
    except (ValueError, KeyError):
        return SdgfToolError("tool_failed", text)


class SdgfClient:
    def __init__(self, url: str, *, token: str | None = None):
        self.url = url
        self.token = token
        self._http: httpx2.AsyncClient | None = None
        self._client: Client | None = None

    async def __aenter__(self) -> Self:
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        self._http = httpx2.AsyncClient(headers=headers)
        await self._http.__aenter__()
        self._client = Client(streamable_http_client(self.url, http_client=self._http))
        try:
            await self._client.__aenter__()
        except BaseException:
            await self._http.__aexit__(None, None, None)
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            if self._client is not None:
                await self._client.__aexit__(exc_type, exc, tb)
        finally:
            if self._http is not None:
                await self._http.__aexit__(exc_type, exc, tb)

    async def call(self, tool: str, /, **args: Any) -> dict[str, Any]:
        """Call one tool; return its result, or raise SdgfToolError."""
        if self._client is None:
            raise RuntimeError("use SdgfClient as `async with SdgfClient(...) as sdgf`")
        result = await self._client.call_tool(tool, args)
        if result.is_error:
            raise _tool_error(result.content[0].text if result.content else "")
        return dict(result.structured_content or {})

    async def wait(self, job_id: str, *, poll_s: float = DEFAULT_POLL_S) -> dict[str, Any]:
        """Wait, however long it takes, until the job stops running; return its status."""
        while True:
            status = await self.call("wait_job", job_id=job_id, timeout_s=poll_s)
            if status["state"] != "running":
                return status


def call(url: str, tool: str, /, *, token: str | None = None, **args: Any) -> dict[str, Any]:
    """One tool call, blocking: a session is opened and closed around it."""

    async def go() -> dict[str, Any]:
        async with SdgfClient(url, token=token) as sdgf:
            return await sdgf.call(tool, **args)

    return anyio.run(go)
