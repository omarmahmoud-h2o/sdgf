"""Tracing a run: one trace per candidate, a span per attempt, check layer, model call and
tool call (README "Tracing with LangSmith").

    candidate ── attempt 0 ── generate ── generator:<backend>   (llm)
                │            │          └ tool:<name>           (tool)
                │            ├ L1 … Ln  (the cascade stops at the first failure)
                │            │   └ judge:<backend> / consistency_judge:<backend>  (llm)
                └ attempt 1 …
    settle → the candidate's outputs (accepted or dropped) and feedback keys

The pipeline runs slots on a thread pool, so the current span lives on a thread-local
stack (like UsageMeter.capture) rather than in contextvars, which don't reach workers.
A candidate's root span is ended by settle on the main thread, because settle can still
drop an accepted record (an L4 near-duplicate within the wave).

Tracing never changes a run: every tracer call is guarded, a failure is logged as a
warning and the run goes on, and artefacts are the same with tracing on or off. Turning
it on makes the tracing service a data destination, recorded per run (TRACE_SINKS_STAGE)
and in the governance report. Held-out keys never leave: L4 issues against the held-out
set are scrubbed before they reach a span.
"""

from __future__ import annotations

import importlib
import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

from sdgf.models.base import DecisionResponse, ModelBackend, ModelResponse, ToolSpec

log = logging.getLogger(__name__)

TRACE_SINKS_STAGE = "trace_sinks"
LANGSMITH = "langsmith"
DEFAULT_LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"


class TracingError(ImportError):
    """Tracing was asked for but can't start (missing package or API key)."""


class Span:
    """A started span; end() it once. The base class is the no-op span."""

    def end(
        self,
        outputs: Mapping[str, Any] | None = None,
        error: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        return None


NULL_SPAN = Span()


@dataclass(frozen=True)
class SpanRef:
    """A candidate's root span, carried by the slot from the worker to settle."""

    handle: Any = None


NULL_REF = SpanRef()


class Tracer:
    """The tracing seam. Subclasses implement _start/_end/_feedback; this class keeps the
    per-thread span stack and makes sure a tracing failure never reaches the run."""

    enabled = False
    sink: dict[str, Any] | None = None  # the data destination this tracer sends to

    def __init__(self) -> None:
        self._local = threading.local()

    # ── adapter primitives ──
    def _start(
        self,
        name: str,
        run_type: str,
        inputs: Mapping[str, Any],
        metadata: Mapping[str, Any],
        tags: Sequence[str],
        parent: Any,
    ) -> Any:
        raise NotImplementedError

    def _end(
        self,
        handle: Any,
        outputs: Mapping[str, Any] | None,
        error: str | None,
        metadata: Mapping[str, Any] | None,
    ) -> None:
        raise NotImplementedError

    def _feedback(self, handle: Any, key: str, value: Any) -> None:
        raise NotImplementedError

    def _flush(self) -> None:
        return None

    # ── public interface ──
    def _stack(self) -> list[Any]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = self._local.stack = []
        return stack

    def _guard(self, what: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # tracing must never fail a run
            log.warning("tracing: %s failed: %s", what, e)
            return None

    @contextmanager
    def candidate(
        self,
        *,
        run_id: str,
        cell_id: str,
        seed: int,
        recipe: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        tags: Sequence[str] = (),
    ) -> Iterator[SpanRef]:
        """Open a candidate's root span on this thread. Leaving the block does not end it;
        settle() does, with the final outcome."""
        if not self.enabled:
            yield NULL_REF
            return
        meta = {"run_id": run_id, "cell_id": cell_id, "seed": seed, **(metadata or {})}
        handle = self._guard(
            "candidate",
            self._start,
            f"candidate {cell_id}",
            "chain",
            {"cell_id": cell_id, "fixed_facts": dict(recipe or {})},
            meta,
            [*tags, f"cell:{cell_id}", f"run:{run_id}"],
            None,
        )
        if handle is None:
            yield NULL_REF
            return
        stack = self._stack()
        stack.append(handle)
        try:
            yield SpanRef(handle)
        finally:
            if stack and stack[-1] is handle:
                stack.pop()

    @contextmanager
    def span(
        self,
        name: str,
        run_type: str = "chain",
        inputs: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Iterator[Span]:
        """A child of this thread's current span. An exception in the block ends the span
        with the error and is re-raised; outside any candidate the span is a no-op."""
        stack = self._stack() if self.enabled else ()
        if not stack:
            yield NULL_SPAN
            return
        handle = self._guard(
            name,
            self._start,
            name,
            run_type,
            dict(inputs or {}),
            dict(metadata or {}),
            (),
            stack[-1],
        )
        if handle is None:
            yield NULL_SPAN
            return
        span = _AdapterSpan(self, handle, name)
        stack.append(handle)
        try:
            yield span
        except BaseException as e:
            span.end(error=f"{type(e).__name__}: {e}")
            raise
        finally:
            if stack and stack[-1] is handle:
                stack.pop()
            span.end()  # no-op if the block already ended it

    def settle(
        self,
        ref: SpanRef,
        *,
        outcome: str,
        layer: str | None = None,
        codes: Sequence[str] = (),
        attempts: int = 0,
        feedback: Mapping[str, Any] | None = None,
    ) -> None:
        """End a candidate's root span with its final outcome and attach feedback keys."""
        if not self.enabled or ref.handle is None:
            return
        outputs = {"outcome": outcome, "layer": layer, "codes": list(codes), "attempts": attempts}
        self._guard("settle", self._end, ref.handle, outputs, None, None)
        keys = {
            "accepted": int(outcome == "accepted"),
            "attempts": attempts,
            **({"failed_layer": layer} if layer else {}),
            **(feedback or {}),
        }
        for key, value in keys.items():
            if value is not None:
                self._guard(f"feedback {key}", self._feedback, ref.handle, key, value)

    def flush(self) -> None:
        if self.enabled:
            self._guard("flush", self._flush)


class _AdapterSpan(Span):
    def __init__(self, tracer: Tracer, handle: Any, name: str):
        self.tracer, self.handle, self.name = tracer, handle, name
        self.ended = False

    def end(
        self,
        outputs: Mapping[str, Any] | None = None,
        error: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if self.ended:
            return
        self.ended = True
        self.tracer._guard(self.name, self.tracer._end, self.handle, outputs, error, metadata)


class NullTracer(Tracer):
    """Tracing off: every call is a no-op."""


# ── in-memory adapter (tests, debugging) ─────────────────────────


@dataclass
class TraceNode:
    name: str
    run_type: str
    inputs: dict[str, Any]
    metadata: dict[str, Any]
    tags: list[str]
    parent: TraceNode | None = None
    children: list[TraceNode] = field(default_factory=list)
    outputs: dict[str, Any] | None = None
    error: str | None = None
    feedback: dict[str, Any] = field(default_factory=dict)
    ended: bool = False

    def child(self, name: str) -> TraceNode:
        """The first child with this name (raises KeyError if none)."""
        for c in self.children:
            if c.name == name:
                return c
        raise KeyError(f"{self.name} has no child {name!r}; has {[c.name for c in self.children]}")

    def names(self) -> list[str]:
        return [c.name for c in self.children]


class RecordingTracer(Tracer):
    """Keeps every trace in memory: roots is one TraceNode per candidate."""

    enabled = True

    def __init__(self, sink: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.sink = dict(sink) if sink is not None else None
        self.roots: list[TraceNode] = []
        self._lock = threading.Lock()

    def _start(self, name, run_type, inputs, metadata, tags, parent):  # type: ignore[no-untyped-def]
        node = TraceNode(name, run_type, dict(inputs), dict(metadata), list(tags), parent)
        with self._lock:
            if parent is None:
                self.roots.append(node)
            else:
                parent.children.append(node)
        return node

    def _end(self, handle, outputs, error, metadata):  # type: ignore[no-untyped-def]
        handle.outputs = dict(outputs) if outputs is not None else handle.outputs
        handle.error = error if error is not None else handle.error
        handle.metadata.update(metadata or {})
        handle.ended = True

    def _feedback(self, handle, key, value):  # type: ignore[no-untyped-def]
        handle.feedback[key] = value


# ── LangSmith adapter ────────────────────────────────────────────


class LangSmithTracer(Tracer):
    """Sends traces to LangSmith as explicit RunTrees (no contextvars). The API key comes
    from LANGSMITH_API_KEY; LANGSMITH_ENDPOINT picks the deployment (default: US cloud)."""

    enabled = True

    def __init__(
        self,
        project: str,
        *,
        tags: Sequence[str] = (),
        client: Any = None,
        endpoint: str | None = None,
    ) -> None:
        super().__init__()
        try:
            self._ls = importlib.import_module("langsmith.run_trees")
        except ImportError as e:
            raise TracingError(
                "tracing to LangSmith needs the optional package 'langsmith'; "
                "install it with: pip install sdgf[tracing]"
            ) from e
        self.endpoint = (
            endpoint or os.environ.get("LANGSMITH_ENDPOINT") or DEFAULT_LANGSMITH_ENDPOINT
        )
        if client is None:
            if not os.environ.get("LANGSMITH_API_KEY"):
                raise TracingError(
                    "tracing to LangSmith needs the environment variable LANGSMITH_API_KEY"
                )
            client = importlib.import_module("langsmith").Client(api_url=self.endpoint)
        self.client = client
        self.project = project
        self.tags = list(tags)
        self.sink = {
            "sink": LANGSMITH,
            "endpoint": self.endpoint,
            "project": project,
            "content": "full",
            "hosting": "provider_api",
        }

    def _start(self, name, run_type, inputs, metadata, tags, parent):  # type: ignore[no-untyped-def]
        extra = {"metadata": dict(metadata)}
        if parent is None:
            run = self._ls.RunTree(
                name=name,
                run_type=run_type,
                inputs=dict(inputs),
                project_name=self.project,
                ls_client=self.client,
                tags=[*self.tags, *tags],
                extra=extra,
            )
        else:
            run = parent.create_child(
                name=name, run_type=run_type, inputs=dict(inputs), extra=extra
            )
        run.post()
        return run

    def _end(self, handle, outputs, error, metadata):  # type: ignore[no-untyped-def]
        handle.end(
            outputs=dict(outputs) if outputs is not None else None,
            error=error,
            metadata=dict(metadata) if metadata else None,
        )
        handle.patch()

    def _feedback(self, handle, key, value):  # type: ignore[no-untyped-def]
        if isinstance(value, bool) or isinstance(value, (int, float)):
            self.client.create_feedback(handle.id, key, score=value, trace_id=handle.trace_id)
        else:
            self.client.create_feedback(handle.id, key, value=value, trace_id=handle.trace_id)

    def _flush(self) -> None:
        flush = getattr(self.client, "flush", None)
        if callable(flush):
            flush()


def make_tracer(kind: str | None, *, project: str, tags: Sequence[str] = ()) -> Tracer:
    """The CLI's --trace: None for no tracing, 'langsmith' for LangSmith."""
    if kind is None:
        return NullTracer()
    if kind == LANGSMITH:
        return LangSmithTracer(project, tags=tags)
    raise ValueError(f"unknown tracer {kind!r}; expected {LANGSMITH!r}")


# ── what a span carries ──────────────────────────────────────────


def layer_outputs(verdict: Any) -> dict[str, Any]:
    """A check layer's span outputs: outcome, codes, messages and details, with L4
    issues against the held-out set scrubbed (no key, no message)."""
    errors = [e.to_dict() for e in verdict.errors]
    if verdict.layer == "L4":
        errors = [_scrub_held_out(e) for e in errors]
    return {
        "outcome": verdict.outcome,
        "codes": [e["code"] for e in errors],
        "errors": errors,
        "details": dict(verdict.details),
    }


def _scrub_held_out(issue: dict[str, Any]) -> dict[str, Any]:
    details = dict(issue.get("details") or {})
    if details.get("source") != "held_out":
        return issue
    details.pop("match", None)
    return {
        **issue,
        "message": "similarity to a held-out item exceeds the threshold",
        "details": details,
    }


# ── model calls ──────────────────────────────────────────────────


class TracedBackend(ModelBackend):
    """Wraps a stage's backend so each call is an llm span under the current span. Name,
    model and hosting are the wrapped backend's (judge selection dispatches on name)."""

    def __init__(self, inner: ModelBackend, stage: str, tracer: Tracer):
        self.inner = inner
        self.name = inner.name  # type: ignore[misc]
        self.model = inner.model
        self.hosting = inner.hosting
        self.stage = stage
        self.tracer = tracer

    def setup(self) -> None:
        self.inner.setup()

    def _meta(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "backend": self.name,
            "model": self.model,
            "hosting": self.hosting,
        }

    def call(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        tools: list[ToolSpec] | None = None,
    ) -> ModelResponse:
        inputs: dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            inputs["tools"] = [t.get("name") for t in tools]
        with self.tracer.span(f"{self.stage}:{self.name}", "llm", inputs, self._meta()) as span:
            response = self.inner.call(prompt, max_tokens, temperature, tools)
            span.end(
                outputs={
                    "text": response.text,
                    "tool_calls": [
                        {"name": c.name, "arguments": c.arguments} for c in response.tool_calls
                    ],
                },
                metadata=_usage_meta(response.input_tokens, response.output_tokens),
            )
            return response

    def decide(self, state: Any, questions: dict[str, dict[str, Any]]) -> DecisionResponse:
        inputs = {"state": state, "questions": questions}
        with self.tracer.span(f"{self.stage}:{self.name}", "llm", inputs, self._meta()) as span:
            response = self.inner.decide(state, questions)
            span.end(
                outputs={"answers": response.answers, "service_model": response.model},
                metadata=_usage_meta(response.input_tokens, response.output_tokens),
            )
            return response


def _usage_meta(inp: int | None, out: int | None) -> dict[str, Any]:
    if inp is None and out is None:
        return {}
    return {
        "usage_metadata": {
            "input_tokens": inp or 0,
            "output_tokens": out or 0,
            "total_tokens": (inp or 0) + (out or 0),
        }
    }
