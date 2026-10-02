"""Token and cost metering per stage (FRAMEWORK_DESIGN.md §8 cost, §9.3 budgets).

MeteredBackend wraps a stage's backend and records every call's input and output tokens
in a UsageMeter. Tokens come from the backend response; a backend that reports none
(MLX, a bare mock) is estimated at one token per four characters of prompt and reply,
and the call is counted as estimated. Cost is estimated from the stage's price in
models.<stage> (input_cost_per_mtok / output_cost_per_mtok, USD per million tokens); a
stage without a price has cost None, so "not priced" is never reported as $0.

Calls made inside UsageMeter.capture() (one pipeline slot, on one thread) go to that
capture, so the pipeline can charge each slot in reservation order and the totals don't
depend on the order concurrent calls finish in. Calls outside any capture (stage 1
keyword expansion) collect as loose usage until take_loose() drains them.
"""

from __future__ import annotations

import json
import math
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

from sdgf.models.base import DecisionResponse, ModelBackend, ModelResponse, ToolSpec
from sdgf.spec.schema import ModelConfig

CHARS_PER_TOKEN = 4
USAGE_STAGE = "usage"  # a run's usage ledger artefact, usage.json, across invocations


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


@dataclass(frozen=True)
class Pricing:
    input_per_mtok: float
    output_per_mtok: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1e6

    @classmethod
    def from_config(cls, config: ModelConfig | None) -> Pricing | None:
        if config is None or config.input_cost_per_mtok is None:
            return None
        assert config.output_cost_per_mtok is not None  # the schema sets both or neither
        return cls(config.input_cost_per_mtok, config.output_cost_per_mtok)


@dataclass
class StageUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_calls: int = 0  # calls whose token counts were estimated from text length
    cost_usd: float | None = None  # None: the stage has no price

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, other: StageUsage) -> None:
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.estimated_calls += other.estimated_calls
        if other.cost_usd is not None:
            self.cost_usd = (self.cost_usd or 0.0) + other.cost_usd

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tokens": self.tokens,
            "estimated_calls": self.estimated_calls,
            "cost_usd": self.cost_usd,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> StageUsage:
        return cls(
            calls=int(d.get("calls", 0)),
            input_tokens=int(d.get("input_tokens", 0)),
            output_tokens=int(d.get("output_tokens", 0)),
            estimated_calls=int(d.get("estimated_calls", 0)),
            cost_usd=None if d.get("cost_usd") is None else float(d["cost_usd"]),
        )


class UsageLedger:
    """Usage per stage, e.g. one slot's calls or a whole run's."""

    def __init__(self, stages: Mapping[str, StageUsage] | None = None):
        self.stages: dict[str, StageUsage] = dict(stages or {})

    def stage(self, name: str) -> StageUsage:
        return self.stages.setdefault(name, StageUsage())

    def add(self, other: UsageLedger) -> None:
        for name in sorted(other.stages):
            self.stage(name).add(other.stages[name])

    @property
    def tokens(self) -> int:
        return sum(s.tokens for s in self.stages.values())

    @property
    def cost_usd(self) -> float | None:
        costs = [s.cost_usd for s in self.stages.values() if s.cost_usd is not None]
        return sum(costs) if costs else None

    def total(self) -> StageUsage:
        out = StageUsage()
        for name in sorted(self.stages):
            out.add(self.stages[name])
        return out

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {name: self.stages[name].to_dict() for name in sorted(self.stages)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Mapping[str, Any]]) -> UsageLedger:
        return cls({name: StageUsage.from_dict(s) for name, s in d.items()})


class UsageMeter:
    """Thread-safe sink for metered calls: into the calling thread's capture, if any."""

    def __init__(self) -> None:
        self._local = threading.local()
        self._lock = threading.Lock()
        self._loose = UsageLedger()

    def record(self, stage: str, usage: StageUsage) -> None:
        held: UsageLedger | None = getattr(self._local, "ledger", None)
        if held is not None:
            held.stage(stage).add(usage)
            return
        with self._lock:
            self._loose.stage(stage).add(usage)

    @contextmanager
    def capture(self) -> Iterator[UsageLedger]:
        self._local.ledger = ledger = UsageLedger()
        try:
            yield ledger
        finally:
            self._local.ledger = None

    def take_loose(self) -> UsageLedger:
        with self._lock:
            loose, self._loose = self._loose, UsageLedger()
        return loose


class MeteredBackend(ModelBackend):
    """Records each call's tokens and estimated cost under `stage`. Name, model and
    hosting are the wrapped backend's, so provenance and endpoints are unchanged."""

    def __init__(
        self, inner: ModelBackend, stage: str, meter: UsageMeter, pricing: Pricing | None = None
    ):
        self.inner = inner
        self.name = inner.name  # type: ignore[misc]
        self.model = inner.model
        self.hosting = inner.hosting
        self.stage = stage
        self.meter = meter
        self.pricing = pricing

    def setup(self) -> None:
        self.inner.setup()

    def call(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        tools: list[ToolSpec] | None = None,
    ) -> ModelResponse:
        response = self.inner.call(prompt, max_tokens, temperature, tools)
        self.meter.record(self.stage, self.usage_of(prompt, tools, response))
        return response

    def decide(self, state: Any, questions: dict[str, dict[str, Any]]) -> DecisionResponse:
        response = self.inner.decide(state, questions)
        inp, out = response.input_tokens, response.output_tokens
        estimated = inp is None or out is None
        if inp is None:
            inp = estimate_tokens(json.dumps([state, questions], sort_keys=True, default=str))
        if out is None:
            out = estimate_tokens(json.dumps(response.answers, sort_keys=True))
        self.meter.record(self.stage, self._usage(inp, out, estimated))
        return response

    def usage_of(
        self, prompt: str, tools: list[ToolSpec] | None, response: ModelResponse
    ) -> StageUsage:
        estimated = response.input_tokens is None or response.output_tokens is None
        inp = response.input_tokens
        if inp is None:
            offered = json.dumps(tools, sort_keys=True) if tools else ""
            inp = estimate_tokens(prompt + offered)
        out = response.output_tokens
        if out is None:
            calls = [{"name": c.name, "arguments": c.arguments} for c in response.tool_calls]
            out = estimate_tokens(
                (response.text or "") + (json.dumps(calls, sort_keys=True) if calls else "")
            )
        return self._usage(inp, out, estimated)

    def _usage(self, inp: int, out: int, estimated: bool) -> StageUsage:
        return StageUsage(
            calls=1,
            input_tokens=inp,
            output_tokens=out,
            estimated_calls=int(estimated),
            cost_usd=self.pricing.cost(inp, out) if self.pricing is not None else None,
        )
