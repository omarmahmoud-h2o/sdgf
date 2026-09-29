"""Keyword generation and bi-directional expansion (FRAMEWORK_DESIGN.md §6.2 step 1).

Ported from DS²-Instruct's keywords_generator.py and prompts.py:

- seed keywords come from the task description and the seeds: keywords the spec or the
  seeds declare, plus one model call over the task description (and seed excerpts when
  seeds.uses includes keyword_seeding);
- each iteration samples the current keywords and asks for ↓ prerequisite
  (foundational) and ↑ advanced (specialised) concepts, keeping only new ones.

Fixes from §12.1: every prompt carries the current keyword list, so the model can see
what already exists (the original always passed found_keywords=[] and only showed a
sample), and an unparseable reply raises KeywordParseError instead of silently falling
back to random mock keywords, which hid a broken backend.

Settings come from coverage.params["keyword_expansion"]; the backend is models.expansion.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field, fields
from typing import Any, Iterable, Mapping, Sequence

from sdgf.models.base import ModelBackend
from sdgf.spec.compile import CompiledSpec

PARAMS_KEY = "keyword_expansion"


class KeywordError(ValueError):
    """Keyword expansion is misconfigured."""


class KeywordParseError(KeywordError):
    """A model reply held no keywords. Raised rather than falling back to mock keywords."""

    def __init__(self, stage: str, response: str | None) -> None:
        self.stage = stage
        self.response = response
        shown = "no text" if not response else repr(response[:200])
        super().__init__(f"keyword {stage} reply had no parseable keywords ({shown})")


# ── parsing ──────────────────────────────────────────────────────


def normalise_keyword(text: str) -> str:
    """Lower-case, underscores for spaces and hyphens, only [a-z0-9_.] (as the original)."""
    kw = re.sub(r"[\s-]+", "_", text.strip().lower())
    return re.sub(r"[^a-z0-9_.]", "", kw).strip("_.")


def parse_keywords(response: str | None) -> list[str]:
    """Port of DS²-Instruct utils.parse_keywords: comma-separated, normalised, deduped."""
    if not response:
        return []
    text = re.sub(r"^\s*(keywords?|concepts?|terms?)\s*:\s*", "", response, flags=re.IGNORECASE)
    parsed = (normalise_keyword(part) for part in re.split(r"[,\n]", text))
    return list(dict.fromkeys(kw for kw in parsed if kw))


# ── settings ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExpansionSettings:
    initial_count: int = 10
    iterations: int = 2
    per_iteration: int = 10  # asked for per direction per iteration
    sample_size: int = 10  # current keywords shown as the sample to expand from
    max_keywords: int | None = None
    directions: tuple[str, ...] = ("prerequisite", "advanced")
    seed_keywords: tuple[str, ...] = ()
    seed_excerpts: int = 5  # seeds shown to the initial prompt under keyword_seeding
    excerpt_chars: int = 400
    example_multiword: str = "cash_flow, interest_rate"
    example_single: str = "liquidity, collateral"

    def __post_init__(self) -> None:
        for name in ("initial_count", "per_iteration", "sample_size"):
            if getattr(self, name) < 1:
                raise KeywordError(f"{PARAMS_KEY}.{name} must be at least 1")
        for name in ("iterations", "seed_excerpts", "excerpt_chars"):
            if getattr(self, name) < 0:
                raise KeywordError(f"{PARAMS_KEY}.{name} must not be negative")
        if self.max_keywords is not None and self.max_keywords < 1:
            raise KeywordError(f"{PARAMS_KEY}.max_keywords must be at least 1")
        unknown = [d for d in self.directions if d not in DIRECTION_PROMPTS]
        if unknown or not self.directions:
            raise KeywordError(
                f"{PARAMS_KEY}.directions must be a non-empty subset of "
                f"{sorted(DIRECTION_PROMPTS)}, got {list(self.directions)}"
            )

    @classmethod
    def from_params(cls, params: Mapping[str, Any] | None) -> ExpansionSettings:
        params = dict(params or {})
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(params) - known)
        if unknown:
            raise KeywordError(f"unknown {PARAMS_KEY} settings: {unknown}")
        for name in ("directions", "seed_keywords"):
            if name in params:
                params[name] = tuple(params[name])
        return cls(**params)


# ── prompts (ported from DS²-Instruct scripts/prompts.py) ────────


def _keyword_list(keywords: Sequence[str]) -> str:
    return ", ".join(keywords) if keywords else "(none yet)"


def initial_prompt(
    task_description: str,
    count: int,
    settings: ExpansionSettings,
    existing: Sequence[str] = (),
    excerpts: Sequence[str] = (),
) -> str:
    examples = ""
    if excerpts:
        examples = "\n\nExample records for this task:\n" + "\n".join(f"- {e}" for e in excerpts)
    return f"""\
Task Context: You are an expert in the domain of this task.

Task Description: {task_description.strip()}{examples}

Existing Keywords: {_keyword_list(existing)}

Instructions: Generate {count} core keywords that represent the most \
essential concepts for this task.

Requirements:
- List exactly {count} core concepts separated by commas
- Use underscores for multi-word concepts (e.g., {settings.example_multiword})
- Single words are acceptable (e.g., {settings.example_single})
- Ensure concepts are different from the existing keywords
- Provide only the comma-separated list without any other text

Core Keywords:"""


def prerequisite_prompt(
    task_description: str,
    sample: Sequence[str],
    count: int,
    settings: ExpansionSettings,
    existing: Sequence[str],
) -> str:
    return f"""\
Task Context: You are an expert in the domain related to: {task_description.strip()}

Sample Keywords: {_keyword_list(sample)}

Existing Keywords: {_keyword_list(existing)}

Instructions: What fundamental concepts, basic terminology, or foundational \
principles should learners understand BEFORE studying the sample keywords? \
Generate {count} prerequisite concepts.

Requirements:
- List {count} prerequisite concepts separated by commas
- Use underscores for multi-word concepts (e.g., {settings.example_multiword})
- Ensure concepts are different from the existing keywords
- Provide only the comma-separated list

Prerequisite Concepts:"""


def advanced_prompt(
    task_description: str,
    sample: Sequence[str],
    count: int,
    settings: ExpansionSettings,
    existing: Sequence[str],
) -> str:
    return f"""\
Task Context: You are an expert in the domain related to: {task_description.strip()}

Sample Keywords: {_keyword_list(sample)}

Existing Keywords: {_keyword_list(existing)}

Instructions: What specialized subfields, cutting-edge developments, or \
expert-level topics BUILD UPON the sample keywords? Generate {count} \
advanced concepts.

Requirements:
- List {count} advanced concepts separated by commas
- Use underscores for multi-word concepts (e.g., {settings.example_multiword})
- Ensure concepts are different from the existing keywords
- Provide only the comma-separated list

Advanced Concepts:"""


def extraction_prompt(
    task_description: str,
    passage: str,
    found_keywords: Sequence[str],
    settings: ExpansionSettings,
) -> str:
    """Retrieval-augmented extraction; found_keywords is the real current list (§12.1)."""
    return f"""\
Task Context: You are an expert in the domain related to: {task_description.strip()}

Current Keywords: {_keyword_list(found_keywords)}

Retrieved Passage:
{passage}

Instructions: Extract additional domain-specific keywords directly from \
the retrieved passage that are missing from the current list.

Requirements:
- Use underscores for multi-word concepts (e.g., {settings.example_multiword})
- Single words are acceptable (e.g., {settings.example_single})
- Focus on domain-specific terminology and concepts
- Avoid generic words or concepts already provided
- Provide only the comma-separated list

Extracted Keywords:"""


DIRECTION_PROMPTS = {"prerequisite": prerequisite_prompt, "advanced": advanced_prompt}


# ── seeds ────────────────────────────────────────────────────────


def _public_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for k, v in value.items():
            if not str(k).startswith("_"):
                yield from _public_strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _public_strings(v)


def seed_excerpt(seed: Mapping[str, Any], max_chars: int) -> str:
    """A seed's message contents (else every public string), whitespace-collapsed."""
    messages = seed.get("messages")
    if isinstance(messages, list) and messages:
        parts = [m.get("content", "") for m in messages if isinstance(m, Mapping)]
    else:
        parts = [s for k, v in seed.items() if k != "keywords" for s in _public_strings({k: v})]
    text = " ".join(" ".join(p.split()) for p in parts if p)
    return text[:max_chars].rstrip()


def declared_seed_keywords(
    seeds: Sequence[Mapping[str, Any]], settings: ExpansionSettings
) -> list[str]:
    """Keywords the settings or the seeds (a `keywords` list or string) declare."""
    raw: list[str] = list(settings.seed_keywords)
    for seed in seeds:
        kws = seed.get("keywords")
        if isinstance(kws, str):
            raw.extend(kws.split(","))
        elif isinstance(kws, list):
            raw.extend(str(k) for k in kws)
    return list(dict.fromkeys(k for k in (normalise_keyword(r) for r in raw) if k))


# ── expansion ────────────────────────────────────────────────────


@dataclass
class KeywordResult:
    keywords: list[str]
    initial: list[str]
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "keywords": list(self.keywords),
            "initial": list(self.initial),
            "history": [dict(h) for h in self.history],
        }


class KeywordExpander:
    def __init__(
        self,
        backend: ModelBackend,
        task_description: str,
        settings: ExpansionSettings | None = None,
        *,
        seeds: Sequence[Mapping[str, Any]] = (),
        use_seed_excerpts: bool = False,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> None:
        if not task_description.strip():
            raise KeywordError("keyword expansion needs a task description")
        self.backend = backend
        self.task_description = task_description
        self.settings = settings or ExpansionSettings()
        self.seeds = tuple(seeds)
        self.use_seed_excerpts = use_seed_excerpts
        self.max_tokens = max_tokens
        self.temperature = temperature

    @classmethod
    def from_spec(
        cls, compiled: CompiledSpec, backend: ModelBackend, **kwargs: Any
    ) -> KeywordExpander:
        spec = compiled.spec
        settings = ExpansionSettings.from_params(spec.coverage.params.get(PARAMS_KEY))
        config = spec.models.expansion
        if config is not None:
            kwargs.setdefault("max_tokens", config.max_tokens)
            kwargs.setdefault("temperature", config.temperature)
        kwargs.setdefault("seeds", compiled.seeds)
        kwargs.setdefault("use_seed_excerpts", "keyword_seeding" in spec.seeds.uses)
        return cls(backend, spec.task.description, settings, **kwargs)

    def _ask(self, stage: str, prompt: str) -> list[str]:
        response = self.backend.call(prompt, self.max_tokens, self.temperature)
        keywords = parse_keywords(response.text)
        if not keywords:
            raise KeywordParseError(stage, response.text)
        return keywords

    def _room(self, current: Sequence[str]) -> int | None:
        cap = self.settings.max_keywords
        return None if cap is None else max(cap - len(current), 0)

    def _add(self, current: list[str], known: set[str], found: Iterable[str]) -> list[str]:
        added = []
        for kw in found:
            room = self._room(current)
            if room == 0:
                break
            if kw not in known:
                known.add(kw)
                current.append(kw)
                added.append(kw)
        return added

    def seed(self) -> tuple[list[str], dict[str, Any]]:
        """Declared seed keywords plus the model's core keywords for the task."""
        s = self.settings
        current = declared_seed_keywords(self.seeds, s)
        excerpts: list[str] = []
        if self.use_seed_excerpts and s.seed_excerpts:
            excerpts = [
                e
                for e in (seed_excerpt(x, s.excerpt_chars) for x in self.seeds[: s.seed_excerpts])
                if e
            ]
        prompt = initial_prompt(self.task_description, s.initial_count, s, current, excerpts)
        known = set(current)
        declared = list(current)
        added = self._add(current, known, self._ask("initial", prompt))
        return current, {"declared": declared, "generated": added}

    def expand_iteration(
        self, iteration: int, current: list[str], rng: random.Random
    ) -> dict[str, Any]:
        """One ↓/↑ round over a sample of `current`, which is extended in place."""
        s = self.settings
        known = set(current)
        sample = rng.sample(current, min(s.sample_size, len(current)))
        record: dict[str, Any] = {
            "iteration": iteration,
            "starting_count": len(current),
            "sample": sample,
            "added": {},
        }
        for direction in s.directions:
            if self._room(current) == 0:
                record["added"][direction] = []
                continue
            prompt = DIRECTION_PROMPTS[direction](
                self.task_description, sample, s.per_iteration, s, list(current)
            )
            record["added"][direction] = self._add(current, known, self._ask(direction, prompt))
        record["ending_count"] = len(current)
        return record

    def run(self, rng: random.Random) -> KeywordResult:
        current, seeded = self.seed()
        result = KeywordResult(keywords=current, initial=list(current))
        result.history.append({"iteration": 0, **seeded, "ending_count": len(current)})
        for i in range(1, self.settings.iterations + 1):
            if self._room(current) == 0:
                break
            result.history.append(self.expand_iteration(i, current, rng))
        return result
