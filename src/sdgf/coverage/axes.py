"""Coverage axes (FRAMEWORK_DESIGN.md §6.2 step 2): resolve every spec axis to concrete
values and weights, and cross them into combinations.

Axis sources:

    fixed              the spec's values and weights
    bloom              the six Bloom levels (DS²-Instruct stage ❷), or the subset listed
    keyword_expansion  the stage 1 keyword list from keywords.py
    retrieval          that list after retrieval extraction (retrieval.py)

A keyword axis puts the spec's own values (if any) first and is always split evenly,
since its values aren't known until expansion runs. Weights count only under
quota_policy "weighted"; otherwise every value of every axis has the same share.

A combination's id is its axis values joined with "|" in axis order (non-strings as
JSON, so the label true is "true"), the same ids the pipeline's fixed-axis grid used.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from sdgf.spec.schema import Axis, CoverageSection

# Port of QUERY_TYPES in src_original/DS2-Instruct/scripts/prompts.py.
BLOOM_LEVELS: dict[str, str] = {
    "Remember": "Create instructions that emphasize recall of factual knowledge, definitions, basic concepts, recognition tasks, and core terminology related to the keyword.",
    "Understand": "Design instructions that require conceptual understanding, explanation of relationships, interpretation, illustrative examples, and meaningful comparisons involving the keyword.",
    "Apply": "Formulate instructions that demand practical use of methods, implementation of procedures, execution of calculations, and real-world application of the keyword.",
    "Analyze": "Develop instructions that involve breaking down complex ideas, identifying patterns, examining relationships, and conducting comparative or structural analysis of the keyword.",
    "Evaluate": "Construct instructions that involve critical judgment, validation of techniques, assessment of alternatives, justification of decisions, and critique of methods related to the keyword.",
    "Create": "Design instructions that foster original thinking, synthesis of ideas, problem innovation, creative design, and novel applications of the keyword.",
}

KEYWORD_SOURCES: tuple[str, ...] = ("keyword_expansion", "retrieval")
CELL_ID_SEP = "|"


class AxisError(ValueError):
    """An axis can't be resolved to values (e.g. a keyword axis with no keyword list)."""


def value_id(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def bloom_description(level: str) -> str:
    try:
        return BLOOM_LEVELS[level]
    except KeyError:
        raise AxisError(f"unknown Bloom level {level!r}; use one of {list(BLOOM_LEVELS)}") from None


@dataclass(frozen=True)
class ResolvedAxis:
    name: str
    source: str
    values: tuple[Any, ...]
    weights: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "values": list(self.values),
            "weights": list(self.weights),
        }


def resolve_axis(
    axis: Axis,
    keywords: Mapping[str, Sequence[str]] | None = None,
    *,
    weighted: bool = False,
) -> ResolvedAxis:
    """Concrete values and weights for one axis; `keywords` maps a keyword source to its list."""
    if axis.source in KEYWORD_SOURCES:
        found = (keywords or {}).get(axis.source)
        if found is None:
            raise AxisError(f"axis {axis.name!r} needs the {axis.source} keyword list")
        values: list[Any] = list(dict.fromkeys([*(axis.values or []), *found]))
        weights = [1.0] * len(values)
    elif axis.source == "bloom":
        values = list(axis.values) if axis.values is not None else list(BLOOM_LEVELS)
        for v in values:
            bloom_description(v)
        weights = list(axis.weights) if weighted and axis.weights else [1.0] * len(values)
    else:
        values = list(axis.values or [])
        weights = list(axis.weights) if weighted and axis.weights else [1.0] * len(values)
    if not values:
        raise AxisError(f"axis {axis.name!r} resolved to no values")
    ids = [value_id(v) for v in values]
    if len(set(ids)) != len(ids):
        raise AxisError(f"axis {axis.name!r} has duplicate values")
    if any(CELL_ID_SEP in i for i in ids):
        raise AxisError(f"axis {axis.name!r}: values may not contain {CELL_ID_SEP!r}")
    return ResolvedAxis(axis.name, axis.source, tuple(values), tuple(weights))


def resolve_axes(
    coverage: CoverageSection, keywords: Mapping[str, Sequence[str]] | None = None
) -> list[ResolvedAxis]:
    weighted = coverage.quota_policy == "weighted"
    return [resolve_axis(a, keywords, weighted=weighted) for a in coverage.axes]


def keyword_sources(coverage: CoverageSection) -> list[str]:
    """The keyword sources the axes need, in KEYWORD_SOURCES order; [] for fixed-only tasks."""
    used = {a.source for a in coverage.axes}
    return [s for s in KEYWORD_SOURCES if s in used]


@dataclass(frozen=True)
class Combo:
    id: str
    params: dict[str, Any]
    share: float  # product of the axis weights


def cross(axes: Sequence[ResolvedAxis]) -> list[Combo]:
    """Every combination of axis values, in axis order (the last axis varies fastest)."""
    per_axis = [list(zip(a.values, a.weights)) for a in axes]
    combos = []
    for picks in itertools.product(*per_axis):
        share = 1.0
        for _, w in picks:
            share *= w
        combos.append(
            Combo(
                id=CELL_ID_SEP.join(value_id(v) for v, _ in picks),
                params={a.name: v for a, (v, _) in zip(axes, picks)},
                share=share,
            )
        )
    return combos
