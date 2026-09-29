from pathlib import Path

import jsonschema
import pytest

from sdgf.judge.interface import (
    Judge,
    JudgeError,
    JudgeField,
    JudgeParseError,
    JudgeResult,
    JudgeSchema,
    RubricCompileError,
    compile_rubric,
)
from sdgf.spec.compile import compile_spec
from sdgf.spec.schema import RubricSection

FAG_TASK = Path(__file__).resolve().parents[1] / "tasks" / "fag" / "task.yaml"


def rubric(reason="never", criteria=None):
    return RubricSection(
        verdict={"values": ["yes", "no"], "description": "is it?"},
        criteria=criteria
        if criteria is not None
        else [
            {"name": "tier", "values": ["A", "B", "C"], "description": "tier"},
            {"name": "realism", "min": 1, "max": 5},
        ],
        reason_required=reason,
    )


def good(**over):
    d = {
        "verdict": "yes",
        "scores": {"tier": "B", "realism": 4},
        "confidence": {"verdict": 0.9, "tier": 0.8, "realism": 0.5},
    }
    d.update(over)
    return d


# ── compilation ──────────────────────────────────────────────────


def test_compile_enum_and_int_criteria():
    s = compile_rubric(rubric())
    assert s.verdict.kind == "enum" and s.verdict_values == ("yes", "no")
    assert s.verdict.description == "is it?"
    tier, realism = s.criteria
    assert tier.kind == "enum" and tier.choices == ("A", "B", "C") and tier.size == 3
    assert realism.kind == "int" and (realism.min, realism.max) == (1, 5)
    assert realism.values() == (1, 2, 3, 4, 5)
    assert s.field_names == ("verdict", "tier", "realism")
    assert s.criterion("realism") is realism
    with pytest.raises(KeyError):
        s.criterion("nope")


def test_json_schema_shape_without_reason():
    js = compile_rubric(rubric()).json_schema()
    jsonschema.Draft202012Validator.check_schema(js)
    assert js["properties"]["verdict"]["enum"] == ["yes", "no"]
    assert js["properties"]["scores"]["properties"]["tier"]["enum"] == ["A", "B", "C"]
    assert js["properties"]["scores"]["properties"]["realism"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 5,
    }
    conf = js["properties"]["confidence"]
    assert conf["required"] == ["verdict", "tier", "realism"]
    assert conf["properties"]["tier"] == {"type": "number", "minimum": 0.0, "maximum": 1.0}
    assert "reason" not in js["properties"]
    assert js["additionalProperties"] is False


def test_reason_only_when_rubric_requires_it():
    never = compile_rubric(rubric("never")).json_schema()
    flagged = compile_rubric(rubric("flagged")).json_schema()
    always = compile_rubric(rubric("always")).json_schema()
    assert "reason" not in never["properties"]
    assert "reason" in flagged["properties"] and "reason" not in flagged["required"]
    assert "reason" in always["properties"] and "reason" in always["required"]


def test_decision_schema_has_no_reason_even_when_required():
    s = compile_rubric(rubric("always"))
    assert "reason" not in s.json_schema(with_reason=False)["properties"]
    forced = compile_rubric(rubric("never")).json_schema(with_reason=True)
    assert "reason" in forced["required"]


def test_needs_reason():
    assert not compile_rubric(rubric("never")).needs_reason(flagged=True)
    s = compile_rubric(rubric("flagged"))
    assert s.needs_reason(flagged=True) and not s.needs_reason()
    assert compile_rubric(rubric("always")).needs_reason()


def test_no_criteria():
    s = compile_rubric(rubric(criteria=[]))
    assert s.field_names == ("verdict",)
    assert s.parse({"verdict": "no", "scores": {}, "confidence": {"verdict": 0.7}}).verdict == "no"


def test_255_choice_limit():
    JudgeField("x", "enum", choices=tuple(str(i) for i in range(255)))
    JudgeField("x", "int", min=0, max=254)
    with pytest.raises(RubricCompileError, match="255"):
        JudgeField("x", "enum", choices=tuple(str(i) for i in range(256)))
    with pytest.raises(RubricCompileError, match="255"):
        JudgeField("x", "int", min=0, max=255)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "enum", "choices": ("a",)},
        {"kind": "enum", "choices": ("a", "a")},
        {"kind": "int", "min": 3, "max": 3},
        {"kind": "int", "min": 1},
        {"kind": "float", "choices": ("a", "b")},
    ],
)
def test_bad_fields_rejected(kwargs):
    with pytest.raises(RubricCompileError):
        JudgeField("x", **kwargs)


def test_bad_schemas_rejected():
    v = JudgeField("verdict", "enum", choices=("a", "b"))
    c = JudgeField("c", "int", min=1, max=3)
    with pytest.raises(RubricCompileError, match="verdict must be an enum"):
        JudgeSchema(JudgeField("verdict", "int", min=0, max=1))
    with pytest.raises(RubricCompileError, match="repeat"):
        JudgeSchema(v, (c, c))
    with pytest.raises(RubricCompileError, match="can't be named"):
        JudgeSchema(v, (JudgeField("verdict", "int", min=1, max=3),))
    with pytest.raises(RubricCompileError, match="reason policy"):
        JudgeSchema(v, (), "sometimes")


def test_fag_rubric_compiles():
    s = compile_rubric(compile_spec(FAG_TASK).spec.rubric)
    assert s.verdict_values == ("breach", "no_breach")
    assert s.field_names == ("verdict", "advice_tier", "realism")
    assert s.criterion("advice_tier").choices == (
        "FACTUAL_INFORMATION",
        "GENERAL_ADVICE",
        "PERSONAL_ADVICE",
    )
    assert s.reason_required == "never"
    assert "reason" not in s.json_schema()["properties"]


# ── parsing ──────────────────────────────────────────────────────


def test_parse_valid():
    r = compile_rubric(rubric()).parse(good())
    assert r == JudgeResult(
        "yes", {"tier": "B", "realism": 4}, {"verdict": 0.9, "tier": 0.8, "realism": 0.5}
    )
    assert r.verdict_confidence == 0.9 and r.min_confidence() == 0.5
    assert r.reason is None and "reason" not in r.to_dict()


@pytest.mark.parametrize(
    "bad, where",
    [
        (good(verdict="maybe"), "verdict"),
        (good(scores={"tier": "Z", "realism": 4}), "scores.tier"),
        (good(scores={"tier": "B", "realism": 6}), "scores.realism"),
        (good(scores={"tier": "B", "realism": 4.0}), "scores.realism"),
        (good(scores={"tier": "B", "realism": True}), "scores.realism"),
        (good(scores={"tier": "B"}), "scores"),
        (good(scores={"tier": "B", "realism": 4, "extra": 1}), "scores"),
        (good(confidence={"verdict": 0.9, "tier": 0.8}), "confidence"),
        (good(confidence={"verdict": 1.5, "tier": 0.8, "realism": 0.5}), "confidence.verdict"),
        (good(confidence={"verdict": True, "tier": 0.8, "realism": 0.5}), "confidence.verdict"),
        (good(reason="because"), "<root>"),
        ({"scores": {}, "confidence": {}}, "<root>"),
        ("yes", "<root>"),
    ],
)
def test_parse_rejects(bad, where):
    with pytest.raises(JudgeParseError) as e:
        compile_rubric(rubric()).parse(bad)
    assert any(err.startswith(where) for err in e.value.errors), e.value.errors


def test_parse_reason_policies():
    flagged = compile_rubric(rubric("flagged"))
    assert flagged.parse(good()).reason is None
    assert flagged.parse(good(reason="because")).reason == "because"
    always = compile_rubric(rubric("always"))
    with pytest.raises(JudgeParseError, match="reason"):
        always.parse(good())
    assert always.parse(good(reason="because")).to_dict()["reason"] == "because"
    # a decision-model answer is parsed without a reason even when the rubric wants one
    assert always.parse(good(), with_reason=False).reason is None


def test_with_reason_keeps_decision():
    r = compile_rubric(rubric()).parse(good())
    r2 = r.with_reason("because")
    assert r2.reason == "because" and r2.verdict == r.verdict and r2.scores == r.scores


# ── Judge ABC ────────────────────────────────────────────────────


class FixedJudge(Judge):
    name = "fixed"

    def judge(self, record):
        return self.schema.parse(good())


def test_judge_abc_and_explain_default():
    j = FixedJudge(compile_rubric(rubric()))
    assert j.judge({}).verdict == "yes"
    assert j.writes_reasons is False
    with pytest.raises(JudgeError, match="fallback"):
        j.explain({}, j.judge({}))
    with pytest.raises(TypeError):
        Judge(compile_rubric(rubric()))
