import pytest

from sdgf.spec.schema import ValidationSection
from sdgf.store.provenance import ProvenanceBuilder
from sdgf.validate.base import (
    Layer,
    LayerVerdict,
    ValidationContext,
    ValidationError,
    ValidationIssue,
)
from sdgf.validate.cascade import Cascade, CascadeError

ISSUE = ValidationIssue("span_not_verbatim", "span not found verbatim in turn 4", "spans[0].text")


class FakeLayer(Layer):
    """Passes, or fails with ISSUE; logs each call into a shared list."""

    def __init__(self, name, log, outcome="pass"):
        self.name = name
        self.log = log
        self.outcome = outcome

    def check(self, record, context):
        self.log.append(self.name)
        if self.outcome == "pass":
            return self.verdict()
        return self.verdict([ISSUE], repairable=self.outcome == "fail_repairable")


def layers(log, **outcomes):
    names = ["L1", "L2", "L3", "L4", "L5", "L6"]
    return [FakeLayer(n, log, outcomes.get(n, "pass")) for n in names]


# ── base ─────────────────────────────────────────────────────────


def test_issue_str_and_dict():
    assert str(ISSUE) == "span_not_verbatim at spans[0].text: span not found verbatim in turn 4"
    assert str(ValidationIssue("x", "bad")) == "x: bad"
    assert ISSUE.to_dict() == {
        "code": "span_not_verbatim",
        "message": "span not found verbatim in turn 4",
        "path": "spans[0].text",
    }
    assert ValidationIssue("x", "m", details={"turn": 4}).to_dict()["details"] == {"turn": 4}


def test_issue_needs_code():
    with pytest.raises(ValidationError):
        ValidationIssue("", "no code")


def test_verdict_helper_outcomes():
    layer = FakeLayer("L2", [])
    assert layer.verdict().outcome == "pass"
    assert layer.verdict([ISSUE]).outcome == "fail_repairable"
    assert layer.verdict([ISSUE], repairable=False).outcome == "fail_hard"
    v = layer.verdict([ISSUE])
    assert v.repairable and not v.hard and not v.passed
    assert v.codes == ("span_not_verbatim",)
    assert v.messages() == (str(ISSUE),)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"layer": "L9", "outcome": "pass"},
        {"layer": "L1", "outcome": "maybe"},
        {"layer": "L1", "outcome": "pass", "errors": (ISSUE,)},
        {"layer": "L1", "outcome": "fail_hard"},
    ],
)
def test_malformed_verdicts_raise(kwargs):
    with pytest.raises(ValidationError):
        LayerVerdict(**kwargs)


# ── ordering ─────────────────────────────────────────────────────


def test_all_pass_runs_every_layer_in_order():
    log = []
    result = Cascade(layers(log)).run({"x": 1})
    assert result.passed and result.outcome == "pass"
    assert log == ["L1", "L2", "L3", "L4", "L5", "L6"]
    assert result.layers_run == tuple(log)
    assert result.failed_layer is None and result.errors == ()


def test_from_config_uses_configured_subset_and_order():
    log = []
    cascade = Cascade.from_config(ValidationSection(layers=["L1", "L2"]), layers(log))
    assert cascade.names == ("L1", "L2")
    cascade.run({})
    assert log == ["L1", "L2"]


def test_from_config_accepts_mapping_and_name_list():
    log = []
    available = {layer.name: layer for layer in layers(log)}
    assert Cascade.from_config(["L2", "L4"], available).names == ("L2", "L4")


def test_from_config_missing_implementation_raises():
    log = []
    with pytest.raises(CascadeError, match=r"L3"):
        Cascade.from_config(["L1", "L3"], [FakeLayer("L1", log)])


def test_from_config_misnamed_mapping_raises():
    with pytest.raises(CascadeError, match="reports name"):
        Cascade.from_config(["L1"], {"L1": FakeLayer("L2", [])})


@pytest.mark.parametrize("names", [["L2", "L1"], ["L1", "L1"], ["L7"]])
def test_constructor_rejects_bad_order(names):
    with pytest.raises(CascadeError):
        Cascade([FakeLayer(n, []) for n in names])


# ── short-circuit ────────────────────────────────────────────────


@pytest.mark.parametrize("failing", ["L1", "L2", "L3", "L4", "L5", "L6"])
@pytest.mark.parametrize("outcome", ["fail_repairable", "fail_hard"])
def test_stops_at_first_failure(failing, outcome):
    log = []
    result = Cascade(layers(log, **{failing: outcome})).run({})
    order = ["L1", "L2", "L3", "L4", "L5", "L6"]
    assert log == order[: order.index(failing) + 1]
    assert result.failed_layer == failing
    assert result.outcome == outcome
    assert result.repairable == (outcome == "fail_repairable")
    assert result.hard == (outcome == "fail_hard")
    assert result.errors == (ISSUE,)


def test_earliest_failure_wins_when_several_would_fail():
    log = []
    result = Cascade(layers(log, L2="fail_repairable", L3="fail_hard")).run({})
    assert result.failed_layer == "L2" and result.repairable
    assert "L3" not in log


def test_context_reaches_layers():
    seen = []

    class CtxLayer(Layer):
        name = "L1"

        def check(self, record, context):
            seen.append((record, context))
            return self.verdict()

    ctx = ValidationContext(cell_id="c1", recipe={"label": True}, attempt=2)
    Cascade([CtxLayer()]).run({"a": 1}, ctx)
    assert seen == [({"a": 1}, ctx)]
    Cascade([CtxLayer()]).run({})
    assert seen[-1][1] == ValidationContext()


def test_layer_returning_wrong_type_or_name_raises():
    class Bad(Layer):
        name = "L1"

        def __init__(self, ret):
            self.ret = ret

        def check(self, record, context):
            return self.ret

    with pytest.raises(CascadeError, match="returned dict"):
        Cascade([Bad({})]).run({})
    with pytest.raises(CascadeError, match="verdict for L2"):
        Cascade([Bad(LayerVerdict("L2", "pass"))]).run({})


def test_layer_exception_propagates():
    class Boom(Layer):
        name = "L1"

        def check(self, record, context):
            raise RuntimeError("bug")

    with pytest.raises(RuntimeError, match="bug"):
        Cascade([Boom()]).run({})


def test_results_recorded_in_provenance():
    log = []
    builder = ProvenanceBuilder("sv", "c1", 7, [])
    Cascade(layers(log, L2="fail_repairable")).run({}, provenance=builder)
    builder.start_repair()
    Cascade(layers(log)).run({}, provenance=builder)
    results = [(r.layer, r.outcome, r.attempt) for r in builder.layer_results]
    assert results[:2] == [("L1", "pass", 0), ("L2", "fail_repairable", 0)]
    assert results[2:] == [(n, "pass", 1) for n in ["L1", "L2", "L3", "L4", "L5", "L6"]]
    assert builder.layer_results[1].errors == (str(ISSUE),)
