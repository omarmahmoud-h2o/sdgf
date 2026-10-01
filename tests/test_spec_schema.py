import copy

import pytest

from sdgf.spec.schema import (
    OPTIONAL_SECTIONS,
    REQUIRED_SECTIONS,
    SpecValidationError,
    TaskSpec,
    parse_spec,
)

MINIMAL = {
    "task": {
        "name": "toy",
        "version": "0.1",
        "type": "classification_spans",
        "generation_mode": "label_first",
        "description": "Toy task for schema tests.",
    },
    "output_schema": {},
    "rubric": {"verdict": {"values": ["pass", "fail"]}},
    "seeds": {"path": "seeds.jsonl"},
    "coverage": {
        "target_size": 10,
        "axes": [{"name": "label", "values": [True, False]}],
    },
    "models": {"generator": {"backend": "mock", "model": "mock-1"}},
    "validation": {"layers": ["L1", "L2"]},
    "thresholds": {},
}


def spec(**overrides):
    data = copy.deepcopy(MINIMAL)
    for key, value in overrides.items():
        data[key] = value
    return data


def error_paths(data):
    with pytest.raises(SpecValidationError) as exc:
        parse_spec(data)
    return [path for path, _ in exc.value.errors], str(exc.value)


def test_minimal_spec_is_valid():
    s = parse_spec(copy.deepcopy(MINIMAL))
    assert isinstance(s, TaskSpec)
    assert s.task.generation_mode == "label_first"
    assert s.tools == []
    assert s.hitl.review_flagged is False
    assert s.budget.max_tokens is None
    assert s.thresholds.governance_violations_max == 0


def test_spec_is_frozen():
    s = parse_spec(copy.deepcopy(MINIMAL))
    with pytest.raises(Exception):
        s.task.name = "other"


@pytest.mark.parametrize("section", REQUIRED_SECTIONS)
def test_missing_required_section_names_it(section):
    data = copy.deepcopy(MINIMAL)
    del data[section]
    paths, message = error_paths(data)
    assert paths == [section]
    assert section in message and "Field required" in message


@pytest.mark.parametrize("section", OPTIONAL_SECTIONS)
def test_optional_sections_may_be_omitted(section):
    data = copy.deepcopy(MINIMAL)
    data.pop(section, None)
    parse_spec(data)


def test_non_mapping_rejected():
    with pytest.raises(SpecValidationError):
        parse_spec(["not", "a", "mapping"])


def test_unknown_key_named():
    data = spec()
    data["task"]["genration_mode"] = "label_first"
    paths, _ = error_paths(data)
    assert "task.genration_mode" in paths


def test_bad_generation_mode_named():
    data = spec()
    data["task"]["generation_mode"] = "whatever"
    paths, _ = error_paths(data)
    assert paths == ["task.generation_mode"]


def test_nested_list_field_named():
    data = spec()
    data["coverage"]["axes"].append({"name": "stance"})  # fixed axis without values
    paths, message = error_paths(data)
    assert paths == ["coverage.axes[1]"]
    assert "values" in message


def test_axis_weights_must_match_values():
    data = spec()
    data["coverage"]["axes"][0]["weights"] = [1.0]
    paths, _ = error_paths(data)
    assert paths == ["coverage.axes[0]"]


def test_balance_must_sum_to_one_and_name_known_axis():
    data = spec()
    data["coverage"]["balance"] = {"label": {"True": 0.6, "False": 0.6}}
    error_paths(data)
    data["coverage"]["balance"] = {"nope": {"a": 1.0}}
    _, message = error_paths(data)
    assert "unknown axis" in message
    data["coverage"]["balance"] = {"label": {"True": 0.5, "False": 0.5}}
    parse_spec(data)


def test_rubric_criterion_scale_limits():
    data = spec()
    data["rubric"]["criteria"] = [{"name": "realism", "min": 0, "max": 255}]
    paths, message = error_paths(data)
    assert paths == ["rubric.criteria[0]"] and "255" in message
    data["rubric"]["criteria"] = [{"name": "realism", "min": 1, "max": 5}]
    parse_spec(data)
    data["rubric"]["criteria"] = [{"name": "r", "min": 1, "max": 5, "values": ["a", "b"]}]
    error_paths(data)


@pytest.mark.parametrize(
    "example, needle",
    [
        ({"record": {"messages": []}, "verdict": "maybe"}, "not a verdict value"),
        ({"record": {"messages": []}, "verdict": "pass", "scores": {"nope": 1}}, "unknown"),
        ({"record": {"messages": []}, "verdict": "pass", "scores": {"realism": 9}}, "9"),
        ({"record": {"messages": []}, "verdict": "pass", "scores": {"realism": True}}, "True"),
        ({"record": {}, "verdict": "pass"}, "record"),
    ],
)
def test_rubric_examples_must_fit_the_rubric(example, needle):
    data = spec()
    data["rubric"]["criteria"] = [{"name": "realism", "min": 1, "max": 5}]
    data["rubric"]["examples"] = [example]
    paths, message = error_paths(data)
    assert paths[0].startswith("rubric") and needle in message
    data["rubric"]["examples"] = [{**example, "verdict": "pass", "scores": {"realism": 3}}]
    if example["record"]:
        assert parse_spec(data).rubric.examples[0].scores == {"realism": 3}


def test_layers_must_be_ordered_and_known():
    data = spec(validation={"layers": ["L2", "L1"]})
    paths, _ = error_paths(data)
    assert paths == ["validation.layers"]
    data = spec(validation={"layers": ["L1", "L7"]})
    paths, _ = error_paths(data)
    assert paths == ["validation.layers[1]"]


def test_l5_requires_judge_model():
    data = spec(validation={"layers": ["L1", "L2", "L5"]})
    _, message = error_paths(data)
    assert "models.judge" in message
    data["models"]["judge"] = {"backend": "mock", "model": "judge-1"}
    parse_spec(data)


def test_reason_required_needs_fallback_judge():
    data = spec()
    data["rubric"]["reason_required"] = "flagged"
    _, message = error_paths(data)
    assert "fallback_judge" in message


def test_governance_violations_threshold_cannot_loosen():
    data = spec(thresholds={"governance_violations_max": 1})
    paths, _ = error_paths(data)
    assert paths == ["thresholds.governance_violations_max"]


def test_unset_thresholds_reported():
    s = parse_spec(spec(thresholds={"fidelity_min": 0.95}))
    unset = s.thresholds.unset()
    assert "fidelity_min" not in unset
    assert "kappa_min" in unset
    assert "governance_violations_max" not in unset


def test_bad_pii_pattern_named():
    data = spec(governance={"extra_pii_patterns": {"acct": "([0-9"}})
    paths, message = error_paths(data)
    assert paths == ["governance.extra_pii_patterns"] and "acct" in message


def test_turn_structure_first_role_must_be_a_role():
    data = spec(output_schema={"turns": {"roles": ["customer", "assistant"], "first_role": "bot"}})
    paths, _ = error_paths(data)
    assert paths == ["output_schema.turns"]


def test_duplicate_tools_rejected():
    data = spec(tools=[{"name": "calc"}, {"name": "calc"}])
    paths, _ = error_paths(data)
    assert paths == ["tools"]


def test_multiple_errors_all_reported():
    data = spec()
    data["coverage"]["target_size"] = 0
    data["models"]["generator"]["temperature"] = 5
    paths, _ = error_paths(data)
    assert set(paths) == {"coverage.target_size", "models.generator.temperature"}
