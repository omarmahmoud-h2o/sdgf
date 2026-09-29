import random
import textwrap

import pytest

from sdgf.spec.hooks import (
    HOOK_SIGNATURES,
    ExtraValidators,
    HookError,
    LabelRule,
    PostProcess,
    SamplerConstraints,
    TaskHooks,
    check_signature,
    load_hooks,
)

ALL_HOOKS = """
def label_rule(record):
    return record["tier"] >= 2

def sampler_constraints(cell, rng):
    if cell.get("invalid"):
        return None
    return {**cell, "pick": rng.choice(["a", "b", "c"])}

def extra_validators(record):
    return [] if record.get("ok") else ["not ok"]

def post_process(record):
    return {**record, "derived": True}
"""


def _write(tmp_path, body: str):
    path = tmp_path / "hooks.py"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_loads_all_four_hooks(tmp_path):
    hooks = load_hooks(_write(tmp_path, ALL_HOOKS))
    assert hooks.present() == list(HOOK_SIGNATURES)
    assert hooks.label_rule({"tier": 3}) is True
    assert hooks.extra_validators({"ok": False}) == ["not ok"]
    assert hooks.post_process({"a": 1}) == {"a": 1, "derived": True}
    assert hooks.sampler_constraints({"invalid": True}, random.Random(0)) is None
    a = hooks.sampler_constraints({}, random.Random(7))
    b = hooks.sampler_constraints({}, random.Random(7))
    assert a == b
    assert hooks.source == textwrap.dedent(ALL_HOOKS)


def test_hooks_satisfy_protocols(tmp_path):
    hooks = load_hooks(_write(tmp_path, ALL_HOOKS))
    assert isinstance(hooks.label_rule, LabelRule)
    assert isinstance(hooks.sampler_constraints, SamplerConstraints)
    assert isinstance(hooks.extra_validators, ExtraValidators)
    assert isinstance(hooks.post_process, PostProcess)


def test_task_directory_accepted(tmp_path):
    _write(tmp_path, ALL_HOOKS)
    assert load_hooks(tmp_path).present() == list(HOOK_SIGNATURES)


def test_subset_of_hooks(tmp_path):
    hooks = load_hooks(_write(tmp_path, "def label_rule(record):\n    return 1\n"))
    assert hooks.present() == ["label_rule"]
    assert hooks.post_process is None


def test_missing_file_is_empty(tmp_path):
    assert load_hooks(tmp_path / "hooks.py") == TaskHooks()
    assert load_hooks(tmp_path).present() == []
    assert load_hooks(None).source == ""


def test_non_hook_names_ignored(tmp_path):
    hooks = load_hooks(_write(tmp_path, "def helper(a, b, c):\n    return 0\n"))
    assert hooks.present() == []


@pytest.mark.parametrize(
    "body, name",
    [
        ("def label_rule():\n    return 1\n", "label_rule"),
        ("def label_rule(record, extra):\n    return 1\n", "label_rule"),
        ("def sampler_constraints(cell):\n    return cell\n", "sampler_constraints"),
        ("def extra_validators(record, *, strict):\n    return []\n", "extra_validators"),
        ("post_process = 42\n", "post_process"),
    ],
)
def test_wrong_signature_raises_naming_hook(tmp_path, body, name):
    with pytest.raises(HookError, match=name):
        load_hooks(_write(tmp_path, body))


def test_optional_extras_allowed():
    def label_rule(record, verbose=False, *, debug=False):
        return 1

    def post_process(*args):
        return args[0]

    check_signature("label_rule", label_rule)
    check_signature("post_process", post_process)


def test_import_error_wrapped(tmp_path):
    with pytest.raises(HookError, match="ZeroDivisionError"):
        load_hooks(_write(tmp_path, "x = 1 / 0\n"))
