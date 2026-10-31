"""--backends plugins for tests/test_cli.py: MockBackend worlds, loaded by `sdgf ... --backends
cli_backends:<name>` in a subprocess. No model or API calls."""

import json
import time

from test_m4_checkpoint import World
from test_pipeline import recipe_from_prompt

# The cell cli_backends:fag_broken never fills: quota 2 at target size 20.
BROKEN = {"product_scope": "corps_act", "label": True, "conversation_length": "single_turn"}


class BrokenWorld(World):
    """Unparseable output for every candidate in BROKEN's cell."""

    def generate(self, call) -> str:
        recipe = recipe_from_prompt(call.prompt)
        if all(recipe.get(k) == v for k, v in BROKEN.items()):
            return "not json at all"
        return super().generate(call)


class UnsureWorld(World):
    """A judge whose every third verdict is below validation.escalation.low_confidence."""

    def __init__(self):
        super().__init__(advise_every=0)
        self.verdicts = 0

    def judge(self, call) -> str:
        self.verdicts += 1
        reply = json.loads(super().judge(call))
        if self.verdicts % 3 == 0:
            reply["confidence"]["verdict"] = 0.4
        return json.dumps(reply)


class SlowWorld(World):
    """World with a generator that takes SLOW_SECONDS a call, so a job is still running
    when tests/test_jobs.py cancels, times out or kills it."""

    SLOW_SECONDS = 0.15

    def generate(self, call) -> str:
        time.sleep(self.SLOW_SECONDS)
        return super().generate(call)


def fag_world(compiled):
    return World().backends()


def fag_broken(compiled):
    return BrokenWorld().backends()


def fag_unsure(compiled):
    return UnsureWorld().backends()


def fag_slow(compiled):
    return SlowWorld().backends()


not_callable = "not a factory"
