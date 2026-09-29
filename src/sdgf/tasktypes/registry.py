"""Task-type registry keyed by name (FRAMEWORK_DESIGN.md §7.1, D1).

A spec declares its type in task.type; resolve() looks it up and checks that the
spec's generation_mode is one the type supports. New types are registered here and
the core pipeline doesn't change.
"""

from __future__ import annotations

from sdgf.spec.schema import TaskSection
from sdgf.tasktypes.base import TaskType, TaskTypeError


class UnknownTaskTypeError(TaskTypeError, KeyError):
    """No task type is registered under the requested name."""

    def __str__(self) -> str:  # KeyError would repr() the message
        return str(self.args[0]) if self.args else ""


class TaskTypeRegistry:
    def __init__(self) -> None:
        self._types: dict[str, TaskType] = {}

    def register(self, task_type: TaskType, *, replace: bool = False) -> TaskType:
        task_type.check_definition()
        if task_type.name in self._types and not replace:
            raise TaskTypeError(f"task type {task_type.name!r} is already registered")
        self._types[task_type.name] = task_type
        return task_type

    def get(self, name: str) -> TaskType:
        try:
            return self._types[name]
        except KeyError:
            known = ", ".join(sorted(self._types)) or "none"
            raise UnknownTaskTypeError(
                f"unknown task type {name!r}; registered task types: {known}"
            ) from None

    def resolve(self, task: TaskSection) -> TaskType:
        """The task type for a spec's task section, with its generation_mode checked."""
        task_type = self.get(task.type)
        task_type.check_mode(task.generation_mode)
        return task_type

    def names(self) -> list[str]:
        return sorted(self._types)

    def __contains__(self, name: object) -> bool:
        return name in self._types


REGISTRY = TaskTypeRegistry()


def register_task_type(task_type: TaskType, *, replace: bool = False) -> TaskType:
    return REGISTRY.register(task_type, replace=replace)


def get_task_type(name: str) -> TaskType:
    return REGISTRY.get(name)
