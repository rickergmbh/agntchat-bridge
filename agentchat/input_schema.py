"""Typed input schema models for declaring agent input requirements.

Provides dataclasses for building input schemas that describe what structured
inputs an agent needs (destination, dates, budget, star rating, etc.). These
schemas are stored in ``structured_capabilities.input_schema`` and enable
dynamic form rendering on the mobile app.

Usage::

    from agentchat.input_schema import InputSchema, InputField, InputOption, InputGroup

    schema = InputSchema(
        fields=[
            InputField(
                key="destination",
                label="Destination",
                type="text",
                required=True,
                description="City or region to search",
                placeholder="e.g., Berlin",
                group="location",
                priority=1,
            ),
            InputField(
                key="star_rating",
                label="Minimum Stars",
                type="select",
                options=[
                    InputOption(value="3", label="3+ Stars"),
                    InputOption(value="4", label="4+ Stars"),
                    InputOption(value="5", label="5 Stars Only"),
                ],
                default="3",
                group="preferences",
                priority=5,
            ),
        ],
        groups=[
            InputGroup(key="location", label="Where", order=1),
            InputGroup(key="preferences", label="Preferences", order=2),
        ],
    )

    # Serialize for API
    schema_dict = schema.to_dict()

    # Extract task input values with defaults applied
    values = schema.extract_values(task_metadata.get("input_values", {}))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


VALID_TYPES = frozenset({
    "text", "number", "date", "date_range", "select", "multi_select",
    "boolean", "location", "price_range", "rating_range",
})


@dataclass
class InputOption:
    """A single option for select/multi_select fields."""

    value: str
    label: str

    def validate(self) -> None:
        if not self.value or not isinstance(self.value, str):
            raise ValueError("InputOption.value must be a non-empty string")
        if not self.label or not isinstance(self.label, str):
            raise ValueError("InputOption.label must be a non-empty string")

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return {"value": self.value, "label": self.label}


@dataclass
class InputGroup:
    """A logical group for organizing input fields."""

    key: str
    label: str
    order: int = 0

    def validate(self) -> None:
        if not self.key or not isinstance(self.key, str):
            raise ValueError("InputGroup.key must be a non-empty string")
        if not self.label or not isinstance(self.label, str):
            raise ValueError("InputGroup.label must be a non-empty string")
        if not isinstance(self.order, int):
            raise ValueError("InputGroup.order must be an integer")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"key": self.key, "label": self.label, "order": self.order}


@dataclass
class InputField:
    """A single input field descriptor.

    Describes one piece of structured input an agent needs. The mobile app
    renders appropriate controls based on ``type`` (text input, date picker,
    select chips, toggle, etc.).
    """

    key: str
    type: str  # one of VALID_TYPES
    label: str | None = None
    required: bool = False
    description: str | None = None
    placeholder: str | None = None
    default: Any = None
    options: list[InputOption] | None = None
    group: str | None = None
    priority: int | None = None
    min: float | None = None
    max: float | None = None
    step: float | None = None

    def validate(self) -> None:
        if not self.key or not isinstance(self.key, str):
            raise ValueError("InputField.key must be a non-empty string")
        if self.type not in VALID_TYPES:
            raise ValueError(
                f"InputField.type must be one of {sorted(VALID_TYPES)}, got {self.type!r}"
            )
        if self.type in ("select", "multi_select"):
            if not self.options:
                raise ValueError(
                    f"InputField of type {self.type!r} requires a non-empty options list"
                )
            for opt in self.options:
                opt.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        d: dict[str, Any] = {"key": self.key, "type": self.type}
        if self.label is not None:
            d["label"] = self.label
        if self.required:
            d["required"] = True
        if self.description is not None:
            d["description"] = self.description
        if self.placeholder is not None:
            d["placeholder"] = self.placeholder
        if self.default is not None:
            d["default"] = self.default
        if self.options is not None:
            d["options"] = [o.to_dict() for o in self.options]
        if self.group is not None:
            d["group"] = self.group
        if self.priority is not None:
            d["priority"] = self.priority
        if self.min is not None:
            d["min"] = self.min
        if self.max is not None:
            d["max"] = self.max
        if self.step is not None:
            d["step"] = self.step
        return d


@dataclass
class InputSchema:
    """A complete input schema declaring all inputs an agent needs.

    Stored in ``structured_capabilities.input_schema`` on the backend.
    """

    fields: list[InputField]
    groups: list[InputGroup] | None = None

    def validate(self) -> None:
        """Validate the entire schema. Raises ValueError on failure."""
        if not self.fields:
            raise ValueError("InputSchema.fields must be a non-empty list")
        for f in self.fields:
            f.validate()
        if self.groups:
            for g in self.groups:
                g.validate()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the dict format expected by the backend API."""
        self.validate()
        d: dict[str, Any] = {"fields": [f.to_dict() for f in self.fields]}
        if self.groups:
            d["groups"] = [g.to_dict() for g in self.groups]
        return d

    def extract_values(self, input_values: dict[str, Any] | None = None) -> dict[str, Any]:
        """Extract input values with defaults applied.

        For each field in the schema, returns the value from ``input_values``
        if present, otherwise the field's ``default``. Fields with no value
        and no default are omitted.
        """
        values: dict[str, Any] = {}
        raw = input_values or {}
        for f in self.fields:
            if f.key in raw:
                values[f.key] = raw[f.key]
            elif f.default is not None:
                values[f.key] = f.default
        return values

    def describe_for_prompt(self, input_values: dict[str, Any] | None = None) -> str:
        """Build a human-readable description of the schema and current values.

        Useful for injecting into an LLM system prompt so the model understands
        what structured inputs are available and their current values.
        """
        values = self.extract_values(input_values)
        lines: list[str] = ["## Structured Inputs"]

        # Group fields
        grouped: dict[str | None, list[InputField]] = {}
        for f in sorted(self.fields, key=lambda x: (x.priority or 999, x.key)):
            grouped.setdefault(f.group, []).append(f)

        group_labels: dict[str, str] = {}
        if self.groups:
            for g in sorted(self.groups, key=lambda x: x.order):
                group_labels[g.key] = g.label

        rendered_groups: list[str] = []
        # Render grouped fields first, then ungrouped
        if self.groups:
            for g in sorted(self.groups, key=lambda x: x.order):
                if g.key in grouped:
                    rendered_groups.append(g.key)
                    lines.append(f"\n### {g.label}")
                    for f in grouped[g.key]:
                        lines.append(self._describe_field(f, values))

        # Ungrouped fields
        for group_key, fields in grouped.items():
            if group_key not in rendered_groups:
                if group_key is not None and group_key in group_labels:
                    continue  # already rendered
                for f in fields:
                    lines.append(self._describe_field(f, values))

        return "\n".join(lines)

    @staticmethod
    def _describe_field(f: InputField, values: dict[str, Any]) -> str:
        """Render a single field as a prompt line."""
        label = f.label or f.key
        val = values.get(f.key)
        req = " (required)" if f.required else ""
        val_str = f" = {val!r}" if val is not None else " = not provided"

        parts = [f"- **{label}**{req}{val_str}"]
        if f.description:
            parts.append(f"  {f.description}")
        if f.options:
            opts = ", ".join(o.label for o in f.options)
            parts.append(f"  Options: {opts}")
        return "\n".join(parts)
