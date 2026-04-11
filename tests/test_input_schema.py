"""Tests for agentchat.input_schema dataclasses."""

import pytest
import sys
import os

# Add SDK to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentchat.input_schema import (
    InputField,
    InputGroup,
    InputOption,
    InputSchema,
    VALID_TYPES,
)


class TestInputOption:
    def test_valid_option(self):
        opt = InputOption(value="3", label="3+ Stars")
        d = opt.to_dict()
        assert d == {"value": "3", "label": "3+ Stars"}

    def test_empty_value_rejected(self):
        opt = InputOption(value="", label="Test")
        with pytest.raises(ValueError, match="value"):
            opt.validate()

    def test_empty_label_rejected(self):
        opt = InputOption(value="x", label="")
        with pytest.raises(ValueError, match="label"):
            opt.validate()


class TestInputGroup:
    def test_valid_group(self):
        g = InputGroup(key="loc", label="Location", order=1)
        d = g.to_dict()
        assert d == {"key": "loc", "label": "Location", "order": 1}

    def test_default_order(self):
        g = InputGroup(key="x", label="X")
        assert g.order == 0

    def test_empty_key_rejected(self):
        g = InputGroup(key="", label="X")
        with pytest.raises(ValueError, match="key"):
            g.validate()


class TestInputField:
    def test_text_field(self):
        f = InputField(key="dest", type="text", label="Destination", required=True)
        d = f.to_dict()
        assert d["key"] == "dest"
        assert d["type"] == "text"
        assert d["label"] == "Destination"
        assert d["required"] is True

    def test_number_field_with_min_max(self):
        f = InputField(key="budget", type="number", min=0, max=10000, step=10)
        d = f.to_dict()
        assert d["min"] == 0
        assert d["max"] == 10000
        assert d["step"] == 10

    def test_select_field(self):
        f = InputField(
            key="stars",
            type="select",
            options=[
                InputOption(value="3", label="3+"),
                InputOption(value="4", label="4+"),
            ],
        )
        d = f.to_dict()
        assert len(d["options"]) == 2
        assert d["options"][0]["value"] == "3"

    def test_select_without_options_rejected(self):
        f = InputField(key="x", type="select")
        with pytest.raises(ValueError, match="options"):
            f.validate()

    def test_multi_select_without_options_rejected(self):
        f = InputField(key="x", type="multi_select")
        with pytest.raises(ValueError, match="options"):
            f.validate()

    def test_invalid_type_rejected(self):
        f = InputField(key="x", type="color_picker")
        with pytest.raises(ValueError, match="color_picker"):
            f.validate()

    def test_all_types_accepted(self):
        for t in VALID_TYPES:
            kwargs = {"key": "test", "type": t}
            if t in ("select", "multi_select"):
                kwargs["options"] = [InputOption(value="a", label="A")]
            f = InputField(**kwargs)
            f.validate()  # should not raise

    def test_optional_fields_omitted(self):
        f = InputField(key="q", type="text")
        d = f.to_dict()
        assert "label" not in d
        assert "required" not in d
        assert "description" not in d
        assert "options" not in d
        assert "group" not in d
        assert "priority" not in d
        assert "min" not in d

    def test_default_value(self):
        f = InputField(key="sort", type="text", default="rating")
        d = f.to_dict()
        assert d["default"] == "rating"


class TestInputSchema:
    def test_valid_schema(self):
        schema = InputSchema(
            fields=[
                InputField(key="dest", type="text", required=True),
                InputField(key="budget", type="number"),
            ],
            groups=[InputGroup(key="main", label="Main", order=1)],
        )
        d = schema.to_dict()
        assert len(d["fields"]) == 2
        assert len(d["groups"]) == 1

    def test_schema_without_groups(self):
        schema = InputSchema(fields=[InputField(key="q", type="text")])
        d = schema.to_dict()
        assert "groups" not in d

    def test_empty_fields_rejected(self):
        schema = InputSchema(fields=[])
        with pytest.raises(ValueError, match="non-empty"):
            schema.validate()

    def test_extract_values_with_defaults(self):
        schema = InputSchema(
            fields=[
                InputField(key="dest", type="text"),
                InputField(key="stars", type="select", default="3",
                           options=[InputOption(value="3", label="3+")]),
                InputField(key="budget", type="number", default=200),
            ],
        )
        values = schema.extract_values({"dest": "Berlin"})
        assert values == {"dest": "Berlin", "stars": "3", "budget": 200}

    def test_extract_values_override_defaults(self):
        schema = InputSchema(
            fields=[
                InputField(key="stars", type="select", default="3",
                           options=[InputOption(value="3", label="3+")]),
            ],
        )
        values = schema.extract_values({"stars": "5"})
        assert values == {"stars": "5"}

    def test_extract_values_empty(self):
        schema = InputSchema(
            fields=[InputField(key="q", type="text")],
        )
        values = schema.extract_values({})
        assert values == {}

    def test_extract_values_none(self):
        schema = InputSchema(
            fields=[InputField(key="q", type="text", default="hello")],
        )
        values = schema.extract_values(None)
        assert values == {"q": "hello"}

    def test_describe_for_prompt(self):
        schema = InputSchema(
            fields=[
                InputField(
                    key="dest", type="text", label="Destination",
                    required=True, description="City to search",
                ),
                InputField(
                    key="stars", type="select", label="Stars",
                    options=[
                        InputOption(value="3", label="3+"),
                        InputOption(value="5", label="5 Only"),
                    ],
                    default="3",
                ),
            ],
        )
        prompt = schema.describe_for_prompt({"dest": "Berlin"})
        assert "Destination" in prompt
        assert "Berlin" in prompt
        assert "(required)" in prompt
        assert "Stars" in prompt
        assert "3+" in prompt

    def test_describe_for_prompt_with_groups(self):
        schema = InputSchema(
            fields=[
                InputField(key="dest", type="text", label="Destination", group="loc"),
                InputField(key="budget", type="number", label="Budget", group="prefs"),
            ],
            groups=[
                InputGroup(key="loc", label="Location", order=1),
                InputGroup(key="prefs", label="Preferences", order=2),
            ],
        )
        prompt = schema.describe_for_prompt({})
        assert "### Location" in prompt
        assert "### Preferences" in prompt
