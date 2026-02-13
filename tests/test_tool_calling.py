"""
Unit tests for tool detection, calculator execution, and readmission tool.

Run with:
    pytest tests/test_tool_calling.py

Run only unit-tagged tests:
    pytest tests/test_tool_calling.py -m unit

Run only tests that exercise tool calling (calculator/readmission flow):
    pytest tests/test_tool_calling.py -m tool_calls

Run without tool-call tests:
    pytest tests/test_tool_calling.py -m "not tool_calls"

Run with verbose output:
    pytest tests/test_tool_calling.py -v
"""

import asyncio
import json
import pytest

from meds_mcp.server.api.cohort_chat import (
    _tool_call_to_dict,
    execute_cohort_tool_call,
)
from meds_mcp.server.tools.calculator import (
    calculator_tool,
    execute_tool_call,
    get_calculator_tool_definition,
    is_simple_calculation,
)
from meds_mcp.server.tools.readmission import (
    get_readmission_prediction,
    get_readmission_tool_definition,
)


# ---------------------------------------------------------------------------
# is_simple_calculation (tool detection)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIsSimpleCalculation:
    """Tests for is_simple_calculation (detecting calculation-style queries)."""

    def test_what_is_number(self):
        assert is_simple_calculation("what is 5 + 3") is True
        assert is_simple_calculation("What is 10 * 2") is True

    def test_calculate_number(self):
        assert is_simple_calculation("calculate 100 / 4") is True
        assert is_simple_calculation("Calculate 7 - 2") is True

    def test_compute_number(self):
        assert is_simple_calculation("compute 6 * 7") is True

    def test_bare_expression(self):
        assert is_simple_calculation("2 + 2") is True
        assert is_simple_calculation("10 * 5") is True
        assert is_simple_calculation("100 / 4") is True

    def test_whats_number(self):
        assert is_simple_calculation("what's 3 + 4") is True
        assert is_simple_calculation("whats 8 / 2") is True

    def test_how_much_is(self):
        assert is_simple_calculation("how much is 5 * 5") is True

    def test_not_simple_calculation(self):
        assert is_simple_calculation("what is the patient's blood pressure") is False
        assert is_simple_calculation("calculate patient age") is False
        assert is_simple_calculation("compute BMI for this patient") is False
        assert is_simple_calculation("show me lab results") is False
        assert is_simple_calculation("") is False

    def test_skip_words_override_pattern(self):
        """Queries that match calculation pattern but contain clinical skip words are False."""
        assert is_simple_calculation("what is patient 123") is False
        assert is_simple_calculation("calculate age from dob") is False
        assert is_simple_calculation("compute weight in kg") is False
        assert is_simple_calculation("blood pressure 120/80") is False

    def test_whitespace_only_query(self):
        """Whitespace-only or empty after strip is not a simple calculation."""
        assert is_simple_calculation("   ") is False
        assert is_simple_calculation("  \t  ") is False

    def test_skip_word_result(self):
        """'result' is a skip word (e.g. lab result)."""
        assert is_simple_calculation("what is the result of 5+3") is False

    def test_what_is_digit_no_operator(self):
        """'what is 5' matches pattern and has no skip word -> True."""
        assert is_simple_calculation("what is 5") is True


# ---------------------------------------------------------------------------
# calculator_tool (expression evaluation)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCalculatorTool:
    """Tests for calculator_tool (expression-only evaluation)."""

    def test_basic_arithmetic(self):
        assert calculator_tool("2 + 2") == "4"
        assert calculator_tool("10 - 3") == "7"
        assert calculator_tool("4 * 5") == "20"
        assert calculator_tool("20 / 4") == "5.0"

    def test_parentheses(self):
        assert calculator_tool("(5 + 3) * 2") == "16"
        assert calculator_tool("10 + 10") == "20"

    def test_invalid_characters(self):
        result = calculator_tool("2 + eval('rm -rf')")
        assert "Error" in result
        assert "Invalid characters" in result

    def test_empty_expression(self):
        result = calculator_tool("")
        assert "Error" in result or result != ""

    def test_division_by_zero(self):
        """Division by zero returns an error string from calculator_tool."""
        result = calculator_tool("5 / 0")
        assert "Error" in result

    def test_float_results(self):
        assert calculator_tool("10 / 4") == "2.5"
        assert calculator_tool("1 / 3")  # non-zero result

    def test_negative_numbers(self):
        assert calculator_tool("-5 + 8") == "3"
        assert calculator_tool("3 - 10") == "-7"

    def test_single_number(self):
        assert calculator_tool("42") == "42"
        assert calculator_tool("0") == "0"

    def test_whitespace_only_expression(self):
        """Whitespace-only passes allowed_chars but eval fails."""
        result = calculator_tool("   ")
        assert "Error" in result

    def test_malformed_expression_syntax(self):
        """Expressions that are invalid Python (e.g. missing operand) return error."""
        result = calculator_tool("5 +")
        assert "Error" in result

    def test_expression_with_only_spaces_around_operator(self):
        assert calculator_tool("  2  +  3  ") == "5"


# ---------------------------------------------------------------------------
# execute_tool_call (LLM tool call execution)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExecuteToolCall:
    """Tests for execute_tool_call (tool call dict -> result string)."""

    def test_calculator_valid(self):
        tool_call = {
            "id": "call_abc",
            "function": {
                "name": "calculator",
                "arguments": json.dumps({"expression": "2 + 2"}),
            },
        }
        result = execute_tool_call(tool_call)
        assert result == "4"

    def test_calculator_valid_args_dict(self):
        """Arguments as dict (e.g. from some clients) are accepted."""
        tool_call = {
            "id": "call_xyz",
            "function": {
                "name": "calculator",
                "arguments": {"expression": "10 * 5"},
            },
        }
        result = execute_tool_call(tool_call)
        assert result == "50"

    def test_calculator_invalid_args_json(self):
        tool_call = {
            "id": "call_1",
            "function": {
                "name": "calculator",
                "arguments": "not valid json {{{",
            },
        }
        result = execute_tool_call(tool_call)
        data = json.loads(result)
        assert data["tool"] == "calculator"
        assert "error" in data and "invalid arguments" in data["error"].lower()

    def test_unknown_tool(self):
        tool_call = {
            "id": "call_2",
            "function": {
                "name": "unknown_tool",
                "arguments": "{}",
            },
        }
        result = execute_tool_call(tool_call)
        data = json.loads(result)
        assert data["tool"] == "unknown_tool"
        assert "error" in data and "unknown" in data["error"].lower()

    def test_invalid_tool_call_not_dict(self):
        result = execute_tool_call(None)
        data = json.loads(result)
        assert data["tool"] == "?"
        assert "error" in data and "dict" in data["error"].lower()

        result = execute_tool_call("not a dict")
        data = json.loads(result)
        assert data["tool"] == "?"
        assert "error" in data

    def test_tool_call_missing_function(self):
        tool_call = {"id": "call_3"}
        result = execute_tool_call(tool_call)
        data = json.loads(result)
        assert "tool" in data
        assert "error" in data
        # Missing function -> name is "" -> unknown tool
        assert "unknown" in data["error"].lower()

    def test_calculator_missing_expression_key(self):
        """Args dict without 'expression' key passes empty string to calculator_tool."""
        tool_call = {
            "id": "call_missing",
            "function": {"name": "calculator", "arguments": "{}"},
        }
        result = execute_tool_call(tool_call)
        assert "Error" in result

    def test_calculator_args_empty_string(self):
        """Empty string for arguments is invalid JSON."""
        tool_call = {
            "id": "call_empty",
            "function": {"name": "calculator", "arguments": ""},
        }
        result = execute_tool_call(tool_call)
        data = json.loads(result)
        assert data["tool"] == "calculator"
        assert "error" in data

    def test_calculator_function_name_case_sensitive(self):
        """Only 'calculator' (lowercase) is recognized; 'Calculator' is unknown."""
        tool_call = {
            "id": "call_case",
            "function": {"name": "Calculator", "arguments": '{"expression": "2+2"}'},
        }
        result = execute_tool_call(tool_call)
        data = json.loads(result)
        assert data["tool"] == "Calculator"
        assert "unknown" in data["error"].lower()

    def test_calculator_function_value_none(self):
        """tool_call['function'] is None -> treated as empty, unknown tool."""
        tool_call = {"id": "call_none", "function": None}
        result = execute_tool_call(tool_call)
        data = json.loads(result)
        assert "tool" in data
        assert "error" in data

    def test_calculator_args_non_dict(self):
        """Arguments parse as non-dict (e.g. list) -> expression becomes '' -> calculator error."""
        tool_call = {
            "id": "call_list",
            "function": {"name": "calculator", "arguments": "[]"},
        }
        result = execute_tool_call(tool_call)
        # calculator_tool("") returns an error string (not JSON)
        assert "Error" in result or (result.startswith("{") and "error" in result)


# ---------------------------------------------------------------------------
# get_calculator_tool_definition
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetCalculatorToolDefinition:
    """Tests for get_calculator_tool_definition (OpenAI-format tool schema)."""

    def test_returns_dict(self):
        definition = get_calculator_tool_definition()
        assert isinstance(definition, dict)

    def test_has_type_and_function(self):
        definition = get_calculator_tool_definition()
        assert definition.get("type") == "function"
        assert "function" in definition
        assert definition["function"].get("name") == "calculator"

    def test_has_required_expression_param(self):
        definition = get_calculator_tool_definition()
        params = definition["function"].get("parameters", {})
        assert "properties" in params
        assert "expression" in params["properties"]
        assert "expression" in params.get("required", [])


# ---------------------------------------------------------------------------
# get_readmission_tool_definition
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetReadmissionToolDefinition:
    """Tests for get_readmission_tool_definition (OpenAI-format tool schema)."""

    def test_returns_dict(self):
        definition = get_readmission_tool_definition()
        assert isinstance(definition, dict)

    def test_has_type_and_function(self):
        definition = get_readmission_tool_definition()
        assert definition.get("type") == "function"
        assert "function" in definition
        assert definition["function"].get("name") == "get_readmission_prediction"

    def test_has_required_person_id_param(self):
        definition = get_readmission_tool_definition()
        params = definition["function"].get("parameters", {})
        assert "properties" in params
        assert "person_id" in params["properties"]
        assert "person_id" in params.get("required", [])

    def test_description_mentions_readmission(self):
        definition = get_readmission_tool_definition()
        desc = definition["function"].get("description", "")
        assert "readmission" in desc.lower()


# ---------------------------------------------------------------------------
# get_readmission_prediction (async)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetReadmissionPrediction:
    """Tests for get_readmission_prediction (async lookup from CSV)."""

    def test_csv_not_found_returns_error_dict(self):
        result = asyncio.run(
            get_readmission_prediction(
                "patient_123",
                csv_path="/nonexistent/path/readmission_labels.csv",
            )
        )
        assert "error" in result
        assert result.get("patient_id") == "patient_123"
        assert result.get("readmission") is None

    def test_csv_missing_patient_id_column(self, tmp_path):
        """CSV without 'patient_id' column returns error."""
        csv_file = tmp_path / "bad.csv"
        csv_file.write_text("readmission,other\nno,x\n", encoding="utf-8")
        result = asyncio.run(
            get_readmission_prediction(
                "patient_1",
                csv_path=str(csv_file),
            )
        )
        assert "error" in result
        assert "patient_id" in result["error"].lower() or "column" in result["error"].lower()
        assert result.get("readmission") is None

    def test_patient_found_readmission_yes(self, tmp_path):
        csv_file = tmp_path / "labels.csv"
        csv_file.write_text(
            "patient_id,readmission\npatient_1,yes\npatient_2,no\n",
            encoding="utf-8",
        )
        result = asyncio.run(
            get_readmission_prediction("patient_1", csv_path=str(csv_file))
        )
        assert "error" not in result
        assert result.get("patient_id") == "patient_1"
        assert result.get("readmission") == "yes"
        assert result.get("predicted_readmission") is True

    def test_patient_found_readmission_no(self, tmp_path):
        csv_file = tmp_path / "labels.csv"
        csv_file.write_text(
            "patient_id,readmission\npatient_1,yes\npatient_2,no\n",
            encoding="utf-8",
        )
        result = asyncio.run(
            get_readmission_prediction("patient_2", csv_path=str(csv_file))
        )
        assert "error" not in result
        assert result.get("patient_id") == "patient_2"
        assert result.get("readmission") == "no"
        assert result.get("predicted_readmission") is False

    def test_patient_not_in_csv_returns_error_dict(self, tmp_path):
        csv_file = tmp_path / "labels.csv"
        csv_file.write_text(
            "patient_id,readmission\npatient_1,yes\n",
            encoding="utf-8",
        )
        result = asyncio.run(
            get_readmission_prediction("unknown_id", csv_path=str(csv_file))
        )
        assert "error" in result
        assert "not found" in result["error"].lower()
        assert result.get("patient_id") == "unknown_id"
        assert result.get("readmission") is None

    def test_predicted_readmission_accepts_1_and_true(self, tmp_path):
        """Labels '1' and 'true' (case-insensitive) map to predicted_readmission True."""
        csv_file = tmp_path / "labels.csv"
        csv_file.write_text(
            "patient_id,readmission\na,1\nb,true\nc,TRUE\n",
            encoding="utf-8",
        )
        for pid in ("a", "b", "c"):
            result = asyncio.run(
                get_readmission_prediction(pid, csv_path=str(csv_file))
            )
            assert result.get("predicted_readmission") is True, f"person_id={pid}"

    def test_empty_or_missing_readmission_label(self, tmp_path):
        """Empty or missing readmission column yields predicted_readmission None."""
        csv_file = tmp_path / "labels.csv"
        csv_file.write_text(
            "patient_id,readmission\nempty,\nmissing,\n",
            encoding="utf-8",
        )
        r1 = asyncio.run(
            get_readmission_prediction("empty", csv_path=str(csv_file))
        )
        r2 = asyncio.run(
            get_readmission_prediction("missing", csv_path=str(csv_file))
        )
        assert r1.get("readmission") in (None, "")
        assert r1.get("predicted_readmission") is None
        assert r2.get("readmission") in (None, "")
        assert r2.get("predicted_readmission") is None


# ---------------------------------------------------------------------------
# Tool call flow: LLM calls tool and tool execution works (both tools)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.tool_calls
class TestLlmToolCallFlowCalculator:
    """Verify that when the LLM returns a calculator tool call, we execute it and get the right result."""

    def test_llm_style_calculator_tool_call_executes_and_returns_result(self):
        """Simulate OpenAI-style tool_call from LLM; execute and assert result."""
        # Same structure as choices[0].message.tool_calls[0] (OpenAI API)
        tool_call = {
            "id": "call_calc_001",
            "type": "function",
            "function": {
                "name": "calculator",
                "arguments": json.dumps({"expression": "3 * 4"}),
            },
        }
        result = execute_tool_call(tool_call)
        assert result == "12"

    def test_llm_style_calculator_via_cohort_path(self):
        """Calculator tool call through cohort chat path (execute_cohort_tool_call) works."""
        tool_call = {
            "id": "call_calc_002",
            "type": "function",
            "function": {
                "name": "calculator",
                "arguments": json.dumps({"expression": "10 + 5"}),
            },
        }
        result = asyncio.run(
            execute_cohort_tool_call(tool_call, patient_ids=["patient_1"])
        )
        assert result == "15"

    def test_llm_message_tool_calls_normalized_then_executed(self):
        """Simulate full flow: message with tool_calls -> _tool_call_to_dict -> execute_tool_call."""
        # As if from response.choices[0].message
        message = {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_xyz",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": json.dumps({"expression": "100 / 4"}),
                    },
                },
            ],
        }
        tool_calls_list = message["tool_calls"]
        tc_dict = _tool_call_to_dict(tool_calls_list[0])
        assert tc_dict["function"]["name"] == "calculator"
        result = execute_tool_call(tc_dict)
        assert result == "25.0"


@pytest.mark.unit
@pytest.mark.tool_calls
class TestLlmToolCallFlowReadmission:
    """Verify that when the LLM returns a readmission tool call, we execute it and get the right result."""

    def test_llm_style_readmission_tool_call_executes_and_returns_result(
        self, tmp_path, monkeypatch
    ):
        """Simulate LLM tool call for get_readmission_prediction; execute via cohort path with temp CSV."""
        csv_file = tmp_path / "readmission_labels.csv"
        csv_file.write_text(
            "patient_id,readmission\ncohort_p1,yes\ncohort_p2,no\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("READMISSION_CSV", str(csv_file))

        tool_call = {
            "id": "call_readm_001",
            "type": "function",
            "function": {
                "name": "get_readmission_prediction",
                "arguments": json.dumps({"person_id": "cohort_p1"}),
            },
        }
        result_str = asyncio.run(
            execute_cohort_tool_call(tool_call, patient_ids=["cohort_p1", "cohort_p2"])
        )
        result = json.loads(result_str)
        assert "error" not in result
        assert result["patient_id"] == "cohort_p1"
        assert result["readmission"] == "yes"
        assert result["predicted_readmission"] is True

    def test_llm_style_readmission_fallback_person_id_from_cohort(self, tmp_path, monkeypatch):
        """When LLM omits person_id, cohort path uses first patient_id from cohort."""
        csv_file = tmp_path / "readmission_labels.csv"
        csv_file.write_text(
            "patient_id,readmission\nfallback_patient,no\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("READMISSION_CSV", str(csv_file))

        # Tool call with empty or missing person_id - cohort provides first id
        tool_call = {
            "id": "call_readm_002",
            "type": "function",
            "function": {
                "name": "get_readmission_prediction",
                "arguments": json.dumps({}),
            },
        }
        result_str = asyncio.run(
            execute_cohort_tool_call(
                tool_call, patient_ids=["fallback_patient", "other"]
            )
        )
        result = json.loads(result_str)
        assert "error" not in result
        assert result["patient_id"] == "fallback_patient"
        assert result["readmission"] == "no"
        assert result["predicted_readmission"] is False

    def test_llm_message_readmission_tool_call_normalized_then_executed(
        self, tmp_path, monkeypatch
    ):
        """Full flow: message with readmission tool_calls -> normalize -> execute_cohort_tool_call."""
        csv_file = tmp_path / "readmission_labels.csv"
        csv_file.write_text(
            "patient_id,readmission\np99,yes\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("READMISSION_CSV", str(csv_file))

        message = {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_r",
                    "type": "function",
                    "function": {
                        "name": "get_readmission_prediction",
                        "arguments": json.dumps({"person_id": "p99"}),
                    },
                },
            ],
        }
        tc_dict = _tool_call_to_dict(message["tool_calls"][0])
        assert tc_dict["function"]["name"] == "get_readmission_prediction"
        result_str = asyncio.run(
            execute_cohort_tool_call(tc_dict, patient_ids=["p99"])
        )
        result = json.loads(result_str)
        assert result["patient_id"] == "p99"
        assert result["readmission"] == "yes"
        assert result["predicted_readmission"] is True
