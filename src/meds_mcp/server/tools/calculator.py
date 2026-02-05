"""
Shared calculator tool for LLM tool-calling.

Provides a safe, expression-only calculator used by cohort chat and demos.
"""

import json
import logging
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)


def calculator_tool(expression: str) -> str:
    """
    Simple calculator tool that evaluates mathematical expressions.

    Args:
        expression: A mathematical expression as a string (e.g., "2 + 2", "10 * 5", "100 / 4")

    Returns:
        The result of the calculation as a string
    """
    logger.info("Calculator tool called with expression: %s", expression)

    try:
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return (
                "Error: Invalid characters in expression. "
                "Only numbers and basic operators (+, -, *, /) are allowed."
            )

        result = eval(expression)
        result_str = str(result)
        logger.info("Calculator result: %s", result_str)
        return result_str
    except Exception as e:
        error_msg = f"Error calculating: {str(e)}"
        logger.error("Calculator error: %s", error_msg)
        return error_msg


def get_calculator_tool_definition() -> Dict[str, Any]:
    """
    Get the tool definition for the calculator in OpenAI format.

    Returns:
        Tool definition dictionary compatible with OpenAI API
    """
    return {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "A calculator tool that evaluates mathematical expressions. "
                "ALWAYS use this tool for ANY mathematical calculation, arithmetic operation, "
                "or numerical computation. Examples: addition (2+2), subtraction (10-5), "
                "multiplication (5*3), division (100/4), or complex expressions ((5+3)*2)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "The mathematical expression to evaluate. Can include numbers, "
                            "operators (+, -, *, /), and parentheses. "
                            "Examples: '2 + 2', '10 * 5', '100 / 4', '(5 + 3) * 2', '10+10'"
                        ),
                    }
                },
                "required": ["expression"],
            },
        },
    }


def execute_tool_call(tool_call: Dict[str, Any]) -> str:
    """
    Execute a tool call and return the result.
    Handles the calculator tool; returns an error for unknown tools.

    Args:
        tool_call: Tool call dictionary from LLM response

    Returns:
        Result of tool execution as a string
    """
    function_name = tool_call.get("function", {}).get("name", "")
    function_args = tool_call.get("function", {}).get("arguments", "{}")

    try:
        args = json.loads(function_args) if isinstance(function_args, str) else function_args
    except json.JSONDecodeError:
        logger.error("Failed to parse tool arguments: %s", function_args)
        return f"Error: Invalid arguments for {function_name}"

    if function_name == "calculator":
        expression = args.get("expression", "")
        return calculator_tool(expression)

    logger.warning("Unknown tool: %s", function_name)
    return f"Error: Unknown tool {function_name}"


def is_simple_calculation(query: str) -> bool:
    """
    Detect if the query is a simple mathematical calculation that doesn't need patient context.

    Args:
        query: User's input query

    Returns:
        True if the query appears to be a simple calculation
    """
    calculation_patterns = [
        r"what is \d+",
        r"calculate \d+",
        r"compute \d+",
        r"\d+\s*[+\-*/]\s*\d+",
        r"what\'?s? \d+",
        r"how much is \d+",
    ]

    query_lower = query.lower().strip()

    for pattern in calculation_patterns:
        if re.search(pattern, query_lower):
            skip_words = [
                "patient",
                "age",
                "weight",
                "height",
                "bmi",
                "blood",
                "pressure",
                "lab",
                "test",
                "result",
            ]
            if not any(word in query_lower for word in skip_words):
                return True

    return False
