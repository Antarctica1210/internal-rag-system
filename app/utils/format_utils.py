"""
JSON formatting utility module

Provides unified JSON serialization and formatting functions to ensure
consistent JSON output across the project.
"""

import json
from typing import Any, Dict


def format_state(state: Dict[str, Any], indent: int = 4) -> str:
    """
    Formats a workflow state (ImportGraphState) as a JSON string.

    Args:
        state: ImportGraphState workflow state dictionary
        indent: Number of spaces for JSON indentation, default 4

    Returns:
        Formatted JSON string

    Example:
        >>> state = {"task_id": "001", "pdf_path": "test.pdf"}
        >>> print(format_state(state))
        {
            "task_id": "001",
            "pdf_path": "test.pdf"
        }
    """

    return json.dumps(state, indent=indent, ensure_ascii=False)


def format_json(data: Any, indent: int = 4, ensure_ascii: bool = False) -> str:
    """
    General-purpose JSON formatting function.

    Args:
        data: Data to format (dict, list, or any JSON-serializable object)
        indent: Number of spaces for JSON indentation, default 4
        ensure_ascii: Whether to escape non-ASCII characters, default False (preserves Chinese and other characters)

    Returns:
        Formatted JSON string

    Example:
        >>> data = {"name": "test", "value": 123}
        >>> print(format_json(data))
        {
            "name": "test",
            "value": 123
        }
    """
    return json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)
