"""JSON trace storage backend."""

from __future__ import annotations

import json
import re
import time
import warnings
from pathlib import Path
from typing import Any

from reagent_flow.exceptions import ReagentError, ReagentWarning, TraceNotFoundError
from reagent_flow.models import Trace, trace_from_dict, trace_to_dict


def _sanitize_name(name: str) -> str:
    """Sanitize a trace name for safe use as a filename component.

    Strips path separators and special characters to prevent path traversal.
    Raises ReagentError if the sanitized name is empty.
    """
    sanitized = re.sub(r"[^\w\-.]", "_", name)
    sanitized = sanitized.strip("._")
    if not sanitized:
        raise ReagentError(f"Trace name sanitizes to empty string: {name!r}")
    return sanitized


def save_trace(trace: Trace, base_dir: str, *, golden: bool = False) -> str:
    """Save a trace to JSON. Returns the file path."""
    safe_name = _sanitize_name(trace.name)

    if golden:
        dir_path = Path(base_dir) / "golden"
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / f"{safe_name}.trace.json"
    else:
        dir_path = Path(base_dir) / "traces"
        dir_path.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        file_path = dir_path / f"{safe_name}_{ts}.trace.json"
        counter = 1
        while file_path.exists():
            file_path = dir_path / f"{safe_name}_{ts}_{counter}.trace.json"
            counter += 1

    data = trace_to_dict(trace)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2, default=_json_fallback)
    return str(file_path)


def _json_fallback(obj: Any) -> str:
    """Fallback serializer that warns when stringifying non-JSON-native types."""
    warnings.warn(
        ReagentWarning(
            f"Non-serializable value of type {type(obj).__name__} was converted to "
            f"string in trace data. Consider converting it before logging."
        ),
        stacklevel=2,
    )
    return str(obj)


def load_trace(path: str) -> Trace:
    """Load a trace from a JSON file."""
    with open(path) as f:
        data: dict[str, Any] = json.load(f)
    return trace_from_dict(data)


def load_golden(base_dir: str, name: str) -> Trace:
    """Load a golden baseline trace."""
    safe_name = _sanitize_name(name)
    path = Path(base_dir) / "golden" / f"{safe_name}.trace.json"
    try:
        return load_trace(str(path))
    except FileNotFoundError:
        raise TraceNotFoundError(f"Golden trace not found: {path}") from None


def find_traces(base_dir: str, name: str) -> list[str]:
    """Find all recorded traces matching a name."""
    safe_name = _sanitize_name(name)
    traces_dir = Path(base_dir) / "traces"
    if not traces_dir.exists():
        return []
    matches = sorted(str(p) for p in traces_dir.glob(f"{safe_name}_*.trace.json"))
    return matches
