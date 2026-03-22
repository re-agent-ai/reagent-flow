"""JSON trace storage backend."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ttrace_ai.exceptions import TraceNotFoundError
from ttrace_ai.models import Trace, trace_from_dict, trace_to_dict


def save_trace(trace: Trace, base_dir: str, *, golden: bool = False) -> str:
    """Save a trace to JSON. Returns the file path."""
    if golden:
        dir_path = Path(base_dir) / "golden"
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / f"{trace.name}.trace.json"
    else:
        dir_path = Path(base_dir) / "traces"
        dir_path.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        file_path = dir_path / f"{trace.name}_{ts}.trace.json"
        counter = 1
        while file_path.exists():
            file_path = dir_path / f"{trace.name}_{ts}_{counter}.trace.json"
            counter += 1

    data = trace_to_dict(trace)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return str(file_path)


def load_trace(path: str) -> Trace:
    """Load a trace from a JSON file."""
    with open(path) as f:
        data: dict[str, Any] = json.load(f)
    return trace_from_dict(data)


def load_golden(base_dir: str, name: str) -> Trace:
    """Load a golden baseline trace."""
    path = Path(base_dir) / "golden" / f"{name}.trace.json"
    try:
        return load_trace(str(path))
    except FileNotFoundError:
        raise TraceNotFoundError(f"Golden trace not found: {path}") from None


def find_traces(base_dir: str, name: str) -> list[str]:
    """Find all recorded traces matching a name."""
    traces_dir = Path(base_dir) / "traces"
    if not traces_dir.exists():
        return []
    matches = sorted(str(p) for p in traces_dir.glob(f"{name}_*.trace.json"))
    return matches
