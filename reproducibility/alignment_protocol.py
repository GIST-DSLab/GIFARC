"""Scientific helpers copied from the actual paper analysis sources.

Only imports/packaging are adapted; function definitions are preserved.
This module itself makes no network calls and does not execute task code.
See source-provenance.json for source hashes and copied-definition hashes.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

BACKCAP_SCHEMA = {'type': 'object', 'properties': {'description': {'type': 'string'}}, 'required': ['description'], 'additionalProperties': False}


def parse_seed_description(text: str) -> tuple[list[str], str]:
    if not isinstance(text, str):
        raise ValueError("seed_description is not a string")
    match = re.search(
        r"(?is)(?:^|\n)\s*#?\s*concepts\s*:\s*(.*?)\s*(?:\n\s*)?#?\s*description\s*:\s*(.*)\s*$",
        text,
    )
    if not match:
        raise ValueError("seed_description does not contain concepts/description blocks")
    concepts_text = re.sub(r"\s+", " ", match.group(1)).strip()
    description = re.sub(r"\s+", " ", match.group(2)).strip()
    concepts = [part.strip() for part in concepts_text.split(",") if part.strip()]
    if not concepts or not description:
        raise ValueError("empty concepts or description")
    return concepts, description


def render_grid(grid: list[list[int]]) -> str:
    """Match Section 5.1/5.2 grid formatting."""
    return json.dumps(grid, indent=2)


def normalize_example_pair(example: Any) -> dict[str, list[list[int]]]:
    if isinstance(example, dict):
        return {"input": example["input"], "output": example["output"]}
    if isinstance(example, list) and len(example) == 2:
        return {"input": example[0], "output": example[1]}
    raise ValueError("stored example is neither {input, output} nor [input, output]")


def to_plain(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return int(value) if float(value).is_integer() else float(value)
    if isinstance(value, tuple):
        return [to_plain(v) for v in value]
    if isinstance(value, list):
        return [to_plain(v) for v in value]
    return value


def validate_grid(value: Any) -> tuple[bool, list[list[int]] | None, str | None]:
    value = to_plain(value)
    if not isinstance(value, list) or not value:
        return False, None, "not_nonempty_list"
    if not all(isinstance(row, list) and row for row in value):
        return False, None, "row_not_nonempty_list"
    width = len(value[0])
    if any(len(row) != width for row in value):
        return False, None, "non_rectangular"
    out: list[list[int]] = []
    for row in value:
        out_row: list[int] = []
        for cell in row:
            if isinstance(cell, bool) or not isinstance(cell, int) or not 0 <= cell <= 9:
                return False, None, "cell_not_int_0_9"
            out_row.append(int(cell))
        out.append(out_row)
    return True, out, None


def body_output_text(body: dict[str, Any]) -> str:
    if body.get("output_text"):
        return str(body["output_text"])
    chunks: list[str] = []
    for output in body.get("output", []) or []:
        for content in output.get("content", []) or []:
            if isinstance(content, dict) and content.get("text"):
                chunks.append(str(content["text"]))
    return "\n".join(chunks)


def is_output_truncated(body: dict[str, Any]) -> bool:
    details = body.get("incomplete_details") or {}
    reason = details.get("reason") if isinstance(details, dict) else None
    if reason in {"max_output_tokens", "length"}:
        return True
    if body.get("status") == "incomplete" and reason:
        return True
    for output in body.get("output", []) or []:
        if output.get("finish_reason") in {"length", "max_output_tokens"}:
            return True
    return False


def cache_key(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def normalize_task_pair(example: Any) -> dict[str, list[list[int]]] | None:
    try:
        pair = normalize_example_pair(example)
    except Exception:
        return None
    ok_input, input_grid, _ = validate_grid(pair.get("input"))
    ok_output, output_grid, _ = validate_grid(pair.get("output"))
    if not ok_input or not ok_output:
        return None
    return {"input": input_grid or [], "output": output_grid or []}


def stored_pairs(task: Any) -> tuple[list[dict[str, list[list[int]]]], list[str]]:
    pairs: list[dict[str, list[list[int]]]] = []
    flags: list[str] = []
    for index, example in enumerate(task.examples):
        pair = normalize_task_pair(example)
        if pair is None:
            flags.append(f"bad_stored_pair:{index}")
            continue
        pairs.append(pair)
    if not pairs:
        flags.append("no_grids")
    return pairs, flags


def build_stored_pair_backcap_prompt(task: Any, pairs: list[dict[str, list[list[int]]]]) -> str:
    selected = pairs[:3]
    lines = [
        "You are analyzing ARC-style input-output grid examples.",
        "You are given only stored input-output examples from one task.",
        "Do not predict any grid.",
        "Describe the transformation rule shown by these examples in 2-4 sentences.",
        "Describe colors by their roles rather than concrete numeric values.",
        "Focus on objects, spatial relations, frame changes, and what changes from input to output.",
        "",
        "[Train Pairs]",
    ]
    for index, pair in enumerate(selected, start=1):
        lines.extend(
            [
                f"Pair {index}:",
                "Input:",
                render_grid(pair["input"]),
                "Output:",
                render_grid(pair["output"]),
                "",
            ]
        )
    lines.extend(
        [
            "Return JSON only with one field:",
            '{ "description": "2-4 sentence natural-language rule description" }',
        ]
    )
    return "\n".join(lines)


def parse_description(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            return stripped
        payload = json.loads(stripped[start : end + 1])
    if isinstance(payload, dict) and isinstance(payload.get("description"), str):
        return payload["description"].strip()
    return stripped


def response_body(raw: dict[str, Any]) -> dict[str, Any] | None:
    if "response" in raw:
        response = raw.get("response") or {}
        if response.get("status_code") != 200:
            return None
        return response.get("body")
    return raw


def extract_caption(path: Path) -> tuple[str | None, list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    body = response_body(raw)
    if body is None:
        return None, ["api_error"]
    if is_output_truncated(body) or body.get("status") == "incomplete":
        return None, ["output_incomplete"]
    text = body_output_text(body)
    if not text.strip():
        return None, ["empty_output"]
    try:
        return parse_description(text), []
    except Exception as exc:  # noqa: BLE001
        return None, [f"parse_failure:{type(exc).__name__}"]


def response_body_for_prompt(prompt: str, *, model: str, reasoning_effort: str) -> dict[str, Any]:
    return {
        "model": model,
        "reasoning": {"effort": reasoning_effort},
        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        "max_output_tokens": 16384,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "gifarc_stored_pair_back_caption",
                "schema": BACKCAP_SCHEMA,
                "strict": True,
            }
        },
    }


def cosine(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / denom) if denom else float("nan")


def retry_allowed(attempts: list[dict]) -> bool:
    return len(attempts) == 1 and attempts[0]['status'] == 'failed'
