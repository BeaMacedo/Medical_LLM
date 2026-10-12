import json
import re

from criteria_pairwise import CRITERIA_PAIRWISE, VERDICT_OPTIONS

_SMART_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})


def _extract_json_block(text: str): #Return the substring from the first '{' to its matching '}', or None.
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    if depth > 0:
        tail = text[start:].rstrip()
        if tail.endswith("```"):
            tail = tail[:-3].rstrip()
        return tail + ("}" * depth)
    return None


def _brace_pad(text: str) -> str:
    depth = text.count("{") - text.count("}")
    return text + ("}" * depth) if depth > 0 else text


def _parse_json(raw_output: str):
    block = _extract_json_block(raw_output)
    if block is None:
        return None

    for candidate in (block, block.translate(_SMART_QUOTES)):
        text = candidate
        for _ in range(8):
            try:
                return json.loads(_brace_pad(text))
            except json.JSONDecodeError as e:
                if e.pos >= len(text):
                    break
                text = text[:e.pos] + text[e.pos + 1:]
    return None


def _regex_verdict(raw_output: str, label: str):
    pattern = r'"' + re.escape(label) + r'"\s*:\s*\{\s*"verdict"\s*:\s*"([^"]*)"'
    match = re.search(pattern, raw_output, re.IGNORECASE)
    return match.group(1) if match else None


def parse_judge_output(raw_output: str) -> dict: #Extract the per-criterion {verdict, reason} pair from a pairwise judge's raw JSON output.
    parsed = {}
    data = _parse_json(raw_output)

    for key, label, _ in CRITERIA_PAIRWISE:
        entry = data.get(label) if isinstance(data, dict) else None
        verdict = entry.get("verdict") if isinstance(entry, dict) else None
        reason = entry.get("reason") if isinstance(entry, dict) else None

        if not (isinstance(verdict, str) and verdict.strip().lower() in VERDICT_OPTIONS):
            fallback = _regex_verdict(raw_output, label) #tries to extract with regex
            if fallback is not None:
                verdict = fallback

        is_valid = isinstance(verdict, str) and verdict.strip().lower() in VERDICT_OPTIONS
        normalized_verdict = verdict.strip().lower() if is_valid else verdict

        parsed[key] = normalized_verdict if is_valid else verdict
        parsed[f"{key}_valid"] = is_valid
        parsed[f"{key}_reason"] = reason

    return parsed
