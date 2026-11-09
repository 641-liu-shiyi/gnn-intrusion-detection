"""Automated grounding checks and the manual review sheet.

The checks only flag candidates for review. A person still decides whether an
alert is factually consistent, using the blank columns in the review sheet.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
SPECULATIVE_TERMS = (
    "attacker", "adversary", "malware", "ransomware", "botnet", "payload", "cve",
    "vulnerabilit", "exfiltrat", "credential", "command and control", "compromis",
    "block", "quarantine", "recommend",
)
MANUAL_COLUMNS = ("manual_factually_consistent", "manual_clear", "manual_notes")


def _as_number(text: str) -> float:
    return float(text.replace(",", ""))


def _numbers_in(value: Any) -> Iterable[float]:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        yield float(value)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _numbers_in(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _numbers_in(item)


def check_alert(text: str, evidence: dict[str, Any], class_names: list[str]) -> dict[str, Any]:
    allowed = set(_numbers_in([evidence["model_output"], evidence["flow"], evidence["window_context"]]))
    unsupported = [n for n in NUMBER.findall(text) if _as_number(n.rstrip(",")) not in allowed]

    output = evidence["model_output"]
    permitted = {output["predicted_class"], output.get("runner_up_class")}
    other_classes = [
        c for c in class_names
        if c not in permitted and re.search(rf"\b{re.escape(c)}\b", text)
    ]
    lowered = text.lower()
    speculative = [term for term in SPECULATIVE_TERMS if term in lowered]
    named = output["predicted_class"].lower() in lowered
    return {
        "unsupported_numbers": unsupported,
        "other_classes_named": other_classes,
        "speculative_terms": speculative,
        "predicted_class_named": named,
        "passed": named and not (unsupported or other_classes or speculative),
    }


def write_review_sheet(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "alert_id", "true_class", "predicted_class", "confidence_percent", "outcome",
        "alert_text", "auto_passed", "auto_unsupported_numbers", "auto_other_classes",
        "auto_speculative_terms", *MANUAL_COLUMNS, "evidence_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def summarize_auto(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total: Counter[str] = Counter()
    passed: Counter[str] = Counter()
    for row in rows:
        total[row["outcome"]] += 1
        passed[row["outcome"]] += bool(row["auto_passed"])
    return {
        "alerts": sum(total.values()),
        "auto_passed": sum(passed.values()),
        "by_outcome": {k: {"alerts": total[k], "auto_passed": passed[k]} for k in sorted(total)},
    }


def _is_yes(value: str) -> bool:
    return value.strip().lower() in {"y", "yes", "1", "true"}


def summarize_review(path: Path) -> dict[str, Any]:
    """Tally a filled-in review sheet by outcome. Blank rows are skipped."""
    tally: dict[str, Counter[str]] = defaultdict(Counter)
    skipped = 0
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            consistent, clear = row["manual_factually_consistent"], row["manual_clear"]
            if not consistent.strip() or not clear.strip():
                skipped += 1
                continue
            tally[row["outcome"]]["reviewed"] += 1
            tally[row["outcome"]]["factually_consistent"] += _is_yes(consistent)
            tally[row["outcome"]]["clear"] += _is_yes(clear)
    by_outcome = {k: dict(v) for k, v in sorted(tally.items())}
    overall: Counter[str] = Counter()
    for counts in tally.values():
        overall.update(counts)
    return {"overall": dict(overall), "by_outcome": by_outcome, "unreviewed_rows": skipped}


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
