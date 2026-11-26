"""Build a self-contained HTML viewer for generated security alerts.

Reads one explanation run directory (review_sheet.csv, alerts.jsonl and
summary.json from scripts/run_explanations.py) and writes a single HTML file
that works offline in any browser. No LLM calls are made.

    python -m scripts.build_alert_viewer
    python -m scripts.build_alert_viewer --run-dir reports/llm_explanations/unsw_gnn_v3 \
        --out reports/llm_explanations/unsw_gnn_v3/alert_viewer.html
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).with_name("alert_viewer_template.html")
DEFAULT_RUN = ROOT / "reports" / "llm_explanations" / "unsw_gnn_v3"
DATASET_NAMES = {"unsw": "UNSW-NB15", "cicids": "CIC-IDS2018"}
OUTCOME_ORDER = {"correct_class": 0, "wrong_attack_class": 1, "false_positive": 2}


def load_run(run_dir: Path) -> dict[str, Any]:
    sheet = run_dir / "review_sheet.csv"
    alerts_file = run_dir / "alerts.jsonl"
    summary_file = run_dir / "summary.json"
    for path in (sheet, alerts_file, summary_file):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")

    checks = {}
    with alerts_file.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                checks[record["alert_id"]] = record["checks"]

    alerts = []
    with sheet.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            alert_id = row["alert_id"]
            if alert_id not in checks:
                raise ValueError(f"{alert_id} is in review_sheet.csv but not in alerts.jsonl")
            alerts.append(
                {
                    "id": alert_id,
                    "true_class": row["true_class"],
                    "outcome": row["outcome"],
                    "text": row["alert_text"],
                    "checks": checks[alert_id],
                    "evidence": json.loads(row["evidence_json"]),
                }
            )

    alerts.sort(
        key=lambda a: (
            OUTCOME_ORDER.get(a["outcome"], 9),
            -(a["evidence"]["model_output"].get("confidence_percent") or 0),
            a["id"],
        )
    )
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    return {
        "dataset": DATASET_NAMES.get(summary.get("dataset"), summary.get("dataset", "")),
        "split": summary.get("split", ""),
        "model": summary.get("model", ""),
        "prompt_version": summary.get("prompt_version", ""),
        "passed": summary.get("automated_checks", {}).get("auto_passed"),
        "alerts": alerts,
    }


def render(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return TEMPLATE.read_text(encoding="utf-8").replace("__ALERT_DATA__", payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path, default=None, help="default: <run-dir>/alert_viewer.html")
    args = parser.parse_args(argv)

    data = load_run(args.run_dir)
    out = args.out or args.run_dir / "alert_viewer.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(data), encoding="utf-8")
    print(f"Wrote {out} with {len(data['alerts'])} alerts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
