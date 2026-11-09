"""Generate and check LLM alert explanations for flows flagged as malicious.

Without an API key, use --dry-run to build evidence and prompts only:

    python -m scripts.run_explanations \
        --predictions results/test_evaluation/unsw/xgb/predictions.npz \
        --graphs data/graphs_unsw \
        --scaler data/features/scaler.joblib \
        --feature-names data/features/feature_names.joblib \
        --out results/explanations/unsw_xgb --dry-run

For real alerts, set ARC_API_KEY and replace --dry-run with --model MODEL_NAME.
After scoring the review sheet by hand:

    python -m scripts.run_explanations --summarize-review results/explanations/unsw_xgb/review_sheet.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from src.explain.arc_client import DEFAULT_BASE_URL, ARCClient, LLMError
from src.explain.evidence import (
    FlowDecoder,
    build_evidence,
    flagged_outcome,
    load_predictions,
    select_alerts,
)
from src.explain.grounding import check_alert, dump_json, summarize_auto, summarize_review, write_review_sheet
from src.explain.prompting import PROMPT_VERSION, build_messages

MAX_CONSECUTIVE_FAILURES = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--graphs", type=Path)
    parser.add_argument("--scaler", type=Path)
    parser.add_argument("--feature-names", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--per-cell", type=int, default=2, help="alerts per (predicted class, outcome) cell")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", help="ARC model name; model names change, so check llm.arc.vt.edu")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--dry-run", action="store_true", help="write evidence and prompts without calling the LLM")
    parser.add_argument("--summarize-review", type=Path, help="tally a hand-scored review sheet and exit")
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> int:
    args = parse_args()
    if args.summarize_review:
        print(json.dumps(summarize_review(args.summarize_review), indent=2))
        return 0

    required = ("predictions", "graphs", "scaler", "feature_names", "out")
    missing = [f"--{name.replace('_', '-')}" for name in required if getattr(args, name) is None]
    if missing:
        sys.exit(f"Missing required arguments: {', '.join(missing)}")

    api_key = os.environ.get("ARC_API_KEY", "")
    if not args.dry_run and (not args.model or not api_key):
        sys.exit("Real runs need --model and the ARC_API_KEY environment variable (or use --dry-run).")

    pred = load_predictions(args.predictions)
    if pred.proba is None:
        print("[WARN] predictions.npz has no y_proba_multiclass; alerts will have no confidence.", file=sys.stderr)

    indices = select_alerts(pred, args.per_cell, args.seed)
    if indices.size == 0:
        sys.exit("No flows were flagged as malicious in these predictions.")
    decoder = FlowDecoder(args.scaler, args.feature_names)
    evidence_rows = build_evidence(pred, indices, args.graphs, decoder)
    write_jsonl(args.out / "evidence.jsonl", evidence_rows)
    print(f"Built evidence for {len(evidence_rows)} alerts from {pred.dataset}/{pred.split}.")

    if args.dry_run:
        write_jsonl(args.out / "prompts.jsonl", [
            {"alert_id": e["alert_id"], "messages": build_messages(e)} for e in evidence_rows
        ])
        print(f"Dry run complete. Wrote {args.out / 'evidence.jsonl'} and prompts.jsonl.")
        return 0

    client = ARCClient(
        api_key=api_key,
        model=args.model,
        cache_path=args.out / "llm_cache.jsonl",
        base_url=args.base_url,
        max_tokens=args.max_tokens,
    )

    alerts: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    consecutive_failures = 0

    for position, (i, evidence) in enumerate(zip(indices, evidence_rows), start=1):
        i = int(i)
        alert_id = evidence["alert_id"]
        try:
            text = client.complete(build_messages(evidence))
        except LLMError as exc:
            failures.append({"alert_id": alert_id, "error": str(exc)})
            print(f"[{position}/{len(evidence_rows)}] FAILED {alert_id}: {exc}", file=sys.stderr)
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print("Stopping after repeated failures. Finished alerts are cached.", file=sys.stderr)
                break
            continue
        consecutive_failures = 0

        checks = check_alert(text, evidence, pred.class_names)
        alerts.append({
            "alert_id": alert_id, "model": args.model, "prompt_version": PROMPT_VERSION,
            "alert_text": text, "checks": checks,
        })
        output = evidence["model_output"]
        review.append({
            "alert_id": alert_id,
            "true_class": pred.class_names[int(pred.y_true[i])],
            "predicted_class": output["predicted_class"],
            "confidence_percent": output["confidence_percent"],
            "outcome": flagged_outcome(int(pred.y_true[i]), int(pred.y_pred[i]), pred.benign_id),
            "alert_text": text,
            "auto_passed": checks["passed"],
            "auto_unsupported_numbers": "; ".join(checks["unsupported_numbers"]),
            "auto_other_classes": "; ".join(checks["other_classes_named"]),
            "auto_speculative_terms": "; ".join(checks["speculative_terms"]),
            "evidence_json": json.dumps(evidence),
        })
        print(f"[{position}/{len(evidence_rows)}] {alert_id} auto_passed={checks['passed']}")

    write_jsonl(args.out / "alerts.jsonl", alerts)
    write_review_sheet(args.out / "review_sheet.csv", review)
    summary = {
        "dataset": pred.dataset,
        "split": pred.split,
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "per_cell": args.per_cell,
        "seed": args.seed,
        "automated_checks": summarize_auto(review),
        "failures": failures,
    }
    dump_json(args.out / "summary.json", summary)
    print(json.dumps(summary["automated_checks"], indent=2))
    print(f"Review sheet: {args.out / 'review_sheet.csv'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
