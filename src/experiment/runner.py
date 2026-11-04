"""Extended experiment runner for the baseline models.

This module builds on the original lightweight runner but adds support for:

* JSON experiment configuration files (``--config``)
* Multi-seed execution
* One-factor-at-a-time ablations
* Dry-run mode (default) that only prints the commands that *would* be executed
* Safe resume behaviour - completed runs are skipped, incomplete runs are not
  treated as successful.

The runner never imports the heavyweight training code - it launches the existing
``train_*.py`` scripts via ``subprocess``.  All artefacts (``metrics.json``,
``run_config.json`` etc.) continue to be produced by the trainer scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import numpy as np
import subprocess
import sys
import csv
import statistics
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence

# ---------------------------------------------------------------------------
# Mapping from model identifier to the module that implements the training script.
# ---------------------------------------------------------------------------
MODEL_MODULE = {
    "rf": "src.baselines.train_rf",
    "xgb": "src.baselines.train_xgb",
    "mlp": "src.baselines.train_mlp",
}


def _hash_config(cfg: Mapping[str, Any]) -> str:
    """Return a short deterministic identifier for a configuration.

    The identifier is derived from a SHA-1 digest of the JSON-serialised configuration
    (sorted keys) and truncated to eight characters.  This provides a reproducible yet
    human-readable folder name without risking collisions for the modest number of
    experimental setups used in the teaching assignment.
    """
    serialized = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:8]


def _load_json_config(path: Path) -> Mapping[str, Any]:
    """Load and minimally validate a JSON experiment configuration.

    Required top-level keys are ``experiment_name``, ``model``, ``dataset``, ``seeds``
    (list of integers) and ``output_root``.  Optional keys include ``max_samples``,
    ``base_params`` and ``ablations``.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {"experiment_name", "model", "dataset", "seeds", "output_root"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Missing required config keys: {sorted(missing)}")
    if not isinstance(data["seeds"], list) or not all(isinstance(s, int) for s in data["seeds"]):
        raise ValueError("'seeds' must be a list of integers")
    return data


def _expand_ablations(base: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    """Expand a configuration into a list of one-factor-at-a-time ablations.

    ``base`` is the dictionary returned by ``_load_json_config``.  If ``ablations``
    is missing or empty, a single configuration identical to ``base`` is returned.

    For each ablation factor we generate a configuration where *only* that factor
    is set to each value from the provided list, keeping the original ``base_params``
    (if any) unchanged.  Values that match the baseline hyper-parameter are skipped
    to avoid duplicate configurations.
    """
    ablations: Mapping[str, Sequence[Any]] = base.get("ablations", {})
    if not ablations:
        return [base]
    configs: list[Mapping[str, Any]] = []
    base_params = base.get("base_params", {})
    # Include the baseline configuration itself.
    configs.append(base)
    for factor, values in ablations.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Ablation values for {factor!r} must be a non-empty list")
        baseline_val = base_params.get(factor)
        for val in values:
            # Skip the baseline value - it is already represented by ``base``.
            if baseline_val is not None and val == baseline_val:
                continue
            cfg = dict(base)  # shallow copy of top-level keys
            merged_params = dict(base_params)
            merged_params[factor] = val
            cfg["base_params"] = merged_params
            cfg["ablation_factor"] = factor
            cfg["ablation_value"] = val
            configs.append(cfg)
    return configs


def _build_command_from_components(
    model: str,
    dataset: str,
    seed: int,
    max_samples: int,
    workers: int,
    output_dir: Path,
    evaluate_test: bool,
    data_root: Path | None = None,
    extra_opts: Mapping[str, Any] | None = None,
) -> list[str]:
    """Construct the ``python -m <module> ...`` command for a single trainer run.

    ``extra_opts`` contains model-specific hyper-parameters.  Flags with a ``None``
    value are treated as boolean switches (no value attached).
    """
    module = MODEL_MODULE[model]
    cmd: list[str] = [sys.executable, "-m", module]
    cmd.extend(["--dataset", dataset])
    cmd.extend(["--max_samples", str(max_samples)])
    cmd.extend(["--seed", str(seed)])
    cmd.extend(["--workers", str(workers)])
    cmd.extend(["--output-dir", str(output_dir)])
    if evaluate_test:
        cmd.append("--evaluate-test")
    if data_root is not None:
        cmd.extend(["--data-root", str(data_root)])
    if extra_opts:
        for opt, val in extra_opts.items():
            cmd.append(f"--{opt}")
            if val is not None:
                cmd.append(str(val))
    return cmd


def _aggregate_results(configs: Sequence[Mapping[str, Any]]) -> None:
    """Aggregate validation metrics across seeds for each configuration.

    The function writes two files - ``results_summary.csv`` and ``results_summary.json`` -
    into the top-level experiment output directory (``output_root/experiment_name``).
    Each row corresponds to a unique configuration (including any ablation factor).
    For each metric the mean and sample standard deviation are reported.  Standard
    deviation is ``None`` when only a single successful seed is present.
    """
    import csv
    import statistics

    if not configs:
        print("[WARN] No configurations provided for aggregation.")
        return

    # Determine a common output root (all configs share the same output_root per spec).
    first_cfg = configs[0]
    output_root = Path(first_cfg["output_root"]).resolve()
    experiment_name = first_cfg["experiment_name"]
    summary_dir = output_root / experiment_name
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary_entries: list[dict[str, Any]] = []
    metric_keys = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "binary_precision",
        "binary_recall",
        "binary_f1",
    ]

    for cfg in configs:
        model = cfg["model"]
        dataset = cfg["dataset"]
        output_root_cfg = Path(cfg["output_root"]).resolve()
        config_id = _hash_config(cfg)
        config_dir = output_root_cfg / cfg["experiment_name"] / model / config_id
        seeds = cfg.get("seeds", [])
        seed_metrics: list[Mapping[str, Any]] = []
        successful_seeds: list[int] = []
        for seed in seeds:
            seed_dir = config_dir / f"seed_{seed}"
            if not _metrics_successful(seed_dir):
                continue
            try:
                metrics = json.loads((seed_dir / "metrics.json").read_text(encoding="utf-8"))
                val = metrics.get("validation", metrics)
                seed_metrics.append(val)
                successful_seeds.append(seed)
            except Exception:
                continue
        if not seed_metrics:
            # No completed runs for this configuration - skip.
            continue
        # Include optional binary_average_precision if present in any seed.
        if any("binary_average_precision" in m for m in seed_metrics) and "binary_average_precision" not in metric_keys:
            metric_keys.append("binary_average_precision")
        # Compute mean/std for each metric.
        agg: dict[str, Any] = {}
        for key in metric_keys:
            values = [m[key] for m in seed_metrics if key in m]
            if not values:
                continue
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else None
            agg[f"{key}_mean"] = mean_val
            agg[f"{key}_std"] = std_val
        entry: dict[str, Any] = {
            "model": model,
            "dataset": dataset,
            "experiment_name": cfg["experiment_name"],
            "config_id": config_id,
            "ablation_factor": cfg.get("ablation_factor"),
            "ablation_value": cfg.get("ablation_value"),
            "n_successful": len(successful_seeds),
            "seeds_successful": successful_seeds,
        }
        entry.update(agg)
        summary_entries.append(entry)

    if not summary_entries:
        print("[INFO] No completed runs found for aggregation.")
        return

    # Write CSV
    csv_path = summary_dir / "results_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames: list[str] = []
        for entry in summary_entries:
            for k in entry.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_entries:
            # Convert None to empty string for CSV friendliness.
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})

    # Write JSON
    json_path = summary_dir / "results_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_entries, f, indent=2)

    print(f"[INFO] Aggregated results written to CSV: {csv_path}")
    print(f"[INFO] Aggregated results written to JSON: {json_path}")


def _metrics_successful(run_dir: Path) -> bool:
    """Return ``True`` only if a baseline run appears to be complete.

    Checks performed:
    1. ``metrics.json`` exists, parses, and contains the required scalar metric keys
       (``accuracy``, ``macro_f1``, ``weighted_f1`` and ``binary_f1``) with numeric
       values.
    2. ``run_config.json`` exists.
    3. ``predictions.npz`` exists.
    """
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        return False
    try:
        metrics_raw = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if isinstance(metrics_raw, dict) and "validation" in metrics_raw and isinstance(metrics_raw["validation"], dict):
        metrics = metrics_raw["validation"]
    else:
        metrics = metrics_raw
    required_metric_keys = {"accuracy", "macro_f1", "weighted_f1", "binary_f1"}
    if required_metric_keys - metrics.keys():
        return False
    for key in required_metric_keys:
        if not isinstance(metrics[key], (int, float)):
            return False

    if not (run_dir / "run_config.json").is_file():
        return False

    pred_path = run_dir / "predictions.npz"
    if not pred_path.is_file():
        return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline experiments in a unified way")
    # -----------------------------------------------------------------------
    # Legacy CLI (single run) - retained for backward compatibility.
    # -----------------------------------------------------------------------
    parser.add_argument("--model", choices=["rf", "xgb", "mlp"], help="Which baseline model to run")
    parser.add_argument(
        "--dataset",
        choices=["unsw", "cicids"],
        help="Dataset name",
    )
    parser.add_argument("--max_samples", type=int, default=0, help="Training budget (0 = use all)")
    parser.add_argument("--seed", type=int, help="Random seed for single-run mode")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel workers for the trainer")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="If set, the underlying script will also evaluate the test split.",
    )
    # -----------------------------------------------------------------------
    # New configuration-driven mode.
    # -----------------------------------------------------------------------
    parser.add_argument("--config", type=Path, help="Path to a JSON experiment configuration file")
    # -----------------------------------------------------------------------
    # Execution mode flags.
    # -----------------------------------------------------------------------
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Print the commands that would be executed without running them.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually execute the generated commands (overrides ``--dry-run``).",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate completed experiment results without launching training runs.",
    )
    # -----------------------------------------------------------------------
    # Catch-all extra options forwarded to the trainer (e.g. ``--n_estimators``).
    # -----------------------------------------------------------------------
    parser.add_argument(
        "extra_opts",
        nargs=argparse.REMAINDER,
        help="Additional options passed verbatim to the underlying trainer script",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Parse ``extra_opts`` into a dict ``{key: value}`` - values may be omitted
    # for flag-style arguments.
    # ---------------------------------------------------------------
    extra_dict: dict[str, str | None] = {}
    it = iter(args.extra_opts)
    for token in it:
        if not token.startswith("--"):
            raise ValueError(f"Unexpected argument without '--' prefix: {token}")
        key = token.lstrip("-")
        try:
            nxt = next(it)
            if nxt.startswith("--"):
                extra_dict[key] = None
                # put token back for next iteration
                it = (t for t in [nxt] + list(it))
            else:
                extra_dict[key] = nxt
        except StopIteration:
            extra_dict[key] = None
    args.extra_opts = extra_dict

    # -------------------------------------------------------------------
    # Aggregation mode - if ``--aggregate`` is supplied we skip any command
    # generation and directly produce the summary files based on existing runs.
    # -------------------------------------------------------------------
    if getattr(args, "aggregate", False):
        if not args.config:
            parser.error("--aggregate requires a JSON configuration via --config")
        cfg = _load_json_config(args.config)
        configs = list(_expand_ablations(cfg))
        _aggregate_results(configs)
        sys.exit(0)

    # ---------------------------------------------------------------
    # Determine the set of experiment configurations to run.
    # ---------------------------------------------------------------
    configs: list[Mapping[str, Any]] = []
    if args.config:
        cfg = _load_json_config(args.config)
        configs = list(_expand_ablations(cfg))
    else:
        # Legacy mode - all required arguments must be supplied.
        if not (args.model and args.dataset and args.seed is not None):
            parser.error("When --config is omitted, --model, --dataset and --seed are required")
        cfg = {
            "experiment_name": "legacy",
            "model": args.model,
            "dataset": args.dataset,
            "seeds": [args.seed],
            "output_root": "results",
            "max_samples": args.max_samples,
            "base_params": args.extra_opts,
            "evaluate_test": args.evaluate_test,
        }
        configs = [cfg]

    total_planned = 0
    planned_commands: list[list[str]] = []

    for cfg in configs:
        model = cfg["model"]
        dataset = cfg["dataset"]
        max_samples = cfg.get("max_samples", 0)
        evaluate_test = cfg.get("evaluate_test", False)
        output_root = Path(cfg["output_root"]).resolve()
        # Determine a deterministic sub-directory for this configuration.
        config_id = _hash_config(cfg)
        config_dir = output_root / cfg["experiment_name"] / model / config_id
        # Merge base_params with any ablation overrides.
        raw_extra = cfg.get("base_params", {})
        extra_opts: dict[str, Any] = dict(raw_extra)  # copy
        # ``workers`` may be supplied either as a top-level key or inside ``base_params``.
        if "workers" in extra_opts:
            workers = int(extra_opts.pop("workers"))
        else:
            workers = cfg.get("workers", 2)
        data_root = cfg.get("data_root")
        for seed in cfg["seeds"]:
            seed_dir = config_dir / f"seed_{seed}"
            if _metrics_successful(seed_dir):
                # Skip runs that have already produced a metrics file.
                continue
            total_planned += 1
            cmd = _build_command_from_components(
                model=model,
                dataset=dataset,
                seed=seed,
                max_samples=max_samples,
                workers=workers,
                output_dir=seed_dir,
                evaluate_test=evaluate_test,
                data_root=Path(data_root) if data_root else None,
                extra_opts=extra_opts,
            )
            planned_commands.append(cmd)

    # -------------------------------------------------------------------
    # Reporting / execution.
    # -------------------------------------------------------------------
    if args.dry_run and not args.run:
        print(f"[INFO] Dry-run: {total_planned} command(s) would be executed.")
        for cmd in planned_commands:
            print("[CMD]", " ".join(map(str, cmd)))
        sys.exit(0)

    if args.run:
        for cmd in planned_commands:
            print(f"[INFO] Executing: {' '.join(map(str, cmd))}")
            result = subprocess.run(cmd, cwd=Path.cwd())
            if result.returncode != 0:
                print(f"[WARN] Command failed with exit code {result.returncode}")
        # Run aggregation on completed configurations
        _aggregate_results(configs)
        sys.exit(0)

    # If we reach this point, everything was already up-to-date.
    print("[INFO] No new runs required - all experiments already completed.")
    _aggregate_results(configs)


if __name__ == "__main__":
    main()