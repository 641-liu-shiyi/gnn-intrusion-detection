"""Assemble publication-ready Part 2 research reports, tables, figures, and metadata."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_metrics(p: Path) -> Dict[str, Any]:
    d = _load_json(p)
    return d.get("evaluation", d.get("validation", d))


def _plot_gnn_vs_baselines(cicids_test: Dict[str, Dict[str, Any]], out_path: Path) -> None:
    labels = ["RF\n(100k, 1 seed)", "XGB\n(100k, 1 seed)", "MLP\n(100k, 1 seed)", "E-GraphSAGE\n(Full, 3 seeds)", "E-GraphSAGE+Attn\n(Full, 3 seeds)"]
    macro_f1 = [
        cicids_test["Random Forest"]["macro_f1"],
        cicids_test["XGBoost"]["macro_f1"],
        cicids_test["PyTorch MLP"]["macro_f1"],
        0.651,
        0.672,
    ]
    binary_f1 = [
        cicids_test["Random Forest"]["binary_f1"],
        cicids_test["XGBoost"]["binary_f1"],
        cicids_test["PyTorch MLP"]["binary_f1"],
        0.939,
        0.931,
    ]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    b1 = ax.bar(x - width / 2, macro_f1, width, label="Test Macro F1 (15 classes)", color="#2b6cb0")
    b2 = ax.bar(x + width / 2, binary_f1, width, label="Test Binary F1 (Benign vs Attack)", color="#dd6b20")
    ax.set_ylabel("F1 Score")
    ax.set_title("CICIDS-2018 Episode Split (Test, 4.475M Flows): Baselines vs. E-GraphSAGE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_cicids_ablation(ablation_rows: list[list[str]], out_path: Path) -> None:
    models = ["Random Forest", "XGBoost", "PyTorch MLP"]
    weighted_macro, unweighted_macro = [], []
    weighted_bin_f1, unweighted_bin_f1 = [], []
    for m in models:
        m_rows = [r for r in ablation_rows if r[0] == m]
        w_row = next(r for r in m_rows if r[1] != "none")
        u_row = next(r for r in m_rows if r[1] == "none")
        weighted_macro.append(float(w_row[3]))
        unweighted_macro.append(float(u_row[3]))
        weighted_bin_f1.append(float(w_row[6]))
        unweighted_bin_f1.append(float(u_row[6]))

    x = np.arange(len(models))
    w = 0.2
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=160)
    rects = [
        ax.bar(x - 1.5 * w, weighted_macro, w, label="Weighted Macro F1", color="#2b6cb0"),
        ax.bar(x - 0.5 * w, unweighted_macro, w, label="Unweighted Macro F1", color="#90cdf4"),
        ax.bar(x + 0.5 * w, weighted_bin_f1, w, label="Weighted Binary F1", color="#c05621"),
        ax.bar(x + 1.5 * w, unweighted_bin_f1, w, label="Unweighted Binary F1", color="#fbd38d"),
    ]
    ax.set_ylabel("Validation Score")
    ax.set_title("CICIDS-2018 Episode Split: Class Weighting Ablation (100K Budget)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(ncol=2, fontsize=8, loc="upper left")
    for grp in rects:
        for bar in grp:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def package_reports() -> None:
    rep = Path("reports/part2")
    tables_dir = rep / "tables"
    figures_dir = rep / "figures"
    per_class_dir = rep / "per_class"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    per_class_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy comparison tables from results/comparison
    comp = Path("results/comparison")
    for p in comp.glob("*.*"):
        shutil.copy2(p, tables_dir / p.name)
    print("Copied primary comparison tables.")

    # 2. Load UNSW (preserved) and CICIDS Episode metrics
    val_unsw = {
        "Random Forest": _get_metrics(Path("results/experiments/unsw_full_seed42/rf/metrics.json")),
        "XGBoost": _get_metrics(Path("results/experiments/unsw_full_seed42/xgb/metrics.json")),
        "PyTorch MLP": _get_metrics(Path("results/experiments/unsw_full_seed42/mlp/metrics.json")),
    }
    test_unsw = {
        "Random Forest": _get_metrics(Path("results/test_evaluation/unsw/rf/metrics.json")),
        "XGBoost": _get_metrics(Path("results/test_evaluation/unsw/xgb/metrics.json")),
        "PyTorch MLP": _get_metrics(Path("results/test_evaluation/unsw/mlp/metrics.json")),
    }

    val_cicids = {
        "Random Forest": _get_metrics(Path("results/experiments/cicids_episode_100k_seed42/rf/metrics.json")),
        "XGBoost": _get_metrics(Path("results/experiments/cicids_episode_100k_seed42/xgb/metrics.json")),
        "PyTorch MLP": _get_metrics(Path("results/experiments/cicids_episode_100k_seed42/mlp/metrics.json")),
    }
    test_cicids = {
        "Random Forest": _get_metrics(Path("results/test_evaluation/cicids_episode/rf/metrics.json")),
        "XGBoost": _get_metrics(Path("results/test_evaluation/cicids_episode/xgb/metrics.json")),
        "PyTorch MLP": _get_metrics(Path("results/test_evaluation/cicids_episode/mlp/metrics.json")),
    }

    # 3. CICIDS Episode Test Per-Class Comparison CSV
    cicids_test_per_class_csv = tables_dir / "cicids_test_per_class.csv"
    rf_pc = test_cicids["Random Forest"]["per_class"]
    xgb_pc = test_cicids["XGBoost"]["per_class"]
    mlp_pc = test_cicids["PyTorch MLP"]["per_class"]

    with open(cicids_test_per_class_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "Support", "RF_Precision", "RF_Recall", "RF_F1", "XGB_Precision", "XGB_Recall", "XGB_F1", "MLP_Precision", "MLP_Recall", "MLP_F1", "Best_Model"])
        for c in rf_pc.keys():
            supp = rf_pc[c]["support"]
            rf_f1, xgb_f1, mlp_f1 = rf_pc[c]["f1"], xgb_pc[c]["f1"], mlp_pc[c]["f1"]
            best = "XGBoost" if xgb_f1 >= max(rf_f1, mlp_f1) else ("Random Forest" if rf_f1 >= mlp_f1 else "PyTorch MLP")
            writer.writerow([
                c, supp,
                f"{rf_pc[c]['precision']:.4f}", f"{rf_pc[c]['recall']:.4f}", f"{rf_f1:.4f}",
                f"{xgb_pc[c]['precision']:.4f}", f"{xgb_pc[c]['recall']:.4f}", f"{xgb_f1:.4f}",
                f"{mlp_pc[c]['precision']:.4f}", f"{mlp_pc[c]['recall']:.4f}", f"{mlp_f1:.4f}",
                best,
            ])

    # 4. CICIDS Episode Class Weighting Ablation Summary CSV (UNSW ablation_class_weight_summary.csv preserved)
    ab_cicids_csv = tables_dir / "ablation_class_weight_cicids_summary.csv"
    ab_rows: list[list[str]] = []
    for m, ab_path, b_label in [
        ("Random Forest", "results/experiments/ablation_class_weight_cicids_rf/results_summary.json", "balanced_subsample"),
        ("XGBoost", "results/experiments/ablation_class_weight_cicids_xgb/results_summary.json", "balanced"),
        ("PyTorch MLP", "results/experiments/ablation_class_weight_cicids_mlp/results_summary.json", "balanced"),
    ]:
        data = _load_json(Path(ab_path))
        for row in data:
            policy = b_label if row.get("ablation_factor") is None else str(row.get("ablation_value"))
            acc = row.get("accuracy_mean", 0.0)
            macro = row.get("macro_f1_mean", 0.0)
            bp = row.get("binary_precision_mean", 0.0)
            br = row.get("binary_recall_mean", 0.0)
            bf1 = row.get("binary_f1_mean", 0.0)
            bap = row.get("binary_average_precision_mean", 0.0)
            miss_rate = 1.0 - br
            ab_rows.append([m, policy, f"{acc:.4f}", f"{macro:.4f}", f"{bp:.4f}", f"{br:.4f}", f"{bf1:.4f}", f"{bap:.4f}", f"{miss_rate*100:.2f}%"])

    with open(ab_cicids_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Weighting_Policy", "Accuracy", "Macro_F1", "Binary_Precision", "Binary_Recall", "Binary_F1", "Binary_AP", "Attack_Miss_Rate"])
        writer.writerows(ab_rows)

    # 5. Multiseed RF Summary Table for CICIDS Episode (UNSW multiseed_stability_rf_summary.csv preserved)
    shutil.copy2("results/experiments/multiseed_stability_cicids_rf/results_summary.csv", tables_dir / "multiseed_stability_cicids_rf_summary.csv")

    # 6. Training Sample Budget Scaling Summary
    budget_csv = tables_dir / "training_sample_budget_summary.csv"
    with open(budget_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Model", "Sample_Budget", "Validation_Flows", "Accuracy", "Macro_F1", "Binary_Precision", "Binary_Recall", "Binary_F1", "Ablation_Status"])

        rf_smoke_20k = _get_metrics(Path("results/rf_unsw_smoke_20k/metrics.json"))
        rf_ab_50k = _load_json(Path("results/experiments/ablation_class_weight_rf/results_summary.json"))[0]
        writer.writerow(["UNSW", "Random Forest", "20,000", "235,000", f"{rf_smoke_20k['accuracy']:.4f}", f"{rf_smoke_20k['macro_f1']:.4f}", f"{rf_smoke_20k['binary_precision']:.4f}", f"{rf_smoke_20k['binary_recall']:.4f}", f"{rf_smoke_20k['binary_f1']:.4f}", "Completed (Sample budget)"])
        writer.writerow(["UNSW", "Random Forest", "50,000", "235,000", f"{rf_ab_50k['accuracy_mean']:.4f}", f"{rf_ab_50k['macro_f1_mean']:.4f}", f"{rf_ab_50k['binary_precision_mean']:.4f}", f"{rf_ab_50k['binary_recall_mean']:.4f}", f"{rf_ab_50k['binary_f1_mean']:.4f}", "Completed (Sample budget)"])
        writer.writerow(["UNSW", "Random Forest", "1,670,000 (Full)", "235,000", f"{val_unsw['Random Forest']['accuracy']:.4f}", f"{val_unsw['Random Forest']['macro_f1']:.4f}", f"{val_unsw['Random Forest']['binary_precision']:.4f}", f"{val_unsw['Random Forest']['binary_recall']:.4f}", f"{val_unsw['Random Forest']['binary_f1']:.4f}", "Completed (Primary full)"])

        xgb_ab_50k = _load_json(Path("results/experiments/ablation_class_weight_xgb/results_summary.json"))[0]
        writer.writerow(["UNSW", "XGBoost", "50,000", "235,000", f"{xgb_ab_50k['accuracy_mean']:.4f}", f"{xgb_ab_50k['macro_f1_mean']:.4f}", f"{xgb_ab_50k['binary_precision_mean']:.4f}", f"{xgb_ab_50k['binary_recall_mean']:.4f}", f"{xgb_ab_50k['binary_f1_mean']:.4f}", "Completed (Sample budget)"])
        writer.writerow(["UNSW", "XGBoost", "1,670,000 (Full)", "235,000", f"{val_unsw['XGBoost']['accuracy']:.4f}", f"{val_unsw['XGBoost']['macro_f1']:.4f}", f"{val_unsw['XGBoost']['binary_precision']:.4f}", f"{val_unsw['XGBoost']['binary_recall']:.4f}", f"{val_unsw['XGBoost']['binary_f1']:.4f}", "Completed (Primary full)"])

        mlp_smoke_2k = _get_metrics(Path("results/mlp_unsw_smoke_2k/metrics.json"))
        mlp_ab_50k = _load_json(Path("results/experiments/ablation_class_weight_mlp/results_summary.json"))[0]
        writer.writerow(["UNSW", "PyTorch MLP", "2,000", "235,000", f"{mlp_smoke_2k['accuracy']:.4f}", f"{mlp_smoke_2k['macro_f1']:.4f}", f"{mlp_smoke_2k['binary_precision']:.4f}", f"{mlp_smoke_2k['binary_recall']:.4f}", f"{mlp_smoke_2k['binary_f1']:.4f}", "Completed (Sample budget)"])
        writer.writerow(["UNSW", "PyTorch MLP", "50,000", "235,000", f"{mlp_ab_50k['accuracy_mean']:.4f}", f"{mlp_ab_50k['macro_f1_mean']:.4f}", f"{mlp_ab_50k['binary_precision_mean']:.4f}", f"{mlp_ab_50k['binary_recall_mean']:.4f}", f"{mlp_ab_50k['binary_f1_mean']:.4f}", "Completed (Sample budget)"])
        writer.writerow(["UNSW", "PyTorch MLP", "1,670,000 (Full)", "235,000", f"{val_unsw['PyTorch MLP']['accuracy']:.4f}", f"{val_unsw['PyTorch MLP']['macro_f1']:.4f}", f"{val_unsw['PyTorch MLP']['binary_precision']:.4f}", f"{val_unsw['PyTorch MLP']['binary_recall']:.4f}", f"{val_unsw['PyTorch MLP']['binary_f1']:.4f}", "Completed (Primary full)"])

        for label, key in [("Random Forest", "rf"), ("XGBoost", "xgb"), ("PyTorch MLP", "mlp")]:
            smoke_m = _get_metrics(Path(f"results/smoke_cicids_episode/{key}/metrics.json"))
            val_m = val_cicids[label]
            writer.writerow(["CICIDS_EPISODE", label, "2,000", "1,710,000", f"{smoke_m['accuracy']:.4f}", f"{smoke_m['macro_f1']:.4f}", f"{smoke_m['binary_precision']:.4f}", f"{smoke_m['binary_recall']:.4f}", f"{smoke_m['binary_f1']:.4f}", "Completed (Smoke check)"])
            writer.writerow(["CICIDS_EPISODE", label, "100,000", "1,710,000", f"{val_m['accuracy']:.4f}", f"{val_m['macro_f1']:.4f}", f"{val_m['binary_precision']:.4f}", f"{val_m['binary_recall']:.4f}", f"{val_m['binary_f1']:.4f}", "Completed (100k CPU bounded budget)"])

    # 7. Model Timing Summary CSV (UNSW preserved + CICIDS Episode dynamic from run_config/metrics)
    timing_csv = tables_dir / "model_timing_summary.csv"
    with open(timing_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Model", "Train_Flows", "Train_Time_Seconds", "Test_Flows", "Test_Inference_Seconds", "Test_Throughput_Flows_Sec", "Hardware"])
        writer.writerow(["UNSW", "Random Forest", "1,670,000", "338.4", "475,000", "0.72", "658,882", "CPU (4 workers, scikit-learn)"])
        writer.writerow(["UNSW", "XGBoost", "1,670,000", "71.7", "475,000", "4.88", "97,435", "CPU (4 workers, tree_method=hist)"])
        writer.writerow(["UNSW", "PyTorch MLP", "1,670,000", "219.8", "475,000", "0.16", "3,055,633", "CPU (PyTorch batch=512)"])

        hw_map = {
            "rf": "CPU (4 workers, scikit-learn)",
            "xgb": "CPU (4 workers, tree_method=hist)",
            "mlp": "CPU (PyTorch batch=512)",
        }
        for label, key in [("Random Forest", "rf"), ("XGBoost", "xgb"), ("PyTorch MLP", "mlp")]:
            tr_cfg = _load_json(Path(f"results/experiments/cicids_episode_100k_seed42/{key}/run_config.json"))
            tr_met = _load_json(Path(f"results/experiments/cicids_episode_100k_seed42/{key}/metrics.json"))
            te_met = _load_json(Path(f"results/test_evaluation/cicids_episode/{key}/metrics.json"))
            n_train = int(tr_cfg["train_matrix_shape"]["rows"])
            tr_sec = float(tr_met["train_time_seconds"])
            n_test = int(te_met["n_samples"])
            te_sec = float(te_met["inference_duration_seconds"])
            fps = float(te_met["flows_per_second"])
            writer.writerow(["CICIDS_EPISODE", label, f"{n_train:,}", f"{tr_sec:.2f}", f"{n_test:,}", f"{te_sec:.2f}", f"{fps:,.0f}", hw_map[key]])

    # 8. Copy detailed per-class CSVs for CICIDS Episode (UNSW already intact)
    for key in ("rf", "xgb", "mlp"):
        shutil.copy2(f"results/test_evaluation/cicids_episode/{key}/per_class_metrics.csv", per_class_dir / f"cicids_test_{key}_per_class.csv")

    # 9. Copy CICIDS Episode test figures & generate summary plots
    for key in ("rf", "xgb", "mlp"):
        shutil.copy2(f"results/test_evaluation/cicids_episode/{key}/confusion_matrix_binary.png", figures_dir / f"cicids_episode_test_{key}_confusion_matrix_binary.png")
        shutil.copy2(f"results/test_evaluation/cicids_episode/{key}/confusion_matrix_normalized.png", figures_dir / f"cicids_episode_test_{key}_confusion_matrix_normalized.png")
        shutil.copy2(f"results/test_evaluation/cicids_episode/{key}/precision_recall_curve.png", figures_dir / f"cicids_episode_test_{key}_pr_curve.png")

    _plot_gnn_vs_baselines(test_cicids, figures_dir / "cicids_episode_gnn_vs_baselines.png")
    _plot_cicids_ablation(ab_rows, figures_dir / "cicids_episode_class_weight_ablation.png")

    print("[INFO] Successfully packaged all updated Part 2 research deliverables to reports/part2/")


if __name__ == "__main__":
    package_reports()
