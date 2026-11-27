"""Generate comprehensive comparison tables (Markdown, LaTeX, CSV) for Part 2 baselines, validation vs test, CICIDS episode split, cross-dataset transfer, and GNN comparison."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_metric_dict(path: Path) -> Dict[str, Any]:
    raw = _load_json(path)
    if "evaluation" in raw:
        return raw["evaluation"]
    if "validation" in raw:
        return raw["validation"]
    return raw


def _load_sample_count(path: Path, split_key: str) -> int:
    raw = _load_json(path)
    if "n_samples" in raw and isinstance(raw["n_samples"], int):
        return int(raw["n_samples"])
    m = _load_metric_dict(path)
    if "confusion_matrix" in m:
        return int(sum(sum(row) for row in m["confusion_matrix"]))
    raise ValueError(f"Could not determine sample count from {path} ({split_key})")


def _write_baseline_table(out_dir: Path, stem: str, results: Dict[str, Dict[str, Any]]) -> None:
    csv_path = out_dir / f"{stem}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Model",
                "Accuracy",
                "Macro_Precision",
                "Macro_Recall",
                "Macro_F1",
                "Weighted_Precision",
                "Weighted_Recall",
                "Weighted_F1",
                "Binary_Precision",
                "Binary_Recall",
                "Binary_F1",
                "Binary_AP",
            ]
        )
        for model, m in results.items():
            writer.writerow(
                [
                    model,
                    f"{m['accuracy']:.6f}",
                    f"{m['macro_precision']:.6f}",
                    f"{m['macro_recall']:.6f}",
                    f"{m['macro_f1']:.6f}",
                    f"{m['weighted_precision']:.6f}",
                    f"{m['weighted_recall']:.6f}",
                    f"{m['weighted_f1']:.6f}",
                    f"{m['binary_precision']:.6f}",
                    f"{m['binary_recall']:.6f}",
                    f"{m['binary_f1']:.6f}",
                    f"{m.get('binary_average_precision', 0.0):.6f}",
                ]
            )

    md_lines = [
        "| Model | Accuracy | Macro F1 | Weighted F1 | Binary Precision | Binary Recall | Binary F1 | Binary AP |",
        "|---|---|---|---|---|---|---|---|",
    ]
    tex_lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"  \toprule",
        r"  Model & Accuracy & Macro\,F1 & Weighted\,F1 & Bin.\,Prec & Bin.\,Rec & Binary\,F1 & Binary\,AP \\",
        r"  \midrule",
    ]
    for model, m in results.items():
        ap = m.get("binary_average_precision", 0.0)
        md_lines.append(
            f"| {model} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['weighted_f1']:.4f} | "
            f"{m['binary_precision']:.4f} | {m['binary_recall']:.4f} | {m['binary_f1']:.4f} | {ap:.4f} |"
        )
        tex_lines.append(
            f"  {model} & {m['accuracy']:.4f} & {m['macro_f1']:.4f} & {m['weighted_f1']:.4f} & "
            f"{m['binary_precision']:.4f} & {m['binary_recall']:.4f} & {m['binary_f1']:.4f} & {ap:.4f} \\\\"
        )
    tex_lines.extend([r"  \bottomrule", r"\end{tabular}"])

    (out_dir / f"{stem}.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    (out_dir / f"{stem}.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def _write_val_vs_test_table(
    out_dir: Path,
    stem: str,
    val_results: Dict[str, Dict[str, Any]],
    test_results: Dict[str, Dict[str, Any]],
) -> None:
    csv_path = out_dir / f"{stem}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Model",
                "Split",
                "Accuracy",
                "Macro_F1",
                "Weighted_F1",
                "Binary_Precision",
                "Binary_Recall",
                "Binary_F1",
                "Binary_AP",
            ]
        )
        for m in val_results:
            for split_label, d in [("Validation", val_results[m]), ("Held-Out Test", test_results[m])]:
                writer.writerow(
                    [
                        m,
                        split_label,
                        f"{d['accuracy']:.6f}",
                        f"{d['macro_f1']:.6f}",
                        f"{d['weighted_f1']:.6f}",
                        f"{d['binary_precision']:.6f}",
                        f"{d['binary_recall']:.6f}",
                        f"{d['binary_f1']:.6f}",
                        f"{d.get('binary_average_precision', 0.0):.6f}",
                    ]
                )

    md_lines = [
        "| Model | Split | Accuracy | Macro F1 | Weighted F1 | Binary Precision | Binary Recall | Binary F1 | Binary AP |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    tex_lines = [
        r"\begin{tabular}{llrrrrrrr}",
        r"  \toprule",
        r"  Model & Split & Accuracy & Macro\,F1 & W-F1 & Precision & Recall & Binary\,F1 & Average\,Precision \\",
        r"  \midrule",
    ]
    for m in val_results:
        v = val_results[m]
        t = test_results[m]
        val_ap = v.get("binary_average_precision", 0.0)
        test_ap = t.get("binary_average_precision", 0.0)
        md_lines.append(
            f"| {m} | Validation | {v['accuracy']:.4f} | {v['macro_f1']:.4f} | {v['weighted_f1']:.4f} | "
            f"{v['binary_precision']:.4f} | {v['binary_recall']:.4f} | {v['binary_f1']:.4f} | {val_ap:.4f} |"
        )
        md_lines.append(
            f"| {m} | Held-Out Test | {t['accuracy']:.4f} | {t['macro_f1']:.4f} | {t['weighted_f1']:.4f} | "
            f"{t['binary_precision']:.4f} | {t['binary_recall']:.4f} | {t['binary_f1']:.4f} | {test_ap:.4f} |"
        )
        tex_lines.append(
            f"  {m} & Val & {v['accuracy']:.4f} & {v['macro_f1']:.4f} & {v['weighted_f1']:.4f} & "
            f"{v['binary_precision']:.4f} & {v['binary_recall']:.4f} & {v['binary_f1']:.4f} & {val_ap:.4f} \\\\"
        )
        tex_lines.append(
            f"  {m} & Test & {t['accuracy']:.4f} & {t['macro_f1']:.4f} & {t['weighted_f1']:.4f} & "
            f"{t['binary_precision']:.4f} & {t['binary_recall']:.4f} & {t['binary_f1']:.4f} & {test_ap:.4f} \\\\"
        )
    tex_lines.extend([r"  \bottomrule", r"\end{tabular}"])

    (out_dir / f"{stem}.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    (out_dir / f"{stem}.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def generate_reports() -> None:
    comp_dir = Path("results/comparison")
    comp_dir.mkdir(parents=True, exist_ok=True)

    models = [("Random Forest", "rf"), ("XGBoost", "xgb"), ("PyTorch MLP", "mlp")]

    # 1. UNSW (preserved paths)
    unsw_val_results = {
        label: _load_metric_dict(Path(f"results/experiments/unsw_full_seed42/{key}/metrics.json"))
        for label, key in models
    }
    unsw_test_paths = {
        label: Path(f"results/test_evaluation/unsw/{key}/metrics.json") for label, key in models
    }
    unsw_test_results = {label: _load_metric_dict(p) for label, p in unsw_test_paths.items()}

    # 2. Corrected CICIDS Episode Split
    cicids_val_results = {
        label: _load_metric_dict(Path(f"results/experiments/cicids_episode_100k_seed42/{key}/metrics.json"))
        for label, key in models
    }
    cicids_test_paths = {
        label: Path(f"results/test_evaluation/cicids_episode/{key}/metrics.json") for label, key in models
    }
    cicids_test_results = {label: _load_metric_dict(p) for label, p in cicids_test_paths.items()}

    # Preserve UNSW tables unchanged; write corrected CICIDS Episode tables
    _write_baseline_table(comp_dir, "baseline_comparison_cicids", cicids_val_results)
    _write_baseline_table(comp_dir, "baseline_comparison_cicids_test", cicids_test_results)
    _write_val_vs_test_table(comp_dir, "val_vs_test_cicids", cicids_val_results, cicids_test_results)

    # 3. Full Test Comparison Summary (CSV)
    with open(comp_dir / "final_test_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Dataset",
                "Model",
                "Test_Flows",
                "Accuracy",
                "Macro_Precision",
                "Macro_Recall",
                "Macro_F1",
                "Weighted_Precision",
                "Weighted_Recall",
                "Weighted_F1",
                "Binary_Precision",
                "Binary_Recall",
                "Binary_F1",
                "Binary_AP",
            ]
        )
        for label, t in unsw_test_results.items():
            n_flows = _load_sample_count(unsw_test_paths[label], "test")
            writer.writerow(
                [
                    "UNSW",
                    label,
                    n_flows,
                    f"{t['accuracy']:.6f}",
                    f"{t['macro_precision']:.6f}",
                    f"{t['macro_recall']:.6f}",
                    f"{t['macro_f1']:.6f}",
                    f"{t['weighted_precision']:.6f}",
                    f"{t['weighted_recall']:.6f}",
                    f"{t['weighted_f1']:.6f}",
                    f"{t['binary_precision']:.6f}",
                    f"{t['binary_recall']:.6f}",
                    f"{t['binary_f1']:.6f}",
                    f"{t.get('binary_average_precision', 0):.6f}",
                ]
            )
        for label, t in cicids_test_results.items():
            n_flows = _load_sample_count(cicids_test_paths[label], "test")
            writer.writerow(
                [
                    "CICIDS_EPISODE",
                    label,
                    n_flows,
                    f"{t['accuracy']:.6f}",
                    f"{t['macro_precision']:.6f}",
                    f"{t['macro_recall']:.6f}",
                    f"{t['macro_f1']:.6f}",
                    f"{t['weighted_precision']:.6f}",
                    f"{t['weighted_recall']:.6f}",
                    f"{t['weighted_f1']:.6f}",
                    f"{t['binary_precision']:.6f}",
                    f"{t['binary_recall']:.6f}",
                    f"{t['binary_f1']:.6f}",
                    f"{t.get('binary_average_precision', 0):.6f}",
                ]
            )

    # 4. Per-Class F1 Table (CICIDS Episode Test; UNSW preserved in reports/part2/tables/unsw_test_per_class.md)
    cicids_classes = list(cicids_test_results["Random Forest"]["per_class"].keys())
    cicids_pc_md = [
        "| Attack Class | Test Support | RF F1 | XGBoost F1 | MLP F1 | Best Baseline |",
        "|---|---|---|---|---|---|",
    ]
    cicids_pc_tex = [
        r"\begin{tabular}{lrrrrl}",
        r"  \toprule",
        r"  Class & Support & RF\,F1 & XGBoost\,F1 & MLP\,F1 & Best \\",
        r"  \midrule",
    ]
    for c in cicids_classes:
        rf_f1 = cicids_test_results["Random Forest"]["per_class"][c]["f1"]
        xgb_f1 = cicids_test_results["XGBoost"]["per_class"][c]["f1"]
        mlp_f1 = cicids_test_results["PyTorch MLP"]["per_class"][c]["f1"]
        supp = cicids_test_results["Random Forest"]["per_class"][c]["support"]
        best = "XGBoost" if xgb_f1 >= max(rf_f1, mlp_f1) else ("Random Forest" if rf_f1 >= mlp_f1 else "PyTorch MLP")
        c_tex = c.replace("_", r"\_")
        cicids_pc_md.append(f"| {c} | {supp:,} | {rf_f1:.4f} | {xgb_f1:.4f} | {mlp_f1:.4f} | **{best}** |")
        cicids_pc_tex.append(f"  {c_tex} & {supp:,} & {rf_f1:.4f} & {xgb_f1:.4f} & {mlp_f1:.4f} & {best} \\\\")
    cicids_pc_tex.extend([r"  \bottomrule", r"\end{tabular}"])
    (comp_dir / "cicids_test_per_class.md").write_text("\n".join(cicids_pc_md) + "\n", encoding="utf-8")
    (comp_dir / "cicids_test_per_class.tex").write_text("\n".join(cicids_pc_tex) + "\n", encoding="utf-8")

    # 5. Cross-Dataset Transfer Comparison (Corrected Episode Split, Val & Test)
    cross_specs = [
        ("UNSW -> CICIDS (Episode)", "val", "unsw_to_cicids_val"),
        ("UNSW -> CICIDS (Episode)", "test", "unsw_to_cicids_test"),
        ("CICIDS (Episode) -> UNSW", "val", "cicids_to_unsw_val"),
        ("CICIDS (Episode) -> UNSW", "test", "cicids_to_unsw_test"),
    ]
    cross_rows: List[Dict[str, Any]] = []
    for direction, split, folder in cross_specs:
        for label, key in models:
            mpath = Path("results/cross_dataset_episode") / folder / key / "metrics.json"
            if not mpath.is_file():
                continue
            m = _load_json(mpath)
            n_eval = _load_sample_count(mpath, split)
            cross_rows.append(
                {
                    "direction": direction,
                    "split": split,
                    "model": label,
                    "n_flows": n_eval,
                    "accuracy": m["accuracy"],
                    "binary_precision": m["binary_precision"],
                    "binary_recall": m["binary_recall"],
                    "binary_f1": m["binary_f1"],
                    "binary_average_precision": m["binary_average_precision"],
                }
            )

    with open(comp_dir / "cross_dataset_comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Direction",
                "Split",
                "Model",
                "Target_Flows",
                "Accuracy",
                "Binary_Precision",
                "Binary_Recall",
                "Binary_F1",
                "Binary_AP",
            ]
        )
        for r in cross_rows:
            writer.writerow(
                [
                    r["direction"],
                    r["split"],
                    r["model"],
                    r["n_flows"],
                    f"{r['accuracy']:.6f}",
                    f"{r['binary_precision']:.6f}",
                    f"{r['binary_recall']:.6f}",
                    f"{r['binary_f1']:.6f}",
                    f"{r['binary_average_precision']:.6f}",
                ]
            )

    cross_md = [
        "| Direction | Split | Model | Target Flows | Accuracy | Binary Precision | Binary Recall | Binary F1 | Binary AP |",
        "|---|---|---|---:|---|---|---|---|---|",
    ]
    cross_tex = [
        r"\begin{tabular}{lllrrrrrr}",
        r"  \toprule",
        r"  Direction & Split & Model & Flows & Accuracy & Precision & Recall & Binary\,F1 & Average\,Precision \\",
        r"  \midrule",
    ]
    for r in cross_rows:
        cross_md.append(
            f"| {r['direction']} | {r['split']} | {r['model']} | {r['n_flows']:,} | {r['accuracy']:.4f} | "
            f"{r['binary_precision']:.4f} | {r['binary_recall']:.4f} | {r['binary_f1']:.4f} | {r['binary_average_precision']:.4f} |"
        )
        d_tex = r["direction"].replace("->", r"$\to$")
        cross_tex.append(
            f"  {d_tex} & {r['split']} & {r['model']} & {r['n_flows']:,} & {r['accuracy']:.4f} & "
            f"{r['binary_precision']:.4f} & {r['binary_recall']:.4f} & {r['binary_f1']:.4f} & {r['binary_average_precision']:.4f} \\\\"
        )
    cross_tex.extend([r"  \bottomrule", r"\end{tabular}"])
    (comp_dir / "cross_dataset_comparison.md").write_text("\n".join(cross_md) + "\n", encoding="utf-8")
    (comp_dir / "cross_dataset_comparison.tex").write_text("\n".join(cross_tex) + "\n", encoding="utf-8")

    # 6. Baselines vs Reference E-GraphSAGE (Corrected CICIDS Episode Test, 4,475,000 flows)
    # Reference GNN values from member3/results_cicids_episode/summary_3seeds.txt & README.md
    gnn_rows = [
        {
            "Model": "Random Forest (balanced_subsample)",
            "Family": "Tabular Baseline",
            "Train_Budget": "100,000",
            "Seeds": "1 (seed 42)",
            "Accuracy": f"{cicids_test_results['Random Forest']['accuracy']:.4f}",
            "Macro_F1": f"{cicids_test_results['Random Forest']['macro_f1']:.4f}",
            "Weighted_F1": f"{cicids_test_results['Random Forest']['weighted_f1']:.4f}",
            "Binary_F1": f"{cicids_test_results['Random Forest']['binary_f1']:.4f}",
            "Binary_AP": f"{cicids_test_results['Random Forest']['binary_average_precision']:.4f}",
        },
        {
            "Model": "XGBoost (balanced)",
            "Family": "Tabular Baseline",
            "Train_Budget": "100,000",
            "Seeds": "1 (seed 42)",
            "Accuracy": f"{cicids_test_results['XGBoost']['accuracy']:.4f}",
            "Macro_F1": f"{cicids_test_results['XGBoost']['macro_f1']:.4f}",
            "Weighted_F1": f"{cicids_test_results['XGBoost']['weighted_f1']:.4f}",
            "Binary_F1": f"{cicids_test_results['XGBoost']['binary_f1']:.4f}",
            "Binary_AP": f"{cicids_test_results['XGBoost']['binary_average_precision']:.4f}",
        },
        {
            "Model": "PyTorch MLP (balanced)",
            "Family": "Tabular Baseline",
            "Train_Budget": "100,000",
            "Seeds": "1 (seed 42)",
            "Accuracy": f"{cicids_test_results['PyTorch MLP']['accuracy']:.4f}",
            "Macro_F1": f"{cicids_test_results['PyTorch MLP']['macro_f1']:.4f}",
            "Weighted_F1": f"{cicids_test_results['PyTorch MLP']['weighted_f1']:.4f}",
            "Binary_F1": f"{cicids_test_results['PyTorch MLP']['binary_f1']:.4f}",
            "Binary_AP": f"{cicids_test_results['PyTorch MLP']['binary_average_precision']:.4f}",
        },
        {
            "Model": "E-GraphSAGE (mean, inverse_sqrt)",
            "Family": "Graph Neural Network",
            "Train_Budget": "13,295,000 (Full)",
            "Seeds": "3 (0, 1, 2)",
            "Accuracy": "0.9825",
            "Macro_F1": "0.651",
            "Weighted_F1": "0.9833",
            "Binary_F1": "0.939",
            "Binary_AP": "N/A",
        },
        {
            "Model": "E-GraphSAGE + Attention (inverse_sqrt)",
            "Family": "Graph Neural Network",
            "Train_Budget": "13,295,000 (Full)",
            "Seeds": "3 (0, 1, 2)",
            "Accuracy": "0.9813",
            "Macro_F1": "0.672",
            "Weighted_F1": "0.9825",
            "Binary_F1": "0.931",
            "Binary_AP": "N/A",
        },
    ]

    with open(comp_dir / "gnn_vs_baselines_cicids_episode.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(gnn_rows[0].keys()))
        writer.writeheader()
        writer.writerows(gnn_rows)

    gnn_md = [
        "| Model | Family | Train Budget | Seeds | Test Accuracy | Test Macro F1 | Test Weighted F1 | Test Binary F1 | Test Binary AP |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    gnn_tex = [
        r"\begin{tabular}{llrrrrrrr}",
        r"  \toprule",
        r"  Model & Train Budget & Seeds & Acc & Macro\,F1 & W-F1 & Binary\,F1 & Binary\,AP \\",
        r"  \midrule",
    ]
    for r in gnn_rows:
        gnn_md.append(
            f"| {r['Model']} | {r['Family']} | {r['Train_Budget']} | {r['Seeds']} | {r['Accuracy']} | {r['Macro_F1']} | {r['Weighted_F1']} | {r['Binary_F1']} | {r['Binary_AP']} |"
        )
        m_tex = r["Model"].replace("_", r"\_")
        gnn_tex.append(
            f"  {m_tex} & {r['Train_Budget']} & {r['Seeds']} & {r['Accuracy']} & {r['Macro_F1']} & {r['Weighted_F1']} & {r['Binary_F1']} & {r['Binary_AP']} \\\\"
        )
    gnn_tex.extend([r"  \bottomrule", r"\end{tabular}"])
    (comp_dir / "gnn_vs_baselines_cicids_episode.md").write_text("\n".join(gnn_md) + "\n", encoding="utf-8")
    (comp_dir / "gnn_vs_baselines_cicids_episode.tex").write_text("\n".join(gnn_tex) + "\n", encoding="utf-8")

    print("[INFO] All comparison reports successfully written to:", comp_dir)


if __name__ == "__main__":
    generate_reports()
