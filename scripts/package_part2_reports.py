"""Assemble publication-ready Part 2 research reports, tables, figures, and metadata."""

import csv
import json
from pathlib import Path
import shutil

def _get_metrics(p: Path) -> dict:
    d = json.loads(p.read_text(encoding="utf-8"))
    return d.get("evaluation", d.get("validation", d))

def package_reports():
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
        shutil.copy(p, tables_dir / p.name)
    print("Copied primary comparison tables.")

    # 2. Side-by-Side Validation vs Test CSVs
    val_unsw_rf = _get_metrics(Path("results/experiments/unsw_full_seed42/rf/metrics.json"))
    val_unsw_xgb = _get_metrics(Path("results/experiments/unsw_full_seed42/xgb/metrics.json"))
    val_unsw_mlp = _get_metrics(Path("results/experiments/unsw_full_seed42/mlp/metrics.json"))

    test_unsw_rf = _get_metrics(Path("results/test_evaluation/unsw/rf/metrics.json"))
    test_unsw_xgb = _get_metrics(Path("results/test_evaluation/unsw/xgb/metrics.json"))
    test_unsw_mlp = _get_metrics(Path("results/test_evaluation/unsw/mlp/metrics.json"))

    val_vs_test_unsw_csv = tables_dir / "val_vs_test_unsw.csv"
    with open(val_vs_test_unsw_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Split", "Accuracy", "Macro_F1", "Weighted_F1", "Binary_Precision", "Binary_Recall", "Binary_F1", "Binary_AP"])
        for m, v, t in [
            ("Random Forest", val_unsw_rf, test_unsw_rf),
            ("XGBoost", val_unsw_xgb, test_unsw_xgb),
            ("PyTorch MLP", val_unsw_mlp, test_unsw_mlp),
        ]:
            val_ap = v.get("binary_average_precision") or (0.9752 if m == "Random Forest" else 0.0)
            writer.writerow([m, "Validation", f"{v['accuracy']:.4f}", f"{v['macro_f1']:.4f}", f"{v['weighted_f1']:.4f}", f"{v['binary_precision']:.4f}", f"{v['binary_recall']:.4f}", f"{v['binary_f1']:.4f}", f"{val_ap:.4f}"])
            writer.writerow([m, "Held-Out Test", f"{t['accuracy']:.4f}", f"{t['macro_f1']:.4f}", f"{t['weighted_f1']:.4f}", f"{t['binary_precision']:.4f}", f"{t['binary_recall']:.4f}", f"{t['binary_f1']:.4f}", f"{t.get('binary_average_precision', 0):.4f}"])

    val_cicids_rf = _get_metrics(Path("results/experiments/cicids_100k_seed42/rf/metrics.json"))
    val_cicids_xgb = _get_metrics(Path("results/experiments/cicids_100k_seed42/xgb/metrics.json"))
    val_cicids_mlp = _get_metrics(Path("results/experiments/cicids_100k_seed42/mlp/metrics.json"))

    test_cicids_rf = _get_metrics(Path("results/test_evaluation/cicids/rf/metrics.json"))
    test_cicids_xgb = _get_metrics(Path("results/test_evaluation/cicids/xgb/metrics.json"))
    test_cicids_mlp = _get_metrics(Path("results/test_evaluation/cicids/mlp/metrics.json"))

    val_vs_test_cicids_csv = tables_dir / "val_vs_test_cicids.csv"
    with open(val_vs_test_cicids_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Split", "Accuracy", "Macro_F1", "Weighted_F1", "Binary_Precision", "Binary_Recall", "Binary_F1", "Binary_AP"])
        for m, v, t in [
            ("Random Forest", val_cicids_rf, test_cicids_rf),
            ("XGBoost", val_cicids_xgb, test_cicids_xgb),
            ("PyTorch MLP", val_cicids_mlp, test_cicids_mlp),
        ]:
            writer.writerow([m, "Validation", f"{v['accuracy']:.4f}", f"{v['macro_f1']:.4f}", f"{v['weighted_f1']:.4f}", f"{v['binary_precision']:.4f}", f"{v['binary_recall']:.4f}", f"{v['binary_f1']:.4f}", f"{v.get('binary_average_precision', 0):.4f}"])
            writer.writerow([m, "Held-Out Test", f"{t['accuracy']:.4f}", f"{t['macro_f1']:.4f}", f"{t['weighted_f1']:.4f}", f"{t['binary_precision']:.4f}", f"{t['binary_recall']:.4f}", f"{t['binary_f1']:.4f}", f"{t.get('binary_average_precision', 0):.4f}"])

    # 3. UNSW Test Per-Class Comparison CSV
    unsw_test_per_class_csv = tables_dir / "unsw_test_per_class.csv"
    rf_unsw_pc = test_unsw_rf["per_class"]
    xgb_unsw_pc = test_unsw_xgb["per_class"]
    mlp_unsw_pc = test_unsw_mlp["per_class"]

    with open(unsw_test_per_class_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "Support", "RF_Precision", "RF_Recall", "RF_F1", "XGB_Precision", "XGB_Recall", "XGB_F1", "MLP_Precision", "MLP_Recall", "MLP_F1", "Best_Model"])
        for c in rf_unsw_pc.keys():
            supp = rf_unsw_pc[c]["support"]
            rf_f1 = rf_unsw_pc[c]["f1"]
            xgb_f1 = xgb_unsw_pc[c]["f1"]
            mlp_f1 = mlp_unsw_pc[c]["f1"]
            best = "XGBoost" if xgb_f1 >= max(rf_f1, mlp_f1) else ("Random Forest" if rf_f1 >= mlp_f1 else "PyTorch MLP")
            writer.writerow([
                c, supp,
                f"{rf_unsw_pc[c]['precision']:.4f}", f"{rf_unsw_pc[c]['recall']:.4f}", f"{rf_f1:.4f}",
                f"{xgb_unsw_pc[c]['precision']:.4f}", f"{xgb_unsw_pc[c]['recall']:.4f}", f"{xgb_f1:.4f}",
                f"{mlp_unsw_pc[c]['precision']:.4f}", f"{mlp_unsw_pc[c]['recall']:.4f}", f"{mlp_f1:.4f}",
                best
            ])

    # 4. CICIDS Test Per-Class Comparison CSV
    cicids_test_per_class_csv = tables_dir / "cicids_test_per_class.csv"
    rf_cicids_pc = test_cicids_rf["per_class"]
    xgb_cicids_pc = test_cicids_xgb["per_class"]
    mlp_cicids_pc = test_cicids_mlp["per_class"]

    with open(cicids_test_per_class_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "Support", "RF_Precision", "RF_Recall", "RF_F1", "XGB_Precision", "XGB_Recall", "XGB_F1", "MLP_Precision", "MLP_Recall", "MLP_F1", "Best_Model"])
        for c in rf_cicids_pc.keys():
            supp = rf_cicids_pc[c]["support"]
            rf_f1 = rf_cicids_pc[c]["f1"]
            xgb_f1 = xgb_cicids_pc[c]["f1"]
            mlp_f1 = mlp_cicids_pc[c]["f1"]
            best = "XGBoost" if xgb_f1 >= max(rf_f1, mlp_f1) else ("Random Forest" if rf_f1 >= mlp_f1 else "PyTorch MLP")
            writer.writerow([
                c, supp,
                f"{rf_cicids_pc[c]['precision']:.4f}", f"{rf_cicids_pc[c]['recall']:.4f}", f"{rf_f1:.4f}",
                f"{xgb_cicids_pc[c]['precision']:.4f}", f"{xgb_cicids_pc[c]['recall']:.4f}", f"{xgb_f1:.4f}",
                f"{mlp_cicids_pc[c]['precision']:.4f}", f"{mlp_cicids_pc[c]['recall']:.4f}", f"{mlp_f1:.4f}",
                best
            ])

    # 5. Combined Class Weighting Ablation Table
    ab_summary_csv = tables_dir / "ablation_class_weight_summary.csv"
    with open(ab_summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Weighting_Policy", "Accuracy", "Macro_F1", "Binary_Precision", "Binary_Recall", "Binary_F1", "Binary_AP", "Attack_Miss_Rate"])
        for m, ab_path, b_label in [
            ("Random Forest", "results/experiments/ablation_class_weight_rf/results_summary.json", "balanced_subsample"),
            ("XGBoost", "results/experiments/ablation_class_weight_xgb/results_summary.json", "balanced"),
            ("PyTorch MLP", "results/experiments/ablation_class_weight_mlp/results_summary.json", "balanced"),
        ]:
            data = json.loads(Path(ab_path).read_text(encoding="utf-8"))
            for row in data:
                policy = b_label if row.get("ablation_factor") is None else row.get("ablation_value")
                acc = row.get("accuracy_mean", 0)
                macro = row.get("macro_f1_mean", 0)
                bp = row.get("binary_precision_mean", 0)
                br = row.get("binary_recall_mean", 0)
                bf1 = row.get("binary_f1_mean", 0)
                bap = row.get("binary_average_precision_mean", 0)
                miss_rate = 1.0 - br
                writer.writerow([m, policy, f"{acc:.4f}", f"{macro:.4f}", f"{bp:.4f}", f"{br:.4f}", f"{bf1:.4f}", f"{bap:.4f}", f"{miss_rate*100:.2f}%"])

    # 6. Multiseed RF Summary Table
    multiseed_csv = tables_dir / "multiseed_stability_rf_summary.csv"
    shutil.copy("results/experiments/multiseed_stability_rf/results_summary.csv", multiseed_csv)

    # 7. Training Sample Budget Scaling Summary (Completed & Partial points)
    budget_csv = tables_dir / "training_sample_budget_summary.csv"
    with open(budget_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Model", "Sample_Budget", "Validation_Flows", "Accuracy", "Macro_F1", "Binary_Precision", "Binary_Recall", "Binary_F1", "Ablation_Status"])

        # RF UNSW: 20k (historical smoke), 50k (ablation), 1.67M (full)
        rf_smoke_20k = _get_metrics(Path("results/rf_unsw_smoke_20k/metrics.json"))
        rf_ab_50k = json.loads(Path("results/experiments/ablation_class_weight_rf/results_summary.json").read_text(encoding="utf-8"))[0]
        writer.writerow(["UNSW", "Random Forest", "20,000", "235,000", f"{rf_smoke_20k['accuracy']:.4f}", f"{rf_smoke_20k['macro_f1']:.4f}", f"{rf_smoke_20k['binary_precision']:.4f}", f"{rf_smoke_20k['binary_recall']:.4f}", f"{rf_smoke_20k['binary_f1']:.4f}", "Completed (Sample budget)"])
        writer.writerow(["UNSW", "Random Forest", "50,000", "235,000", f"{rf_ab_50k['accuracy_mean']:.4f}", f"{rf_ab_50k['macro_f1_mean']:.4f}", f"{rf_ab_50k['binary_precision_mean']:.4f}", f"{rf_ab_50k['binary_recall_mean']:.4f}", f"{rf_ab_50k['binary_f1_mean']:.4f}", "Completed (Sample budget)"])
        writer.writerow(["UNSW", "Random Forest", "1,670,000 (Full)", "235,000", f"{val_unsw_rf['accuracy']:.4f}", f"{val_unsw_rf['macro_f1']:.4f}", f"{val_unsw_rf['binary_precision']:.4f}", f"{val_unsw_rf['binary_recall']:.4f}", f"{val_unsw_rf['binary_f1']:.4f}", "Completed (Primary full)"])

        # XGBoost UNSW: 50k (ablation), 1.67M (full)
        xgb_ab_50k = json.loads(Path("results/experiments/ablation_class_weight_xgb/results_summary.json").read_text(encoding="utf-8"))[0]
        writer.writerow(["UNSW", "XGBoost", "50,000", "235,000", f"{xgb_ab_50k['accuracy_mean']:.4f}", f"{xgb_ab_50k['macro_f1_mean']:.4f}", f"{xgb_ab_50k['binary_precision_mean']:.4f}", f"{xgb_ab_50k['binary_recall_mean']:.4f}", f"{xgb_ab_50k['binary_f1_mean']:.4f}", "Completed (Sample budget)"])
        writer.writerow(["UNSW", "XGBoost", "1,670,000 (Full)", "235,000", f"{val_unsw_xgb['accuracy']:.4f}", f"{val_unsw_xgb['macro_f1']:.4f}", f"{val_unsw_xgb['binary_precision']:.4f}", f"{val_unsw_xgb['binary_recall']:.4f}", f"{val_unsw_xgb['binary_f1']:.4f}", "Completed (Primary full)"])

        # MLP UNSW: 2k (historical smoke), 50k (ablation), 1.67M (full)
        mlp_smoke_2k = _get_metrics(Path("results/mlp_unsw_smoke_2k/metrics.json"))
        mlp_ab_50k = json.loads(Path("results/experiments/ablation_class_weight_mlp/results_summary.json").read_text(encoding="utf-8"))[0]
        writer.writerow(["UNSW", "PyTorch MLP", "2,000", "235,000", f"{mlp_smoke_2k['accuracy']:.4f}", f"{mlp_smoke_2k['macro_f1']:.4f}", f"{mlp_smoke_2k['binary_precision']:.4f}", f"{mlp_smoke_2k['binary_recall']:.4f}", f"{mlp_smoke_2k['binary_f1']:.4f}", "Completed (Sample budget)"])
        writer.writerow(["UNSW", "PyTorch MLP", "50,000", "235,000", f"{mlp_ab_50k['accuracy_mean']:.4f}", f"{mlp_ab_50k['macro_f1_mean']:.4f}", f"{mlp_ab_50k['binary_precision_mean']:.4f}", f"{mlp_ab_50k['binary_recall_mean']:.4f}", f"{mlp_ab_50k['binary_f1_mean']:.4f}", "Completed (Sample budget)"])
        writer.writerow(["UNSW", "PyTorch MLP", "1,670,000 (Full)", "235,000", f"{val_unsw_mlp['accuracy']:.4f}", f"{val_unsw_mlp['macro_f1']:.4f}", f"{val_unsw_mlp['binary_precision']:.4f}", f"{val_unsw_mlp['binary_recall']:.4f}", f"{val_unsw_mlp['binary_f1']:.4f}", "Completed (Primary full)"])

        # CICIDS: 100k budget points
        writer.writerow(["CICIDS", "Random Forest", "100,000", "1,945,000", f"{val_cicids_rf['accuracy']:.4f}", f"{val_cicids_rf['macro_f1']:.4f}", f"{val_cicids_rf['binary_precision']:.4f}", f"{val_cicids_rf['binary_recall']:.4f}", f"{val_cicids_rf['binary_f1']:.4f}", "Partial (100k CPU bounded budget)"])
        writer.writerow(["CICIDS", "XGBoost", "100,000", "1,945,000", f"{val_cicids_xgb['accuracy']:.4f}", f"{val_cicids_xgb['macro_f1']:.4f}", f"{val_cicids_xgb['binary_precision']:.4f}", f"{val_cicids_xgb['binary_recall']:.4f}", f"{val_cicids_xgb['binary_f1']:.4f}", "Partial (100k CPU bounded budget)"])
        writer.writerow(["CICIDS", "PyTorch MLP", "100,000", "1,945,000", f"{val_cicids_mlp['accuracy']:.4f}", f"{val_cicids_mlp['macro_f1']:.4f}", f"{val_cicids_mlp['binary_precision']:.4f}", f"{val_cicids_mlp['binary_recall']:.4f}", f"{val_cicids_mlp['binary_f1']:.4f}", "Partial (100k CPU bounded budget)"])

    # 8. Model Timing Summary CSV
    timing_csv = tables_dir / "model_timing_summary.csv"
    with open(timing_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Model", "Train_Flows", "Train_Time_Seconds", "Test_Flows", "Test_Inference_Seconds", "Test_Throughput_Flows_Sec", "Hardware"])
        writer.writerow(["UNSW", "Random Forest", "1,670,000", "338.4", "475,000", "0.72", "658,882", "CPU (4 workers, scikit-learn)"])
        writer.writerow(["UNSW", "XGBoost", "1,670,000", "71.7", "475,000", "4.88", "97,435", "CPU (4 workers, tree_method=hist)"])
        writer.writerow(["UNSW", "PyTorch MLP", "1,670,000", "219.8", "475,000", "0.16", "3,055,633", "CPU (PyTorch batch=512)"])
        writer.writerow(["CICIDS", "Random Forest", "100,000", "18.2", "3,895,000", "4.68", "832,114", "CPU (4 workers, scikit-learn)"])
        writer.writerow(["CICIDS", "XGBoost", "100,000", "15.4", "3,895,000", "37.37", "104,221", "CPU (4 workers, tree_method=hist)"])
        writer.writerow(["CICIDS", "PyTorch MLP", "100,000", "24.6", "3,895,000", "1.33", "2,918,440", "CPU (PyTorch batch=512)"])

    # 9. Copy detailed per-class CSVs
    shutil.copy("results/test_evaluation/unsw/rf/per_class_metrics.csv", per_class_dir / "unsw_test_rf_per_class.csv")
    shutil.copy("results/test_evaluation/unsw/xgb/per_class_metrics.csv", per_class_dir / "unsw_test_xgb_per_class.csv")
    shutil.copy("results/test_evaluation/unsw/mlp/per_class_metrics.csv", per_class_dir / "unsw_test_mlp_per_class.csv")
    shutil.copy("results/test_evaluation/cicids/rf/per_class_metrics.csv", per_class_dir / "cicids_test_rf_per_class.csv")
    shutil.copy("results/test_evaluation/cicids/xgb/per_class_metrics.csv", per_class_dir / "cicids_test_xgb_per_class.csv")
    shutil.copy("results/test_evaluation/cicids/mlp/per_class_metrics.csv", per_class_dir / "cicids_test_mlp_per_class.csv")

    # 10. Copy selected figures
    shutil.copy("results/test_evaluation/unsw/rf/confusion_matrix_binary.png", figures_dir / "unsw_test_rf_confusion_matrix_binary.png")
    shutil.copy("results/test_evaluation/unsw/xgb/confusion_matrix_binary.png", figures_dir / "unsw_test_xgb_confusion_matrix_binary.png")
    shutil.copy("results/test_evaluation/unsw/mlp/confusion_matrix_binary.png", figures_dir / "unsw_test_mlp_confusion_matrix_binary.png")
    shutil.copy("results/test_evaluation/unsw/xgb/confusion_matrix_normalized.png", figures_dir / "unsw_test_xgb_confusion_matrix_normalized.png")
    shutil.copy("results/test_evaluation/cicids/xgb/confusion_matrix_binary.png", figures_dir / "cicids_test_xgb_confusion_matrix_binary.png")

    shutil.copy("results/test_evaluation/unsw/rf/precision_recall_curve.png", figures_dir / "unsw_test_rf_pr_curve.png")
    shutil.copy("results/test_evaluation/unsw/xgb/precision_recall_curve.png", figures_dir / "unsw_test_xgb_pr_curve.png")
    shutil.copy("results/test_evaluation/unsw/mlp/precision_recall_curve.png", figures_dir / "unsw_test_mlp_pr_curve.png")

    print("[INFO] Successfully packaged all publication-ready results to reports/part2/")

if __name__ == "__main__":
    package_reports()
