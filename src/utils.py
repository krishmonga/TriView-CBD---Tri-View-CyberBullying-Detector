"""
Plotting and reporting utilities.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import Dict, List


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_training_history(history: Dict, model_name: str, save_dir: str):
    ensure_dir(save_dir)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_loss"], label="Train Loss")
    if "val_loss" in history:
        ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name} — Loss")
    ax1.legend()

    ax1b = ax1.twinx()
    if "learning_rates" in history:
        ax1b.plot(history["learning_rates"], color="gray", alpha=0.4, linestyle="--", label="LR")
        ax1b.set_ylabel("Learning Rate")

    ax2.plot(history["train_acc"], label="Train Acc")
    ax2.plot(history["val_acc"], label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{model_name} — Accuracy")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{model_name}_training.png"), dpi=150)
    plt.close(fig)


def plot_confusion_matrix_with_metrics(labels, preds, class_names,
                                       model_name: str, save_dir: str):
    ensure_dir(save_dir)
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title(f"{model_name} — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{model_name}_confusion.png"), dpi=150)
    plt.close(fig)


def plot_model_comparison(results: Dict[str, Dict], save_dir: str):
    ensure_dir(save_dir)
    models = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

    x = np.arange(len(models))
    width = 0.18
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (m, lab) in enumerate(zip(metrics, labels)):
        vals = [results[model].get(m, 0) * 100 for model in models]
        ax.bar(x + i * width, vals, width, label=lab)

    ax.set_ylabel("Score (%)")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([m.replace("_", " ").title() for m in models], rotation=25, ha="right")
    ax.legend()
    ax.set_ylim(70, 100)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "model_comparison.png"), dpi=150)
    plt.close(fig)


def plot_kfold_results(kfold_results: Dict, save_dir: str):
    ensure_dir(save_dir)
    models = list(kfold_results.keys())
    means = [kfold_results[m]["mean_accuracy"] * 100 for m in models]
    stds = [kfold_results[m]["std_accuracy"] * 100 for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", edgecolor="black")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("5-Fold Cross-Validation Results (Mean ± Std)")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in models], rotation=25, ha="right")
    ax.set_ylim(70, 100)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{m:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "kfold_comparison.png"), dpi=150)
    plt.close(fig)


def create_comprehensive_report(all_results: Dict, kfold_results: Dict,
                                save_path: str):
    ensure_dir(os.path.dirname(save_path))
    report = {
        "single_split_results": {},
        "kfold_results": {},
    }

    for name, res in all_results.items():
        report["single_split_results"][name] = {
            k: float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in res.items()
            if k not in ("predictions", "labels", "history", "confusion_matrix")
        }

    for name, res in kfold_results.items():
        report["kfold_results"][name] = {
            "mean_accuracy": float(res["mean_accuracy"]),
            "std_accuracy": float(res["std_accuracy"]),
            "mean_precision": float(res.get("mean_precision", 0)),
            "std_precision": float(res.get("std_precision", 0)),
            "mean_recall": float(res.get("mean_recall", 0)),
            "std_recall": float(res.get("std_recall", 0)),
            "mean_f1_score": float(res["mean_f1_score"]),
            "std_f1_score": float(res.get("std_f1_score", 0)),
            "fold_accuracies": [float(a) for a in res["fold_accuracies"]],
            "fold_f1_scores": [float(f) for f in res["fold_f1_scores"]],
            "fold_precisions": [float(p) for p in res.get("fold_precisions", [])],
            "fold_recalls": [float(r) for r in res.get("fold_recalls", [])],
        }

    with open(save_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {save_path}")
    return report


def generate_latex_tables(all_results: Dict, kfold_results: Dict,
                          save_dir: str):
    """Generate LaTeX tables matching the paper format."""
    ensure_dir(save_dir)

    # Table: Baseline Comparison (Table III in paper)
    lines = [
        r"\begin{table}[htbp]",
        r"\caption{Baseline Model Comparison}",
        r"\label{tab:baseline_comp}",
        r"\centering",
        r"\begin{tabular}{|l|c|c|c|c|}",
        r"\hline",
        r"Model & Accuracy & Precision & Recall & F1-score \\",
        r"\hline",
    ]
    for name, res in all_results.items():
        display = name.replace("_", " ").title()
        if name == "trifuse":
            display = "TriFuse (Proposed)"
        lines.append(
            f"{display} & {res['accuracy']*100:.2f}\\% & "
            f"{res['precision']*100:.2f}\\% & "
            f"{res['recall']*100:.2f}\\% & "
            f"{res['f1_score']*100:.2f}\\% \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    path = os.path.join(save_dir, "baseline_comparison.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines))

    # Table: K-Fold CV (Table VII in paper)
    lines = [
        r"\begin{table}[htbp]",
        r"\caption{Stratified 5-Fold Cross-Validation Results (Mean $\pm$ Std)}",
        r"\label{tab:cross_val}",
        r"\centering",
        r"\begin{tabular}{|l|c|c|}",
        r"\hline",
        r"Model & Accuracy & F1-score \\",
        r"\hline",
    ]
    for name, res in kfold_results.items():
        display = name.replace("_", " ").title()
        if name == "trifuse":
            display = r"\textbf{TriFuse (Proposed)}"
        ma = res["mean_accuracy"] * 100
        sa = res["std_accuracy"] * 100
        mf = res["mean_f1_score"] * 100
        sf = res.get("std_f1_score", 0) * 100
        lines.append(
            f"{display} & {ma:.2f} $\\pm$ {sa:.2f}\\% & "
            f"{mf:.2f} $\\pm$ {sf:.2f}\\% \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    path = os.path.join(save_dir, "kfold_results.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines))

    # Table: K-Fold CV with all 4 metrics (extended version)
    lines = [
        r"\begin{table}[htbp]",
        r"\caption{Stratified 5-Fold Cross-Validation — Full Metrics (Mean $\pm$ Std)}",
        r"\label{tab:cross_val_full}",
        r"\centering",
        r"\resizebox{\columnwidth}{!}{",
        r"\begin{tabular}{|l|c|c|c|c|}",
        r"\hline",
        r"Model & Accuracy & Precision & Recall & F1-score \\",
        r"\hline",
    ]
    for name, res in kfold_results.items():
        display = name.replace("_", " ").title()
        if name == "trifuse":
            display = r"\textbf{TriFuse (Proposed)}"
        ma = res["mean_accuracy"] * 100
        sa = res["std_accuracy"] * 100
        mp = res.get("mean_precision", 0) * 100
        sp = res.get("std_precision", 0) * 100
        mr = res.get("mean_recall", 0) * 100
        sr = res.get("std_recall", 0) * 100
        mf = res["mean_f1_score"] * 100
        sf = res.get("std_f1_score", 0) * 100
        lines.append(
            f"{display} & {ma:.2f} $\\pm$ {sa:.2f}\\% & "
            f"{mp:.2f} $\\pm$ {sp:.2f}\\% & "
            f"{mr:.2f} $\\pm$ {sr:.2f}\\% & "
            f"{mf:.2f} $\\pm$ {sf:.2f}\\% \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"}", r"\end{table}"]
    path = os.path.join(save_dir, "kfold_results_full.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines))

    # Table: Ablation Study (single-branch + pairwise + no-attention)
    ablation_keys = [k for k in all_results
                     if k in ("lexical_only", "semantic_only", "structural_only",
                              "lexical_semantic", "lexical_structural",
                              "semantic_structural", "no_attention", "trifuse")]
    if ablation_keys:
        display_map = {
            "lexical_only": "Lexical Only",
            "semantic_only": "Semantic Only",
            "structural_only": "Structural Only",
            "lexical_semantic": "Lexical + Semantic",
            "lexical_structural": "Lexical + Structural",
            "semantic_structural": "Semantic + Structural",
            "no_attention": "No Attention",
            "trifuse": "TriFuse (Full)",
        }
        lines = [
            r"\begin{table}[htbp]",
            r"\caption{Ablation Study Results}",
            r"\label{tab:ablation_full}",
            r"\centering",
            r"\begin{tabular}{|l|c|c|c|c|}",
            r"\hline",
            r"Model Variant & Accuracy & Precision & Recall & F1-score \\",
            r"\hline",
        ]
        for key in ablation_keys:
            res = all_results[key]
            display = display_map.get(key, key)
            lines.append(
                f"{display} & {res['accuracy']*100:.2f}\\% & "
                f"{res['precision']*100:.2f}\\% & "
                f"{res['recall']*100:.2f}\\% & "
                f"{res['f1_score']*100:.2f}\\% \\\\"
            )
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
        path = os.path.join(save_dir, "ablation_full.tex")
        with open(path, "w") as f:
            f.write("\n".join(lines))

    print(f"LaTeX tables saved to {save_dir}")
