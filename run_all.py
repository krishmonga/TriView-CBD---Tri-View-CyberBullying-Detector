#!/usr/bin/env python3
"""
Run TriFuse experiments on both Davidson and Shahane datasets,
generate results files for each, and produce a combined comparison log.

Usage:
    python run_all.py              # Full run on both datasets
    python run_all.py --quick      # Quick 10-epoch run for testing
    python run_all.py --results    # Just regenerate combined log from existing results
"""

import os, sys, json, subprocess, argparse, gc
from datetime import datetime
import yaml

# Fix Windows console encoding for Unicode box-drawing characters
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.abspath(__file__))


def run_dataset(ds_name, data_path, output_dir, quick=False):
    """Run main.py for a single dataset. Returns exit code."""
    config_path = os.path.join(ROOT, "configs", "config.yaml")
    with open(config_path) as f:
        base_config = yaml.safe_load(f)

    ds_config = dict(base_config)
    rel_out = os.path.relpath(output_dir, ROOT)
    ds_config["paths"] = {
        "base_output": rel_out + "/",
        "models_dir": rel_out + "/models/",
        "plots_dir": rel_out + "/plots/",
        "results_dir": rel_out + "/results/",
        "logs_dir": rel_out + "/logs/",
    }
    tmp_config = os.path.join(ROOT, "configs", f"config_{ds_name}.yaml")
    with open(tmp_config, "w") as f:
        yaml.dump(ds_config, f, default_flow_style=False)

    cmd = [
        sys.executable, os.path.join(ROOT, "main.py"),
        "--mode", "full", "--data_path", data_path, "--config", tmp_config,
    ]
    if quick:
        cmd.append("--quick")

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "run.log")

    print(f"\n{'='*70}")
    print(f"  STARTING: {ds_name} ({data_path})")
    print(f"  Output:   {output_dir}")
    print(f"  Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    with open(log_path, "w", encoding="utf-8") as log_file:
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", cwd=ROOT,
            )
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_file.write(line)
                log_file.flush()
            proc.wait()
        except Exception as e:
            print(f"\n  ERROR running {ds_name}: {e}")
            log_file.write(f"\nERROR: {e}\n")
            return -1

    code = proc.returncode
    print(f"\n  Finished {ds_name} -- exit code {code}")
    if code == -9:
        print(f"  WARNING: Process was killed (OOM). Partial results may exist.")
    return code


def load_report(output_dir):
    """Load comprehensive_report.json from an output directory."""
    candidates = [
        os.path.join(output_dir, "results", "comprehensive_report.json"),
        os.path.join(output_dir, "comprehensive_report.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            with open(path, "r") as f:
                return json.load(f)
    return {}


def find_existing_reports():
    """Search known directories for existing reports."""
    reports = {}
    for ds_name, dirs in [
        ("davidson", ["DavidsonResult", "Davisonoutputs", "outputs_davidson"]),
        ("shahane", ["outputs_shahane", "outputs"]),
    ]:
        for d in dirs:
            r = load_report(os.path.join(ROOT, d))
            if r:
                reports[ds_name] = r
                break
    return reports


def create_combined_log(reports, log_path):
    """Create a side-by-side comparison log file."""
    lines = []
    lines.append("=" * 90)
    lines.append("  TriFuse -- Combined Results Comparison")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 90)
    lines.append("")

    datasets = list(reports.keys())
    if not datasets:
        lines.append("  NO RESULTS AVAILABLE")
        text = "\n".join(lines)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(text)
        return

    # Single-split
    lines.append("-" * 90)
    lines.append("  SINGLE-SPLIT RESULTS")
    lines.append("-" * 90)
    all_models = sorted(set(
        m for ds in datasets for m in reports[ds].get("single_split_results", {})
    ))
    header = f"  {'Model':<22s}"
    for ds in datasets:
        header += f" | {ds+' Acc':>14s} {ds+' F1':>14s}"
    lines.append(header)
    lines.append("  " + "-" * (22 + 31 * len(datasets)))
    for model in all_models:
        row = f"  {'* TRIFUSE *' if model == 'trifuse' else model:<22s}"
        for ds in datasets:
            r = reports[ds].get("single_split_results", {}).get(model, {})
            acc = r.get("accuracy", 0) * 100
            f1 = r.get("f1_score", 0) * 100
            if acc > 0:
                row += f" | {acc:>13.2f}% {f1:>13.2f}%"
            else:
                row += f" | {'N/A':>13s}  {'N/A':>13s} "
        lines.append(row)

    # K-fold
    lines.append("")
    lines.append("-" * 90)
    lines.append("  K-FOLD CROSS-VALIDATION RESULTS (Mean +/- Std)")
    lines.append("-" * 90)
    kfold_models = sorted(set(
        m for ds in datasets for m in reports[ds].get("kfold_results", {})
    ))
    header = f"  {'Model':<22s}"
    for ds in datasets:
        header += f" | {ds+' Acc':>14s} {ds+' F1':>14s}"
    lines.append(header)
    lines.append("  " + "-" * (22 + 31 * len(datasets)))
    for model in kfold_models:
        row = f"  {'* TRIFUSE *' if model == 'trifuse' else model:<22s}"
        for ds in datasets:
            r = reports[ds].get("kfold_results", {}).get(model, {})
            ma = r.get("mean_accuracy", 0) * 100
            sa = r.get("std_accuracy", 0) * 100
            mf = r.get("mean_f1_score", 0) * 100
            sf = r.get("std_f1_score", 0) * 100
            if ma > 0:
                row += f" | {ma:5.2f}+/-{sa:4.2f}% {mf:5.2f}+/-{sf:4.2f}%"
            else:
                row += f" | {'N/A':>13s}  {'N/A':>13s} "
        lines.append(row)

    # Winner summary
    lines.append("")
    lines.append("=" * 90)
    lines.append("  WINNER SUMMARY")
    lines.append("=" * 90)
    for ds in datasets:
        lines.append(f"\n  Dataset: {ds}")
        sr = reports[ds].get("single_split_results", {})
        if sr:
            best = max(sr, key=lambda m: sr[m].get("accuracy", 0))
            lines.append(f"    Best: {best} ({sr[best]['accuracy']*100:.2f}%)")
            tf = sr.get("trifuse", {}).get("accuracy", 0) * 100
            lines.append(f"    TriFuse: {tf:.2f}%")
            if best == "trifuse":
                lines.append("    >> TriFuse is the BEST model!")
            else:
                lines.append(f"    Gap: {sr[best]['accuracy']*100 - tf:.2f}% behind {best}")

    text = "\n".join(lines)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


def clear_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--results", action="store_true",
                        help="Regenerate combined log from existing results only")
    parser.add_argument("--datasets", nargs="+", default=["shahane", "davidson"])
    args = parser.parse_args()

    ds_configs = {
        "shahane": ("dataset/", os.path.join(ROOT, "outputs_shahane")),
        "davidson": ("dataset_davidson/", os.path.join(ROOT, "outputs_davidson")),
    }

    if args.results:
        reports = find_existing_reports()
        for ds in args.datasets:
            if ds not in reports and ds in ds_configs:
                r = load_report(ds_configs[ds][1])
                if r:
                    reports[ds] = r
        if reports:
            create_combined_log(reports, os.path.join(ROOT, "combined_results_log.txt"))
            with open(os.path.join(ROOT, "combined_results.json"), "w") as f:
                json.dump(reports, f, indent=2)
        else:
            print("No existing reports found!")
        return

    # Prepare Davidson if needed
    if "davidson" in args.datasets:
        dd = os.path.join(ROOT, "dataset_davidson")
        if not os.path.isdir(dd) or not any(f.endswith(".csv") for f in os.listdir(dd)):
            subprocess.run([sys.executable, os.path.join(ROOT, "prepare_davidson.py")],
                           cwd=ROOT, check=True)

    reports = {}
    for ds in args.datasets:
        if ds not in ds_configs:
            continue
        data_path, out_dir = ds_configs[ds]
        code = run_dataset(ds, data_path, out_dir, quick=args.quick)
        r = load_report(out_dir)
        if r:
            reports[ds] = r
        clear_gpu()

    # Also check for pre-existing reports
    existing = find_existing_reports()
    for ds, r in existing.items():
        if ds not in reports:
            reports[ds] = r

    if reports:
        create_combined_log(reports, os.path.join(ROOT, "combined_results_log.txt"))
        with open(os.path.join(ROOT, "combined_results.json"), "w") as f:
            json.dump(reports, f, indent=2)
    else:
        print("\nNo results generated. Check run.log files for details.")


if __name__ == "__main__":
    main()
