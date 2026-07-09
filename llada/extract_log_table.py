#!/usr/bin/env python3
"""Extract evaluation metrics from CORA/LLaDA log files.

Usage:
  python extract_log_table.py output/other_tasks/mbpp_g256_s256_full
  python extract_log_table.py output/other_tasks/math_g256_s256_4shot_limit20 --format csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


RUNTIME_PATTERNS = {
    "tokens": re.compile(r"^\[GPU\d+\]\s+Number of tokens:\s+(" + FLOAT + r")"),
    "time_s": re.compile(r"^\[GPU\d+\]\s+Generation time:\s+(" + FLOAT + r")"),
    "tps": re.compile(r"^\[GPU\d+\]\s+Tokens per second:\s+(" + FLOAT + r")"),
    "step_ratio": re.compile(r"^\[GPU\d+\]\s+CORA actual denoising step ratio:\s+(" + FLOAT + r")"),
    "fast_accept": re.compile(r"^\[GPU\d+\]\s+CORA fast accept tokens:\s+(" + FLOAT + r")"),
    "residual_accept": re.compile(r"^\[GPU\d+\]\s+CORA residual accept tokens:\s+(" + FLOAT + r")"),
    "avg_residual_score": re.compile(r"^\[GPU\d+\]\s+CORA avg residual score:\s+(" + FLOAT + r")"),
    "eot_truncated": re.compile(r"^\[GPU\d+\]\s+CORA EoT truncated samples:\s+(" + FLOAT + r")"),
}


def clean_cell(cell: str) -> str:
    return cell.strip().replace("_", "").replace("↑", "").replace("±", "")


def parse_metric_rows(text: str) -> dict[str, str]:
    """Parse lm-eval markdown tables and return compact metric columns."""
    metrics: dict[str, str] = {}
    last_task = ""

    for line in text.splitlines():
        if not line.startswith("|") or "Metric" in line or set(line.strip()) <= {"|", "-", ":"}:
            continue

        cells = [clean_cell(c) for c in line.strip().strip("|").split("|")]
        if len(cells) < 7:
            continue

        task = cells[0] or last_task
        if task:
            last_task = task

        metric = cells[4] if len(cells) > 4 else ""
        value = cells[6] if len(cells) > 6 else ""
        stderr = cells[8] if len(cells) > 8 else ""
        filt = cells[2] if len(cells) > 2 else ""
        nshot = cells[3] if len(cells) > 3 else ""

        if not metric or not value:
            continue
        if metric not in {"exact_match", "pass_at_1", "pass@1", "non_greedy_accuracy", "acc"}:
            continue

        # Group rows have empty n-shot. Keep them as aggregate metrics.
        if task.startswith("score_non_greedy_robustness_math") and metric == "non_greedy_accuracy":
            key = "math_non_greedy_acc"
        elif filt in {"flexible-extract", "strict-match"}:
            key = f"{filt}_{metric}".replace("-", "_")
        elif metric in {"pass_at_1", "pass@1"}:
            key = "pass_at_1"
        else:
            key = metric

        if key not in metrics:
            metrics[key] = value
            if stderr and stderr != "N/A":
                metrics[f"{key}_stderr"] = stderr
        if nshot:
            metrics["n_shot"] = nshot

    return metrics


def parse_log(path: Path) -> dict[str, str]:
    text = path.read_text(errors="replace")
    row: dict[str, str] = {"method": path.stem, "log": str(path)}

    for key, pattern in RUNTIME_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            row[key] = matches[-1]

    row.update(parse_metric_rows(text))
    return row


def format_float(value: str) -> str:
    if value == "":
        return ""
    try:
        x = float(value)
    except ValueError:
        return value
    if abs(x) >= 100:
        return f"{x:.2f}"
    if abs(x) >= 10:
        return f"{x:.4g}"
    return f"{x:.4f}"


def print_markdown(rows: list[dict[str, str]], fields: list[str]) -> None:
    print("| " + " | ".join(fields) + " |")
    print("| " + " | ".join(["---"] * len(fields)) + " |")
    for row in rows:
        print("| " + " | ".join(format_float(row.get(f, "")) for f in fields) + " |")


def print_csv(rows: list[dict[str, str]], fields: list[str]) -> None:
    writer = csv.DictWriter(
        __import__("sys").stdout,
        fieldnames=fields,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Log files or directories containing *.log")
    parser.add_argument("--format", choices=["md", "csv"], default="md")
    args = parser.parse_args()

    log_files: list[Path] = []
    for item in args.paths:
        path = Path(item)
        if path.is_dir():
            log_files.extend(sorted(path.glob("*.log")))
        else:
            log_files.append(path)

    rows = [parse_log(p) for p in log_files if p.exists()]

    fields = [
        "method",
        "tokens",
        "time_s",
        "tps",
        "step_ratio",
        "fast_accept",
        "residual_accept",
        "eot_truncated",
        "n_shot",
        "flexible_extract_exact_match",
        "strict_match_exact_match",
        "pass_at_1",
        "math_non_greedy_acc",
    ]

    if args.format == "csv":
        print_csv(rows, fields)
    else:
        print_markdown(rows, fields)


if __name__ == "__main__":
    main()
