"""
CodeQL is a *baseline* detector (Section 3.4.3: "standalone static analysis
baselines are Bandit alone, Semgrep alone and CodeQL alone"), not one of
the 16 model features (see static_features.py's docstring for why). This
module reads the CodeQL results already shipped with SecurityEval
(Result/testcases_copilot/*.csv, Result/testcases_incoder/*.csv -- one CSV
per CWE, no header row, path column at index 4 like
'/CWE-079/codeql_1.py') and returns a binary flagged indicator per sample,
for use only when scoring the CodeQL-alone and three-way-union/intersection
baselines against the hybrid pipeline.
"""
from __future__ import annotations

import csv
from pathlib import Path


def load_codeql_flags(result_dir: Path) -> set[tuple[str, str]]:
    """Returns the set of (CWE, filename) keys that CodeQL flagged at least
    once, across every per-CWE results CSV in result_dir."""
    flagged: set[tuple[str, str]] = set()
    paths = sorted(Path(result_dir).glob("results_cwe_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No archived CodeQL reports in {result_dir}")
    for csv_path in paths:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for row in csv.reader(fh):
                if len(row) < 5:
                    continue
                rel_path = row[4]  # e.g. '/CWE-079/codeql_1.py'
                parts = [p for p in rel_path.split("/") if p]
                if len(parts) != 2:
                    continue
                cwe, fname = parts
                flagged.add((cwe, fname))
    return flagged


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", type=Path, required=True)
    args = ap.parse_args()
    flags = load_codeql_flags(args.result_dir)
    print(f"{len(flags)} distinct (CWE, filename) samples flagged by CodeQL in {args.result_dir}")
