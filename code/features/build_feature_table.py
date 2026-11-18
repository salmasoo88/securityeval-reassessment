"""
Joins the manifest (data/build_manifest.py) with:
  - the 16 static-analysis features (static_features.py)
  - the 46 structural/lexical/token features (structural_features.py)
  - the CodeQL baseline flag (codeql_baseline.py, baseline-only, not a
    model feature)
  - the raw source text (kept as a column so TF-IDF can be fit per-fold
    inside the modelling pipeline -- see model/pipeline.py -- rather than
    globally, which would leak test-fold vocabulary into training folds)

Output: features/feature_table.csv, one row per sample, 260 rows.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from static_features import compute_static_features, validate_coverage, zero_row, STATIC_FEATURE_NAMES  # noqa: E402
from structural_features import compute_structural_features, ALL_FEATURE_NAMES  # noqa: E402
from codeql_baseline import load_codeql_flags  # noqa: E402
from build_manifest import build_manifest  # noqa: E402

GENERATOR_DIR = {"copilot": "Testcases_Copilot", "incoder": "Testcases_InCoder"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--securityeval-root", type=Path, required=True)
    ap.add_argument("--static-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "feature_table.csv")
    args = ap.parse_args()

    root = args.securityeval_root
    manifest_rows = build_manifest(root)

    bandit_paths = [args.static_dir / f"bandit_{g}.json" for g in GENERATOR_DIR]
    semgrep_paths = [args.static_dir / f"semgrep_{g}.json" for g in GENERATOR_DIR]
    valid_keys = {(m["generator"], m["cwe"], m["sample_id"]) for m in manifest_rows}
    validate_coverage(bandit_paths, semgrep_paths, valid_keys)
    static_feats = compute_static_features(bandit_paths, semgrep_paths)
    codeql_copilot = load_codeql_flags(root / "Result" / "testcases_copilot")
    codeql_incoder = load_codeql_flags(root / "Result" / "testcases_incoder")

    fieldnames = (
        ["source_filename", "generator", "benchmark_version", "cwe", "sample_id",
         "prompt_group_key", "label", "codeql_flagged"]
        + STATIC_FEATURE_NAMES
        + ALL_FEATURE_NAMES
        + ["source_text"]
    )

    if set(static_feats) - valid_keys:
        raise ValueError("Unmatched analyzer findings")
    if any(m["manual_label"] not in ("0", "1") for m in manifest_rows):
        raise ValueError("Manifest has missing or nonbinary labels")
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for m in manifest_rows:
            key = (m["cwe"], m["sample_id"])
            path = root / m["source_filename"]
            source = path.read_text(encoding="utf-8")

            codeql_set = codeql_copilot if m["generator"] == "copilot" else codeql_incoder

            row = {
                "source_filename": m["source_filename"],
                "generator": m["generator"],
                "benchmark_version": m["benchmark_version"],
                "cwe": m["cwe"],
                "sample_id": m["sample_id"],
                "prompt_group_key": m["prompt_group_key"],
                "label": m["manual_label"],
                "codeql_flagged": 1 if key in codeql_set else 0,
                **static_feats.get((m["generator"], *key), zero_row()),
                **compute_structural_features(source),
                "source_text": source,
            }
            w.writerow(row)

    print(f"Wrote {len(manifest_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
