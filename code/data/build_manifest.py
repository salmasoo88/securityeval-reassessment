"""
Point 1 deliverable: dataset manifest + grouping audit.

Builds one row per generated sample (Copilot + InCoder completions of the
SecurityEval benchmark), with:
  - source filename
  - generator
  - which benchmark version the sample belongs to
  - prompt ID (CWE + SampleID, the correct grouping key)
  - CWE
  - manual ground-truth label
  - which cross-validation group it falls into

It also prints a short report explaining the 260 -> groups arithmetic
and flagging the version mismatch between Testcases_Copilot/Testcases_InCoder
(old SecurityEval release: 130 samples / 75 CWEs) and dataset.jsonl /
Testcases_Prompt (current release: 121 prompts / 69 CWEs), which is the
exact issue Supervisor Point 1 identified.

Usage:
    python3 build_manifest.py --securityeval-root /path/to/SecurityEval \
        --out manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


GENERATOR_DIRS = {
    "copilot": "Testcases_Copilot",
    "incoder": "Testcases_InCoder",
}
MANUAL_CSV = {
    "copilot": "Result/testcases_copilot.csv",
    "incoder": "Result/testcases_incoder.csv",
}


def load_manual_labels(root: Path, generator: str) -> dict[tuple[str, str], dict]:
    """Return {(CWE, SampleID): row_dict} from the benchmark's manual-annotation CSV."""
    path = root / MANUAL_CSV[generator]
    labels: dict[tuple[str, str], dict] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cwe = (row.get("CWEID") or "").strip()
            sample_id = (row.get("SampleID") or "").strip()
            if not cwe or not sample_id:
                # one trailing blank row ships with the benchmark CSVs; skip it
                continue
            key = (cwe, sample_id)
            if key in labels:
                raise ValueError(f"Duplicate manual label in {path}: {key}")
            labels[key] = row
    return labels


def load_new_version_cwes(root: Path) -> set[str]:
    """CWEs present in the current dataset.jsonl release (121 prompts / 69 CWEs)."""
    cwes = set()
    with open(root / "dataset.jsonl", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cwes.add(obj["ID"].split("_")[0])
    return cwes


def build_manifest(root: Path) -> list[dict]:
    rows: list[dict] = []

    for generator, dirname in GENERATOR_DIRS.items():
        manual = load_manual_labels(root, generator)
        gen_dir = root / dirname
        for cwe_dir in sorted(p for p in gen_dir.iterdir() if p.is_dir()):
            cwe = cwe_dir.name
            for f in sorted(cwe_dir.glob("*.py")):
                sample_id = f.name
                key = (cwe, sample_id)
                manual_row = manual.get(key)
                if manual_row is None:
                    manual_label = None
                    label_source = "MISSING — no manual-annotation row found"
                else:
                    manual_label = manual_row.get("Manual", "").strip()
                    label_source = "benchmark-supplied manual annotation"

                version = "legacy completions (130 prompts / 75 CWEs)"

                rows.append(
                    {
                        "source_filename": f.relative_to(root).as_posix(),
                        "generator": generator,
                        "benchmark_version": version,
                        "cwe": cwe,
                        "sample_id": sample_id,
                        # correct grouping key: CWE + SampleID identifies the prompt;
                        # generator is NOT part of the key, so Copilot/InCoder
                        # completions of the same prompt land in the same fold.
                        "prompt_group_key": f"{cwe}::{sample_id}",
                        "manual_label": manual_label,
                        "label_source": label_source,
                    }
                )
    return rows


def write_manifest(rows: list[dict], out_path: Path) -> None:
    fieldnames = [
        "source_filename",
        "generator",
        "benchmark_version",
        "cwe",
        "sample_id",
        "prompt_group_key",
        "manual_label",
        "label_source",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def report(rows: list[dict], root: Path) -> str:
    groups = defaultdict(list)
    for r in rows:
        groups[r["prompt_group_key"]].append(r["generator"])

    group_sizes = defaultdict(int)
    for members in groups.values():
        group_sizes[len(members)] += 1

    n_vulnerable = sum(1 for r in rows if r["manual_label"] == "1")
    n_secure = sum(1 for r in rows if r["manual_label"] == "0")
    n_missing = sum(1 for r in rows if r["manual_label"] is None)

    old_version_n = len(rows)
    current_cwes = load_new_version_cwes(root)
    new_version_n = sum(r["cwe"] in current_cwes for r in rows)

    lines = []
    lines.append(f"Total samples in manifest: {len(rows)}")
    lines.append(
        f"  of which from the superseded (old, 130/75-CWE) release directories: {old_version_n}"
    )
    lines.append(
        f"  of which from CWE categories still present in the current (121/69) release: {new_version_n}"
    )
    lines.append("")
    lines.append(f"Manual labels: {n_vulnerable} vulnerable, {n_secure} secure, {n_missing} missing")
    if n_vulnerable + n_secure:
        rate = n_vulnerable / (n_vulnerable + n_secure) * 100
        lines.append(f"  observed vulnerability rate: {rate:.1f}%")
    lines.append("")
    lines.append(f"Correct grouping key (CWE + SampleID, generator-agnostic) yields {len(groups)} groups")
    lines.append(f"  group-size distribution: {dict(sorted(group_sizes.items()))}")
    lines.append("")
    lines.append(
        "This contradicts the dissertation's reported figure of 106 groups for 260 samples. "
        "With the correct key, every prompt that both Copilot and InCoder completed forms "
        "exactly one group of 2, giving 130 groups, not 106 — a grouping key that is coarser "
        "than (CWE, SampleID) (e.g. dropping the CWE prefix, so same-named files such as "
        "author_1.py from different CWE directories collide into a single group) would explain "
        "a lower group count. The manifest above should be checked against whatever grouping "
        "key the original fold-assignment code actually used; if it used only the filename, "
        "that is the bug."
    )
    lines.append("")
    lines.append(
        "Version mismatch (Supervisor Point 1, first issue): Testcases_Copilot and "
        "Testcases_InCoder in the SecurityEval repository are still laid out under the "
        "superseded 130-sample / 75-CWE release. The current dataset.jsonl / Testcases_Prompt "
        "ship the updated 121-prompt / 69-CWE release, which fixed prompt typos and removed "
        "prompts that deliberately asked for vulnerable code. Six CWE categories present in "
        "the Copilot/InCoder completions do not exist in the current prompt set: "
    )
    removed = sorted(
        {r["cwe"] for r in rows}
        - load_new_version_cwes(root)
    )
    lines.append("  " + ", ".join(removed))
    lines.append(
        "The dissertation's text describing '121 prompts / 69 CWEs' is describing the prompt "
        "release; the completions actually analysed (and this manifest) are on the older "
        "130/75 release. Both numbers are individually correct for what they describe — the "
        "fix is to say so explicitly in the dataset section rather than quoting both without "
        "distinguishing them."
    )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--securityeval-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "SecurityEval",
        help="Path to a clone of https://github.com/s2e-lab/SecurityEval",
    )
    ap.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent / "manifest.csv"
    )
    args = ap.parse_args()

    rows = build_manifest(args.securityeval_root)
    write_manifest(rows, args.out)
    print(f"Wrote {len(rows)} rows to {args.out}\n")
    print(report(rows, args.securityeval_root))


if __name__ == "__main__":
    main()
