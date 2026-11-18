"""
Appendix A.1: the 16 static-analysis numeric features, computed from the
Bandit and Semgrep JSON produced by data/run_static_analysis.sh.

Per the dissertation text (3.3.1) and the explicit wording of sa_intersect
("1 if both Bandit and Semgrep reported a finding"), these 16 features are
derived from Bandit + Semgrep only. CodeQL results are read separately
(see codeql_baseline.py) and used only as one of the standalone/ensemble
*baselines* in Section 3.4.3 -- they are NOT one of the 16 model features.
This is the fix for Supervisor Point 6's diagram/text inconsistency: the
features correctly exclude CodeQL; it is Figure 3.3's "Static analysis:
Bandit, Semgrep, CodeQL" box and the main text that need rewording, not
this feature set.

Severity/confidence encodings follow Appendix A.1 exactly:
  Bandit severity/confidence: LOW=1, MEDIUM=2, HIGH=3, 0 if no findings.
  Semgrep severity: INFO=1, WARNING=2, ERROR=3.
"""
from __future__ import annotations

import json
from pathlib import Path

BANDIT_SEV = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
BANDIT_CONF = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
SEMGREP_SEV = {"INFO": 1, "WARNING": 2, "ERROR": 3}

STATIC_FEATURE_NAMES = [
    "bandit_flagged", "bandit_n", "bandit_n_rules", "bandit_max_sev",
    "bandit_max_conf", "bandit_sev_sum", "bandit_high",
    "semgrep_flagged", "semgrep_n", "semgrep_n_rules", "semgrep_max_sev",
    "semgrep_sev_sum",
    "sa_union", "sa_intersect", "sa_total_findings", "sa_agreement",
]


def _bandit_by_file(bandit_json_paths: list[Path]) -> dict[str, list[dict]]:
    """Map a sample's path (as Bandit reports it, e.g.
    'Testcases_Copilot/CWE-020/author_1.py') to its list of findings."""
    by_file: dict[str, list[dict]] = {}
    for p in bandit_json_paths:
        data = json.loads(Path(p).read_text(encoding="utf8"))
        if data["errors"]:
            raise ValueError(f"Analyzer errors in {p}: {data['errors']}")
        for finding in data["results"]:
            by_file.setdefault(finding["filename"], []).append(finding)
    return by_file


def _semgrep_by_file(semgrep_json_paths: list[Path]) -> dict[str, list[dict]]:
    by_file: dict[str, list[dict]] = {}
    for p in semgrep_json_paths:
        data = json.loads(Path(p).read_text(encoding="utf8"))
        if data["errors"]:
            raise ValueError(f"Analyzer errors in {p}: {data['errors']}")
        for finding in data["results"]:
            by_file.setdefault(finding["path"], []).append(finding)
    return by_file


def _norm_key(path: str) -> tuple[str, str, str]:
    """Normalise a tool-reported path to (generator, CWE, filename) so Bandit's and
    Semgrep's different path spellings can be joined on the same key."""
    parts = path.replace(chr(92), "/").split("/")
    # find the CWE-xxx component and take it plus the filename
    cwe = next((p for p in parts if p.startswith("CWE-")), None)
    fname = parts[-1]
    generators = {"Testcases_Copilot": "copilot", "Testcases_InCoder": "incoder"}
    matches = [generators[p] for p in parts if p in generators]
    if len(matches) != 1 or cwe is None or parts[-2] != cwe or not fname.endswith(".py"):
        raise ValueError(f"Unexpected analyzer path: {path}")
    return matches[0], cwe, fname


def validate_coverage(bandit_paths, semgrep_paths, expected):
    """Require positive evidence that each analyzer visited every input."""
    for tool, paths in (("bandit", bandit_paths), ("semgrep", semgrep_paths)):
        scanned = set()
        for path in paths:
            report = json.loads(path.read_text(encoding="utf8"))
            if report["errors"]:
                raise ValueError(f"Analyzer errors in {path}: {report['errors']}")
            files = (set(report["metrics"]) - {"_totals"} if tool == "bandit"
                     else report["paths"]["scanned"])
            keys = [_norm_key(file) for file in files]
            file_field = "filename" if tool == "bandit" else "path"
            findings = {_norm_key(result[file_field]) for result in report["results"]}
            unique_keys = set(keys)
            if len(keys) != len(unique_keys) or scanned & unique_keys:
                raise ValueError(f"Duplicate {tool} scan coverage in {path}")
            if not findings <= unique_keys:
                raise ValueError(f"{tool} finding without scan coverage in {path}")
            scanned.update(unique_keys)
        if scanned != expected:
            raise ValueError(f"{tool} scan coverage mismatch: "
                             f"{len(expected - scanned)} missing, {len(scanned - expected)} extra")


def compute_static_features(
    bandit_json_paths: list[Path], semgrep_json_paths: list[Path]
) -> dict[tuple[str, str, str], dict]:
    """Returns {(generator, CWE, filename): {feature_name: value}} for every sample
    that appears in either tool's output. Samples with no findings from
    either tool still need to be covered by the caller with an
    no-findings row (this function only knows about flagged samples)."""
    bandit_raw = _bandit_by_file(bandit_json_paths)
    semgrep_raw = _semgrep_by_file(semgrep_json_paths)

    bandit_by_key: dict[tuple[str, str, str], list[dict]] = {}
    for path, findings in bandit_raw.items():
        bandit_by_key.setdefault(_norm_key(path), []).extend(findings)

    semgrep_by_key: dict[tuple[str, str, str], list[dict]] = {}
    for path, findings in semgrep_raw.items():
        semgrep_by_key.setdefault(_norm_key(path), []).extend(findings)

    keys = set(bandit_by_key) | set(semgrep_by_key)
    out: dict[tuple[str, str, str], dict] = {}

    for key in keys:
        b_findings = bandit_by_key.get(key, [])
        s_findings = semgrep_by_key.get(key, [])

        b_sevs = [BANDIT_SEV.get(f["issue_severity"], 0) for f in b_findings]
        b_confs = [BANDIT_CONF.get(f["issue_confidence"], 0) for f in b_findings]
        b_rules = {f["test_id"] for f in b_findings}

        s_sevs = [SEMGREP_SEV.get(f.get("extra", {}).get("severity", "").upper(), 0) for f in s_findings]
        s_rules = {f.get("check_id") for f in s_findings}

        bandit_flagged = 1 if b_findings else 0
        semgrep_flagged = 1 if s_findings else 0

        out[key] = {
            "bandit_flagged": bandit_flagged,
            "bandit_n": len(b_findings),
            "bandit_n_rules": len(b_rules),
            "bandit_max_sev": max(b_sevs) if b_sevs else 0,
            "bandit_max_conf": max(b_confs) if b_confs else 0,
            "bandit_sev_sum": sum(b_sevs),
            "bandit_high": sum(1 for s in b_sevs if s == 3),
            "semgrep_flagged": semgrep_flagged,
            "semgrep_n": len(s_findings),
            "semgrep_n_rules": len(s_rules),
            "semgrep_max_sev": max(s_sevs) if s_sevs else 0,
            "semgrep_sev_sum": sum(s_sevs),
            "sa_union": 1 if (bandit_flagged or semgrep_flagged) else 0,
            "sa_intersect": 1 if (bandit_flagged and semgrep_flagged) else 0,
            "sa_total_findings": len(b_findings) + len(s_findings),
            "sa_agreement": 1 if (bandit_flagged == semgrep_flagged) else 0,
        }
    return out


def zero_row() -> dict:
    return {name: int(name == "sa_agreement") for name in STATIC_FEATURE_NAMES}


if __name__ == "__main__":
    import argparse
    import csv

    ap = argparse.ArgumentParser()
    ap.add_argument("--static-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    feats = compute_static_features(
        [args.static_dir / "bandit_copilot.json", args.static_dir / "bandit_incoder.json"],
        [args.static_dir / "semgrep_copilot.json", args.static_dir / "semgrep_incoder.json"],
    )
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["generator", "cwe", "sample_id"] + STATIC_FEATURE_NAMES)
        w.writeheader()
        for (generator, cwe, fname), row in sorted(feats.items()):
            w.writerow({"generator": generator, "cwe": cwe, "sample_id": fname, **row})
    print(f"Wrote {len(feats)} flagged-sample rows to {args.out}")
    print("(samples absent from this file get an no-findings static feature row --")
    print(" build_feature_table.py fills those in from the manifest.)")
