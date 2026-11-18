"""Audit frozen scanner evidence and run a size-matched transfer sensitivity."""
import argparse
import ast
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
for name in ("code-root", "results-root", "benchmark", "archived-scans", "out"):
    parser.add_argument("--" + name, required=True, type=Path)
args = parser.parse_args()
sys.path.insert(0, str(args.code_root))
from evaluate import KEY, confusion, fit, load_table, measures, scores

df = load_table(args.results_root / "feature_table.csv")
stored = pd.read_csv(args.results_root / "prompt_cv/predictions.csv")
assert stored[KEY + ["label"]].equals(df[KEY + ["label"]])
args.out.mkdir(parents=True, exist_ok=True)

# Scan coverage is observed at file level. A finding's CWE metadata is a rule tag,
# not a manually adjudicated target weakness.
archives = {}
scan_rows = []
for generator, part in df.groupby("generator"):
    report = json.loads((args.archived_scans / f"semgrep_{generator}.json").read_text(encoding="utf8"))
    def identity(path):
        return tuple(path.replace(chr(92), "/").split("/")[-2:])
    scanned = [identity(p) for p in report["paths"]["scanned"]]
    expected = set(part[["cwe", "sample_id"]].itertuples(index=False, name=None))
    assert len(scanned) == len(set(scanned)) and set(scanned) == expected
    flags = set()
    for finding in report["results"]:
        cwe, sample = identity(finding["path"])
        assert (cwe, sample) in expected
        flags.add((cwe, sample))
        tags = sorted(set(re.findall(r"CWE-\d+", json.dumps(finding["extra"].get("metadata", {}).get("cwe", [])))))
        normalized = {f"CWE-{int(tag.split('-')[1]):03d}" for tag in tags}
        scan_rows.append(dict(generator=generator, cwe=cwe, sample_id=sample,
                              rule=finding["check_id"], line=finding["start"]["line"],
                              tags=";".join(tags), target_tag_match=cwe in normalized))
    assert set(part.loc[part.semgrep_flagged.eq(1), ["cwe", "sample_id"]].itertuples(index=False, name=None)) == flags
    selected = [row for row in scan_rows if row["generator"] == generator]
    archives[generator] = dict(version=report["version"], files_scanned=len(scanned),
        errors=len(report["errors"]), skipped_rule_entries=len(report["skipped_rules"]),
        profiling_rule_entries=len(report["time"]["rules"]), findings=len(selected),
        rule_ids_with_findings=len({r["rule"] for r in selected}), flagged_files=len(flags),
        findings_with_cwe_tags=sum(bool(r["tags"]) for r in selected),
        files_with_target_tag_match=len({(r["cwe"], r["sample_id"]) for r in selected if r["target_tag_match"]}))
pd.DataFrame(scan_rows).to_csv(args.out / "archived_semgrep_findings.csv", index=False)

codeql = {}
disagreements = []
for generator, part in df.groupby("generator"):
    p = args.benchmark / "Result" / f"testcases_{generator}.csv"
    summary = pd.read_csv(p).dropna(subset=["CWEID", "SampleID"]).rename(columns={"CWEID": "cwe", "SampleID": "sample_id"})
    joined = part.merge(summary[["cwe", "sample_id", "Manual", "CodeQL"]], on=["cwe", "sample_id"],
                        how="left", validate="one_to_one")
    assert joined.label.eq(joined.Manual).all() and joined.CodeQL.isin([0, 1]).all()
    unequal = joined.loc[joined.codeql_flagged.ne(joined.CodeQL), KEY + ["label", "codeql_flagged", "CodeQL"]]
    disagreements.append(unequal)
    codeql[generator] = dict(both=int((joined.codeql_flagged.eq(1) & joined.CodeQL.eq(1)).sum()),
        summary_only=int((joined.codeql_flagged.eq(0) & joined.CodeQL.eq(1)).sum()),
        export_only=int((joined.codeql_flagged.eq(1) & joined.CodeQL.eq(0)).sum()),
        export_only_positives=int(unequal.loc[unequal.codeql_flagged.eq(1), "label"].sum()))
pd.concat(disagreements).to_csv(args.out / "codeql_source_disagreements.csv", index=False)

fingerprints = defaultdict(set)
parse_errors = 0
for row in df.itertuples(index=False):
    try:
        tree = ast.parse(row.source_text, type_comments=True)
    except (SyntaxError, ValueError, RecursionError):
        parse_errors += 1
        continue
    fingerprints[ast.dump(tree)].add(row.prompt_group_key)
collisions = sorted(sorted(groups) for groups in fingerprints.values() if len(groups) > 1)
with (args.out / "scanner_and_source_audit.json").open("w", encoding="utf8") as handle:
    json.dump(dict(archived_semgrep=archives, codeql=codeql, ast_parse_errors=parse_errors,
                   cross_prompt_ast_collisions=collisions), handle, indent=2)
    handle.write("\n")
print("Finished frozen scanner and source audit.", flush=True)

# Fixed outer folds and target rows. The overlap condition includes the source
# counterpart of every test prompt. Its remaining source rows are sampled from
# the disjoint pool without replacement, so training size is the same per fit.
y = df.label.to_numpy()
generator_column = df.generator.to_numpy()
fold_column = stored.fold.to_numpy()
transfer_rows, train_rows = [], []
for direction, (source, target) in enumerate((("copilot", "incoder"), ("incoder", "copilot"))):
    for fold in sorted(set(fold_column)):
        disjoint = np.flatnonzero((generator_column == source) & (fold_column != fold))
        overlap_source = np.flatnonzero((generator_column == source) & (fold_column == fold))
        test = np.flatnonzero((generator_column == target) & (fold_column == fold))
        test_prompts = set(df.iloc[test].prompt_group_key)
        assert set(df.iloc[overlap_source].prompt_group_key) == test_prompts
        assert not (test_prompts & set(df.iloc[disjoint].prompt_group_key))
        for rep in [-1, *range(5)]:
            if rep == -1:
                train = disjoint
            else:
                rng = np.random.default_rng([42, direction, fold, rep])
                extra = rng.choice(disjoint, len(disjoint) - len(overlap_source), replace=False)
                train = np.sort(np.r_[overlap_source, extra])
            assert len(train) == len(disjoint) and len(set(train)) == len(train)
            assert len(set(y[train])) == 2
            model = fit(df, y, train, "random_forest")
            held = scores(model, "random_forest", df.iloc[test])
            for idx, score in zip(test, held):
                transfer_rows.append(dict(source=source, target=target, fold=fold, rep=rep, row=int(idx),
                                          label=int(y[idx]), score=float(score), pred=int(score >= 0.5)))
            train_rows.extend(dict(source=source, target=target, fold=fold, rep=rep, row=int(idx))
                              for idx in train)
        print(f"Transfer {source} to {target}, fold {fold} done", flush=True)
predictions = pd.DataFrame(transfer_rows)
legacy = pd.read_csv(args.results_root / "prompt_cv/transfer_predictions.csv")
original = legacy.loc[legacy.protocol.eq("group_disjoint"), ["source", "target", "row", "score"]]
check = predictions.loc[predictions.rep.eq(-1)].merge(original, on=["source", "target", "row"],
                                                     validate="one_to_one")
assert len(check) == len(original) == 260 and np.allclose(check.score_x, check.score_y, rtol=0, atol=1e-12)
summaries = []
for (source, target, rep), part in predictions.groupby(["source", "target", "rep"]):
    assert len(part) == 130 and part.row.is_unique
    summaries.append(dict(source=source, target=target, rep=rep,
                          **measures(confusion(part.label.to_numpy(), part.pred.to_numpy()))))
predictions.to_csv(args.out / "matched_transfer_predictions.csv", index=False)
pd.DataFrame(train_rows).to_csv(args.out / "matched_transfer_training_rows.csv", index=False)
pd.DataFrame(summaries).to_csv(args.out / "matched_transfer_metrics.csv", index=False)

