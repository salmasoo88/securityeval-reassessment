"""Describe benchmark labels and compare matched human reference implementations."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from evaluate import load_table
from structural_features import compute_structural_features

METRICS = ("loc", "n_chars", "cyclomatic", "n_calls", "n_imports", "max_depth",
           "ast_entropy", "keyword_risk_score", "n_try", "comment_ratio")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("feature-table", "securityeval-root", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    df = load_table(args.feature_table)
    by_cwe = df.groupby(["cwe", "generator"]).label.agg(samples="size", vulnerable="sum").reset_index()
    by_cwe.to_csv(args.out / "cwe_label_counts.csv", index=False)
    labels = df.pivot(index="prompt_group_key", columns="generator", values="label")
    assert labels.notna().all().all()
    counts = labels.groupby(["copilot", "incoder"]).size().rename("prompt_count").reset_index()
    counts.to_csv(args.out / "paired_generator_labels.csv", index=False)
    refs = []
    for path in sorted((args.securityeval_root / "Testcases_Insecure_Code").glob("*/*.py")):
        refs.append(dict(cwe=path.parent.name, sample_id=path.name,
                         **compute_structural_features(path.read_text(encoding="utf8"))))
    human = pd.DataFrame(refs)
    ai = df.loc[df.label == 1].groupby(["cwe", "sample_id"])[list(METRICS)].mean().reset_index()
    matched = ai.merge(human, on=["cwe", "sample_id"], suffixes=("_generated", "_human"),
                       validate="one_to_one")
    matched[["cwe", "sample_id", *[f"{m}_{s}" for m in METRICS for s in ("generated", "human")]]].to_csv(
        args.out / "human_reference_pairs.csv", index=False)
    output = []
    for metric in METRICS:
        diffs = matched[f"{metric}_generated"] - matched[f"{metric}_human"]
        by_group = diffs.groupby(matched.cwe).median()
        plus, minus = (by_group > 0).sum(), (by_group < 0).sum()
        p = binomtest(plus, plus + minus).pvalue if plus + minus else 1.0
        output.append(dict(metric=metric, human_examples=len(human), vulnerable_generated=int(df.label.sum()),
                           vulnerable_prompt_groups=len(ai), matched_prompts=len(matched),
                           matched_cwes=len(by_group), median_prompt_difference=np.median(diffs),
                           cwe_positive=plus, cwe_negative=minus, p_cwe_sign_test=p))
    adj = 0
    for i, row in enumerate(sorted(output, key=lambda x: x["p_cwe_sign_test"])):
        adj = max(adj, min(1, (len(output) - i) * row["p_cwe_sign_test"]))
        row["p_holm"] = adj
    pd.DataFrame(output).to_csv(args.out / "human_reference_metrics.csv", index=False)


if __name__ == "__main__":
    main()

