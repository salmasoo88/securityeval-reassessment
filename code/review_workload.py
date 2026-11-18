"""Compare file triage decisions with paired resampling of held-out prompt groups."""
import argparse
import csv
from pathlib import Path

import numpy as np


def read(path):
    with path.open(newline="", encoding="utf8") as file:
        return list(csv.DictReader(file))


def run(predictions, benchmark, out):
    rows = read(predictions)
    summary = {(g, r["CWEID"], r["SampleID"]): (int(r["CodeQL"]), int(r["Manual"]))
               for g in ("copilot", "incoder")
               for r in read(benchmark / "Result" / f"testcases_{g}.csv") if r["SampleID"].strip()}
    keys = [(r["generator"], r["cwe"], r["sample_id"]) for r in rows]
    assert len(keys) == len(set(keys)) == len(summary) == 260 and set(keys) == set(summary)
    labels = np.array([int(row["label"]) for row in rows])
    assert labels.sum() == 184 and all(summary[k][1] == y for k, y in zip(keys, labels))
    methods = ("random_forest_nested", "random_forest_fixed", "union2", "union3")
    decisions = {m: np.array([int(float(row[m])) for row in rows]) for m in methods}
    decisions["union3_summary"] = np.maximum(decisions["union2"], [summary[k][0] for k in keys])
    assert all(np.isin(v, [0, 1]).all() for v in decisions.values())
    group = np.array([row["group"] for row in rows])
    names = sorted(set(group))
    groups = [np.flatnonzero(group == g) for g in names]
    assert len(groups) == 130 and all(len(ix) == 2 for ix in groups)
    assert all(len({rows[i]["fold"] for i in ix}) == 1 for ix in groups)
    draws = np.random.default_rng(42).integers(len(groups), size=(5000, len(groups)))
    output = []
    for comparator in methods[1:] + ("union3_summary",):
        forest = decisions["random_forest_nested"]
        baseline = decisions[comparator]
        for label, outcome in [(None, "all_flags"), (1, "benchmark_positive"), (0, "benchmark_negative")]:
            keep = np.ones(len(rows), dtype=bool) if label is None else labels == label
            added = keep & (forest == 1) & (baseline == 0)
            dropped = keep & (forest == 0) & (baseline == 1)
            delta = added.astype(int) - dropped.astype(int)
            per_group = np.array([delta[ix].sum() for ix in groups])
            draws_scaled = 100 * per_group[draws].sum(axis=1) / len(rows)
            lo, hi = np.quantile(draws_scaled, [.025, .975])
            net = int(delta.sum())
            assert net == int(forest[keep].sum() - baseline[keep].sum())
            output.append(dict(comparator=comparator, outcome=outcome, forest_only=int(added.sum()),
                               comparator_only=int(dropped.sum()), net_flags=net, per_100=100 * net / len(rows),
                               lo_per_100=float(lo), hi_per_100=float(hi)))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "paired_review_workload.csv").open("w", newline="", encoding="utf8") as file:
        writer = csv.DictWriter(file, fieldnames=output[0])
        writer.writeheader()
        writer.writerows(output)
    with (out / "file_decisions.csv").open("w", newline="", encoding="utf8") as file:
        fields = ["generator", "cwe", "sample_id", "label", "prompt_group_key", "group", "fold", *decisions]
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow(dict(row, **{name: values[i] for name, values in decisions.items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("predictions", "benchmark", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    run(args.predictions, args.benchmark, args.out)
