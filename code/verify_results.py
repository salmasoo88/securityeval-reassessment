"""Check saved predictions and evaluate the benchmark-summary CodeQL sensitivity."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import KEY, CLASSIFIER_NAMES, confusion, group_swap_p, load_table, measures


def verify(table, evaluation, root, out):
    df = load_table(table)
    predictions = pd.read_csv(evaluation / "predictions.csv", float_precision="round_trip")
    inner = pd.read_csv(evaluation / "inner_folds.csv")
    selection = pd.read_csv(evaluation / "inner_selection.csv")
    transfer = pd.read_csv(evaluation / "transfer_predictions.csv")
    assert predictions[KEY + ["label"]].equals(df[KEY + ["label"]])
    assert predictions.groupby("group").fold.nunique().eq(1).all()
    assert set(predictions.fold) == set(range(5))
    for fold, held in predictions.groupby("fold"):
        train_rows = set(predictions.index) - set(held.index)
        valid = inner[inner.outer_fold == fold].set_index("row")
        assert valid.index.is_unique and set(valid.index) == train_rows
        assert predictions.loc[valid.index, "group"].groupby(valid.inner_validation_fold).size().size == 5
        assert pd.DataFrame({"g": predictions.loc[valid.index, "group"],
                             "f": valid.inner_validation_fold}).groupby("g").f.nunique().eq(1).all()
        winners = selection[(selection.fold == fold) & selection.selected]
        assert len(winners) == 1
        assert held.selected_name.eq(winners.classifier.iloc[0]).all()
        assert held.selected_nested.equals(held[winners.classifier.iloc[0] + "_nested"])
        for classifier in CLASSIFIER_NAMES:
            score = held[classifier + "_score"]
            cutoff = held[classifier + "_threshold"]
            assert np.isfinite(score).all() and cutoff.nunique() == 1
            assert ((score >= cutoff).astype(int) == held[classifier + "_nested"]).all()
    for (protocol, source, target), part in transfer.groupby(["protocol", "source", "target"]):
        assert set(part.row) == set(df.index[df.generator == target]) and part.row.is_unique
        if protocol == "group_disjoint":
            assert (predictions.loc[part.row, "fold"].to_numpy() == part.fold).all()
    for _, result in pd.read_csv(evaluation / "metrics.csv").iterrows():
        counts = confusion(df.label.to_numpy(), predictions[result.method].to_numpy(dtype=int))
        for key, val in measures(counts).items():
            assert np.isclose(result[key], val), (result.method, key)
    flags = []
    for g in df.generator.unique():
        file = root / "Result" / f"testcases_{g}.csv"
        part = pd.read_csv(file).dropna(subset=["CWEID", "SampleID"])
        flags.append(part.rename(columns={"CWEID": "cwe", "SampleID": "sample_id"}).assign(generator=g))
    benchmark = df[KEY + ["label"]].merge(pd.concat(flags)[KEY + ["CodeQL", "Manual"]], how="left",
                                          on=KEY, validate="one_to_one")
    assert benchmark.Manual.eq(df.label).all() and benchmark.CodeQL.isin([0, 1]).all()
    y = df.label.to_numpy()
    rf = predictions.random_forest_nested.to_numpy()
    codeql = benchmark.CodeQL.to_numpy(dtype=int)
    union = np.maximum(codeql, predictions.union2)
    rows = []
    for name, pred in (("benchmark_summary_codeql", codeql), ("union3_using_benchmark_summary", union)):
        difference = (rf == y).astype(int) - (pred == y).astype(int)
        paired = pd.Series(difference).groupby(predictions.group).sum().to_numpy()
        rows.append(dict(method=name, **measures(confusion(y, pred)),
                         delta_rf_accuracy=float(difference.mean()),
                         p_cluster_accuracy_unadjusted=group_swap_p(paired)))
    pd.DataFrame(rows).to_csv(out, index=False)
    strata = []
    for unit in ("generator", "cwe"):
        for category, part in predictions.groupby(unit):
            for method in ("random_forest_nested", "random_forest_fixed", "selected_nested", "union2", "union3"):
                strata.append(dict(unit=unit, category=category, method=method,
                                   n=len(part), positives=int(part.label.sum()),
                                   **measures(confusion(part.label.to_numpy(), part[method].to_numpy(dtype=int)))))
    pd.DataFrame(strata).to_csv(evaluation / "stratified_metrics.csv", index=False)
    print(f"Verified {evaluation.name}: prediction identities, inner/outer split isolation, thresholds, transfer coverage and metric arithmetic")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("feature-table", "evaluation", "securityeval-root", "out"):
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    verify(args.feature_table, args.evaluation, args.securityeval_root, args.out)

